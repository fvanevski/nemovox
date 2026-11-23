"""Read-only offline replay of Nemovox source evidence from cached ASR chunks.

No client, model, FFmpeg, GPU, or mutable ASR job is needed. The replay uses
native CTC parsing only; it never claims to reconstruct optional prior
subinterval/forced-alignment or speaker-embedding results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from nvidia_transcribe.alignment import recover_chunk_alignment
from nvidia_transcribe.asr_integrity import audit_asr_generation
from nvidia_transcribe.chunking import ChunkSpec
from nvidia_transcribe.prompts import cleanup_generation_loops
from nvidia_transcribe.source_events import build_source_evidence_ledger
from nvidia_transcribe.speakers import normalize_chunk_speakers
from nvidia_transcribe.timing_evidence import audit_ctc_timing_units


def _hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def replay_cached_source_evidence(root: Path, *, forensic_incomplete: bool = False) -> dict[str, Any]:
    """Authenticate each cached raw chunk and build the shadow evidence ledger.

    Legacy complete manifests without `asr_run_status` are supported. An
    incomplete response, missing chunk, or changed raw digest is a hard error.
    """
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("asr_run_status") not in (None, "completed") and not forensic_incomplete:
        raise ValueError("Cached ASR manifest is not completed")
    meta = manifest.get("media") or {}
    fingerprint = meta.get("fingerprint_sha256")
    if not fingerprint:
        raise ValueError("Missing media fingerprint in cached manifest")
    entries = manifest.get("chunks") or []
    if not entries:
        raise ValueError("No cached ASR chunk entries")

    specs: list[ChunkSpec] = []
    chunk_units: list[list[dict[str, Any]]] = []
    raw_digests: list[str] = []
    raw_texts: list[str] = []
    diar: list[list[dict[str, Any]]] = []
    seen_indices: set[int] = set()
    rejected_chunks: list[dict[str, Any]] = []

    for entry in entries:
        if entry.get("status") != "completed" and not forensic_incomplete:
            raise ValueError("Cached ASR chunk was not completed")
        idx = int(entry["chunk_index"])
        if idx in seen_indices:
            raise ValueError(f"Duplicate chunk index {idx}")
        seen_indices.add(idx)
        spec = ChunkSpec(
            chunk_index=idx,
            core_start_s=float(entry["core_start_s"]),
            core_end_s=float(entry["core_end_s"]),
            infer_start_s=float(entry["infer_start_s"]),
            infer_end_s=float(entry["infer_end_s"]),
        )
        # The expected relative path prevents a hostile/stale absolute path in
        # an archived manifest from causing reads outside the selected job.
        raw_path = root / "chunks" / f"chunk_{idx:03d}.raw.json"
        raw_dump = raw_path.read_text(encoding="utf-8")
        digest = _hash(raw_dump)
        if digest != entry.get("raw_response_sha256"):
            raise ValueError(f"Cached raw-response digest mismatch at chunk {idx}")
        raw = json.loads(raw_dump)
        integrity = audit_asr_generation(raw)
        if integrity["status"] != "complete" or entry.get("status") != "completed":
            if not forensic_incomplete:
                raise ValueError(f"Cached chunk {idx} is incomplete ({integrity['reason']})")
            rejected_chunks.append({
                "chunk_index": idx, "raw_response_sha256": digest,
                "finish_status": integrity["status"], "reason": integrity["reason"],
                "manifest_chunk_status": entry.get("status"),
                "infer_start_s": spec.infer_start_s, "infer_end_s": spec.infer_end_s,
                "decision": "no_accepted_lexical_observations_from_this_chunk",
            })
            continue
        text = cleanup_generation_loops(
            (((raw.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
        )
        rec = recover_chunk_alignment(
            raw_text=text,
            ctc_payload=raw.get("ctc_timestamps"),
            source_offset_s=spec.infer_start_s,
            chunk_index=idx,
            model_path=None,
        )
        audited, _ = audit_ctc_timing_units(
            rec["units"], infer_start_s=spec.infer_start_s,
            infer_end_s=spec.infer_end_s,
        )
        normalized, _ = normalize_chunk_speakers(
            chunk_index=idx,
            units=audited,
            diarization=rec["diarization"],
            speaker_mapping=rec["speaker_mapping"],
        )
        specs.append(spec)
        chunk_units.append(normalized)
        raw_digests.append(digest)
        raw_texts.append(text)
        diar.append(rec["diarization"])

    ledger = build_source_evidence_ledger(
        media_fingerprint_sha256=str(fingerprint),
        source_language=str(manifest.get("source_language") or "unverified"),
        chunk_specs=specs,
        chunk_units=chunk_units,
        raw_response_sha256=raw_digests,
        raw_source_texts=raw_texts,
        diarization_by_chunk=diar,
    )
    ledger["replay_provenance"] = {
        "mode": ("offline_forensic_partial" if rejected_chunks else "offline_native_ctc_only"),
        "rejected_or_incomplete_chunks": rejected_chunks,
        "asr_manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "limitations": [
            "no audio listening or VAD",
            "no original subinterval or external forced-aligner replay",
            "no speaker embedding/global identity replay",
            "no canonical source-event acceptance",
        ],
    }
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-dir", type=Path, required=True, help="Completed cached ASR job root")
    parser.add_argument("--output", type=Path, required=True, help="New private JSON file outside ASR job root")
    parser.add_argument("--forensic-incomplete", action="store_true",
                        help="Skip incomplete chunks as explicit coverage gaps; never mark them accepted")
    ns = parser.parse_args()
    root = ns.asr_dir.resolve()
    output = ns.output.resolve()
    if output == root or root in output.parents:
        parser.error("--output must be outside the immutable ASR job")
    if output.exists():
        parser.error("--output already exists (no clobber)")
    ledger = replay_cached_source_evidence(root, forensic_incomplete=ns.forensic_incomplete)
    payload = json.dumps(ledger, ensure_ascii=False, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
    print(json.dumps({
        "schema": ledger["schema_version"],
        "mode": ledger["mode"],
        "output": str(output),
        "output_sha256": _hash(payload),
        "asr_manifest_sha256": ledger["replay_provenance"]["asr_manifest_sha256"],
        "rejected_or_incomplete_chunks": ledger["replay_provenance"]["rejected_or_incomplete_chunks"],
        "diagnostics": ledger["diagnostics"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
