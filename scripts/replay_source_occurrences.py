"""CPU-only replay of sealed private ASR observations; no model or media access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace
from collections import Counter

from nvidia_transcribe.alignment import recover_chunk_alignment
from nvidia_transcribe.prompts import cleanup_generation_loops
from nvidia_transcribe.source_occurrences import bind_occurrence_inventory, source_observation_id
from nvidia_transcribe.occurrence_locality import assess_occurrence_locality


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != args.archive_sha256:
            raise ValueError("Private diagnostic archive digest mismatch")
    raw_by_sha, ledgers = {}, {}
    with tarfile.open(args.archive, "r:gz") as archive:
        # Read regular JSON members only; never extract or execute archive files.
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            if member.size > 32 * 1024 * 1024:
                raise ValueError("Oversize JSON member in diagnostic archive")
            with archive.extractfile(member) as stream:
                data = stream.read()
            try:
                obj = json.loads(data)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(obj, dict):
                continue
            digest = sha(data)
            if "asr_observations" in obj and "media_fingerprint_sha256" in obj:
                ledgers[digest] = obj
            if obj.get("choices") and "ctc_timestamps" in obj:
                raw_by_sha[digest] = obj
    eligible = [doc for doc in ledgers.values() if len(doc["asr_observations"]) == 16
                and all(row["raw_response_sha256"] in raw_by_sha for row in doc["asr_observations"])]
    if len(eligible) != 1:
        raise ValueError("Expected one unique 16-chunk source ledger with all exact raw bytes")
    ledger = eligible[0]
    summary, details = [], []
    for row in sorted(ledger["asr_observations"], key=lambda r: r["chunk_index"]):
        raw = raw_by_sha[row["raw_response_sha256"]]
        spec = SimpleNamespace(**{k: row[k] for k in (
            "chunk_index", "infer_start_s", "infer_end_s", "core_start_s", "core_end_s")})
        text = raw["choices"][0]["message"]["content"]
        projected = cleanup_generation_loops(text)
        rec = recover_chunk_alignment(
            raw_text=projected, ctc_payload=raw["ctc_timestamps"],
            source_offset_s=spec.infer_start_s, chunk_index=spec.chunk_index, model_path=None,
        )
        inv = bind_occurrence_inventory(
            observation_id=source_observation_id(ledger["media_fingerprint_sha256"],
                row["raw_response_sha256"], spec),
            raw_text=text, projected_text=projected, units=rec["units"],
            native_observations=rec["diagnostic"]["native_ctc_observations"],
        )
        units = rec["units"]
        ids = {u["source_unit_id"] for u in units}
        assert len(ids) == len(units)
        assert ids == set(inv["working_source_unit_ids"])
        linked = [o["source_unit_id"] for o in inv["native_ctc_observations"] if "source_unit_id" in o]
        assert len(linked) == len(set(linked)) and set(linked) <= ids
        raw_tokens = {t["id"] for turn in inv["recognized_turns"] for t in turn["tokens"]}
        token_refs = [t for u in units for t in u["raw_recognized_token_ids"]]
        if inv["projection_status"] == "exact":
            assert set(token_refs) == raw_tokens and len(token_refs) == len(raw_tokens)
        else:
            assert not token_refs
        audited, locality = assess_occurrence_locality(
            units, infer_start_s=spec.infer_start_s, infer_end_s=spec.infer_end_s)
        assert {u["source_unit_id"] for u in audited} == ids
        assert sorted(uid for span in locality["spans"] for uid in span["source_unit_ids"]) == sorted(ids)
        summary.append({
            "chunk_index": spec.chunk_index,
            "raw_response_sha256": row["raw_response_sha256"],
            "crop": [spec.infer_start_s, spec.infer_end_s],
            "raw_turns": len(inv["recognized_turns"]),
            "raw_tokens": len(raw_tokens), "working_units": len(units),
            "native_observations": len(inv["native_ctc_observations"]),
            "provisional_associations": len(linked),
            "untimed_working_units": sum(not u["valid_timing"] for u in units),
            "association_statuses": dict(Counter(o["association_status"] for o in inv["native_ctc_observations"])),
            "locality_spans": len(locality["spans"]),
            "locality_span_statuses": dict(Counter(s["locality_status"] for s in locality["spans"])),
            "locality_findings": dict(Counter(f["reason"] for f in locality["findings"])),
            "post_locality_untimed_units": sum(not u["valid_timing"] for u in audited),
            "projection_status": inv["projection_status"],
            "acoustic_locality_certified": False,
        })
        details.append({"inventory": inv, "working_units": units,
                        "locality": locality, "audited_units": audited})
    if len({r["chunk_index"] for r in summary}) != 16:
        raise ValueError("Repeated or incomplete accepted chunk identity")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "archive_sha256": args.archive_sha256, "chunks": summary,
        "note": "Raw replay only; no acoustic truth, boundary-selection or timing-improvement claim."
    }, indent=2), encoding="utf-8")
    (args.output_dir / "private-details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Replayed 16 authenticated raw responses; inspect per-chunk association statuses.")


if __name__ == "__main__":
    main()
