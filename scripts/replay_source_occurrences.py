"""CPU-only replay of sealed private ASR observations; no model or media access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from collections import Counter, defaultdict

from nvidia_transcribe.alignment import recover_chunk_alignment
from nvidia_transcribe.prompts import cleanup_generation_loops
from nvidia_transcribe.source_occurrences import bind_occurrence_inventory, source_observation_id
from nvidia_transcribe.occurrence_locality import assess_occurrence_locality
from nvidia_transcribe.chunking import ChunkSpec, reconcile_chunk_boundaries
from nvidia_transcribe.speakers import normalize_chunk_speakers
from nvidia_transcribe.timing_evidence import audit_ctc_timing_units
from nvidia_transcribe.utterances import reconstruct_speaker_utterances


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
    specs, baseline_chunks, candidate_chunks = [], [], []
    # Freeze historical speaker hypotheses for this matched comparison.
    # This isolates selection/locality behavior from new speaker reconciliation.
    voice_hypotheses = defaultdict(set)
    for unit in ledger.get("lexical_observations", []):
        if unit.get("local_speaker_id") and unit.get("global_speaker_hypothesis"):
            voice_hypotheses[unit["local_speaker_id"]].add(unit["global_speaker_hypothesis"])

    def normalize(units, native, spec):
        normalized, _ = normalize_chunk_speakers(chunk_index=spec.chunk_index,
            units=units, diarization=native["diarization"],
            speaker_mapping=native["speaker_mapping"])
        for unit in normalized:
            candidates = voice_hypotheses[unit["local_speaker_id"]]
            unit["global_speaker_id"] = next(iter(candidates)) if len(candidates) == 1 else "SPK_UNK"
        return normalized
    for row in sorted(ledger["asr_observations"], key=lambda r: r["chunk_index"]):
        raw = raw_by_sha[row["raw_response_sha256"]]
        spec = ChunkSpec(**{k: row[k] for k in (
            "chunk_index", "infer_start_s", "infer_end_s", "core_start_s", "core_end_s")})
        specs.append(spec)
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
        baseline_units, _ = audit_ctc_timing_units(
            units, infer_start_s=spec.infer_start_s, infer_end_s=spec.infer_end_s)
        baseline_chunks.append(normalize(baseline_units, rec, spec))
        candidate_chunks.append(normalize(audited, rec, spec))
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
    all_ids = {u["source_unit_id"] for group in candidate_chunks for u in group}
    def selection(chunk_units):
        selected, diagnostic = reconcile_chunk_boundaries(chunk_units, specs)
        dispositions = diagnostic["source_dispositions"]
        assert len(dispositions) == len(all_ids)
        assert {d["source_unit_id"] for d in dispositions} == all_ids
        assert all(set(d["related_source_unit_ids"]) <= all_ids for d in dispositions)
        selected_ids = {u["source_unit_id"] for u in selected}
        assert len(selected_ids) == len(selected)
        assert selected_ids == {d["source_unit_id"] for d in dispositions
                                if d["status"].startswith("selected_")}
        utterances = reconstruct_speaker_utterances(selected,
            source_language=ledger["source_language"],
            chunk_windows={c.chunk_index: (c.infer_start_s, c.infer_end_s) for c in specs})
        assert {uid for u in utterances for uid in u["source_unit_ids"]} == selected_ids
        return selected, dispositions, utterances

    baseline, old_dispositions, old_rows = selection(baseline_chunks)
    candidate, new_dispositions, new_rows = selection(candidate_chunks)
    old_ids = {u["source_unit_id"] for u in baseline}
    new_ids = {u["source_unit_id"] for u in candidate}
    def metrics(units, dispositions, rows):
        return {"dispositions": dict(Counter(d["status"] for d in dispositions)),
                "selected_units": len(units),
                "selected_untimed_units": sum(not u["valid_timing"] for u in units),
                "utterances": len(rows),
                "wholly_untimed_utterances": sum(u["source_start"] is None for u in rows)}
    comparison = {
        "control": "same candidate lexical association, historical fixed speaker hypotheses, existing timing audit, unit ownership",
        "candidate": "same input and speaker hypotheses, pre-selection locality and coherent span ownership",
        "control_metrics": metrics(baseline, old_dispositions, old_rows),
        "candidate_metrics": metrics(candidate, new_dispositions, new_rows),
        "selected_only_by_control": len(old_ids - new_ids),
        "selected_only_by_candidate": len(new_ids - old_ids),
        "all_original_observations": len(all_ids),
        "acoustic_truth_accepted": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "archive_sha256": args.archive_sha256, "chunks": summary,
        "matched_selection_comparison": comparison,
        "note": "Matched offline policy ablation, not an exact historical executable or independent acoustic truth."
    }, indent=2), encoding="utf-8")
    (args.output_dir / "private-selection.json").write_text(json.dumps({
        "control_dispositions": old_dispositions, "candidate_dispositions": new_dispositions,
        "control_utterances": old_rows, "candidate_utterances": new_rows,
        "selected_only_by_control": sorted(old_ids - new_ids),
        "selected_only_by_candidate": sorted(new_ids - old_ids),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "private-details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Replayed 16 authenticated raw responses; inspect per-chunk association statuses.")


if __name__ == "__main__":
    main()
