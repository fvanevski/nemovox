"""Replay changed occurrence selection on sealed, already observed acoustic crops.

No media extraction, model acquisition, or new worker calls. The acquisition
geometry is frozen to the actually observed campaign, not the new scheduler's
counterfactual plan. Report changes for adjudication rather than require a
particular correction count or force historical output equality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from nvidia_transcribe import source_quality as quality
from nvidia_transcribe.occurrence_resolution import resolve_occurrences, resolution_receipts
from scripts.evaluate_occurrence_recovery import POLICY, verified_archive, write_json
from scripts.replay_recovery_schedule import ARCHIVE_SHA, TESTED_SOURCE, REGIONS

SUPPLEMENT_SHA = "a512f262035860a9f3aaa9ac78b523b517c1cc673573af05eacc9eee2b4ffc4d"
SUPPLEMENT_SOURCE = "ebdc06f0bd2997c06d73e5e32158e07beb82e3a6"


def summarize(rec, rows, decisions, receipts):
    original_ids = {r["id"] for r in rec["utterances"]}
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate canonical row IDs")
    return {
        "original_rows": len(original_ids),
        "canonical_rows": len(rows),
        "missing_original_row_ids": sorted(original_ids - set(ids)),
        "new_row_ids": sorted(set(ids) - original_ids),
        "untimed_before": sum(r.get("source_start") is None for r in rec["utterances"]),
        "untimed_after": sum(r.get("source_start") is None for r in rows),
        "decisions": dict(Counter(d["status"] for d in decisions)),
        "receipts": dict(Counter(r["status"] for r in receipts)),
    }


def replay(archive, supplement, root):
    members = verified_archive(archive, expected_sha=ARCHIVE_SHA, expected_source=TESTED_SOURCE)
    extra = verified_archive(supplement, expected_sha=SUPPLEMENT_SHA, expected_source=SUPPLEMENT_SOURCE)
    public, private = {}, {}
    for case in REGIONS:
        prefix = "private/phase-b-live/" + case + "/"
        rec = json.loads(members[prefix + "recording/recording.contract.json"])
        manifest = json.loads(members[prefix + "recording/manifest.json"])
        ledger_bytes = members[prefix + "recording/source-evidence.v1.json"]
        if hashlib.sha256(ledger_bytes).hexdigest() != manifest["source_evidence"]["sha256"]:
            raise ValueError("Archived ledger digest mismatch")
        ledger = json.loads(ledger_bytes)
        report = json.loads(members[prefix + "source-quality.json"])
        frozen = report["acoustic_observations"]
        if quality._digest(rec) != report["recording_sha256"]:
            raise ValueError("Archived recording digest mismatch")
        crops = {tuple(o["crop"]): o for o in frozen}
        if len(crops) != len(frozen):
            raise ValueError("Duplicate frozen crops")
        for o in frozen:
            if quality._digest(o["raw_response"]) != o["raw_response_sha256"]:
                raise ValueError("Acoustic response digest mismatch")
        requested = []
        def observe(client, media, crop, out, index):
            key = tuple(crop)
            if key not in crops or key in requested:
                raise ValueError("Replay would require new or repeated inference")
            requested.append(key)
            return deepcopy(crops[key])
        windows = deepcopy(report["windows"])
        with patch.object(quality, "_observe", side_effect=observe):
            observations, events, unresolved, acquisition = quality.acquire_acoustic_evidence(
                rec, SimpleNamespace(duration_s=rec["source"]["duration_s"]), windows,
                asr_scope=lambda: nullcontext(None), output_dir=root, policy=POLICY)
        if set(requested) != set(crops) or quality._digest(observations) != quality._digest(frozen):
            raise ValueError("Replay did not consume identical frozen acoustic observations")
        recovery = deepcopy(report["independent_recovery"])
        targets = recovery["targets"]
        if recovery.get("observations"):
            recovery["status"] = "observed"
        rows, decisions = resolve_occurrences(rec, events, observations, targets, recovery, POLICY)
        receipts = resolution_receipts(rec, ledger, rows, decisions, POLICY)
        public[case] = {**summarize(rec, rows, decisions, receipts),
                        "frozen_acoustic_requests": len(requested),
                        "frozen_independent_requests": recovery["requests"],
                        "legacy_subinterval_attempted": any(
                            d.get("recovery_attempted") for d in rec["alignment_diagnostics"])}
        private[case] = {"rows": rows, "decisions": decisions, "receipts": receipts,
                         "events": events, "cross_model_events": recovery["cross_model_events"],
                         "unresolved": unresolved, "acquisition": acquisition}
        if case == "boundary":
            targeted = json.loads(extra["private/replay-and-targeted/private-targeted-resolution.json"])
            worker = json.loads(extra["private/replay-and-targeted/private-targeted-worker.json"])
            if worker["observations"] != targeted["worker_observations"]:
                raise ValueError("Targeted worker observations disagree")
            rows, decisions = resolve_occurrences(rec, [], observations, [targeted["target"]], worker, POLICY)
            receipts = resolution_receipts(rec, ledger, rows, decisions, POLICY)
            public["targeted_boundary"] = {**summarize(rec, rows, decisions, receipts),
                                            "frozen_independent_requests": worker["requests"]}
            private["targeted_boundary"] = {"rows": rows, "decisions": decisions, "receipts": receipts,
                                             "cross_model_events": worker["cross_model_events"]}
    return public, private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--supplement", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    public, private = replay(args.archive, args.supplement, args.output_dir)
    write_json(args.output_dir / "summary.json", {
        "status": "completed", "new_model_inference": False,
        "archive_sha256": ARCHIVE_SHA, "supplement_sha256": SUPPLEMENT_SHA,
        "regions": public})
    write_json(args.output_dir / "private-replay.json", private)


if __name__ == "__main__":
    main()
