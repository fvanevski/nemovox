"""Replay the qualified acoustic recovery evidence, then optionally challenge
the one deferred, explicitly bracketed original occurrence with CPU Whisper.

No GPU client or model is acquired. A changed primary plan is a counterfactual
plan, not a fresh acoustic run. Independent admission is compared on the
exact prior observed tasks and unchanged acoustic receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from nvidia_transcribe.media import probe_media, validate_wav_file
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.occurrence_resolution import resolution_receipts, resolve_occurrences
from nvidia_transcribe.recovery_planner import plan_recovery_windows
from nvidia_transcribe.source_recovery import schedule_independent, acquire_independent_evidence
from scripts.evaluate_occurrence_recovery import POLICY, verified_archive, write_json

ARCHIVE_SHA="4d46e32e6f2459cb348389aa4362b6a3fdfd2dddba56391160cc8ef2773d6755"
TESTED_SOURCE="029f969b43c58d53fd27975ff122c097ead04167"
REGIONS=("compression","boundary","late_gap","low_flag_control","multilingual_control")


def _admission_metrics(tasks):
    return {"states":dict(Counter(t["state"] for t in tasks)),
            "admitted_evidence_ids":len({uid for t in tasks
                if t["state"] in ("admitted","observed") for uid in t["evidence_ids"]}),
            "total_evidence_ids":len({uid for t in tasks for uid in t["evidence_ids"]}),
            "admitted_target_seconds":round(sum(t["window"][1]-t["window"][0]
                for t in tasks if t["state"] in ("admitted","observed") and t["window"]),6)}


def replay(archive):
    members=verified_archive(archive,expected_sha=ARCHIVE_SHA,expected_source=TESTED_SOURCE)
    public,private={},{}
    boundary=None
    for case in REGIONS:
        root="private/phase-b-live/"+case+"/"
        rec=json.loads(members[root+"recording/recording.contract.json"])
        manifest=json.loads(members[root+"recording/manifest.json"])
        ledger_data=members[root+"recording/source-evidence.v1.json"]
        if hashlib.sha256(ledger_data).hexdigest()!=manifest["source_evidence"]["sha256"]:
            raise ValueError("Archived recording ledger hash mismatch")
        ledger=json.loads(ledger_data)
        quality=json.loads(members[root+"source-quality.json"])
        canonical=json.loads(members[root+"canonical-source.contract.json"])
        before=deepcopy(rec)
        request=rec["processing"]["boundary_reconciliation"]["semantic_review_candidates"]
        windows,deferred=plan_recovery_windows(rec,ledger,request,None,POLICY)
        old=quality["recovery_tasks"]
        new=[*windows,*deferred]
        old_ids={uid for t in old for uid in t["evidence_ids"]}
        new_ids={uid for t in new for uid in t["evidence_ids"]}
        if old_ids!=new_ids:
            raise ValueError("Changed planning lost or invented evidence IDs")
        if len({t["task_id"] for t in new})!=len(new):
            raise ValueError("Duplicate task identities")
        # Do not pretend that a newly planned crop has been observed.
        # Re-admit the independent worker on the exact old acquisition.
        targets,independent_deferred=schedule_independent(
            rec,quality["acoustic_observations"],windows=old,policy=POLICY)
        receipts=resolution_receipts(rec,ledger,canonical["utterances"],quality["decisions"],POLICY)
        if rec!=before:
            raise ValueError("Replay mutated original recording")
        public[case]={"primary_plan_before":_admission_metrics(old),
                      "primary_plan_candidate":_admission_metrics(new),
                      "all_evidence_ids_preserved":True,
                      "prior_independent_kinds":[t["kind"] for t in quality["independent_recovery"]["targets"]],
                      "candidate_independent_kinds":[t["kind"] for t in targets],
                      "candidate_independent_deferred":len(independent_deferred),
                      "candidate_independent_planned_audio_s":round(sum(
                          b-a for t in targets for a,b in t["crops"]),6),
                      "prior_resolution_statuses":dict(Counter(r["status"] for r in quality["occurrence_resolutions"])),
                      "candidate_resolution_statuses":dict(Counter(r["status"] for r in receipts)),
                      "new_audio_executed":False}
        private[case]={"candidate_windows":windows,"candidate_deferred":deferred,
                       "candidate_independent_targets":targets,
                       "candidate_independent_deferred":independent_deferred,
                       "candidate_resolution_receipts":receipts}
        if case=="boundary":
            untimed=[r for r in rec["utterances"] if r["source_start"] is None and r["source_end"] is None]
            if len(untimed)!=1:
                raise ValueError("Retained boundary control no longer identifies one untimed original")
            uid=untimed[0]["id"]
            prior=[t for t in quality["independent_recovery"]["deferred_targets"]
                   if t["kind"]=="untimed" and t["ids"]==[uid]]
            now=[t for t in targets if t["kind"]=="untimed" and t["ids"]==[uid]]
            if len(prior)!=1 or len(now)!=1 or now[0]["bounds"]!=prior[0]["bounds"]:
                raise ValueError("Explicit previously deferred target not admitted")
            planned=[w for w in windows if uid in w["source_ids"] and
                     w["window"][0]<=now[0]["bounds"][0]<now[0]["bounds"][1]<=w["window"][1]]
            if not planned:
                raise ValueError("Candidate primary plan did not admit the targeted bracket")
            public[case]["previously_deferred_bracket_admitted"]=True
            public[case]["bracket_seconds"]=now[0]["bounds"]
            public[case]["primary_target_window_seconds"]=min(w["window"][1]-w["window"][0] for w in planned)
            expected=json.loads(members["public/summaries/live_summary.json"])["regions"][case]["input_audio_sha256"]
            boundary=(rec,ledger,quality,now[0],expected)
    return public,private,boundary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",required=True,type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--boundary-wav",type=Path,
                        help="Optional exact retained boundary input WAV; enables one CPU crop pair")
    parser.add_argument("--model-resources",type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    public,private,boundary=replay(args.archive)
    write_json(args.output_dir/"private-replay.json",private)
    result={"status":"completed","archive_sha256":ARCHIVE_SHA,
            "comparison":"candidate primary plan and independent re-admission on frozen prior observed tasks",
            "regions":public,"new_worker":"not_requested","acoustic_truth_accepted":False}
    write_json(args.output_dir/"summary.json",result)
    if args.boundary_wav is None:
        return
    rec,ledger,quality,target,expected=boundary
    duration,digest=validate_wav_file(args.boundary_wav)
    if digest!=expected or abs(duration-rec["source"]["duration_s"])>.01:
        raise ValueError("Boundary WAV must match the qualified original input bytes")
    media=probe_media(args.boundary_wav,audio_stream_index=0)
    resource=load_model_resources(args.model_resources).resolve("source_challenger",require_local_paths=True)
    report=acquire_independent_evidence(rec,[target],media,challenger_resource=resource,
            output_dir=args.output_dir,policy=POLICY)
    write_json(args.output_dir/"private-targeted-worker.json",report)
    if report["status"]!="observed":
        result["new_worker"]="blocked_or_failed"
        write_json(args.output_dir/"summary.json",result)
        raise RuntimeError("Targeted CPU worker failed")
    rows,decisions=resolve_occurrences(rec,[],quality["acoustic_observations"],
                                     [target],report,POLICY)
    write_json(args.output_dir/"private-targeted-resolution.json",{
        "target":target,"rows":rows,"decisions":decisions,
        "worker_observations":report["observations"],
        "cross_model_events":report.get("cross_model_events",[])})
    # A narrow positive or unresolved result is acceptable. The targeted
    # run is not a fresh qualification of the candidate primary scheduler.
    result["new_worker"]="observed"
    result["targeted_worker"]={"requests":report["requests"],"audio_seconds":report["audio_seconds"],
        "wall_s":report.get("wall_s"),"receipt_sha256":report.get("receipt_sha256"),
        "decisions":dict(Counter(d["status"] for d in decisions)),
        "original_target_ids":target["ids"],
        "target_still_untimed":any(r["id"] in target["ids"] and r["source_start"] is None for r in rows),
        "canonical_rows_before":len(rec["utterances"]),"canonical_rows_after":len(rows),
        "acoustic_truth_accepted":False}
    write_json(args.output_dir/"summary.json",result)


if __name__=="__main__":
    main()
