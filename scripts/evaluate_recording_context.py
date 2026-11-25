"""Qualify native recording context and shared scheduling on authenticated inputs.

No ASR/translation inference in planning mode. Optional integrated source-quality
mode uses the actual production orchestrator on one bounded authenticated crop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.recording_context import run_recording_context
from nvidia_transcribe.resource_lifecycle import ModelLifecycle, managed_asr_client, managed_text_client
from nvidia_transcribe.source_quality import SourceQualityPolicy, load_current_ledger, plan_windows, review_current_source
from nvidia_transcribe.ninfer_messages_client import QUALITY_TIMEOUT_S


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-resources", type=Path, required=True)
    parser.add_argument("--lifecycle-config", type=Path)
    parser.add_argument("--live-media", type=Path, help="One exact bounded source; production source-quality path")
    args = parser.parse_args()
    if args.live_media and len(args.recording) != 1:
        parser.error("--live-media requires exactly one recording")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=False)
    recordings = []
    for path in args.recording:
        rec = json.loads(path.read_text(encoding="utf-8"))
        ledger = load_current_ledger(rec,path)
        if args.live_media and rec["source"]["duration_s"] > 180.:
            raise ValueError("Live qualification is bounded to <=180s")
        recordings.append((path,rec,ledger))
    registry = load_model_resources(args.model_resources)
    lifecycle = ModelLifecycle(config_path=args.lifecycle_config)
    resource = registry.resolve("translation_review")
    def text_scope():
        return managed_text_client(lifecycle,resource,wire_protocol="ninfer_messages",
                                   timeout_s=QUALITY_TIMEOUT_S)
    policy = SourceQualityPolicy(max_windows=4,max_extra_contexts=2,
        max_acoustic_audio_s=240.,max_acoustic_wall_s=420.,
        max_independent_targets=2,max_independent_audio_s=100.,max_independent_wall_s=420.,
        max_text_cases=2,scene_max_cases=1)
    write(root/"resources.json", {"translation_review_fingerprint":resource.fingerprint(),
        "resource":resource.name,"adapter":resource.adapter,"model":resource.model})
    if args.live_media:
        path,rec,ledger = recordings[0]
        asr = registry.resolve("asr")
        challenger = registry.resolve("source_challenger",require_local_paths=True)
        client = NemotronVLLMClient(base_url=asr.endpoint,model=asr.model,
                                   api_key=asr.credential(),timeout_s=900.)
        canonical, report = review_current_source(rec,recording_path=path,media_path=args.live_media,
            output_dir=root,text_scope=text_scope,asr_scope=lambda:managed_asr_client(lifecycle,asr,client),
            policy=policy,challenger_resource=challenger)
        write(root/"source-quality.json",report)
        write(root/"canonical-source.contract.json",canonical)
        write(root/"summary.json",{"status":report["status"],"context":report["recording_context"]["status"],
             "calls":report["recording_context"]["calls"],"audio_requests":report["audio_requests"],
             "semantic_task_count":sum("semantic_occurrence_request" in w["reasons"] for w in report["recovery_tasks"])})
        return
    summaries = []
    with text_scope() as text:
        for i,(path,rec,ledger) in enumerate(recordings):
            report = run_recording_context(rec,ledger,client=text,policy=policy.recording_context)
            req = rec["processing"]["boundary_reconciliation"]["semantic_review_candidates"]
            original = plan_windows(rec,ledger,req,None,policy)
            combined = {**req,"occurrence_requests":[*req.get("occurrence_requests",[]),
                                                       *report["occurrence_requests"]]}
            candidate = plan_windows(rec,ledger,combined,None,policy)
            write(root/f"context-{i:02d}.json",{"recording_path":str(path),"report":report,
                "baseline_admitted":original[0],"baseline_deferred":original[1],
                "context_admitted":candidate[0],"context_deferred":candidate[1]})
            summaries.append({"case":i,"recording_sha256":report["recording_sha256"],
                "status":report["status"],"calls":report["calls"],
                "eligible":len(report["eligible_source_ids"]),"covered":len(report["covered_source_ids"]),
                "proposals":len(report["proposals"]),"rejected":len(report["rejected_proposals"]),
                "occurrence_requests":len(report["occurrence_requests"]),
                "native_token_totals":report["native_token_totals"],
                "all_attempt_token_usage_reported":report["all_attempt_token_usage_reported"],
                "baseline_admitted":len(original[0]),"context_admitted":len(candidate[0]),
                "source_truth_accepted":False})
    write(root/"summary.json",{"mode":"context_and_planning_only","cases":summaries})


if __name__ == "__main__":
    main()
