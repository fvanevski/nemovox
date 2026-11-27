"""Bounded recording-review qualification on private existing native MT outputs.

Development runner only: no Riva/ASR recomputation except an explicitly
nominated acoustic crop. Use separate private output directories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from nvidia_transcribe import connected_units
from nvidia_transcribe.media import compute_file_sha256
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.ninfer_messages_client import QUALITY_COMPLETION_TOKENS, QUALITY_TIMEOUT_S
from nvidia_transcribe.resource_lifecycle import ModelLifecycle, managed_text_client
from nvidia_transcribe.review_evidence import ReviewEvidenceProvider
from nvidia_transcribe.translation_review import TranslationReviewConfig, review_translated_utterances
from nvidia_transcribe.moss_recording import validate_moss_contract


def write(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")


def select(rows, ranges):
    wanted=set()
    for spec in ranges:
        pieces=spec.split(":")
        if len(pieces)==1:
            lo=hi=int(pieces[0])
        elif len(pieces)==2:
            lo,hi=map(int,pieces)
        else:
            raise ValueError("Target specification must be N or N:M (inclusive native indices)")
        if not 0<=lo<=hi:
            raise ValueError("Invalid target range")
        wanted.update(range(lo,hi+1))
    existing={r["timestamp_provenance"]["native_event_index"]:r["id"] for r in rows}
    if not wanted or not wanted<=existing.keys():
        raise ValueError("Target index is not an observed native event")
    wanted_ids={existing[i] for i in wanted}
    selected=set(wanted_ids)
    for group in connected_units.groups(rows):
        if any(rows[i]["id"] in selected for i in group):
            selected.update(rows[i]["id"] for i in group)
    return wanted_ids,selected


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--contract",required=True)
    p.add_argument("--contract-sha256",required=True)
    p.add_argument("--model-resources",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--target",nargs="+",required=True)
    p.add_argument("--modes",nargs="+",choices=["english","bilingual"], default=["bilingual"])
    p.add_argument("--resume-check",action="store_true",help="Replay successful current-version selection from cache.")
    p.add_argument("--media")
    p.add_argument("--adaptive",action="store_true")
    p.add_argument("--probe-visual",type=int)
    p.add_argument("--probe-question")
    p.add_argument("--probe-acoustic",type=int)
    args=p.parse_args()
    if args.adaptive and not args.media:
        p.error("--adaptive requires explicitly admitted original media")
    if (args.probe_visual is not None or args.probe_acoustic is not None) and not args.adaptive:
        p.error("Explicit observation probes require --adaptive")
    if args.probe_visual is not None and not args.probe_question:
        p.error("Supply a neutral, observational --probe-question")
    input_path=Path(args.contract)
    sha=compute_file_sha256(input_path)
    if sha!=args.contract_sha256:
        raise ValueError("Private baseline contract identity mismatch")
    contract=json.loads(input_path.read_text())
    validate_moss_contract(contract,media_path=args.media)
    rows=contract["utterances"]
    requested,selected=select(rows,args.target)
    registry=load_model_resources(args.model_resources)
    review=registry.resolve("translation_review")
    vision=registry.resolve("visual_inspection")
    asr=registry.resolve("asr")
    lifecycle=ModelLifecycle()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=False)
    write(output/"input.json",{"contract_sha256":sha,"language":contract["source_language"],
        "requested_ids":sorted(requested),"selected_ids":sorted(selected),
        "review_resource_fingerprint":review.fingerprint(),"visual_resource_fingerprint":vision.fingerprint(),
        "adaptive":args.adaptive,"modes":args.modes})
    with managed_text_client(lifecycle,review,timeout_s=QUALITY_TIMEOUT_S,
                             reasoning_effort="xhigh",wire_protocol="ninfer_messages") as client:
        provider=ReviewEvidenceProvider(media_path=args.media,contract=contract,lifecycle=lifecycle,
            review_resource=review,visual_resource=vision,acoustic_resource=asr,review_client=client) if args.adaptive else None
        for mode in args.modes:
            root=output/mode
            root.mkdir()
            cfg=TranslationReviewConfig(profile="balanced",style="natural",target_language="en",
                base_url=review.endpoint,model=review.model,resource_fingerprint=review.fingerprint(),
                wire_protocol="ninfer_messages",thinking_budget=2048,reasoning_effort="xhigh",
                max_tokens=QUALITY_COMPLETION_TOKENS,timeout_s=QUALITY_TIMEOUT_S,context_mode=mode)
            started=time.perf_counter()
            revised,summary=review_translated_utterances(rows,source_language=contract["source_language"],
                config=cfg,text_client=client,evidence_provider=provider,
                target_utterance_ids=selected,checkpoint_path=root/"review.checkpoint.json",resume=False)
            connected_units.validate_selected_units(revised,require_complete=True)
            for old,new in zip(rows,revised):
                for key in ("id","source_text","speaker_id","source_start","source_end","machine_translated_text"):
                    if old.get(key)!=new.get(key):
                        raise ValueError("Canonical or primary translation modified: "+str(key))
                if old["id"] not in selected and old!=new:
                    raise ValueError("Non-target occurrence modified")
            write(root/"reviewed.contract.json",{**contract,"utterances":revised,"translation_review":summary})
            write(root/"report.json",{"mode":mode,"wall_s":time.perf_counter()-started,
                "summary":summary,"selected_ids":sorted(selected)})
            print(json.dumps({"mode":mode,"status":summary["status"],"changed":summary["changed"],
                 "deferred":summary["deferred_comparison_ids"],"api_calls":summary["api_calls"]}),flush=True)
            if args.resume_check:
                replay,receipt=review_translated_utterances(rows,source_language=contract["source_language"],
                    config=cfg,text_client=client,evidence_provider=provider,target_utterance_ids=selected,
                    checkpoint_path=root/"review.checkpoint.json",resume=True)
                write(root/"resume.json",{"equal_rows":replay==revised,"summary":receipt})
        for kind,index,question in (("visual",args.probe_visual,args.probe_question),
                                   ("acoustic",args.probe_acoustic,"Re-observe spoken wording.")):
            if index is None:
                continue
            uid=next(r["id"] for r in rows if r["timestamp_provenance"]["native_event_index"]==index)
            observation=provider.collect(uid,{"kind":kind,"reference_id":uid,"question":question})
            write(output/f"explicit-{kind}-probe.json",observation)
            print(json.dumps({"explicit_probe":kind,"status":observation["status"]}),flush=True)


if __name__=="__main__":
    main()
