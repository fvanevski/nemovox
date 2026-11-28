"""Private matched review evaluation: freeze source-bound selections, then run each policy.

Uses the production review and managed lifecycle, never the embedded destination subtitle
stream. The historical policy can run this same driver using its own authenticated checkout.
A selective output is an experiment, not an ordinary complete-recording render.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys

from nvidia_transcribe import connected_units
from nvidia_transcribe import translation_review
from nvidia_transcribe.media import compute_file_sha256
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.moss_recording import validate_moss_contract
from nvidia_transcribe.presentation import admit_observations
from nvidia_transcribe.refinement import load_glossary
from nvidia_transcribe.resource_lifecycle import ModelLifecycle, managed_text_client
from nvidia_transcribe.review_evidence import ReviewEvidenceProvider
from nvidia_transcribe.translation_review import TranslationReviewConfig, review_translated_utterances


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    path=Path(path)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(path)


def verify(path, expected):
    actual=compute_file_sha256(Path(path))
    if actual!=expected:
        raise ValueError(f"Input digest mismatch: {Path(path).name}")
    return actual


def spread(values, limit):
    if len(values)<=limit:
        return values
    return [values[round(i*(len(values)-1)/(limit-1))] for i in range(limit)]


def select_targets(contract, windows, include):
    """Select mechanism controls without treating model status as semantic truth."""
    rows=contract["utterances"]
    by_id={connected_units.uid(r):r for r in rows}
    chosen=defaultdict(set)
    def add(uid, reason):
        if uid not in by_id:
            raise ValueError(f"Unknown occurrence {uid}")
        chosen[uid].add(reason)
    for uid in include:
        add(uid,"documented_development_case")
    for i,(lo,hi) in enumerate(windows):
        if not 0<=lo<hi<=contract["source"]["duration_s"]:
            raise ValueError("Invalid reserved comparison window")
        for uid,r in by_id.items():
            s,e=r.get("source_start"),r.get("source_end")
            if s is not None and e is not None and s<hi and e>lo:
                add(uid,f"reserved_comparison_window_{i+1}")
    # All small historic visual sets, or an ordered spread of larger sets;
    # include acoustic negatives without assuming that every query was wasteful.
    assessments=contract["translation_review"].get("evidence_assessments",[])
    for kind,limit in (("visual",8),("acoustic",3)):
        for item in spread([r for r in assessments if r.get("kind")==kind],limit):
            add(item["target_id"],f"prior_{kind}_acquisition")
            if item.get("reference_id") in by_id:
                add(item["reference_id"],f"prior_{kind}_reference")
    rejected=[uid for uid,r in by_id.items() if r.get("translation_review_status","").startswith("review_rejected")]
    for uid in spread(rejected,5):
        add(uid,"comparative_rejection_control")
    gender=re.compile(r"\b(?:he|she|him|her|his|hers|they|them|their)\b",re.I)
    changed=[uid for uid,r in by_id.items() if any(ch in r["source_text"] for ch in "他她它")
             and {x.casefold() for x in gender.findall(r["machine_translated_text"])}
                 !={x.casefold() for x in gender.findall(r["translated_text"])}]
    for uid in spread(changed,5):
        add(uid,"referent_change_candidate")
    variants=defaultdict(lambda:defaultdict(list))
    for term in contract["translation_review"].get("discourse_terms",[]):
        variants[term["source"]][term["english"].casefold()].append(term["accepted_target_id"])
    conflicts=[v for v in variants.values() if len(v)>1]
    for group in conflicts[:4]:
        for occurrences in list(group.values())[:4]:
            add(occurrences[0],"terminology_variant_control")
    # Include the full atomic candidate, even when a reserved window straddles it.
    for block in connected_units.groups(rows):
        ids=[connected_units.uid(rows[i]) for i in block]
        if any(uid in chosen for uid in ids):
            for uid in ids:
                if uid not in chosen:
                    add(uid,"connected_allocation_member")
    if not chosen:
        raise ValueError("Empty evaluation selection")
    return [{"id":uid,"reasons":sorted(chosen[uid])} for uid in by_id if uid in chosen]


def freeze(args):
    baseline=read(args.baseline)
    verify(args.baseline,args.baseline_sha256)
    source=baseline["source"]
    if source["content_sha256"]!=args.media_sha256:
        raise ValueError("Media identity does not match the retained contract")
    if baseline["translation_review"]["status"]!="completed":
        raise ValueError("Use an ordinary completed baseline")
    admit_observations(read(args.presentation),source)
    plan={"schema":"nemovox-semantic-evaluation/v1","baseline_sha256":args.baseline_sha256,
          "media_sha256":args.media_sha256,
          "presentation_sha256":compute_file_sha256(Path(args.presentation)),
          "windows":args.reserved_window,"selection":select_targets(baseline,args.reserved_window,args.include),
          "controls":{"context_max_bytes":args.context_max_bytes,"evidence_budget":args.evidence_budget,
                      "visual_max_frames":args.visual_max_frames,"visual_thinking_budget":args.visual_thinking_budget}}
    path=Path(args.plan)
    if path.exists():
        raise FileExistsError("Freeze once; use the same plan for both policies")
    path.parent.mkdir(parents=True,exist_ok=True)
    write(path,plan)
    print(json.dumps({"plan":str(path),"sha256":compute_file_sha256(path),
                      "selected":len(plan["selection"])}))
    return 0


def run(args):
    root=Path(args.source_root).resolve()
    if not Path(translation_review.__file__).resolve().is_relative_to(root/"src"):
        raise ValueError("Wrong review implementation imported")
    current=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if current!=args.expected_commit:
        raise ValueError("Wrong evaluation source commit")
    verify(args.plan,args.plan_sha256)
    plan=read(args.plan)
    if plan.get("schema")!="nemovox-semantic-evaluation/v1":
        raise ValueError("Unknown selection schema")
    verify(args.baseline,plan["baseline_sha256"])
    verify(args.presentation,plan["presentation_sha256"])
    baseline=read(args.baseline)
    validate_moss_contract(baseline,media_path=args.media)
    if baseline["source"]["content_sha256"]!=plan["media_sha256"]:
        raise ValueError("Evaluation media mismatch")
    observations=admit_observations(read(args.presentation),baseline["source"])
    rows=baseline["utterances"]
    ids={r["id"] for r in plan["selection"]}
    if len(ids)!=len(plan["selection"]) or not ids<={connected_units.uid(r) for r in rows}:
        raise ValueError("Invalid selected occurrences")
    report=baseline["translation_review"]
    settings=report["review_policy"]
    resources=load_model_resources(args.model_resources)
    review=resources.resolve("translation_review",resource_name=args.review_resource)
    visual=(review if resources.visual_inherits_review and args.visual_resource is None else
            resources.resolve("visual_inspection",resource_name=args.visual_resource))
    acoustic=resources.resolve("asr",resource_name=args.asr_resource)
    if review.fingerprint()!=report["resource_fingerprint"]:
        raise ValueError("Review resource changed since baseline")
    for evidence in report.get("supplementary_evidence",[]):
        old=evidence.get("provider") or {}
        for key,res in (("visual_resource",visual),("acoustic_resource",acoustic)):
            if old.get(key) and old[key]!=res.fingerprint():
                raise ValueError(f"Evidence resource mismatch: {key}")
    gl=load_glossary(args.glossary)
    if gl.digest()!=report["glossary_digest"]:
        raise ValueError("Glossary changed since baseline")
    control=plan["controls"]
    cfg=TranslationReviewConfig(profile=report["profile"],style=report["style"],
        target_language=settings["target_language"],context_utterances_before=settings["context_before"],
        context_utterances_after=settings["context_after"],batch_size=report["batch_size"],
        base_url=review.endpoint,model=review.model,resource_fingerprint=review.fingerprint(),
        api_key=review.credential(),timeout_s=600.,max_tokens=settings["max_tokens"],
        use_json_schema=True,temperature=settings["temperature"],reasoning_effort=settings["reasoning_effort"],
        wire_protocol=settings["wire_protocol"],thinking_budget=settings["thinking_budget"],
        context_mode=report["recording_context_receipts"][0]["representation"],
        context_max_bytes=control["context_max_bytes"],max_evidence_requests_per_batch=control["evidence_budget"])
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    manifest={"source_commit":current,"policy":args.policy,"plan_sha256":args.plan_sha256,
              "review_resource":review.fingerprint(),"visual_resource":visual.fingerprint(),
              "acoustic_resource":acoustic.fingerprint(),"glossary_digest":gl.digest()}
    marker=out/"evaluation-inputs.json"
    if marker.exists() and read(marker)!=manifest:
        raise ValueError("Evaluation output directory belongs to another run")
    write(marker,manifest)
    manager=ModelLifecycle()
    with managed_text_client(manager,review,timeout_s=cfg.timeout_s,
                             reasoning_effort=cfg.reasoning_effort,wire_protocol=cfg.wire_protocol) as client:
        provider=ReviewEvidenceProvider(media_path=args.media,contract=baseline,lifecycle=manager,
            review_resource=review,visual_resource=visual,acoustic_resource=acoustic,
            review_client=client,max_frames=control["visual_max_frames"],
            visual_thinking_budget=control["visual_thinking_budget"],
            **({"presentation_observations":observations} if args.policy=="candidate" else {}))
        result,summary=review_translated_utterances(rows,source_language=baseline["source_language"],
            config=cfg,glossary=gl,text_client=client,checkpoint_path=out/"translation-review.checkpoint.json",
            target_utterance_ids=ids,evidence_provider=provider)
    if len(result)!=len(rows):
        raise ValueError("Recording occurrence count changed")
    immutable=("id","source_text","source_start","source_end","speaker_id","machine_translated_text")
    selected=[]
    for old,new in zip(rows,result):
        if any(old.get(k)!=new.get(k) for k in immutable):
            raise ValueError("Source or primary MT changed")
        if connected_units.uid(old) in ids:
            selected.append({"id":connected_units.uid(old),"before":old,"after":new})
        elif old!=new:
            raise ValueError("Nonselected row changed")
    connected_units.validate_selected_units(result,require_complete=True)
    write(out/"selected-results.json",selected)
    write(out/"review-summary.json",summary)
    print(json.dumps({"selected":len(selected),"status":summary["status"],
                     "reviewed":summary["reviewed"],"api_calls":summary["api_calls"],
                     "elapsed_s":summary["elapsed_s"],"execution":summary.get("review_execution")}))
    return 0 if summary["status"]=="completed" and summary["reviewed"]==len(ids) else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    f=sub.add_parser("freeze")
    for name in ("baseline","baseline-sha256","media-sha256","presentation","plan"):
        f.add_argument("--"+name,required=True)
    f.add_argument("--reserved-window",type=float,nargs=2,action="append",default=[])
    f.add_argument("--include",action="append",default=[])
    f.add_argument("--context-max-bytes",type=int,required=True)
    f.add_argument("--evidence-budget",type=int,required=True)
    f.add_argument("--visual-max-frames",type=int,required=True)
    f.add_argument("--visual-thinking-budget",type=int,required=True)
    e=sub.add_parser("run")
    for name in ("baseline","presentation","plan","plan-sha256","media","source-root","expected-commit",
                 "model-resources","output-dir"):
        e.add_argument("--"+name,required=True)
    e.add_argument("--policy",choices=("reference","candidate"),required=True)
    for name in ("glossary","review-resource","visual-resource","asr-resource"):
        e.add_argument("--"+name)
    args=parser.parse_args(argv)
    return freeze(args) if args.command=="freeze" else run(args)


if __name__=="__main__":
    sys.exit(main())
