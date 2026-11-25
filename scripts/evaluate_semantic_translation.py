"""Bounded C2 qualification with production MT/review/visual interfaces.

Compare six versus twelve targets on identical primary MT from a current
bounded public job. Optional real-model controls use explicit synthetic text
and actual rendered pixels; they are not fabricated ASR observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.ninfer_messages_client import QUALITY_TIMEOUT_S
from nvidia_transcribe.refinement import GlossaryConfig, load_glossary
from nvidia_transcribe.resource_lifecycle import ModelLifecycle, managed_text_client
from nvidia_transcribe.scene_evidence import SceneEvidencePolicy, inspect_scene, apply_decision
from nvidia_transcribe.semantic_context import accepted_row_context
from nvidia_transcribe.translation import TranslationConfig, translate_utterances
from nvidia_transcribe.translation_review import TranslationReviewConfig, review_translated_utterances


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()


def identity(rows):
    return [(r.get("utterance_id") or r.get("id"),r.get("source_text"),
        r.get("source_start"),r.get("source_end"),r.get("speaker_id"),
        r.get("source_unit_ids"),r.get("semantic_context"),r.get("machine_translated_text"))
        for r in rows]


def review_metrics(report):
    totals={"input_tokens":0,"output_tokens":0,"reasoning_tokens":0,"final_answer_tokens":0}
    missing=0
    attempts=report.get("native_attempts",[])
    for attempt in attempts:
        usage=attempt.get("usage") or {}
        inp,out=usage.get("input_tokens"),usage.get("output_tokens")
        thinking=(usage.get("output_tokens_details") or {}).get("thinking_tokens")
        if (type(inp) is not int or inp<0 or type(out) is not int or out<0
                or type(thinking) is not int or not 0<=thinking<=out):
            missing+=1
            continue
        totals["input_tokens"]+=inp
        totals["output_tokens"]+=out
        totals["reasoning_tokens"]+=thinking
        totals["final_answer_tokens"]+=out-thinking
    return {**{k:report.get(k) for k in ("status","batch_size","reviewed","changed","rejected_polarity",
        "comparative_rejections","dependent_passage_rejections","joined_passage_rejections","comparative_calls",
        "completion_retries","api_calls","elapsed_s","deferred_comparison_ids","errors")},
        "native_usage_totals_reported_attempts":totals,"native_attempt_count":len(attempts),
        "native_usage_complete":missing==0 and len(attempts)==report.get("api_calls"),
        "subdivision_count":len(report.get("subdivisions",[]))}


def config_for(resource, report=None):
    policy=(report or {}).get("review_policy",{})
    return TranslationReviewConfig(profile=(report or {}).get("profile","balanced"),
        style=(report or {}).get("style","natural"),target_language=policy.get("target_language","en"),
        batch_size=12,context_utterances_before=policy.get("context_before",5),
        context_utterances_after=policy.get("context_after",4),
        base_url=resource.endpoint,model=resource.model,resource_fingerprint=resource.fingerprint(),
        timeout_s=QUALITY_TIMEOUT_S,max_tokens=policy.get("max_tokens",8192),
        temperature=policy.get("temperature",0.),wire_protocol="ninfer_messages",
        thinking_budget=policy.get("thinking_budget",2048),
        reasoning_effort=policy.get("reasoning_effort","xhigh"))


def compare_translated(path,root,resource,manager,gl):
    data=path.read_bytes()
    doc=json.loads(data)
    rows=doc["utterances"]
    report=doc["translation_review"]
    quality=doc["source_quality"]
    duration=doc["source"]["duration_s"]
    if (not 0<duration<=180 or not 12<=len(rows)<=120
            or quality.get("schema")!="nemovox-current-source-quality/v1"
            or quality.get("media_fingerprint_sha256")!=doc["source"]["fingerprint_sha256"]
            or quality.get("source_language")!=doc["source_language"]
            or doc["translation"]["backend"]!="riva"
            or report.get("batch_size")!=12
            or report.get("status") not in ("completed","completed_with_deferred_comparisons")
            or report.get("resource_fingerprint")!=resource.fingerprint()
            or report.get("glossary_digest")!=gl.digest()
            or report.get("review_policy",{}).get("wire_protocol")!="ninfer_messages"
            or doc.get("condensation",{}).get("status") not in ("disabled",None)):
        raise ValueError("Require bounded current public candidate job, identical review resource/glossary and no condensation")
    ids=[r.get("utterance_id") or r.get("id") for r in rows]
    if len(ids)!=len(set(ids)) or any(not uid for uid in ids) or any(
            not isinstance(r.get("machine_translated_text"),str) for r in rows):
        raise ValueError("Incomplete or ambiguous primary MT")
    before=deepcopy(rows)
    cfg=config_for(resource,report)
    with managed_text_client(manager,resource,wire_protocol="ninfer_messages",timeout_s=QUALITY_TIMEOUT_S) as client:
        baseline,b_report=review_translated_utterances(rows,source_language=doc["source_language"],
            config=replace(cfg,batch_size=6),glossary=gl,text_client=client,resume=False)
    if rows!=before or identity(baseline)!=identity(rows) or path.read_bytes()!=data:
        raise ValueError("Baseline comparison changed original source/primary MT")
    details=[]
    for a,b in zip(rows,baseline):
        details.append({"source_id":a.get("utterance_id") or a.get("id"),
            "source_text":a["source_text"],"source_quality_status":a.get("source_quality_status"),
            "prior_source_hypotheses":a.get("prior_source_hypotheses"),
            "prior_scene_source_hypothesis":a.get("prior_scene_source_hypothesis"),
            "semantic_context":a.get("semantic_context"),
            "primary_mt":a["machine_translated_text"],"candidate_12":a["translated_text"],
            "candidate_12_proposal":a.get("translation_reviewed_text"),
            "candidate_12_status":a.get("translation_review_status"),
            "candidate_12_comparison":a.get("translation_comparative_review"),
            "baseline_6":b["translated_text"],"baseline_6_proposal":b.get("translation_reviewed_text"),
            "baseline_6_status":b.get("translation_review_status"),
            "baseline_6_comparison":b.get("translation_comparative_review")})
    write(root/"private-passage-comparison.json",details)
    write(root/"private-six-review.json",{"rows":baseline,"report":b_report})
    write(root/"matched-summary.json",{"input_sha256":hashlib.sha256(data).hexdigest(),
        "primary_identity_sha256":digest(identity(rows)),"rows":len(rows),"source_language":doc["source_language"],
        "candidate_12":review_metrics(report),"baseline_6":review_metrics(b_report),
        "different_final_rows":sum(a["translated_text"]!=b["translated_text"] for a,b in zip(rows,baseline)),
        "candidate_status_counts":dict(Counter(a.get("translation_review_status") for a in rows)),
        "baseline_status_counts":dict(Counter(a.get("translation_review_status") for a in baseline)),
        "semantic_quality_requires_adjudication":True})


def controls(root,registry,manager,resource):
    # Explicit language/negation/glossary examples; no inferred ASR truth.
    cases=[
        ("fr", [{"id":"fr-negative","source_text":"Il ne va pas fermer la porte."},
                {"id":"fr-idiom","source_text":"Ils se sont fichus de nous !"},
                {"id":"fr-restriction","source_text":"Il ne voit que la porte."},
                {"id":"fr-name","source_text":"Nadia attend devant la porte."},
                {"id":"fr-idiom-context","source_text":"Ils se sont fichus de nous en nous donnant de fausses pistes."},
                {"id":"fr-direction-a","speaker_id":"A","source_text":"Il a mis directement"},
                {"id":"fr-direction-b","speaker_id":"A","source_text":"le cap à l'ouest."}],
         GlossaryConfig(translation_examples=(("fermer la porte","close the door"),("porte","door")))),
        ("zh", [{"id":"zh-negative","source_text":"不要打开门。"},
                {"id":"zh-positive","source_text":"他已经打开门了。"},
                {"id":"zh-repeat","source_text":"等等，等等！"}],
         GlossaryConfig(translation_examples=(("打开门","open the door"),))),
    ]
    riva=registry.resolve("riva_translation",require_local_paths=True)
    if riva.device!="cuda:0":
        raise ValueError("Explicit existing CUDA Riva resource required")
    tc=TranslationConfig(backend="riva",riva_model_path=riva.checkpoint_path,
        riva_python=riva.worker_python,riva_device=riva.device,resource_fingerprint=riva.fingerprint())
    primary=[]
    for lang,rows,gl in cases:
        out,report=translate_utterances(rows,source_language=lang,config=tc,glossary=gl,
            lifecycle=manager,riva_resource_name=riva.name,resume=False)
        if not report.get("context_example_rows") or any(
                not r.get("machine_translated_text") for r in out):
            raise ValueError("Control did not exercise actual Riva examples")
        primary.append((lang,out,report,gl))
    results={}
    with managed_text_client(manager,resource,wire_protocol="ninfer_messages",timeout_s=QUALITY_TIMEOUT_S) as client:
        for lang,rows,mt,gl in primary:
            out,report=review_translated_utterances(rows,source_language=lang,
                config=config_for(resource),glossary=gl,text_client=client,resume=False)
            if identity(out)!=identity(rows):
                raise ValueError("Control changed source/primary MT")
            results[lang]={"primary":mt,"review":review_metrics(report),"rows":out}
        write(root/"private-language-controls.json",results)
        # Known actual pixels, not source dialogue or fake acoustic receipts.
        clip=root/"visual-control.mp4"
        proc=subprocess.run(["ffmpeg","-hide_banner","-nostdin","-loglevel","error",
            "-f","lavfi","-i","color=c=black:s=640x360:r=5:d=4",
            "-vf","drawtext=text=Nadia:fontcolor=white:fontsize=72:x=(w-tw)/2:y=(h-th)/2",
            "-c:v","libx264","-pix_fmt","yuv420p",str(clip)],capture_output=True,text=True,timeout=60)
        if proc.returncode:
            raise RuntimeError("Visual control generation failed: "+proc.stderr[-1000:])
        row={"id":"visual-control","source_text":"Nadia attend.","source_start":1.,"source_end":3.}
        media=SimpleNamespace(path=clip,duration_s=4.,source_language="fr")
        frames,reply=inspect_scene(client,media,row,root,SceneEvidencePolicy(max_frames=3))
        context=accepted_row_context(row,[],frames)
        mismatch=accepted_row_context({**row,"source_text":"Nadya attend."},[],frames)
        # The known rendered text is Nadia. A misread frame is recorded as
        # model error, never treated as verified identity or speech.
        original=deepcopy(row)
        if frames:
            applied=apply_decision(row,{"decision":{"action":"caption","candidate_id":frames[0]["id"],
                "replace_span":"Nadia attend.","materially_resolved":True}},[],frames)
            if applied or row!=original:
                raise ValueError("Visual-only text changed a spoken source row")
        write(root/"private-visual-control.json",{"known_rendered_text":"Nadia","frames":frames,
            "response":reply,"accepted_literal_context":context,"mismatched_source_context":mismatch,
            "source_mutated":row!=original,"synthetic_pixels_not_asr":True,
            "known_literal_observed":any(f.get("visible_text")=="Nadia" or f.get("caption")=="Nadia" for f in frames)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--translated",type=Path,help="Bounded candidate public translated.contract.json")
    parser.add_argument("--controls",action="store_true",help="Actual Riva/native review and known-pixel controls")
    parser.add_argument("--glossary",type=Path)
    parser.add_argument("--model-resources",required=True,type=Path)
    parser.add_argument("--lifecycle-config",type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    if not (args.translated or args.controls):
        parser.error("Specify --translated and/or --controls")
    root=args.output_dir
    root.mkdir(parents=True,exist_ok=False)
    registry=load_model_resources(args.model_resources)
    resource=registry.resolve("translation_review")
    manager=ModelLifecycle(config_path=args.lifecycle_config)
    if args.translated:
        compare_translated(args.translated,root,resource,manager,load_glossary(args.glossary))
    if args.controls:
        controls(root,registry,manager,resource)


if __name__=="__main__":
    main()
