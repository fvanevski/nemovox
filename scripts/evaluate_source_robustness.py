"""Bounded, private source-robustness experiments on a sealed intake.

This is a qualification driver, not an alternate production pipeline. No
canonical row is deleted or retimed. A fresh source-quality implementation
must be judged separately from these model observations.
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import time

from nvidia_transcribe.source_quality import SourceQualityPolicy, _lex


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def write(path, value):
    path=Path(path)
    tmp=path.with_name("."+path.name+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    os.replace(tmp,path)


def read_intake(path, expected_sha):
    if sha(path)!=expected_sha:
        raise ValueError("Private intake archive digest mismatch")
    prefix="source-robustness-intake/"
    wanted=[
        "retained/recording/asr/manifest.json",
        "retained/recording/asr/recording.contract.json",
        "retained/recording/asr/source-evidence.v1.json",
        "retained/source-quality/output/source-quality.json",
        "provenance/media.json","provenance/models.json",
        "analysis/scene-decisions.json",
    ]
    with tarfile.open(path,"r:gz") as archive:
        members=archive.getmembers()
        names=[m.name for m in members]
        if (len(set(names))!=len(names) or any(
                not m.isfile() or not m.name.startswith(prefix)
                or ".." in PurePosixPath(m.name).parts for m in members)):
            raise ValueError("Unsafe private intake member inventory")
        def raw(name):
            m=archive.getmember(prefix+name)
            if m.size>100_000_000:
                raise ValueError("Unexpectedly large intake member")
            with archive.extractfile(m) as stream:
                return stream.read()
        manifest=json.loads(raw("manifest.json"))
        entries={e["relative_path"]:e for e in manifest["entries"]}
        if len(entries)!=len(manifest["entries"]):
            raise ValueError("Duplicate manifest members")
        out={}
        for name in wanted:
            b=raw(name)
            entry=entries[name]
            if (len(b)!=entry["byte_count"] or
                    hashlib.sha256(b).hexdigest()!=entry["sha256"]):
                raise ValueError("Intake payload mismatch: "+name)
            out[name]=json.loads(b)
    return manifest,out


def displaced_sequences(ledger, *, minimum_units=6):
    """Nominate exact recognized-order matches with disjoint CTC hypotheses.

    The shared crop must contain the earlier hypothesized words; the later
    projection must fall outside that shared crop. The match is a reason to
    listen, not a finding of echo or speaker identity.
    """
    if minimum_units<2:
        raise ValueError("Minimum sequence too short")
    parents={a["id"]:a for a in ledger.get("asr_observations",[])}
    exact={i["asr_observation_id"] for i in ledger.get(
        "recognized_occurrence_inventories",[]) if i.get("projection_status")=="exact"}
    by_chunk={}
    for u in ledger.get("lexical_observations",[]):
        if u["asr_observation_id"] in exact and u.get("raw_recognized_token_ids") and _lex(u["text"]):
            by_chunk.setdefault(u["chunk_index"],[]).append(u)
    for items in by_chunk.values():
        items.sort(key=lambda u:u["source_ordinal"])
    by_index={p["chunk_index"]:p for p in parents.values()}
    result=[]
    def bounds(u):
        a,b=u.get("ctc_start_s"),u.get("ctc_end_s")
        if isinstance(a,(int,float)) and isinstance(b,(int,float)) and a<b:
            return a,b
        return None
    for index,left in sorted(by_chunk.items()):
        right=by_chunk.get(index+1,[])
        if not right:
            continue
        a,b=by_index[index],by_index[index+1]
        shared=(max(a["infer_start_s"],b["infer_start_s"]),
                min(a["infer_end_s"],b["infer_end_s"]))
        if shared[0]>=shared[1]:
            continue
        lkeys=[_lex(u["text"]) for u in left]
        rkeys=[_lex(u["text"]) for u in right]
        for match in SequenceMatcher(None,lkeys,rkeys,autojunk=False).get_matching_blocks():
            run=[]
            def flush():
                if len(run)<minimum_units or len({_lex(x["text"]) for x,_ in run})<3:
                    return
                before,after=zip(*run)
                result.append({
                    "left_chunk":index,"right_chunk":index+1,
                    "shared_crop":shared,
                    "left_unit_ids":[u["id"] for u in before],
                    "right_unit_ids":[u["id"] for u in after],
                    "left_bounds":[bounds(u) for u in before],
                    "right_bounds":[bounds(u) for u in after],
                    "left_turn_ids":sorted({u["recognized_turn_id"] for u in before}),
                    "right_turn_ids":sorted({u["recognized_turn_id"] for u in after}),
                    "unit_count":len(run),"acoustic_truth_accepted":False,
                })
            for offset in range(match.size):
                x,y=left[match.a+offset],right[match.b+offset]
                xb,yb=bounds(x),bounds(y)
                eligible=(xb and yb and shared[0]<=xb[0]<xb[1]<=shared[1]
                    and shared[1]<=yb[0]<yb[1]<=b["core_end_s"]
                    and x.get("boundary_disposition")=="selected_provisional"
                    and y.get("boundary_disposition")=="selected_provisional")
                if eligible:
                    run.append((x,y))
                else:
                    flush()
                    run=[]
            flush()
    return result


def validate_plan(plan,duration):
    probes=plan.get("local_probes",[])
    targets=plan.get("independent_targets",[])
    primary=plan.get("primary_chunks",[])
    if (len(primary)>2 or len(set(primary))!=len(primary)
            or any(type(i) is not int or i<0 for i in primary)
            or len(probes)+2*len(primary)>12 or len(targets)>6):
        raise ValueError("Probe count exceeds bounded experiment")
    def valid(bounds):
        return (isinstance(bounds,list) and len(bounds)==2
                and all(type(v) in (int,float) for v in bounds)
                and 0<=bounds[0]<bounds[1]<=duration)
    ids=[x["id"] for x in probes+targets]
    if len(set(ids))!=len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+",i) for i in ids):
        raise ValueError("Invalid probe identifiers")
    for p in probes:
        if not valid(p["bounds"]) or p["bounds"][1]-p["bounds"][0]>60:
            raise ValueError("Invalid localized probe")
    for t in targets:
        if (not valid(t["bounds"]) or len(t["crops"])!=2 or t["crops"][0]==t["crops"][1]
                or not all(valid(c) and c[0]<=t["bounds"][0]<t["bounds"][1]<=c[1]
                           for c in t["crops"])):
            raise ValueError("Invalid independent target geometry")
    if sum(c[1]-c[0] for t in targets for c in t["crops"])>240:
        raise ValueError("Independent experiment audio budget exceeded")


def acoustic(args,root,data,plan,registry):
    from nvidia_transcribe.client import NemotronVLLMClient, InferenceRequestConfig
    from nvidia_transcribe.media import probe_media, extract_wav_segment, validate_wav_file
    from nvidia_transcribe.resource_lifecycle import ModelLifecycle, managed_asr_client
    from nvidia_transcribe.source_recovery import acquire_independent_evidence
    from nvidia_transcribe.asr_integrity import audit_asr_generation
    from nvidia_transcribe.acoustic_observations import reconstruct_acoustic_observation

    provenance=data["provenance/media.json"]
    original=provenance["observed"]
    path=args.media or Path(original["path"])
    if sha(path)!=provenance["current_full_sha256"]:
        raise ValueError("Original media full-byte identity mismatch")
    media=probe_media(path,audio_stream_index=original["selected_audio_stream"])
    if (media.fingerprint_sha256!=original["fingerprint_sha256"]
            or abs(media.duration_s-original["duration_s"])>.01):
        raise ValueError("Original media fingerprint or duration mismatch")
    rec=data["retained/recording/asr/recording.contract.json"]
    models=data["provenance/models.json"]
    language=rec["source_language"]
    validate_plan(plan,media.duration_s)
    resource=registry.resolve("asr",require_local_paths=True)
    if (resource.adapter!="nemotron_ctc" or resource.model!=models["asr"]["served_model"]
            or not resource.checkpoint_path
            or Path(resource.checkpoint_path).resolve()!=Path(models["asr"]["model_path"]).resolve()):
        raise ValueError("Admitted ASR model differs from retained baseline identity")
    manifest=data["retained/recording/asr/manifest.json"]
    chunks={c["chunk_index"]:c for c in manifest["chunks"]}
    work=[]
    for idx in plan.get("primary_chunks",[]):
        c=chunks[idx]
        work.extend([
            (f"chunk-{idx}-baseline",[c["infer_start_s"],c["infer_end_s"]],c["wav_sha256"]),
            (f"chunk-{idx}-no-left-context",[c["core_start_s"],c["infer_end_s"]],None),
        ])
    work.extend((p["id"],p["bounds"],None) for p in plan.get("local_probes",[]))
    if (len({name for name,_,_ in work})!=len(work)
            or sum(b-a for _,(a,b),_ in work)>600):
        raise ValueError("Native experiment budget or identity conflict")
    inputs=root/"inputs"
    inputs.mkdir()
    prepared=[]
    for name,(lo,hi),expected in work:
        wav=inputs/(name+".wav")
        extract_wav_segment(media.path,start_s=lo,duration_s=hi-lo,out_wav=wav,
                            audio_stream_index=media.selected_audio_stream)
        duration,digest=validate_wav_file(wav)
        if abs(duration-(hi-lo))>.1 or (expected and digest!=expected):
            raise ValueError("Native crop does not match registered input: "+name)
        prepared.append((name,(lo,hi),wav,digest,duration))
    manager=ModelLifecycle()
    results=[]
    client=NemotronVLLMClient(base_url=resource.endpoint,model=resource.model,
                             api_key=resource.credential(),timeout_s=900.)
    write(root/"native-summary.json",{"status":"in_progress","resource":resource.fingerprint(),
                                     "requests":results})
    with managed_asr_client(manager,resource,client) as stage:
        for name,crop,wav,digest,duration in prepared:
            if sha(wav)!=digest:
                raise ValueError("Prepared WAV changed before execution")
            receipt={"id":name,"crop":crop,"audio_sha256":digest,"audio_seconds":duration,
                     "status":"started"}
            results.append(receipt)
            write(root/"native-summary.json",{"status":"in_progress","requests":results})
            started=time.monotonic()
            try:
                r=stage.transcribe(wav,InferenceRequestConfig(
                    mode="speaker",source_language=language,capture_ctc_timestamps=True,
                    require_ctc_timestamps=False,source_offset_s=crop[0]))
                raw=r["raw_response"]
                write(root/(name+".raw.json"),raw)
                audit=audit_asr_generation(raw)
                receipt.update({"status":"returned","integrity":audit,
                                "raw_sha256":sha(root/(name+".raw.json"))})
                rows,inventory=reconstruct_acoustic_observation(
                    raw,audio_sha256=digest,crop=crop,language=language)
                write(root/(name+".observation.json"),{"rows":rows,"inventory":inventory})
            except Exception as exc:
                receipt.update({"status":"error","error":type(exc).__name__,"detail":str(exc)})
                raise
            finally:
                receipt["wall_s"]=round(time.monotonic()-started,4)
                write(root/"native-summary.json",{"status":"in_progress","requests":results})
    write(root/"native-summary.json",{"status":"complete","resource":resource.fingerprint(),
                                     "requests":results})
    targets=[{"kind":"occurrence","target_id":t["id"],"bounds":t["bounds"],"crops":t["crops"],
              "ids":[],"task_ids":[t["id"]],"evidence_ids":[t["id"]],"chunks":[]}
             for t in plan.get("independent_targets",[])]
    if targets:
        independent=registry.resolve("source_challenger",require_local_paths=True)
        report=acquire_independent_evidence(rec,targets,media,challenger_resource=independent,
                  output_dir=root,policy=SourceQualityPolicy(),lifecycle=manager)
        write(root/"independent.json",report)
        if report["status"]!="observed":
            raise RuntimeError("Independent observation stage not completed")


def scene(root,data,registry):
    from copy import deepcopy
    from nvidia_transcribe.resource_lifecycle import ModelLifecycle, managed_text_client
    from nvidia_transcribe.scene_evidence import assess_case_payload, apply_decision, SceneDecisionError

    cases=data["analysis/scene-decisions.json"]["cases"]
    recording=data["retained/recording/asr/recording.contract.json"]
    rows={r["id"]:r for r in recording["utterances"]}
    resource=registry.resolve("translation_review",require_local_paths=True)
    results=[]
    with managed_text_client(ModelLifecycle(),resource,wire_protocol="ninfer_messages",
                             timeout_s=900.) as client:
        for entry in cases:
            # Reconstructed historical failed inputs are explicitly not wire
            # receipts. Successful initial cases are matched positive controls.
            x=entry.get("failed_case_input_reconstruction") or entry["reconstructed_initial"]
            payload,schema=x["input"],x["schema"]
            case={"source_id":payload["target"]["id"],"historical_error":bool(entry.get("error")),
                  "historical_input_class":"reconstructed","failed_assessments":[],
                  "assessment_recovery_attempts":0}
            results.append(case)
            try:
                assessed=assess_case_payload(client,payload,schema,case,system=x["system_prompt"])
                case["assessment"]=assessed
                row=deepcopy(rows[case["source_id"]])
                case["would_apply_supported_source_change"]=apply_decision(
                    row,assessed,payload["acoustic_alternatives"],payload["visual_observations"])
                case["counterfactual_row"]=row
                case["status"]="valid_decision"
            except SceneDecisionError as exc:
                case["status"]="rejected"
                case["error"]=str(exc)
            finally:
                write(root/"scene-comparison.json",{"status":"in_progress","cases":results})
    write(root/"scene-comparison.json",{"status":"complete","resource":resource.fingerprint(),
                                       "cases":results})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",required=True,type=Path)
    p.add_argument("--archive-sha256",required=True)
    p.add_argument("--output-dir",required=True,type=Path)
    p.add_argument("--phase",choices=["replay","acoustic","scene"],required=True)
    p.add_argument("--plan",type=Path)
    p.add_argument("--media",type=Path)
    p.add_argument("--model-resources",type=Path)
    args=p.parse_args()
    os.umask(0o077)
    root=args.output_dir
    root.mkdir(exist_ok=False)
    manifest,data=read_intake(args.archive,args.archive_sha256)
    write(root/"input-identity.json",{"archive_sha256":args.archive_sha256,
          "job_id":manifest["job_id"],"inspection_anchor":manifest["anchor"],"phase":args.phase})
    if args.phase=="replay":
        ledger=data["retained/recording/asr/source-evidence.v1.json"]
        quality=data["retained/source-quality/output/source-quality.json"]
        results={}
        for minimum in (4,6,8):
            nominations=displaced_sequences(ledger,minimum_units=minimum)
            for n in nominations:
                ids=set(n["right_unit_ids"])
                n["existing_tasks"]=[{"task_id":t["task_id"],"state":t["state"],
                    "window":t["window"],"reason":t["reason"]} for t in quality["recovery_tasks"]
                    if ids.intersection(t.get("source_unit_ids",[]))]
            results[str(minimum)]=nominations
        write(root/"sequence-nominations.json",results)
        return
    if args.model_resources is None:
        p.error("--model-resources required for actual inference")
    from nvidia_transcribe.model_resources import load_model_resources
    registry=load_model_resources(args.model_resources)
    if args.phase=="acoustic":
        if args.plan is None:
            p.error("--plan required for acoustic experiment")
        plan=json.loads(args.plan.read_bytes())
        write(root/"probe-plan.json",plan)
        acoustic(args,root,data,plan,registry)
    else:
        scene(root,data,registry)


if __name__=="__main__":
    main()
