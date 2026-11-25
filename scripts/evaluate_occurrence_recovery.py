"""Bounded host qualification of the shared occurrence recovery path.

Run with python -m scripts.evaluate_occurrence_recovery from the checkout.
Replay does not acquire models. Worker uses only cached CPU Whisper. Live
uses the actual recording/current-source pipeline, with semantic review
explicitly excluded from this acoustic experiment (not a new CLI default).
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from types import SimpleNamespace

from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.media import extract_wav_segment, probe_media, validate_wav_file
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.occurrence_dispositions import finalize_boundary_dispositions
from nvidia_transcribe.recovery_planner import plan_recovery_windows
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media
from nvidia_transcribe.resource_lifecycle import ModelLifecycle, managed_asr_client
from nvidia_transcribe.source_quality import SourceQualityPolicy, review_current_source, _bounds, _lex
from nvidia_transcribe.source_recovery import acquire_independent_evidence, independent_crops
from scripts.evaluate_chunk_geometry import (
    DIAGNOSTIC_SHA, CHECKPOINT, REGIONS, BudgetedClient, read_authority,
    sha_file, write_json, inspect_run,
)

A3_SHA="805ab0e43275171b16c0beebbc6c938d15ed5b055d4a6e5b342180f46d61eefe"
A3_SOURCE="13c0bb89637f1b451e8a3e7b115fa250a9b8de7d"
POLICY=SourceQualityPolicy(max_windows=4,max_extra_contexts=2,
    max_acoustic_audio_s=240.,max_acoustic_wall_s=420.,
    max_independent_targets=2,max_independent_audio_s=100.,max_independent_wall_s=420.)


def verified_archive(path):
    if sha_file(path)!=A3_SHA:
        raise ValueError("A3 archive digest mismatch")
    members={}
    with tarfile.open(path,"r:gz") as archive:
        for member in archive.getmembers():
            parts=PurePosixPath(member.name)
            if parts.is_absolute() or ".." in parts.parts:
                raise ValueError("Unsafe archive member")
            if member.isdir():
                continue
            if not member.isfile() or member.size>32*1024*1024:
                raise ValueError("Unsafe or oversized archive member")
            if len(parts.parts)<2:
                raise ValueError("Expected single archive root")
            name="/".join(parts.parts[1:])
            if name in members:
                raise ValueError("Duplicate archive member")
            members[name]=archive.extractfile(member).read()
    manifest=json.loads(members["manifest.json"])
    if manifest["tested_source_sha"]!=A3_SOURCE:
        raise ValueError("A3 tested source mismatch")
    expected={a["path"]:a for a in manifest["artifacts"]}
    if len(expected)!=len(manifest["artifacts"]) or set(expected)!=set(members)-{"manifest.json"}:
        raise ValueError("Archive manifest inventory mismatch")
    for name,a in expected.items():
        data=members[name]
        if len(data)!=a["bytes"] or hashlib.sha256(data).hexdigest()!=a["sha256"]:
            raise ValueError("Archive artifact digest mismatch")
    return members


def update_dispositions(ledger):
    """Apply only the qualified metadata correction to frozen A3 evidence."""
    inv=ledger["recognized_occurrence_inventories"]
    original=[d for i in inv for s in i["locality"]["spans"] for d in s["selection_dispositions"]]
    fixed=finalize_boundary_dispositions(original)
    by_id={d["source_unit_id"]:d for d in fixed}
    for i in inv:
        for s in i["locality"]["spans"]:
            s["selection_dispositions"]=[by_id[d["source_unit_id"]] for d in s["selection_dispositions"]]
    for u in ledger["lexical_observations"]:
        d=by_id[u["id"]]
        u.update(boundary_decision=d,boundary_disposition=d["status"])
    return sum(a!=b for a,b in zip(original,fixed))


def plan_metrics(windows,deferred):
    tasks=[*windows,*deferred]
    return {"admitted":len(windows),"deferred_or_unlocalized":len(deferred),
            "evidence_ids":len({uid for w in tasks for uid in w["evidence_ids"]}),
            "admitted_evidence_ids":len({uid for w in windows for uid in w["evidence_ids"]}),
            "reasons":dict(Counter(r for w in tasks for r in w["reasons"])),
            "admitted_target_seconds":round(sum(b-a for a,b in (w["window"] for w in windows)),4)}


def replay(args,root):
    members=verified_archive(args.a3_archive)
    results,private={},{}
    for name in sorted(members):
        if not name.startswith("runs/") or not name.endswith("/source-evidence.v1.json"):
            continue
        parent=name.rsplit("/",1)[0]
        rec=json.loads(members[parent+"/recording.contract.json"])
        manifest=json.loads(members[parent+"/manifest.json"])
        led=json.loads(members[name])
        if manifest["source_evidence"]["sha256"]!=hashlib.sha256(members[name]).hexdigest():
            raise ValueError("Ledger is not bound to archived manifest")
        changed=update_dispositions(led)
        req=rec["processing"]["boundary_reconciliation"]["semantic_review_candidates"]
        prior=deepcopy(rec)
        windows,deferred=plan_recovery_windows(rec,led,req,None,SourceQualityPolicy())
        if rec!=prior:
            raise ValueError("Planning mutated source recording")
        if len({w["task_id"] for w in windows+deferred})!=len(windows+deferred):
            raise ValueError("Duplicate recovery task identity")
        ablated={k:v for k,v in led.items() if k!="recognized_occurrence_inventories"}
        aw,ad=plan_recovery_windows(rec,ablated,req,None,SourceQualityPolicy())
        results[parent]={"full":plan_metrics(windows,deferred),"without_inventory":plan_metrics(aw,ad),
                         "corrected_dispositions":changed}
        private[parent]={"windows":windows,"deferred":deferred}
    if len(results)!=16:
        raise ValueError("Expected 16 authenticated A3 recording runs")
    write_json(root/"private-tasks.json",private)
    write_json(root/"summary.json",{"status":"completed","archive_sha256":A3_SHA,
        "runs":results,"corrected_dispositions":sum(r["corrected_dispositions"] for r in results.values()),
        "comparison":"same new scheduler with/without occurrence inventory; not prior executable",
        "acoustic_truth_accepted":False})


def original_media(args):
    authority=read_authority(args.diagnostic_archive)
    paths=[Path(p) for p in authority["media_paths"] if Path(p).is_file()]
    path=args.media or (paths[0] if paths else None)
    if path is None:
        raise FileNotFoundError("Original media relocated; supply identical --media")
    media=probe_media(path,audio_stream_index=authority["selected_audio_stream"])
    if (media.fingerprint_sha256!=authority["fingerprint_sha256"]
            or abs(media.duration_s-authority["duration_s"])>.01):
        raise ValueError("Diagnostic media identity mismatch")
    return media


def challenger(registry):
    resource=registry.resolve("source_challenger",require_local_paths=True)
    if resource.adapter!="whisper_cpu" or resource.device!="cpu":
        raise ValueError("Explicit cached CPU challenger required")
    return resource


def worker(args,root,registry):
    media=original_media(args)
    # Registered CPU worker control, not a claim of missing speech.
    bounds=(1212.,1224.)
    target={"kind":"occurrence","target_id":"registered-worker-control",
            "bounds":bounds,"crops":independent_crops(bounds,media.duration_s),
            "ids":[],"task_ids":["worker-control"],"evidence_ids":["registered-input"],"chunks":[]}
    report=acquire_independent_evidence(
        {"source_language":"fr"},[target],media,challenger_resource=challenger(registry),
        output_dir=root,policy=POLICY)
    write_json(root/"private-worker.json",report)
    write_json(root/"summary.json",{"status":report["status"],"requests":report["requests"],
        "audio_seconds":report["audio_seconds"],"resource":report.get("resource"),
        "receipt_sha256":report.get("receipt_sha256"),"wall_s":report.get("wall_s"),
        "observations":len(report["observations"]),"acoustic_truth_accepted":False,
        "errors":[e["error"] for e in report["errors"]]})
    if report["status"]!="observed":
        raise RuntimeError("CPU challenger control did not produce validated observations")


def trace_ids(row):
    ids={row["id"]}
    for key in ("prior_source_hypothesis","prior_scene_source_hypothesis"):
        if isinstance(row.get(key),dict):
            ids.update(trace_ids(row[key]))
    for old in row.get("prior_source_hypotheses",[]):
        ids.update(trace_ids(old))
    return ids


def live(args,root,registry):
    media=original_media(args)
    resource=registry.resolve("asr",require_local_paths=True)
    cpu=challenger(registry)
    if (resource.adapter!="nemotron_ctc" or not resource.checkpoint_path
            or Path(resource.checkpoint_path).resolve()!=Path(CHECKPOINT).resolve()):
        raise ValueError("Expected exact cached Nemotron checkpoint")
    client=NemotronVLLMClient(base_url=resource.endpoint,model=resource.model,
                             api_key=resource.credential(),timeout_s=900.)
    bounded=BudgetedClient(client,root/"private-requests.json")
    # Same regions as A3, now only one current geometry per region. No
    # repeated matched geometry run and no full episode.
    cases=[(name,media,region,"fr") for name,region,_,_ in REGIONS]
    if args.control_media:
        control=probe_media(args.control_media,audio_stream_index=args.control_audio_stream)
        if not args.control_language or args.control_language=="fr":
            raise ValueError("Independent multilingual control language required")
        lo=args.control_start
        hi=min(control.duration_s,lo+args.control_seconds)
        if not 0<=lo<hi<=control.duration_s or hi-lo>60.:
            raise ValueError("Invalid bounded multilingual control")
        cases.append(("multilingual_control",control,(lo,hi),args.control_language))
    summary={}
    for name,source,region,language in cases:
        case=root/name
        case.mkdir()
        wav=case/"input.wav"
        extract_wav_segment(source.path,start_s=region[0],duration_s=region[1]-region[0],
                            out_wav=wav,audio_stream_index=source.selected_audio_stream)
        actual,digest=validate_wav_file(wav)
        if abs(actual-(region[1]-region[0]))>.05:
            raise ValueError("Evaluation input duration mismatch")
        manager=ModelLifecycle()
        with ExitStack() as stack:
            bounded.client=stack.enter_context(managed_asr_client(manager,resource,client))
            bounded.run=name+"/primary"
            out=process_media(wav,output_dir=case/"recording",client=bounded,
                config=RecordingPipelineConfig(source_language=language,mode="speaker",
                    chunk_policy="fixed",chunk_seconds=90.,context_seconds=10.,
                    speaker_embeddings="none",alignment_fallback="off",resume=False,
                    model_path=resource.checkpoint_path,asr_resource_fingerprint=resource.fingerprint()))
            before=deepcopy(out["contract"])
            metrics,_=inspect_run(out,region,digest)
            @contextmanager
            def acoustic_scope():
                bounded.run=name+"/recovery"
                try:
                    yield bounded
                finally:
                    # One owned ASR acquisition per region; release before
                    # the actual review_current_source CPU worker phase.
                    stack.close()
            canonical,report=review_current_source(out["contract"],recording_path=out["contract_path"],
                media_path=wav,output_dir=case,text_scope=lambda:None,
                asr_scope=acoustic_scope,challenger_resource=cpu,policy=POLICY,
                semantic_review=False)
        write_json(case/"canonical-source.contract.json",canonical)
        write_json(case/"source-quality.json",report)
        if out["contract"]!=before:
            raise ValueError("Primary source recording mutated")
        rows=canonical["utterances"]
        if len({r["id"] for r in rows})!=len(rows):
            raise ValueError("Duplicate canonical source identity")
        original_ids={r["id"] for r in before["utterances"]}
        represented={uid for r in rows for uid in trace_ids(r)}
        if not original_ids<=represented:
            raise ValueError("Original source identity lost during resolution")
        if any(_bounds(r) and _bounds(r)[1]>actual+.01 for r in rows):
            raise ValueError("Canonical bounds outside evaluation media")
        recovery=report["independent_recovery"]
        summary[name]={"absolute_media_region_s":region,"input_audio_sha256":digest,
            "source_media_fingerprint":source.fingerprint_sha256,"language":language,
            "primary":metrics,"original_source_ids_accounted_for":len(original_ids),
            "canonical_rows":len(rows),"primary_untimed":sum(not _bounds(r) for r in before["utterances"]),
            "canonical_untimed":sum(not _bounds(r) for r in rows),
            "decisions":dict(Counter(d["status"] for d in report["decisions"])),
            "task_states":dict(Counter(t["state"] for t in report["recovery_tasks"])),
            "occurrence_resolutions":dict(Counter(r["status"] for r in report["occurrence_resolutions"])),
            "acoustic_acquisition":report["acoustic_acquisition"],
            "independent":{"status":recovery["status"],"requests":recovery["requests"],
                "audio_seconds":recovery["audio_seconds"],"wall_s":recovery.get("wall_s"),
                "errors":[e["error"] for e in recovery["errors"]],
                "cross_model_events":len(recovery.get("cross_model_events",[])),
                "deferred_targets":len(recovery["deferred_targets"])},
            "semantic_review_requested":False,"acoustic_truth_accepted":False}
        write_json(root/"summary.json",{"status":"in_progress","regions":summary})
    write_json(root/"summary.json",{"status":"completed","regions":summary,
        "multilingual_control":"executed" if args.control_media else "not_available",
        "total_asr_requests":len(bounded.requests),
        "total_asr_audio_seconds":round(sum(r["audio_seconds"] for r in bounded.requests),4),
        "comparison":"same current primary recording before/after bounded acoustic recovery",
        "semantic_review_requested":False,"acoustic_truth_accepted":False})


def qwen(args,root,registry):
    """One real ordered-text control plus synthetic negative replays of its output."""
    from nvidia_transcribe.alignment import recover_chunk_alignment
    from nvidia_transcribe.qwen_aligner import QwenAlignerConfig,QwenForcedAlignerAdapter,map_language_for_qwen_aligner
    from nvidia_transcribe.selective_alignment import AdmissionPolicy,apply_selective_recovery
    candidates=[]
    for path in sorted(args.live_output_dir.glob("*/source-quality.json")):
        report=json.loads(path.read_text())
        rec=json.loads((path.parent/"canonical-source.contract.json").read_text())
        language=rec["source_language"]
        if not map_language_for_qwen_aligner(language):
            continue
        for e in report["independent_recovery"].get("cross_model_events",[]):
            b=_bounds(e)
            words=_lex(e["source_text"])
            if (b and 1.<=b[1]-b[0]<=12. and len(words)>=3 and words!=tuple(reversed(words))):
                candidates.append((path.parent,language,e))
        if not candidates:
            for row in rec["utterances"]:
                b=_bounds(row)
                prov=row.get("timestamp_provenance",{})
                words=_lex(row["source_text"])
                if (b and 1.<=b[1]-b[0]<=12. and len(words)>=3 and words!=tuple(reversed(words))
                        and prov.get("backend")=="separate_acoustic_contexts"
                        and len(set(prov.get("audio_sha256",[])))>=2
                        and not str(row.get("source_quality_status","")).startswith("unresolved")):
                    candidates.append((path.parent,language,row))
    if not candidates:
        write_json(root/"summary.json",{"status":"blocked_no_credible_ordered_text_control"})
        return
    case,language,event=candidates[0]
    source=probe_media(case/"input.wav")
    a,b=_bounds(event)
    lo,hi=max(0.,a-.6),min(source.duration_s,b+.6)
    wav=root/"control.wav"
    extract_wav_segment(source.path,start_s=lo,duration_s=hi-lo,out_wav=wav)
    _,digest=validate_wav_file(wav)
    raw="<spk:0>"+event["source_text"]
    native=recover_chunk_alignment(raw_text=raw,ctc_payload={"words":[],"error":"controlled_untimed_input"},
        source_offset_s=lo,chunk_index=0,model_path=None)
    if not native["units"]:
        raise ValueError("Ordered control has no recognized units")
    # Deliberately supplied test interval from positive acoustic event
    # evidence, NOT a new model diarization claim.
    native["diarization"]=[{"speaker":native["units"][0]["speaker"],"start":a,"end":b,"valid_timing":True}]
    resource=registry.resolve("forced_alignment",require_local_paths=True)
    if resource.device!="cuda:0":
        raise ValueError("GPU Qwen control requires explicit cuda:0 resource")
    outputs=[]
    def run(aligner):
        return apply_selective_recovery(native=deepcopy(native),raw_text=raw,
            chunk_start_s=lo,chunk_end_s=hi,chunk_index=0,source_language=language,
            aligner=aligner,extract_wav=lambda start,end:crop_control(source.path,root,start,end),
            policy=AdmissionPolicy(mode="fine"))
    class Capture:
        def __init__(self,adapter):
            self.adapter=adapter
        def align(self,**kwargs):
            output=self.adapter.align(**kwargs)
            outputs.append(output)
            return output
    with QwenForcedAlignerAdapter(QwenAlignerConfig(model_path=resource.checkpoint_path,
            worker_python=resource.worker_python,device=resource.device)) as adapter:
        positive=run(Capture(adapter))
    metrics=adapter.metrics
    if not outputs:
        write_json(root/"summary.json",{"status":"blocked_control_not_admitted","metrics":metrics})
        return
    class Replay:
        def __init__(self,mode):
            self.mode=mode
        def align(self,**kwargs):
            value=deepcopy(outputs[0])
            if self.mode=="wrong_order":
                value["units"]=list(reversed(value["units"]))
            else:
                for u in value["units"]:
                    u.update(end=u.get("start"),valid_timing=True)
            return value
    negatives={mode:run(Replay(mode)) for mode in ("wrong_order","zero_duration")}
    def count(result):
        return result["diagnostic"]["external_aligner"]["recovered_units"]
    write_json(root/"private-control.json",{"event":event,"source_case":case.name,"crop":[lo,hi],
        "input_audio_sha256":digest,"positive":positive,"actual_output":outputs[0],
        "negative_replays":negatives,"metrics":metrics})
    passed=count(positive)>0 and all(count(r)==0 for r in negatives.values())
    write_json(root/"summary.json",{"status":"completed" if passed else "control_not_qualified",
        "positive_recovered_units":count(positive),"negative_recovered_units":{m:count(r) for m,r in negatives.items()},
        "metrics":metrics,"negative_controls":"synthetic perturbations of real worker response",
        "word_timing_truth_accepted":False})
    if not passed:
        raise RuntimeError("Qwen positive/negative control not qualified")


def crop_control(media,root,start,end):
    path=root/"alignment.wav"
    extract_wav_segment(media,start_s=start,duration_s=end-start,out_wav=path)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",required=True,choices=["replay","worker","live","qwen"])
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--a3-archive",type=Path)
    parser.add_argument("--diagnostic-archive",type=Path)
    parser.add_argument("--media",type=Path)
    parser.add_argument("--model-resources",type=Path)
    parser.add_argument("--control-media",type=Path)
    parser.add_argument("--control-language")
    parser.add_argument("--control-audio-stream",type=int,default=0)
    parser.add_argument("--control-start",type=float,default=0.)
    parser.add_argument("--control-seconds",type=float,default=40.)
    parser.add_argument("--live-output-dir",type=Path)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    if args.phase=="replay":
        replay(args,args.output_dir)
    else:
        registry=load_model_resources(args.model_resources)
        {"worker":worker,"live":live,"qwen":qwen}[args.phase](args,args.output_dir,registry)


if __name__=="__main__":
    main()
