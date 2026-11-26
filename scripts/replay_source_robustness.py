"""Replay the sealed issue-24 comparison through current production selectors.

This is a private counterfactual using previously acquired observations,
not a fresh acquisition or a claim that today's scheduler produced them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path, PurePosixPath
import tarfile

from nvidia_transcribe.source_quality import SourceQualityPolicy, project_acoustic_observation
from nvidia_transcribe.recovery_planner import plan_recovery_windows
from nvidia_transcribe.targeted_occurrences import adjudicate_sequences, recover_unlocalized_events
from nvidia_transcribe.scene_evidence import assess_case_payload


def load_archive(path,digest):
    with path.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=digest:
            raise ValueError("Comparison archive digest mismatch")
    prefix="source-robustness-comparison/"
    with tarfile.open(path,"r:gz") as t:
        members=t.getmembers()
        names=[m.name for m in members]
        if (len(names)!=len(set(names)) or any(not m.isfile() or not m.name.startswith(prefix)
                or ".." in PurePosixPath(m.name).parts for m in members)):
            raise ValueError("Unsafe comparison archive")
        def read(name):
            with t.extractfile(prefix+name) as stream:
                return stream.read()
        manifest=json.loads(read("manifest.json"))
        entries={e["path"]:e for e in manifest["entries"]}
        if len(entries)!=len(manifest["entries"]):
            raise ValueError("Duplicate manifest entries")
        result={}
        for name,e in entries.items():
            data=read(name)
            if len(data)!=e["bytes"] or hashlib.sha256(data).hexdigest()!=e["sha256"]:
                raise ValueError("Comparison payload mismatch: "+name)
            if name.endswith(".json"):
                result[name]=json.loads(data)
    return manifest,result


def replay(data):
    rec=data["baseline/retained/recording/asr/recording.contract.json"]
    ledger=data["baseline/retained/recording/asr/source-evidence.v1.json"]
    request=rec["processing"]["boundary_reconciliation"]["semantic_review_candidates"]
    policy=SourceQualityPolicy()
    admitted,deferred=plan_recovery_windows(rec,ledger,request,None,policy)
    receipts={r["id"]:r for r in data["acoustic/native-summary.json"]["requests"]}
    old=data["acoustic/independent.json"]
    observations,targets,windows=[],[],[]
    # These labels identify the previously registered private experiment
    # arms; no lexical content or episode timestamps appear in production.
    bindings={
        "sequence_origin":(["earlier-boundary"],"independent-earlier"),
        "sequence_displacement":(["later-projection","chunk-2-no-left-context"],"independent-later"),
        "untimed_event_context":(["untimed-broad","untimed-shifted"],"independent-untimed"),
    }
    for reason,(names,target_name) in bindings.items():
        matches=[w for w in admitted if reason in w["reasons"]]
        if len(matches)!=1:
            raise ValueError("Expected unique admitted discovery task for "+reason)
        task=deepcopy(matches[0])
        task["state"]="observed"
        windows.append(task)
        target=next(t for t in old["targets"] if t["target_id"]==target_name)
        targets.append({**target,"task_ids":[task["task_id"]],"evidence_ids":task["evidence_ids"]})
        for name in names:
            r=receipts[name]
            raw=data["acoustic/"+name+".raw.json"]
            obs=project_acoustic_observation(raw,r["audio_sha256"],r["crop"],
                rec["source_language"],actual=r["audio_seconds"],model=raw.get("model"))
            obs["task_ids"]=[task["task_id"]]
            obs["chunks"]=task["chunks"]
            if name=="chunk-2-no-left-context":
                if list(task.get("context_removed_crop",[]))!=list(r["crop"]):
                    raise ValueError("Counterfactual core geometry mismatch")
                obs["context_removed_task_ids"]=[task["task_id"]]
            observations.append(obs)
    before=deepcopy(rec)
    rows,echo=adjudicate_sequences(rec,ledger,windows,observations,targets,old["observations"])
    rows,untimed=recover_unlocalized_events({**rec,"utterances":rows},windows,observations,
                                          targets,old["observations"])
    if rec!=before:
        raise ValueError("Original recording mutated")
    # Replay first replies through the current validator, not a fresh model.
    scenes=[]
    for case in data["scene/scene-comparison.json"]["cases"]:
        entry=(case.get("failed_assessments") or [case["assessment"]])[0]
        class Client:
            def complete(self,**kw):
                return deepcopy(entry["response"])
        fresh={}
        try:
            result=assess_case_payload(Client(),entry["input"],entry["schema"],fresh)
            scenes.append({"source_id":case["source_id"],"status":"valid",
                           "normalization":result.get("normalization"),
                           "corrective_requests":fresh.get("assessment_recovery_attempts",0)})
        except ValueError as exc:
            scenes.append({"source_id":case["source_id"],"status":"rejected","error":str(exc)})
    return {"evidence_class":"counterfactual_replay_of_sealed_observations",
        "admitted":admitted,"deferred":deferred,"rebound_tasks":windows,
        "sequence_decisions":echo,"untimed_decisions":untimed,"scene_decisions":scenes,
        "original_rows":len(rec["utterances"]),"candidate_rows":len(rows),
        "canonical_rows":rows,"original_unchanged":True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--sha256",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    os.umask(0o077)
    manifest,data=load_archive(args.archive,args.sha256)
    result=replay(data)
    result["input_identity"]=manifest["input"]
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2)


if __name__=="__main__":
    main()
