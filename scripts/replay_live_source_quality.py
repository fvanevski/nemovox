"""Counterfactual selection of retained *live* evidence, no inference.

Authenticate the two input archives and bind observations to the unchanged
primary recording. This is not a fresh normal production acquisition.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath
from copy import deepcopy

from scripts.evaluate_source_robustness import read_intake, sha
from nvidia_transcribe.source_quality import SourceQualityPolicy, _digest, _bounds
from nvidia_transcribe.targeted_occurrences import adjudicate_sequences, recover_unlocalized_events
from nvidia_transcribe.failed_ctc_recovery import recover_failed_ctc_events
from nvidia_transcribe.recovery_planner import plan_recovery_windows


def live_inputs(path,expected):
    if sha(path)!=expected:
        raise ValueError("Live archive digest mismatch")
    with tarfile.open(path,"r:gz") as t:
        members=t.getmembers()
        names=[m.name for m in members]
        if len(set(names))!=len(names) or any(not m.isfile() or PurePosixPath(m.name).is_absolute()
                or ".." in PurePosixPath(m.name).parts for m in members):
            raise ValueError("Unsafe live archive")
        def read(name):
            with t.extractfile(name) as f:
                return f.read()
        manifest=json.loads(read("manifest.json"))
        if set(names)!=set(manifest["files"])|{"manifest.json","SHA256SUMS"}:
            raise ValueError("Live manifest membership mismatch")
        wanted={"french/source-quality.json","mandarin/source-quality.json",
                "mandarin/asr/recording.contract.json","mandarin/asr/source-evidence.v1.json"}
        result={}
        hashes={}
        for name,entry in manifest["files"].items():
            data=read(name)
            digest=hashlib.sha256(data).hexdigest()
            if len(data)!=entry["bytes"] or digest!=entry["sha256"]:
                raise ValueError("Live manifest mismatch "+name)
            hashes[name]=digest
            if name in wanted:
                result[name]=json.loads(data)
        hashes["manifest.json"]=hashlib.sha256(read("manifest.json")).hexdigest()
        checks={}
        for line in read("SHA256SUMS").decode().splitlines():
            digest,name=line.split(maxsplit=1)
            if name in checks:
                raise ValueError("Duplicate checksum path")
            checks[name]=digest
        if hashes!=checks:
            raise ValueError("Live checksum inventory mismatch")
        return manifest,result


def replay(recording,ledger,report):
    if _digest(recording)!=report["recording_sha256"]:
        raise ValueError("Source-quality recording binding mismatch")
    if ledger["media_fingerprint_sha256"]!=recording["source"]["fingerprint_sha256"]:
        raise ValueError("Occurrence ledger recording identity mismatch")
    before=deepcopy(recording)
    policy=SourceQualityPolicy()
    observed=report["windows"]
    native=report["acoustic_observations"]
    independent=report["independent_recovery"]
    targets=independent["targets"]
    other=independent["observations"]
    rows,echo=adjudicate_sequences(recording,ledger,observed,native,targets,other)
    rows,untimed=recover_unlocalized_events({**recording,"utterances":rows},observed,native,targets,other)
    rows,failed=recover_failed_ctc_events({**recording,"utterances":rows},ledger,observed,native,targets,other,policy)
    if recording!=before:
        raise ValueError("Input recording mutated")
    admitted,deferred=plan_recovery_windows(recording,ledger,
        recording["processing"]["boundary_reconciliation"]["semantic_review_candidates"],None,policy)
    return {"sequence_decisions":echo,"untimed_event_decisions":untimed,"failed_ctc_decisions":failed,
        "original_rows":len(before["utterances"]),"original_untimed":sum(not _bounds(r) for r in before["utterances"]),
        "candidate_rows":len(rows),"candidate_untimed":sum(not _bounds(r) for r in rows),
        "candidate_timed":sum(bool(_bounds(r)) for r in rows),"canonical_rows":rows,
        "new_structural_plan":{"admitted":admitted,"deferred":deferred},
        "evidence_class":"counterfactual_targeted_selection_of_retained_live_observations"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--intake",type=Path,required=True)
    p.add_argument("--intake-sha256",required=True)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--sha256",required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    os.umask(0o077)
    _,intake=read_intake(args.intake,args.intake_sha256)
    manifest,live=live_inputs(args.archive,args.sha256)
    output={"live_evidence_commit":manifest["candidate"],
        "french":replay(intake["retained/recording/asr/recording.contract.json"],
            intake["retained/recording/asr/source-evidence.v1.json"],live["french/source-quality.json"]),
        "mandarin":replay(live["mandarin/asr/recording.contract.json"],
            live["mandarin/asr/source-evidence.v1.json"],live["mandarin/source-quality.json"])}
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(output,f,ensure_ascii=False,indent=2)


if __name__=="__main__":
    main()
