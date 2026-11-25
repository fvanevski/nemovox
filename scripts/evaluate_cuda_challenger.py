"""Matched CUDA challenger qualification on the sealed #19 CPU input/crops.

Uses the production worker and lifecycle, not a second recognition pathway.
No text or primary-ASR calls. The frozen CPU outputs are observations, not
gold transcription or a mandatory identical-output target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from copy import deepcopy

from nvidia_transcribe.media import probe_media, validate_wav_file
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.resource_lifecycle import ModelLifecycle
from nvidia_transcribe.source_quality import SourceQualityPolicy, _digest
from nvidia_transcribe.source_recovery import acquire_independent_evidence
from nvidia_transcribe.occurrence_resolution import resolve_occurrences

ARCHIVE_SHA="206ada956e77b9db1263de090fbbeac139c78e1fa26670bc24963cbadcad3498"
MEDIA_SHA="30234a1c67ae9ed310ed6c76e4b97d4d0c9d54a7220129ac0a05c61eb368f314"
POLICY=SourceQualityPolicy(max_windows=4,max_extra_contexts=2,
    max_acoustic_audio_s=240.,max_acoustic_wall_s=420.,
    max_independent_targets=2,max_independent_audio_s=100.,max_independent_wall_s=420.,
    scene_max_cases=1)


def read_archive(path):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest()!=ARCHIVE_SHA:
            raise ValueError("Prior CPU qualification archive digest mismatch")
    members={}
    with tarfile.open(path,"r:gz") as archive:
        for member in archive:
            parts=PurePosixPath(member.name).parts
            if not parts or parts[0]!="packet" or ".." in parts:
                raise ValueError("Unexpected archive path")
            if member.isdir():
                continue
            if not member.isfile() or member.size>16*1024*1024:
                raise ValueError("Invalid archive member")
            name="/".join(parts[1:])
            if name in members:
                raise ValueError("Duplicate archive member")
            members[name]=archive.extractfile(member).read()
    manifest=json.loads(members["manifest.json"])
    if set(manifest)!=set(members)-{"manifest.json"}:
        raise ValueError("Archive inventory mismatch")
    for name,entry in manifest.items():
        payload=members[name]
        if len(payload)!=entry["bytes"] or hashlib.sha256(payload).hexdigest()!=entry["sha256"]:
            raise ValueError("Archive member digest mismatch")
    return members


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",required=True,type=Path)
    parser.add_argument("--media",required=True,type=Path)
    parser.add_argument("--model-resources",required=True,type=Path)
    parser.add_argument("--lifecycle-config",type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    members=read_archive(args.archive)
    base="live-acoustic-recording/"
    rec=json.loads(members[base+"recording.contract.json"])
    manifest=json.loads(members[base+"manifest.json"])
    report=json.loads(members["live-acoustic-eval/source-quality.json"])
    ledger_bytes=members[base+"source-evidence.v1.json"]
    if (manifest["recording_contract_sha256"]!=hashlib.sha256(members[base+"recording.contract.json"]).hexdigest()
        or manifest["source_evidence"]["sha256"]!=hashlib.sha256(ledger_bytes).hexdigest()
        or report["recording_sha256"]!=_digest(rec)):
        raise ValueError("Recording/ledger authority mismatch")
    duration,media_sha=validate_wav_file(args.media)
    media=probe_media(args.media,audio_stream_index=rec["source"]["selected_audio_stream"])
    if (media_sha!=MEDIA_SHA or abs(duration-rec["source"]["duration_s"])>.01
        or media.fingerprint_sha256!=rec["source"]["fingerprint_sha256"]):
        raise ValueError("Exact matched original WAV required")
    prior=report["independent_recovery"]
    targets=prior["targets"]
    if len(targets)!=2 or len(prior["observations"])!=4 or prior["resource"]["device"]!="cpu":
        raise ValueError("Unexpected CPU baseline")
    resource=load_model_resources(args.model_resources).resolve("source_challenger",require_local_paths=True)
    manager=ModelLifecycle(config_path=args.lifecycle_config)
    root=args.output_dir
    root.mkdir(parents=True,exist_ok=False)
    before=_digest(rec)
    candidate=acquire_independent_evidence(rec,deepcopy(targets),media,
        challenger_resource=resource,output_dir=root,policy=POLICY,lifecycle=manager)
    (root/"cuda-worker.json").write_text(json.dumps(candidate,ensure_ascii=False,indent=2))
    if candidate["status"]!="observed":
        raise RuntimeError("CUDA worker did not produce validated observations; inspect cuda-worker.json")
    def keys(result):
        return {o["id"]:(o["audio_sha256"],o["crop"]) for o in result["observations"]}
    if keys(candidate)!=keys(prior) or candidate["resource"]["checkpoint_sha256"]!=prior["resource"]["checkpoint_sha256"]:
        raise ValueError("Matched crop or checkpoint identity mismatch")
    # Resolver control uses frozen native observations and only the fresh
    # independent evidence; no invented newly acquired native events.
    rows,decisions=resolve_occurrences(rec,[],report["acoustic_observations"],targets,deepcopy(candidate),POLICY)
    if _digest(rec)!=before:
        raise ValueError("Original recording mutated")
    (root/"private-resolution.json").write_text(json.dumps({"rows":rows,"decisions":decisions},ensure_ascii=False,indent=2))
    (root/"summary.json").write_text(json.dumps({"status":"observed",
        "archive_sha256":ARCHIVE_SHA,"media_sha256":MEDIA_SHA,"matched_crops":4,
        "checkpoint_equivalence":True,"cpu_wall_s":prior["wall_s"],
        "cuda_wall_s":candidate["wall_s"],"audio_seconds":candidate["audio_seconds"],
        "cuda_resource":candidate["resource"],"cpu_receipt":prior["receipt_sha256"],
        "cuda_receipt":candidate["receipt_sha256"],"resolution_decisions":len(decisions),
        "source_truth_accepted":False},indent=2))


if __name__=="__main__":
    main()
