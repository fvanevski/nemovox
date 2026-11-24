"""Bounded primary-ASR recovery by smaller, overlapping source excerpts.

A failed parent is never a transcript. Accepted leaves are independent raw
observations with actual source offsets; ordinary reconciliation handles
their overlap. Checkpoint identity is distinct from final sequential indices.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from nvidia_transcribe.asr_integrity import IncompleteASRGeneration, audit_asr_generation
from nvidia_transcribe.media import validate_wav_file


def split_spec(spec):
    """Bisect the owned core, with bounded overlapping inference context."""
    if spec.core_duration_s < 12.0:
        return []
    mid=round((spec.core_start_s+spec.core_end_s)/2,6)
    pad=min(2.0,spec.core_duration_s/8)
    return [
        replace(spec,core_end_s=mid,
                infer_start_s=max(spec.infer_start_s,round(spec.core_start_s-pad,6)),
                infer_end_s=min(spec.infer_end_s,round(mid+pad,6))),
        replace(spec,core_start_s=mid,
                infer_start_s=max(spec.infer_start_s,round(mid-pad,6)),
                infer_end_s=min(spec.infer_end_s,round(spec.core_end_s+pad,6))),
    ]


def recover_generation(parent,*,root,signature,resume,extract,transcribe,write):
    """At most two split levels (six new requests), no synthetic parent.

    Reuse verified complete leaf receipts across interruption. A rejected
    child may be subdivided but cannot supply transcript content. Transport
    errors propagate; only audited generation incompleteness triggers splits.
    """
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    def visit(spec,key,depth):
        wav=root/f"{key}.wav"
        raw=root/f"{key}.raw.json"
        receipt=root/f"{key}.receipt.json"
        identity={"schema":"primary-asr-rechunk/v1","signature":signature,
                  "spec":spec.to_dict(),"node":key}
        entry=None
        response=None
        if resume and receipt.is_file() and raw.is_file() and wav.is_file():
            try:
                saved=json.loads(receipt.read_text())
                duration,digest=validate_wav_file(wav,expected_sample_rate_hz=16000)
                data=raw.read_bytes()
                if (saved["identity"]==identity and saved["wav_sha256"]==digest
                        and saved["raw_response_sha256"]==hashlib.sha256(data).hexdigest()
                        and abs(duration-spec.infer_duration_s)<=.1):
                    response=json.loads(data)
                    if audit_asr_generation(response)["status"]=="complete":
                        return [(spec,wav,response,{**saved,"recovery_checkpoint_reused":True})]
                    if audit_asr_generation(response)["status"]=="incomplete" and depth<2 and split_spec(spec):
                        entry=saved
            except (OSError,ValueError,KeyError,TypeError):
                entry=None
        if entry is None:
            extract(spec,wav)
            duration,digest=validate_wav_file(wav,expected_sample_rate_hz=16000)
            if abs(duration-spec.infer_duration_s)>.1:
                raise ValueError("Recovery WAV does not match inference window")
            result=transcribe(spec,wav)
            response=result["raw_response"]
            integrity=audit_asr_generation(response)
            data=json.dumps(response,ensure_ascii=False,indent=2)
            raw_sha=hashlib.sha256(data.encode()).hexdigest()
            # Preserve rejected bytes even if a terminal node is retried later.
            if integrity["status"]!="complete":
                write(root/"rejected"/f"{key}_{raw_sha}.raw.json",data)
            write(raw,data)
            entry={**spec.to_dict(),"identity":identity,
                   "status":"completed" if integrity["status"]=="complete" else "incomplete_generation",
                   "wav_path":str(wav),"wav_sha256":digest,"actual_wav_duration_s":duration,
                   "raw_response_path":str(raw),"raw_response_sha256":raw_sha,
                   "generation_integrity":integrity,"finish_reason":integrity["finish_reason"],
                   "usage":response.get("usage"),"latency_s":float(result["latency_s"]),
                   "recovery_checkpoint_reused":False}
            write(receipt,json.dumps(entry,ensure_ascii=False,indent=2))
        if entry["generation_integrity"]["status"]=="complete":
            return [(spec,wav,response,entry)]
        children=split_spec(spec) if depth<2 and entry["generation_integrity"]["status"]=="incomplete" else []
        if not children:
            raise IncompleteASRGeneration(f"Primary ASR recovery exhausted at {key}, "
                f"core {spec.core_start_s}-{spec.core_end_s}; raw evidence retained at {raw}")
        return [leaf for i,child in enumerate(children)
                for leaf in visit(child,f"{key}-{i}",depth+1)]
    children=split_spec(parent)
    if not children:
        raise IncompleteASRGeneration("Primary ASR core too short for further bounded subdivision")
    return [leaf for i,child in enumerate(children) for leaf in visit(child,str(i),1)]
