"""Exact-input current-job source-quality checkpoint, not a historical import."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os

from nvidia_transcribe.media import probe_media
from nvidia_transcribe.source_quality import _digest, load_current_ledger

SCHEMA="nemovox-source-quality-checkpoint/v1"


def identity(recording, recording_path, media_path, policy, resources):
    ledger=load_current_ledger(recording,recording_path)
    source=recording["source"]
    media=probe_media(media_path,audio_stream_index=source["selected_audio_stream"])
    if (media.fingerprint_sha256!=source["fingerprint_sha256"]
            or abs(media.duration_s-source["duration_s"])>.01):
        raise ValueError("Source-quality checkpoint media does not match original recording")
    # A conservative code fingerprint invalidates checkpoints after any
    # implementation change; no assertion of immutable served model weights.
    root=Path(__file__).parent
    code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    return _digest({"schema":SCHEMA,"recording":_digest(recording),"ledger":_digest(ledger),
        "media":media.fingerprint_sha256,"audio_stream":source["selected_audio_stream"],
        "policy":asdict(policy),"resources":resources,"code":code})


def reusable(report):
    context=report.get("recording_context")
    recovery=report.get("independent_recovery") or {}
    scene=report.get("scene_evidence") or {}
    return (report.get("status") in ("complete","partial")
        and report.get("semantic_review_requested") is True
        and report.get("original_recording_modified") is False
        and isinstance(context,dict)
        and not any(a.get("status")=="failed" for a in context.get("attempts",[]))
        and not recovery.get("errors") and recovery.get("status")!="error"
        and not scene.get("errors"))


def run_checkpoint(path, key, recording, *, resume, run):
    """Reuse only an authenticated completed stage, including its honest
    unresolved/deferred dispositions. Never cache an execution error.
    """
    path=Path(path)
    if resume and path.is_file():
        try:
            doc=json.loads(path.read_bytes())
            payload=doc["payload"]
            canonical,report=payload["canonical"],payload["report"]
            if (doc["schema"]==SCHEMA and doc["input_sha256"]==key
                    and doc["payload_sha256"]==_digest(payload)
                    and report["recording_sha256"]==_digest(recording)
                    and canonical["source"]==recording["source"]
                    and canonical["source_language"]==recording["source_language"]
                    and canonical["source_quality"]["recording_sha256"]==report["recording_sha256"]
                    and reusable(report)):
                return canonical,report,"reused_exact_input"
        except (OSError,ValueError,KeyError,TypeError,AttributeError):
            pass
    # Remove the old admission receipt before executing a new attempt.
    # A failed fresh run must not resurrect a previous successful result.
    path.unlink(missing_ok=True)
    canonical,report=run()
    if reusable(report):
        payload={"canonical":canonical,"report":report}
        doc={"schema":SCHEMA,"input_sha256":key,"payload_sha256":_digest(payload),"payload":payload}
        tmp=path.with_name("."+path.name+".tmp")
        tmp.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding="utf-8")
        os.replace(tmp,path)
        return canonical,report,"computed"
    return canonical,report,"computed_not_reusable"
