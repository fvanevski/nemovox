"""Fail-closed ASR generation integrity; independent of acoustic word validity.

A model-generated length-limit loop is not a completed transcription, even
when a text-loop cleaner leaves a plausible-looking short fragment. Keep the
raw response, hash, and diagnostics available for an owned retry. Never
manufacture source speech from a failed generation.
"""

from __future__ import annotations

from typing import Any

from nvidia_transcribe.prompts import cleanup_generation_loops


class IncompleteASRGeneration(RuntimeError):
    """A chunk is checkpointed as incomplete, not silently rendered as complete."""


def audit_asr_generation(raw_response: dict[str, Any]) -> dict[str, Any]:
    """Return non-content metadata; original text/CTC must remain immutable."""
    choice = (raw_response.get("choices") or [{}])[0]
    raw = str((choice.get("message") or {}).get("content") or "")
    finish = choice.get("finish_reason")
    clean = cleanup_generation_loops(raw)
    # One or two genuine repeated interjections cannot trigger this rule.
    pathological_loop = len(raw) >= 2048 and len(clean) * 5 < len(raw)
    if finish == "length":
        status, reason = "incomplete", "generation_limit_exhausted"
    elif pathological_loop:
        status, reason = "incomplete", "pathological_generation_loop"
    elif finish not in ("stop", None):
        status, reason = "review", "unrecognized_finish_reason"
    else:
        status, reason = "complete", None
    return {
        "schema": "nemovox-asr-generation-integrity/v1",
        "status": status,
        "reason": reason,
        "finish_reason": finish,
        "raw_text_chars": len(raw),
        "post_loop_guard_chars": len(clean),
        "loop_guard_removed_chars": len(raw) - len(clean),
        "ctc_metadata_present": isinstance(raw_response.get("ctc_timestamps"), dict),
        "source_text_not_modified": True,
    }
