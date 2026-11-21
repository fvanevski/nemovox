"""Admit useful untimed speech windows only after all native CTC recovery has finished.

This module never re-aligns native-timed units or changes recognized text/speaker
identity. A candidate requires a finite, uniquely localized speech interval.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nvidia_transcribe.alignment import (
    _apply_verified_sequence_timings,
    _contiguous_speaker_spans,
    _has_mixed_speaker_overlap,
    _join_units_for_aligner,
)


@dataclass(frozen=True)
class AdmissionPolicy:
    mode: str = "subtitle"   # subtitle (default) | fine
    max_window_s: float = 20.0
    max_candidates_per_chunk: int = 16
    min_subtitle_chars: int = 3
    min_untimed_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in {"subtitle", "fine"}:
            raise ValueError("Unsupported Qwen admission mode")
        if not math.isfinite(self.max_window_s) or not 0.5 <= self.max_window_s <= 180:
            raise ValueError("Invalid max_window_s")
        if not 1 <= self.max_candidates_per_chunk <= 100:
            raise ValueError("Invalid max_candidates_per_chunk")


@dataclass(frozen=True)
class RecoveryCandidate:
    lo: int
    hi: int
    text: str
    start_s: float
    end_s: float
    untimed_units: int


def _finite_interval(start: Any, end: Any) -> tuple[float, float] | None:
    if isinstance(start, bool) or isinstance(end, bool):
        return None
    try:
        a, b = float(start), float(end)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b):
        return None
    return a, b


def _speaker_key(value: Any) -> str:
    s = str(value or "").strip().lower()
    found = re.search(r"(?:spk[_:]|<spk:)(\d+)", s)
    return found.group(1) if found else s


def plan_selective_recovery(
    *,
    native: dict[str, Any],
    raw_text: str,
    chunk_start_s: float,
    chunk_end_s: float,
    policy: AdmissionPolicy,
) -> tuple[list[RecoveryCandidate], list[dict[str, Any]]]:
    """Return strictly localized candidates and bounded reason-coded decisions."""
    units = native.get("units") or []
    diar = native.get("diarization") or []
    decisions: list[dict[str, Any]] = []
    candidates: list[RecoveryCandidate] = []
    bounds = _finite_interval(chunk_start_s, chunk_end_s)
    if bounds is None:
        return [], [{"status": "skipped_invalid_chunk_window"}]
    chunk_start_s, chunk_end_s = bounds

    diar_valid: list[tuple[float, float, str]] = []
    for d in diar:
        if not d.get("valid_timing"):
            continue
        interval = _finite_interval(d.get("start"), d.get("end"))
        if interval is not None:
            diar_valid.append((*interval, _speaker_key(d.get("speaker"))))

    for lo, hi in _contiguous_speaker_spans(units):
        span = units[lo:hi]
        untimed = [u for u in span if not u.get("valid_timing")]
        if not untimed:
            continue
        decision: dict[str, Any] = {"span_range": [lo, hi], "untimed_units": len(untimed)}
        decisions.append(decision)
        text = _join_units_for_aligner(span)
        chars = len("".join(c for c in text if c.isalnum()))
        if not text or chars == 0:
            decision["status"] = "skipped_empty_or_artifact"
            continue
        if any(not str(u.get("text") or "").strip() for u in span):
            decision["status"] = "skipped_invalid_unit"
            continue
        if policy.mode == "subtitle" and (
            chars < policy.min_subtitle_chars
            or len(untimed) / len(span) < policy.min_untimed_fraction
        ):
            decision["status"] = "skipped_low_subtitle_value"
            continue
        spk_keys = {_speaker_key(u.get("speaker")) for u in span}
        if len(spk_keys) != 1:
            decision["status"] = "skipped_mixed_speaker_units"
            continue
        speaker = next(iter(spk_keys))
        valid_in_span = [
            interval for u in span
            if u.get("valid_timing")
            if (interval := _finite_interval(u.get("start"), u.get("end"))) is not None
        ]
        matching_diar = [
            (s, e) for s, e, who in diar_valid
            if who == speaker and e > chunk_start_s and s < chunk_end_s
        ]
        interval: tuple[float, float] | None = None
        if valid_in_span:
            # A partly timed utterance is itself a reliable localization anchor.
            interval = (
                max(chunk_start_s, min(s for s, _ in valid_in_span) - 1.2),
                min(chunk_end_s, max(e for _, e in valid_in_span) + 1.2),
            )
        elif len(matching_diar) == 1:
            # Multiple same-speaker diarization windows cannot be matched to
            # an untimed sentence uniquely. Do not guess by index.
            s, e = matching_diar[0]
            interval = max(chunk_start_s, s - 0.35), min(chunk_end_s, e + 0.35)
        else:
            # An untimed sentence needs *both* neighboring time anchors.
            left = next(
                (_finite_interval(u.get("start"), u.get("end")) for u in reversed(units[:lo])
                 if _speaker_key(u.get("speaker")) == speaker and u.get("valid_timing")),
                None,
            )
            right = next(
                (_finite_interval(u.get("start"), u.get("end")) for u in units[hi:]
                 if _speaker_key(u.get("speaker")) == speaker and u.get("valid_timing")),
                None,
            )
            if left is not None and right is not None and right[0] > left[1]:
                interval = max(chunk_start_s, left[1] - 0.15), min(chunk_end_s, right[0] + 0.15)
        if interval is None:
            decision["status"] = "skipped_unanchored_or_ambiguous"
            continue
        start, end = interval
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end
                and 0.25 <= end - start <= policy.max_window_s):
            decision["status"] = "skipped_invalid_or_long_window"
            continue
        if _has_mixed_speaker_overlap(diar, raw_text, window=(start, end)):
            decision["status"] = "skipped_mixed_speaker_overlap"
            continue
        if any(start < prev.end_s and end > prev.start_s for prev in candidates):
            # Do not make overlapping forced-aligner calls and accidentally map
            # one utterance's words into a second speaker/duplicate span.
            decision["status"] = "skipped_ambiguous_shared_window"
            continue
        if len(candidates) >= policy.max_candidates_per_chunk:
            decision["status"] = "skipped_chunk_candidate_limit"
            continue
        decision.update(status="admitted", window=[round(start, 3), round(end, 3)])
        candidates.append(RecoveryCandidate(lo, hi, text, start, end, len(untimed)))
    return candidates, decisions


def apply_selective_recovery(
    *,
    native: dict[str, Any],
    raw_text: str,
    chunk_start_s: float,
    chunk_end_s: float,
    chunk_index: int,
    source_language: str | None,
    aligner: Any,
    extract_wav: Callable[[float, float], Path],
    policy: AdmissionPolicy,
) -> dict[str, Any]:
    """Attach verified external timestamps in place, retaining native provenance."""
    candidates, decisions = plan_selective_recovery(
        native=native, raw_text=raw_text,
        chunk_start_s=chunk_start_s, chunk_end_s=chunk_end_s, policy=policy,
    )
    units = native["units"]
    total_recovered = 0
    for c in candidates:
        decision = next(d for d in decisions if d["span_range"] == [c.lo, c.hi])
        try:
            wav = extract_wav(c.start_s, c.end_s)
            output = aligner.align(
                audio_path=wav, text=c.text,
                language=source_language, source_offset_s=c.start_s,
            )
            if isinstance(output, dict):
                external_units = list(output.get("units") or [])
                diagnostic = dict(output.get("diagnostic") or {})
            else:
                external_units = list(output or [])
                diagnostic = {}
            decision["aligner_status"] = diagnostic.get("status", "unknown")
            decision["aligner_diagnostic"] = diagnostic
            if not external_units:
                decision["status"] = "attempted_no_valid_units"
                continue
            matched = _apply_verified_sequence_timings(
                units, external_units, recovered_status="external_qwen_recovered",
                only_untimed=True, target_range=(c.lo, c.hi),
                min_start_s=c.start_s, max_end_s=c.end_s,
                enforce_neighbor_monotonic=True,
            )
            decision["matched_untimed_units"] = matched["matched_units"]
            decision["rejected_bounds_units"] = matched["rejected_bounds_count"]
            decision["status"] = "recovered" if matched["matched_units"] else "attempted_no_verified_match"
            total_recovered += matched["matched_units"]
        except Exception as exc:
            # No timestamps are fabricated and existing CTC timing survives.
            decision["status"] = "external_error"
            decision["error_type"] = type(exc).__name__
            decision["error"] = str(exc)[:300]

    remaining = sum(not u.get("valid_timing") for u in units)
    native_diag = native.setdefault("diagnostic", {})
    native_diag["external_aligner"] = {
        "backend": "qwen3_forced_aligner",
        "policy": policy.mode,
        "status": ("recovered" if total_recovered else
                   "no_eligible_spans" if not candidates else "no_recovery"),
        "candidate_count": len(candidates),
        "admitted_count": sum(d["status"] in {"recovered", "attempted_no_valid_units",
                          "attempted_no_verified_match", "external_error"} for d in decisions),
        "skipped_count": sum(str(d["status"]).startswith("skipped") for d in decisions),
        "recovered_units": total_recovered,
        "remaining_untimed_units": remaining,
        "localized_spans": decisions,
    }
    if total_recovered:
        native_diag["external_recovered_units"] = total_recovered
        native_diag["untimed_units"] = remaining
        native_diag["status"] = "external_recovered" if remaining == 0 else "partial_external_recovery"
        native["alignment_status"] = (
            "hybrid_ctc_and_external_recovered"
            if any(u.get("alignment_status") == "subinterval_ctc_recovered" for u in units)
            and remaining == 0 else
            "external_recovered" if remaining == 0 else "partial_external_recovery"
        )
    return native
