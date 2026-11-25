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
    _norm_align_tok,
    group_diarization_windows,
)


@dataclass(frozen=True)
class AdmissionPolicy:
    mode: str = "subtitle"   # subtitle (default) | fine
    max_window_s: float = 20.0
    max_candidates_per_chunk: int = 16
    max_recording_candidates: int = 24
    max_recording_audio_s: float = 240.
    min_subtitle_chars: int = 3
    min_untimed_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in {"subtitle", "fine"}:
            raise ValueError("Unsupported Qwen admission mode")
        if not math.isfinite(self.max_window_s) or not 0.5 <= self.max_window_s <= 180:
            raise ValueError("Invalid max_window_s")
        if not 1 <= self.max_candidates_per_chunk <= 100:
            raise ValueError("Invalid max_candidates_per_chunk")
        if not 1 <= self.max_recording_candidates <= 1000 or not (
                math.isfinite(self.max_recording_audio_s) and self.max_recording_audio_s>0):
            raise ValueError("Invalid recording alignment budget")


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

    spans = []
    for lo,hi in _contiguous_speaker_spans(units):
        start=lo
        for i in range(lo+1,hi):
            prev,cur=units[i-1].get("recognized_turn_id"),units[i].get("recognized_turn_id")
            if prev and cur and prev!=cur:
                spans.append((start,i))
                start=i
        spans.append((start,hi))
    untimed_spans_by_speaker: dict[str, int] = {}
    for s_lo, s_hi in spans:
        s_span = units[s_lo:s_hi]
        if any(not u.get("valid_timing") for u in s_span):
            s_keys = {_speaker_key(u.get("speaker")) for u in s_span}
            if len(s_keys) == 1:
                k = next(iter(s_keys))
                untimed_spans_by_speaker[k] = untimed_spans_by_speaker.get(k, 0) + 1

    for lo, hi in spans:
        span = units[lo:hi]
        untimed = [u for u in span if not u.get("valid_timing")]
        if not untimed:
            continue
        decision: dict[str, Any] = {"span_range": [lo, hi], "untimed_units": len(untimed)}
        decisions.append(decision)
        decision["source_unit_ids"]=[u["source_unit_id"] for u in span if u.get("source_unit_id")]
        if any(u.get("requires_acoustic_recovery") for u in span):
            decision["status"] = "skipped_disputed_occurrence_requires_acoustic_evidence"
            decision["source_unit_ids"] = [u["source_unit_id"] for u in span if u.get("source_unit_id")]
            continue
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
        grouped_diar = group_diarization_windows(
            [{"start": s, "end": e, "valid_timing": True} for s, e in matching_diar],
            max_gap_s=0.6,
            pad_s=0.0,
            max_duration_s=policy.max_window_s,
        )
        interval: tuple[float, float] | None = None
        if valid_in_span:
            native_start = min(s for s, _ in valid_in_span)
            native_end = max(e for _, e in valid_in_span)
            untimed_prefix = not span[0].get("valid_timing")
            untimed_suffix = not span[-1].get("valid_timing")
            if untimed_prefix or untimed_suffix:
                # The submitted text includes the entire utterance: a fixed
                # pad near its native-timed words could omit a long untimed
                # lead-in/tail. Require one uniquely grouped same-speaker
                # diarization interval overlapping native timing.
                if len(grouped_diar) != 1 or untimed_spans_by_speaker.get(speaker, 0) != 1:
                    decision["status"] = "skipped_partial_span_unanchored_edge"
                    continue
                diar_start, diar_end = grouped_diar[0]
                if diar_end < native_start - 0.35 or diar_start > native_end + 0.35:
                    decision["status"] = "skipped_partial_span_unanchored_edge"
                    continue
                interval = (
                    max(chunk_start_s, min(diar_start, native_start) - 0.35)
                    if untimed_prefix else max(chunk_start_s, native_start - 0.35),
                    min(chunk_end_s, max(diar_end, native_end) + 0.35)
                    if untimed_suffix else min(chunk_end_s, native_end + 0.35),
                )
            else:
                # Both boundaries have native timing, so small local pads
                # retain the complete utterance without guessing edge audio.
                interval = (
                    max(chunk_start_s, native_start - 1.2),
                    min(chunk_end_s, native_end + 1.2),
                )
        elif len(grouped_diar) == 1 and untimed_spans_by_speaker.get(speaker, 0) == 1:
            # Intra-utterance pauses (<=0.6s) merge into one speech window for a sole untimed span;
            # multiple separated same-speaker windows remain ambiguous.
            s, e = grouped_diar[0]
            interval = max(chunk_start_s, s - 0.35), min(chunk_end_s, e + 0.35)
        else:
            # An untimed sentence needs *both* neighboring time anchors.
            left_same = next(
                (_finite_interval(u.get("start"), u.get("end")) for u in reversed(units[:lo])
                 if _speaker_key(u.get("speaker")) == speaker and u.get("valid_timing")),
                None,
            )
            right_same = next(
                (_finite_interval(u.get("start"), u.get("end")) for u in units[hi:]
                 if _speaker_key(u.get("speaker")) == speaker and u.get("valid_timing")),
                None,
            )
            left = left_same or next(
                (_finite_interval(u.get("start"), u.get("end")) for u in reversed(units[:lo])
                 if u.get("valid_timing")),
                None,
            )
            right = right_same or next(
                (_finite_interval(u.get("start"), u.get("end")) for u in units[hi:]
                 if u.get("valid_timing")),
                None,
            )
            if left is not None and right is not None and right[0] > left[1]:
                between_diar = group_diarization_windows(
                    [
                        {"start": s, "end": e, "valid_timing": True}
                        for s, e in matching_diar
                        if s >= left[1] - 0.25 and e <= right[0] + 0.25
                    ],
                    max_gap_s=0.6,
                    pad_s=0.0,
                    max_duration_s=policy.max_window_s,
                )
                right_pad = 0.15 if right_same is not None else 0.0
                left_pad = 0.15 if left_same is not None else 0.0
                if len(between_diar) == 1:
                    bs, be = between_diar[0]
                    interval = (
                        max(chunk_start_s, left[1] - left_pad, bs - 0.35),
                        min(chunk_end_s, right[0] + right_pad, be + 0.35),
                    )
                elif left_same is not None and right_same is not None:
                    interval = max(chunk_start_s, left[1] - 0.15), min(chunk_end_s, right[0] + 0.15)
        if interval is None:
            decision["status"] = "skipped_unanchored_or_ambiguous"
            continue
        start, end = interval
        # A single diarization interval can contain multiple utterances, even
        # when only one of them is untimed. Do not align that untimed sentence
        # against already-timed adjacent speech. The closest timed unit in
        # canonical reading order constrains the window regardless of speaker;
        # a later same-speaker unit must not supersede an intervening speaker.
        prior_timed = next(
            (v for u in reversed(units[:lo]) if u.get("valid_timing")
             if (v := _finite_interval(u.get("start"), u.get("end"))) is not None),
            None,
        )
        following_timed = next(
            (v for u in units[hi:] if u.get("valid_timing")
             if (v := _finite_interval(u.get("start"), u.get("end"))) is not None),
            None,
        )
        if prior_timed is not None:
            start = max(start, prior_timed[1] - 0.05)
        if following_timed is not None:
            end = min(end, following_timed[0] + 0.05)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end
                and 0.25 <= end - start <= policy.max_window_s):
            decision["status"] = "skipped_conflicting_or_long_window"
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
    planned=None,
) -> dict[str, Any]:
    """Attach verified external timestamps in place, retaining native provenance."""
    candidates, decisions = planned if planned is not None else plan_selective_recovery(
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
            # Forced alignment is evidence about timing of supplied text,
            # not independent speech recognition. Reject reordering or
            # missing lexical material before attaching any candidate time.
            expected = "".join(_norm_align_tok(u.get("text","")) for u in units[c.lo:c.hi])
            actual = "".join(_norm_align_tok(u.get("text","")) for u in external_units)
            external_units = [{**u,"valid_timing":bool(u.get("valid_timing")
                and _finite_interval(u.get("start"),u.get("end")) is not None)}
                for u in external_units]
            decision["invalid_external_timing_units"]=sum(not u["valid_timing"] for u in external_units)
            valid = [v for u in external_units if u.get("valid_timing")
                     if (v := _finite_interval(u.get("start"),u.get("end"))) is not None]
            if not expected or expected != actual:
                decision["status"] = "attempted_no_verified_match"
                decision["rejection_reason"] = "forced_alignment_lexical_order_mismatch"
                continue
            if not valid or any(b[0] < a[0] for a,b in zip(valid,valid[1:])):
                decision["status"] = "attempted_no_verified_match"
                decision["rejection_reason"] = "forced_alignment_invalid_or_reversed_times"
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
