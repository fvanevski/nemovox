"""Non-destructive plausibility audit for word-level CTC timing.

CTC metadata may be structurally valid while acoustically wrong. This module
never substitutes a guessed timestamp or changes the ASR lexical transcript.
Only independently implausible *local* sequence islands and out-of-excerpt
intervals are quarantined; long pauses alone are not evidence of misalignment.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _mid(unit: dict[str, Any]) -> float | None:
    s, e = unit.get("start"), unit.get("end")
    if not unit.get("valid_timing") or not isinstance(s, (float, int)) or not isinstance(e, (float, int)):
        return None
    if not (math.isfinite(s) and math.isfinite(e)):
        return None
    return (float(s) + float(e)) / 2.0


def audit_ctc_timing_units(
    units: list[dict[str, Any]],
    *,
    infer_start_s: float,
    infer_end_s: float,
    outlier_displacement_s: float = 8.0,
    neighbor_consistency_s: float = 2.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit source-ordered CTC units against their actual inference excerpt.

    A word whose midpoint is far from BOTH same-speaker lexical neighbors
    while those neighbors agree is a local sequence outlier. A word outside
    the audio excerpt cannot have a trustworthy CTC time. In either case,
    preserve its hypothesis in explicit fields but leave its time unknown.

    Two-token jumps, ordinary silence and uncertain interleaved speakers
    remain unmodified, with bounded review-only diagnostics.
    """
    if not (math.isfinite(infer_start_s) and math.isfinite(infer_end_s)
            and infer_end_s > infer_start_s):
        raise ValueError("Invalid source excerpt bounds for CTC timing audit")
    copied = [dict(u) for u in units]
    reasons: dict[int, str] = {}
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, unit in enumerate(copied):
        middle = _mid(unit)
        if middle is None or unit.get("alignment_status") not in ("ctc_aligned", None):
            continue
        start, end = float(unit["start"]), float(unit["end"])
        if start < infer_start_s - 0.15 or end > infer_end_s + 0.15:
            reasons[i] = "ctc_outside_inference_excerpt"
        speaker = unit.get("speaker")
        if speaker is not None:
            by_speaker[str(speaker)].append(i)

    gaps: list[dict[str, Any]] = []
    for speaker, indexes in by_speaker.items():
        for j in range(1, len(indexes)):
            before, current = indexes[j - 1], indexes[j]
            prior_mid, this_mid = _mid(copied[before]), _mid(copied[current])
            if prior_mid is None or this_mid is None:
                continue
            # A long pause is never sufficient to erase a word.
            delta = this_mid - prior_mid
            if abs(delta) >= outlier_displacement_s:
                if len(gaps) < 80:
                    gaps.append({
                        "speaker": speaker, "source_unit_before": before,
                        "source_unit_after": current,
                        "delta_s": round(delta, 3),
                        "reason": ("same_speaker_backtrack" if delta < 0
                                   else "large_same_speaker_gap_review_only"),
                    })
                copied[current]["ctc_sequence_gap_s"] = round(delta, 3)

        for j in range(1, len(indexes) - 1):
            prev_i, this_i, next_i = indexes[j - 1:j + 2]
            prev_mid, this_mid, next_mid = (
                _mid(copied[prev_i]), _mid(copied[this_i]), _mid(copied[next_i]))
            if None in (prev_mid, this_mid, next_mid):
                continue
            assert prev_mid is not None and this_mid is not None and next_mid is not None
            if (abs(next_mid - prev_mid) <= neighbor_consistency_s
                    and (this_mid < min(prev_mid, next_mid) - outlier_displacement_s
                         or this_mid > max(prev_mid, next_mid) + outlier_displacement_s)):
                reasons.setdefault(this_i, "ctc_isolated_sequence_outlier")

    for i, reason in reasons.items():
        unit = copied[i]
        unit["ctc_hypothesis_start_s"] = unit.get("start")
        unit["ctc_hypothesis_end_s"] = unit.get("end")
        unit["timing_evidence_status"] = "unverified_ctc_hypothesis"
        unit["timing_evidence_reason"] = reason
        unit["valid_timing"] = False
        unit["start"] = None
        unit["end"] = None
        unit["alignment_status"] = "ctc_timing_suspect"

    return copied, {
        "schema": "nemovox-ctc-timing-evidence/v1",
        "audited_units": len(copied),
        "quarantined_units": len(reasons),
        "quarantine_reasons": {
            reason: sum(value == reason for value in reasons.values())
            for reason in sorted(set(reasons.values()))
        },
        "large_gaps_review_only": gaps,
        "large_gap_count": sum(
            1 for indexes in by_speaker.values() for j in range(1, len(indexes))
            if (_mid(copied[indexes[j]]) is not None
                and _mid(copied[indexes[j - 1]]) is not None
                and abs(_mid(copied[indexes[j]]) - _mid(copied[indexes[j - 1]]))
                >= outlier_displacement_s)
        ),
        "quarantined_source_indices": sorted(reasons),
        "source_timing_not_fabricated": True,
    }
