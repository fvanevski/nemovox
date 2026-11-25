"""Pre-selection occurrence spans and explicit, provisional locality evidence.

Recognized order is not a global acoustic chronology (especially for SOT).
Long pauses partition candidate spans; they never erase speech. Numeric
conflicts are retained separately from usable timing and cannot admit an
independent forced alignment of the disputed wording.
"""
from __future__ import annotations

from collections import defaultdict
import math

from nvidia_transcribe.source_occurrences import stable_id
from nvidia_transcribe.timing_evidence import audit_ctc_timing_units


def _bounds(unit, hypothesis=False):
    if hypothesis and unit.get("ctc_hypothesis_start_s") is not None:
        a, b = unit.get("ctc_hypothesis_start_s"), unit.get("ctc_hypothesis_end_s")
    else:
        if not unit.get("valid_timing"):
            return None
        a, b = unit.get("start"), unit.get("end")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in (a, b)) or a > b:
        return None
    return float(a), float(b)


def assess_occurrence_locality(units, *, infer_start_s, infer_end_s,
                               pause_split_s=0.85, review_gap_s=8.0):
    """Return audited units and immutable-reference candidate span inventory.

    Span identity derives from observed lexical units, not selected time,
    speaker reconciliation or output position. Repartitioning can create new
    candidate spans; the original unit/turn identities remain authoritative.
    """
    if not (0 < pause_split_s <= review_gap_s):
        raise ValueError("Invalid occurrence locality policy")
    audited, audit = audit_ctc_timing_units(
        units, infer_start_s=infer_start_s, infer_end_s=infer_end_s,
        outlier_displacement_s=review_gap_s,
    )
    by_turn = defaultdict(list)
    for u in audited:
        if not u.get("source_unit_id") or not u.get("recognized_turn_id"):
            raise ValueError("Occurrence locality requires bound source units")
        by_turn[u["recognized_turn_id"]].append(u)
    turns = sorted(by_turn.values(), key=lambda g: min(u["source_unit_index"] for u in g))
    spans, findings = [], []
    prior_turn = None
    for turn in turns:
        turn.sort(key=lambda u: u["source_unit_index"])
        group = []
        groups = []
        for u in turn:
            bounds = _bounds(u, hypothesis=True)
            prev = group[-1] if group else None
            before = _bounds(prev, hypothesis=True) if prev else None
            reasons = []
            if prev and before and bounds:
                gap = bounds[0] - before[1]
                if gap >= review_gap_s:
                    reasons.append("large_intra_turn_gap_review_only")
                if bounds[0] < before[0] - 0.2:
                    reasons.append("intra_turn_order_conflict")
                    # Neither of two reversed positions is established as correct.
                    # Preserve both candidate times without selecting a winner.
                    for item in (prev, u):
                        if item.get("valid_timing"):
                            item["ctc_hypothesis_start_s"] = item["start"]
                            item["ctc_hypothesis_end_s"] = item["end"]
                            item["start"] = item["end"] = None
                            item["valid_timing"] = False
                            item["alignment_status"] = "ctc_timing_suspect"
                            item["timing_evidence_status"] = "unverified_ctc_hypothesis"
                            item["timing_evidence_reason"] = "intra_turn_order_conflict"
                for reason in reasons:
                    findings.append({
                        "id": stable_id("locality", [reason, prev["source_unit_id"], u["source_unit_id"]]),
                        "reason": reason, "source_unit_ids": [prev["source_unit_id"], u["source_unit_id"]],
                        "candidate_windows_s": [list(before), list(bounds)],
                        "acoustic_truth_accepted": False,
                    })
                if gap > pause_split_s or "intra_turn_order_conflict" in reasons:
                    groups.append(group)
                    group = []
            elif prev and bool(before) != bool(bounds):
                # Do not assign a time to an untimed segment from its neighbor.
                groups.append(group)
                group = []
            group.append(u)
        if group:
            groups.append(group)
        observed = [_bounds(u, hypothesis=True) for u in turn]
        observed = [b for b in observed if b is not None]
        if observed:
            turn_bounds = (min(a for a, _ in observed), max(b for _, b in observed))
            if prior_turn is not None and turn_bounds[1] < prior_turn[1][0] - review_gap_s:
                # SOT may legitimately emit nonchronological turns. This is a
                # candidate order conflict, NOT a reason to erase usable timing.
                prior, previous_bounds = prior_turn
                refs = [prior[0]["source_unit_id"], turn[0]["source_unit_id"]]
                findings.append({
                    "id": stable_id("locality", ["nonchronological_turn_hypothesis", refs]),
                    "reason": "nonchronological_turn_hypothesis",
                    "source_unit_ids": refs,
                    "candidate_windows_s": [list(previous_bounds), list(turn_bounds)],
                    "acoustic_truth_accepted": False,
                })
            prior_turn = (turn, turn_bounds)
        else:
            prior_turn = None
        for group in groups:
            refs = [u["source_unit_id"] for u in group]
            span_id = stable_id("span", refs)
            usable = [b for u in group if (b := _bounds(u)) is not None]
            hypotheses = [b for u in group if (b := _bounds(u, hypothesis=True)) is not None]
            reasons = sorted({u["timing_evidence_reason"] for u in group
                              if u.get("timing_evidence_reason")})
            disputed = any(u.get("timing_evidence_status") == "unverified_ctc_hypothesis"
                           for u in group)
            status = ("disputed" if disputed else "provisional" if len(usable) == len(group)
                      else "partial" if usable else "unlocalized")
            usable_bounds = ([min(a for a, _ in usable), max(b for _, b in usable)]
                             if usable else None)
            candidate_bounds = ([min(a for a, _ in hypotheses), max(b for _, b in hypotheses)]
                                if hypotheses else None)
            for u in group:
                u["occurrence_span_id"] = span_id
                u["locality_status"] = status
                u["requires_acoustic_recovery"] = disputed
                u["occurrence_bounds_s"] = usable_bounds if status == "provisional" else None
            spans.append({
                "id": span_id, "recognized_turn_id": group[0]["recognized_turn_id"],
                "source_unit_ids": refs, "locality_status": status,
                "candidate_bounds_s": candidate_bounds,
                "usable_bounds_s": usable_bounds if status == "provisional" else None,
                "reasons": reasons, "acoustic_truth_accepted": False,
            })
    return audited, {"schema": "nemovox-occurrence-locality/v1", "spans": spans,
                     "findings": findings, "timing_audit": audit,
                     "acoustic_truth_accepted": False}
