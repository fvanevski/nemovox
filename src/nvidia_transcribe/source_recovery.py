"""Evidence-first bounded source recovery, independent of model runtime.

This planner consumes the immutable source-evidence/v1 observation sidecar,
optional independently acquired acoustic observations and explicit ASR
generation gaps. It never treats VAD silence as a no-speech conclusion,
monotonic alignment as proof of words, or lexical similarity as proof of
one acoustic event. No model is launched by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "nemovox-source-recovery-plan/v1"
ACTIVITY_SCHEMA = "nemovox-audio-activity-evidence/v1"
MAX_WINDOW_S = 24.0
OVERLAP_S = 2.0
KNOWN_ACTIVITY = frozenset({"speech", "lexical_shout", "singing", "music", "sfx", "nonlexical_vocalization", "unknown"})
PRIORITIES = {"generation_incomplete": 0, "acoustic_reference_omission": 1, "activity_without_timed_source_hypothesis": 2}
LANGUAGES = {"fr": "French", "zh": "Chinese"}


def _id(kind: str, payload: Any) -> str:
    value = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return kind + "_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _interval(item: Mapping[str, Any], *, start: str = "start_s", end: str = "end_s") -> tuple[float, float]:
    a, b = item.get(start), item.get(end)
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in (a, b)):
        raise ValueError("Invalid interval: missing/nonfinite time")
    if not (0 <= a < b):
        raise ValueError("Invalid interval: end must exceed start")
    return float(a), float(b)


def _tile(a: float, b: float, *, window: float) -> list[tuple[float, float]]:
    """Overlapping bounded inference windows, without silently dropping tails."""
    if window <= OVERLAP_S:
        raise ValueError("Window must exceed overlap")
    result = []
    cursor = a
    while cursor < b - 1e-6:
        stop = min(b, cursor + window)
        result.append((round(cursor, 6), round(stop, 6)))
        if stop >= b - 1e-6:
            break
        cursor = stop - OVERLAP_S
    return result


def _overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def plan_source_recovery(
    ledger: Mapping[str, Any],
    *,
    activity: Mapping[str, Any] | None = None,
    incomplete: Sequence[Mapping[str, Any]] = (),
    independently_auditioned_misses: Sequence[Mapping[str, Any]] = (),
    window_s: float = MAX_WINDOW_S,
    max_requests: int = 32,
) -> dict[str, Any]:
    """Plan recoverable audio intervals, preserving unserved work and source IDs.

    Activity is a *soft* proposal source; even music-only observations do not
    veto independently heard speech. Missing CTC time never proves no words.
    Generation gaps always get full bounded audio coverage independent of VAD.
    """
    if ledger.get("schema_version") != "nemovox-source-evidence/v1":
        raise ValueError("Expected nemovox-source-evidence/v1")
    media = ledger.get("media_fingerprint_sha256")
    lang = ledger.get("source_language")
    if not isinstance(media, str) or not media:
        raise ValueError("Missing source media fingerprint")
    if not isinstance(window_s, (int, float)) or not math.isfinite(window_s) or not (OVERLAP_S < window_s <= MAX_WINDOW_S):
        raise ValueError("Invalid bounded ASR inference duration")
    if not isinstance(max_requests, int) or isinstance(max_requests, bool) or max_requests < 1:
        raise ValueError("max_requests must be positive")
    if activity is not None and (
        activity.get("schema") != ACTIVITY_SCHEMA
        or activity.get("media_fingerprint_sha256") != media
    ):
        raise ValueError("Acoustic evidence schema/media fingerprint mismatch")
    source_ids = {a["id"] for a in ledger.get("asr_observations", [])}
    timed_units = []
    for unit in ledger.get("lexical_observations", []):
        if unit.get("asr_observation_id") not in source_ids:
            raise ValueError("Lexical observation lacks source ASR identity")
        a, b = unit.get("ctc_start_s"), unit.get("ctc_end_s")
        if unit.get("timing_status") in {"provisional_ctc", "provisional_sequence_gap"} and a is not None and b is not None:
            if not (isinstance(a, (int, float)) and isinstance(b, (int, float)) and
                    math.isfinite(a) and math.isfinite(b) and 0 <= a <= b):
                raise ValueError("Invalid CTC hypothesis interval")
            timed_units.append((float(a), float(b)))

    observations = []
    for a in ledger.get("asr_observations", []):
        observations.append({"id": a["id"], "bounds": _interval(a, start="infer_start_s", end="infer_end_s")})
    raw_proposals: list[dict[str, Any]] = []
    for gap in incomplete:
        a, b = _interval(gap, start="infer_start_s", end="infer_end_s")
        if not gap.get("raw_response_sha256") or gap.get("reason") != "generation_limit_exhausted" and gap.get("reason") != "pathological_generation_loop":
            raise ValueError("Incomplete chunk needs authenticated raw SHA and integrity failure reason")
        raw_proposals.append({
            "start_s": a, "end_s": b, "reason": "generation_incomplete",
            "evidence_ids": [str(gap["raw_response_sha256"])],
            "independently_auditioned": False,
        })
    for event in independently_auditioned_misses:
        if event.get("acoustic_truth") != "independently_auditioned" or not event.get("id"):
            raise ValueError("Manual omission requires an independently auditioned event ID")
        a, b = _interval(event)
        raw_proposals.append({
            "start_s": a, "end_s": b, "reason": "acoustic_reference_omission",
            "evidence_ids": [str(event["id"])],
            "independently_auditioned": True,
        })
    for event in (activity or {}).get("intervals", []):
        a, b = _interval(event)
        if event.get("kind") not in KNOWN_ACTIVITY or not event.get("id") or not event.get("source"):
            raise ValueError("Activity events require kind, evidence ID and model source")
        if event.get("kind") not in {"speech", "lexical_shout"}:
            continue
        # Uncovered is only shorthand for no overlapping *CTC hypothesis*.
        # An untimed model word could still correspond to this acoustic event.
        if any(_overlap(a, b, u0, u1) > 0 for u0, u1 in timed_units):
            continue
        raw_proposals.append({
            "start_s": a, "end_s": b,
            "reason": "activity_without_timed_source_hypothesis",
            "evidence_ids": [str(event["id"])],
            "independently_auditioned": False,
        })

    # Coalesce overlapping proposals without losing their original evidence
    # references or forcing the overlapping turns to be one source event.
    groups: list[dict[str, Any]] = []
    for p in sorted(raw_proposals, key=lambda r: (r["start_s"], r["end_s"], r["reason"])):
        if groups and p["start_s"] < groups[-1]["end_s"] and (
            p["reason"] == groups[-1]["reasons"][0] == "generation_incomplete"
        ):
            groups[-1]["end_s"] = max(groups[-1]["end_s"], p["end_s"])
            groups[-1]["evidence_ids"].extend(p["evidence_ids"])
        else:
            groups.append({
                "start_s": p["start_s"], "end_s": p["end_s"],
                "reasons": [p["reason"]], "evidence_ids": list(p["evidence_ids"]),
                "independently_auditioned": p["independently_auditioned"],
            })

    requests = []
    for group in groups:
        a, b = group["start_s"], group["end_s"]
        for t0, t1 in _tile(a, b, window=float(window_s)):
            ids = sorted(set(group["evidence_ids"]))
            requests.append({
                "id": _id("rec", [media, t0, t1, ids]),
                "start_s": t0, "end_s": t1,
                "region_id": _id("region", [media, a, b, ids]),
                "reason": group["reasons"][0],
                "trigger_evidence_ids": ids,
                "observed_asr_ids": [
                    x["id"] for x in observations
                    if _overlap(t0, t1, *x["bounds"]) > 0
                ],
                "language": lang,
                "challenger_language": LANGUAGES.get(lang),
                "primary_retry_required": group["reasons"][0] == "generation_incomplete",
                "state": "candidate_only_no_source_content_accepted",
                "source_content_accepted": False,
            })
    # Give source-generation failures and independent audio labels priority.
    requests.sort(key=lambda x: (PRIORITIES[x["reason"]], x["start_s"], x["end_s"], x["id"]))
    selected = requests[:max_requests]
    pending = requests[max_requests:]
    return {
        "schema": SCHEMA,
        "media_fingerprint_sha256": media,
        "source_language": lang,
        "policy": {
            "window_s": window_s, "overlap_s": OVERLAP_S,
            "max_requests": max_requests, "no_model_invocation": True,
            "vad_negative_is_not_audio_exclusion": True,
        },
        "requests": selected,
        "unplanned_due_to_budget": pending,
        "status": "partial_budget" if pending else "complete_proposals",
        "diagnostics": {
            "proposal_regions": len(groups),
            "total_requests": len(requests),
            "selected_requests": len(selected),
            "pending_requests": len(pending),
            "possible_content_omission_not_proven_by_ctc": True,
            "source_fidelity_certified": False,
        },
    }


def inspect_forced_alignment(
    words: Sequence[Mapping[str, Any]],
    *,
    source_region: Mapping[str, Any],
    acoustic_pauses: Sequence[Mapping[str, Any]] = (),
    max_zero_ratio: float = 0.4,
    max_word_duration_s: float = 1.2,
) -> dict[str, Any]:
    """Structural rejection. PASS does NOT verify the model's source text."""
    lo, hi = _interval(source_region)
    if not words:
        return {"status": "rejected", "reasons": ["no_words"], "text_verified": False}
    errors: list[str] = []
    zero = 0
    prev_start = lo
    for unit in words:
        a, b = _interval(unit)
        if a < lo - .05 or b > hi + .05:
            errors.append("word_outside_source_region")
        if a < prev_start - .001:
            errors.append("non_monotonic_word_order")
        if b-a <= .001:
            zero += 1
        if b-a > max_word_duration_s:
            errors.append("word_duration_suspect")
        for pause in acoustic_pauses:
            p0, p1 = _interval(pause)
            if a < p0 and b > p1:
                errors.append("crosses_audited_pause")
        prev_start = a
    ratio = zero / len(words)
    if ratio >= max_zero_ratio:
        errors.append("excessive_zero_duration")
    return {
        "status": "rejected" if errors else "structurally_provisional",
        "reasons": sorted(set(errors)),
        "zero_duration_ratio": round(ratio, 6),
        "text_verified": False,
        "timestamp_is_acoustic_truth": False,
    }


def assemble_source_turn_revision(
    ledger: Mapping[str, Any],
    *,
    audio_event_id: str,
    chosen_observation_id: str,
    candidate_source_text: str,
    selection_basis: str,
    source_start_s: float | None = None,
    source_end_s: float | None = None,
) -> dict[str, Any]:
    """Versioned lineage for a proposed source turn, not automatic acceptance."""
    if ledger.get("schema_version") != "nemovox-source-evidence/v1":
        raise ValueError("Incorrect source ledger schema")
    if not audio_event_id or not candidate_source_text.strip():
        raise ValueError("Audio event ID and nonempty source text required")
    if selection_basis not in {"independent_audio_review", "multi_asr_provisional", "legacy_provisional"}:
        raise ValueError("Unknown source-selection evidence class")
    source = {r["id"] for r in ledger.get("asr_observations", [])}
    if chosen_observation_id not in source:
        raise ValueError("Chosen source observation not present in immutable ledger")
    if (source_start_s is None) != (source_end_s is None):
        raise ValueError("Source bounds must be both present or both absent")
    if source_start_s is not None:
        _interval({"start_s": source_start_s, "end_s": source_end_s})
    return {
        "schema": "nemovox-source-turn-revision/v1",
        "source_turn_id": _id("src_turn", [ledger["media_fingerprint_sha256"], audio_event_id]),
        "audio_event_id": audio_event_id,
        "source_language": ledger.get("source_language"),
        "source_text": candidate_source_text,
        "source_observation_ids": [chosen_observation_id],
        "selection_basis": selection_basis,
        "source_status": (
            "reviewed_source_candidate" if selection_basis == "independent_audio_review"
            else "model_source_candidate_provisional"
        ),
        "source_start": source_start_s,
        "source_end": source_end_s,
        "word_alignment_status": "unverified",
        "speaker_identity_status": "unverified",
        "not_acoustic_truth_certificate": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evidence", required=True, type=Path)
    parser.add_argument("--activity-evidence", type=Path)
    parser.add_argument("--incomplete", type=Path, help="Optional JSON array of rejected source chunks")
    parser.add_argument("--auditioned-misses", type=Path, help="Optional JSON array of independently audited missing events")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-requests", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; refuse overwrite")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    plan = plan_source_recovery(
        load(args.source_evidence),
        activity=load(args.activity_evidence) if args.activity_evidence else None,
        incomplete=load(args.incomplete) if args.incomplete else (),
        independently_auditioned_misses=load(args.auditioned_misses) if args.auditioned_misses else (),
        max_requests=args.max_requests,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"status": plan["status"], **plan["diagnostics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
