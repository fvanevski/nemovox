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
PRIORITIES = {"generation_incomplete": 0, "acoustic_reference_omission": 1, "independent_coverage_probe": 2, "activity_without_timed_source_hypothesis": 3}
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
    coverage_floor: Mapping[str, Any] | None = None,
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
        digest = gap.get("raw_response_sha256")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest.lower())
                or gap.get("reason") not in {"generation_limit_exhausted", "pathological_generation_loop"}):
            raise ValueError("Incomplete chunk needs a SHA-256 digest and integrity failure reason")
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
    # Optional independent challenger sweep over source media. This closes
    # the blind-spot class where ALL available VAD/diar detectors missed a
    # short real phrase (as observed for W04c and the H2 opening). It is
    # intentionally opt-in and fully budget-accounted, not a secret default.
    if coverage_floor is not None:
        a, b = _interval(coverage_floor)
        raw_proposals.append({
            "start_s": a, "end_s": b,
            "reason": "independent_coverage_probe",
            "evidence_ids": [_id("coverage", [media, a, b])],
            "independently_auditioned": False,
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

    # Coalesce overlapping *model activity proposals* by reason, independent
    # of other interleaved proposal categories. This merges CPU/GPU detector
    # intervals, NOT distinct acoustic speech events or speaker turns.
    # The full coverage floor retains its own request budget protection.
    groups: list[dict[str, Any]] = []
    for reason in sorted(PRIORITIES, key=PRIORITIES.get):
        within_reason: list[dict[str, Any]] = []
        relevant = sorted((p for p in raw_proposals if p["reason"] == reason),
                          key=lambda p: (p["start_s"], p["end_s"], tuple(p["evidence_ids"])))
        for p in relevant:
            can_coalesce = reason in {
                "generation_incomplete", "activity_without_timed_source_hypothesis"
            }
            if (can_coalesce and within_reason
                    and p["start_s"] < within_reason[-1]["end_s"]):
                previous = within_reason[-1]
                previous["end_s"] = max(previous["end_s"], p["end_s"])
                previous["evidence_ids"].extend(p["evidence_ids"])
                continue
            within_reason.append({
                "start_s": p["start_s"], "end_s": p["end_s"],
                "reasons": [reason], "evidence_ids": list(p["evidence_ids"]),
                "independently_auditioned": p["independently_auditioned"],
            })
        groups.extend(within_reason)

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
    # The independent source-audio floor is explicitly protected against
    # activity-model fan-out. It is scheduled after verified failures/heard
    # misses, but BEFORE unverified detector-triggered proposals. If the
    # budget cannot cover the floor, every unserved window stays visible.
    requests.sort(key=lambda x: (PRIORITIES[x["reason"]], x["start_s"], x["end_s"], x["id"]))
    selected = requests[:max_requests]
    pending = requests[max_requests:]
    total_floor = sum(r["reason"] == "independent_coverage_probe" for r in requests)
    covered_floor = sum(r["reason"] == "independent_coverage_probe" for r in selected)
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
            "coverage_floor_requests": total_floor,
            "coverage_floor_selected": covered_floor,
            "coverage_floor_pending": total_floor - covered_floor,
            "activity_requests": sum(r["reason"] == "activity_without_timed_source_hypothesis" for r in requests),
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
        a, b = unit.get("start_s"), unit.get("end_s")
        if (any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) for x in (a, b))
                or not (0 <= a <= b)):
            errors.append("invalid_word_interval")
            continue
        a, b = float(a), float(b)
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
        "source_revision_id": _id("src_revision", [ledger["media_fingerprint_sha256"], audio_event_id,
                                           chosen_observation_id, candidate_source_text, selection_basis,
                                           source_start_s, source_end_s]),
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


def project_source_turn_to_v1_utterance(revision: Mapping[str, Any]) -> dict[str, Any]:
    """Explicit compatibility adapter: source-event lineage survives v1 MT/render.

    Only a projection for known selected candidates. Does NOT certify
    speaker identity, missing timestamps, or the recognized source words.
    """
    if revision.get("schema") != "nemovox-source-turn-revision/v1":
        raise ValueError("Incorrect source revision schema")
    if not revision.get("source_turn_id") or not revision.get("source_revision_id"):
        raise ValueError("Source-turn identity and immutable revision identity required")
    if not isinstance(revision.get("source_text"), str) or not revision["source_text"].strip():
        raise ValueError("Cannot project an empty source transcript")
    a, b = revision.get("source_start"), revision.get("source_end")
    if (a is None) != (b is None):
        raise ValueError("Cannot project partial unknown source timing")
    if a is not None:
        _interval({"start_s": a, "end_s": b})
    return {
        "id": revision["source_turn_id"],
        "utterance_id": revision["source_turn_id"],
        "source_turn_id": revision["source_turn_id"],
        "source_turn_ids": [revision["source_turn_id"]],
        "source_revision_id": revision["source_revision_id"],
        "source_language": revision.get("source_language"),
        "source_text": revision["source_text"],
        "source_start": a,
        "source_end": b,
        "speaker_id": "SPK_UNK",
        "alignment_status": "unverified",
        "timestamp_provenance": {"source_revision_id": revision["source_revision_id"],
                                 "status": "provisional_not_acoustic_truth"},
        "source_acceptance_status": revision.get("source_status"),
        "source_observation_ids": list(revision.get("source_observation_ids") or []),
        "audio_event_id": revision.get("audio_event_id"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evidence", required=True, type=Path)
    parser.add_argument("--activity-evidence", type=Path)
    parser.add_argument("--incomplete", type=Path, help="Optional JSON array of rejected source chunks")
    parser.add_argument("--auditioned-misses", type=Path, help="Optional JSON array of independently audited missing events")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--coverage-end-s", type=float,
                        help="Opt-in independent whole-recording Qwen coverage sweep end in seconds")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; refuse overwrite")
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    plan = plan_source_recovery(
        load(args.source_evidence),
        activity=load(args.activity_evidence) if args.activity_evidence else None,
        incomplete=load(args.incomplete) if args.incomplete else (),
        independently_auditioned_misses=load(args.auditioned_misses) if args.auditioned_misses else (),
        coverage_floor=({"start_s": 0, "end_s": args.coverage_end_s}
                        if args.coverage_end_s is not None else None),
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
