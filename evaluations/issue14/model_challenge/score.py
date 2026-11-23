"""Deterministic, source-reference-bound model challenge scoring for Nemovox #14.

This evaluator never uses model agreement as acoustic ground truth. Labels must
originate from independently auditioned audio; scoring is a bounded event-level
proxy, not full-episode WER/CER or diarization error rate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Any

REF_SCHEMA = "nemovox-issue14-acoustic-reference/v1"
PRED_SCHEMA = "nemovox-issue14-model-predictions/v1"
RESULT_SCHEMA = "nemovox-issue14-model-challenge-results/v1"
VOCAL_KINDS = {"speech", "singing", "lexical_shout"}
NON_VOCAL_KINDS = {"music", "sfx", "silence"}
KINDS = VOCAL_KINDS | NON_VOCAL_KINDS | {"nonlexical_vocalization", "unknown"}


def _bounds(row: dict[str, Any], lo: float, hi: float) -> tuple[float, float]:
    a, b = row.get("start_s"), row.get("end_s")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (a, b)):
        raise ValueError("Non-finite or missing event time")
    a, b = float(a), float(b)
    if not (lo <= a < b <= hi):
        raise ValueError(f"Event interval [{a},{b}] outside region [{lo},{hi}]")
    return a, b


def _merge(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for a, b in sorted(intervals):
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(b, merged[-1][1])
    return [(a, b) for a, b in merged]


def _length(intervals: list[tuple[float, float]]) -> float:
    return sum(b - a for a, b in _merge(intervals))


def _intersection(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    left, right = _merge(a), _merge(b)
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _subtract(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for lo, hi in _merge(a):
        cursor = lo
        for b_lo, b_hi in _merge(b):
            if b_hi <= cursor or b_lo >= hi:
                continue
            if b_lo > cursor:
                result.append((cursor, min(b_lo, hi)))
            cursor = max(cursor, b_hi)
            if cursor >= hi:
                break
        if cursor < hi:
            result.append((cursor, hi))
    return result


def _edit_distance(left: list[str], right: list[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    prev = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        now = [i]
        for j, other in enumerate(right, 1):
            now.append(min(now[-1] + 1, prev[j] + 1,
                           prev[j - 1] + int(token != other)))
        prev = now
    return prev[-1]


def _lexical_units(text: str, language: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text.casefold())
    if language == "zh":
        return [ch for ch in normalized if unicodedata.category(ch)[0] in {"L", "N"}]
    cleaned = "".join(ch if (unicodedata.category(ch)[0] in {"L", "N"} or ch == "'")
                      else " " for ch in normalized)
    return cleaned.split()


def _normalized_regions(document: dict[str, Any], *, prediction: bool) -> dict[str, dict[str, Any]]:
    expected = PRED_SCHEMA if prediction else REF_SCHEMA
    if document.get("schema") != expected:
        raise ValueError(f"Expected schema {expected!r}")
    regions = document.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("Document must contain a nonempty regions array")
    indexed: dict[str, dict[str, Any]] = {}
    for region in regions:
        region_id = region.get("id")
        if not isinstance(region_id, str) or not region_id or region_id in indexed:
            raise ValueError("Missing/duplicate region ID")
        indexed[region_id] = region
    return indexed


def score(reference: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    """Score available independently heard reference events with explicit gaps."""
    ref = _normalized_regions(reference, prediction=False)
    pred = _normalized_regions(prediction, prediction=True)
    if not reference.get("media_fingerprint_sha256") or (
        reference.get("media_fingerprint_sha256") != prediction.get("media_fingerprint_sha256")
    ):
        raise ValueError("Media fingerprint mismatch or omission")
    model = prediction.get("model")
    if not isinstance(model, dict) or not model.get("repo_id") or not model.get("revision"):
        raise ValueError("Missing exact model ID/revision")
    if set(pred) != set(ref):
        raise ValueError("Prediction must explicitly include each reference region, even skipped runs")
    rows: list[dict[str, Any]] = []
    scored = []
    for rid, region in ref.items():
        trial = pred[rid]
        lo, hi = _bounds({"start_s": region.get("start_s"), "end_s": region.get("end_s")},
                         0.0, float("inf"))
        trial_status = trial.get("status")
        if trial_status not in {"ok", "skipped_unsupported", "blocked", "failed", "not_run"}:
            raise ValueError(f"Invalid trial status for {rid}")
        if trial_status != "ok":
            if trial.get("observations"):
                raise ValueError("Skipped/failed region must not contain predictions")
            rows.append({"id": rid, "status": trial_status, "reason": trial.get("reason") or "not_provided"})
            continue
        if region.get("acoustic_truth") != "independently_auditioned":
            rows.append({"id": rid, "status": "reference_unverified"})
            continue
        events = region.get("events")
        observations = trial.get("observations")
        if not isinstance(events, list) or not isinstance(observations, list):
            raise ValueError("Reference events and predictions must be arrays")
        r: dict[str, list[tuple[float, float]]] = {k: [] for k in KINDS}
        q: dict[str, list[tuple[float, float]]] = {k: [] for k in KINDS}
        for row, buckets in [(x, r) for x in events] + [(x, q) for x in observations]:
            kind = row.get("kind")
            if kind not in KINDS:
                raise ValueError(f"Unknown event class {kind!r} in {rid}")
            buckets[kind].append(_bounds(row, lo, hi))
        reference_speech = _merge(r["speech"] + r["lexical_shout"])
        predicted_speech = _merge(q["speech"] + q["lexical_shout"])
        non_speech_only = _subtract(r["sfx"] + r["music"] + r["silence"],
                                    reference_speech + r["singing"])
        # This is temporal overlap, not proof that the model recognized words.
        speech_hits = 0
        per_event_overlap: list[float] = []
        for a, b in r["speech"] + r["lexical_shout"]:
            overlap = _intersection([(a, b)], predicted_speech)
            per_event_overlap.append(round(overlap / (b-a), 6))
            # Fixed declared proxy: any positive overlap, not a semantic match.
            speech_hits += int(overlap > 0)
        record = {
            "id": rid,
            "status": "scored_acoustic_event_proxy",
            "reference_speech_events": len(per_event_overlap),
            "event_hit_any_overlap": speech_hits,
            "event_coverage_fractions": per_event_overlap,
            "reference_speech_s": round(_length(reference_speech), 4),
            "predicted_speech_s": round(_length(predicted_speech), 4),
            "reference_speech_covered_s": round(_intersection(reference_speech, predicted_speech), 4),
            "non_speech_only_predicted_speech_s": round(_intersection(non_speech_only, predicted_speech), 4),
            "reference_singing_s": round(_length(r["singing"]), 4),
            "predicted_singing_s": round(_length(q["singing"]), 4),
            "speaker_identity_validated": False,
            "source_word_accuracy_validated": False,
            "word_alignment_validated": False,
        }
        # Exact lexical accuracy is evaluated only against explicitly verified
        # private source text. A VAD/diarization output without text has no WER.
        if region.get("source_text_verified") is True:
            gold = region.get("source_text")
            hypothesis = trial.get("source_text")
            language = region.get("language")
            if not isinstance(gold, str) or not isinstance(hypothesis, str) or language not in {"fr", "zh"}:
                raise ValueError(f"Missing exact private source text/model transcript/language in {rid}")
            target = _lexical_units(gold, language)
            emitted = _lexical_units(hypothesis, language)
            record["lexical_metric"] = "CER" if language == "zh" else "WER"
            record["lexical_reference_units"] = len(target)
            record["lexical_edit_count"] = _edit_distance(target, emitted)
            record["lexical_error_ratio"] = (
                round(record["lexical_edit_count"] / len(target), 6) if target else None
            )
            record["source_word_accuracy_validated"] = True
        scored.append(record)
        rows.append(record)
    return {
        "schema": RESULT_SCHEMA,
        "model": model,
        "reference_fingerprint": reference["media_fingerprint_sha256"],
        "reference_sha256": hashlib.sha256(
            json.dumps(reference, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "scored_regions": len(scored),
        "unscored_regions": len(rows) - len(scored),
        "summary": {
            "reference_speech_events": sum(x["reference_speech_events"] for x in scored),
            "event_hit_any_overlap": sum(x["event_hit_any_overlap"] for x in scored),
            "reference_speech_s": round(sum(x["reference_speech_s"] for x in scored), 4),
            "reference_speech_covered_s": round(sum(x["reference_speech_covered_s"] for x in scored), 4),
            "non_speech_only_predicted_speech_s": round(
                sum(x["non_speech_only_predicted_speech_s"] for x in scored), 4
            ),
        },
        "lexical_accuracy": {
            metric: {
                "reference_units": sum(
                    r.get("lexical_reference_units", 0) for r in scored if r.get("lexical_metric") == metric
                ),
                "edit_count": sum(
                    r.get("lexical_edit_count", 0) for r in scored if r.get("lexical_metric") == metric
                ),
            }
            for metric in ("WER", "CER")
        },
        "regions": rows,
        "limitations": [
            "Not speech recognition accuracy or speaker DER",
            "Model predictions are never audio ground truth",
            "Only independently auditioned regions receive event-level scores",
            "Any model absent from a region is explicit, never counted as silence",
            "Transcript CER/WER and word timing require separate exact private source annotations",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("Output exists; use a new path")
    ref = json.loads(args.reference.read_text(encoding="utf-8"))
    pred = json.loads(args.prediction.read_text(encoding="utf-8"))
    result = score(ref, pred)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": result["model"]["repo_id"], "output": str(output),
                      "summary": result["summary"], "scored_regions": result["scored_regions"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
