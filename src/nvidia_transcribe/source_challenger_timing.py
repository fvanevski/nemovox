"""Immutable cross-ASR timing comparison, not transcription or timing authority.

Designed to compare the source-evidence/v1 raw chunk lexical observations
with a bounded, independently generated Whisper word-timestamp experiment.
Output is *review candidates only*. No original source unit is changed and
no speaker or acoustic timing certainty is inferred from model agreement.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from statistics import median
import math
import unicodedata
from typing import Any

SCHEMA = "nemovox-independent-timing-comparison/v1"


def _token(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Source/challenger word must be a string")
    folded = unicodedata.normalize("NFKD", value.casefold())
    return "".join(c for c in folded if c.isalnum() and not unicodedata.combining(c))


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"Invalid {name}")
    return float(value)


def compare_independent_word_timing(
    source_evidence: dict[str, Any],
    frozen_windows: dict[str, Any],
    challenger: dict[str, Any],
    *,
    model_key: str = "whisper_large_v3",
    minimum_matched_words: int = 4,
    review_shift_s: float = 3.0,
) -> dict[str, Any]:
    """Produce source-unit-backed drift hypotheses without accepting new times.

    All identities are checked against their corresponding model and bounded
    crop. Matching compares raw lexical order, not (possibly displaced) CTC
    times or inferred speaker identity. Ambiguous repeated n-grams cannot
    establish unique anchors. Outputs must never be applied automatically.
    """
    if source_evidence.get("schema_version") != "nemovox-source-evidence/v1":
        raise ValueError("Unsupported immutable source evidence schema")
    media = source_evidence.get("media_fingerprint_sha256")
    language = source_evidence.get("source_language")
    if not isinstance(media, str) or not media or language not in {"fr", "zh"}:
        raise ValueError("Missing media fingerprint or source language")
    if minimum_matched_words < 4 or not math.isfinite(review_shift_s) or review_shift_s < 0:
        raise ValueError("Unsafe comparison thresholds")
    model = challenger.get(model_key)
    if not isinstance(model, dict) or not model.get("repo_id") or not model.get("revision"):
        raise ValueError("Missing pinned independent ASR model identity")
    if not isinstance(model.get("cases"), dict):
        raise ValueError("Missing independent ASR case records")
    asrs: dict[int, dict[str, Any]] = {}
    for obs in source_evidence.get("asr_observations", []):
        idx = obs.get("chunk_index")
        if not isinstance(idx, int) or idx in asrs:
            raise ValueError("Duplicate/invalid source ASR chunk")
        _finite(obs.get("infer_start_s"), "inference start")
        _finite(obs.get("infer_end_s"), "inference end")
        if not obs.get("raw_response_sha256") or not obs.get("id"):
            raise ValueError("Missing immutable raw ASR identity")
        asrs[idx] = obs
    source_by_chunk: dict[int, list[dict[str, Any]]] = defaultdict(list)
    identities: set[str] = set()
    for word in source_evidence.get("lexical_observations", []):
        idx = word.get("chunk_index")
        if idx not in asrs or word.get("asr_observation_id") != asrs[idx]["id"]:
            raise ValueError("Source word has unmatched raw ASR owner")
        if not isinstance(word.get("source_ordinal"), int) or not word.get("id") or word["id"] in identities:
            raise ValueError("Invalid or duplicated lexical unit identity")
        identities.add(word["id"])
        if word.get("unit_type") == "word":
            source_by_chunk[idx].append(word)
    for words in source_by_chunk.values():
        words.sort(key=lambda w: w["source_ordinal"])
        if len({w["source_ordinal"] for w in words}) != len(words):
            raise ValueError("Duplicate lexical ordinal")
    results: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    seen_windows: set[str] = set()
    for window in frozen_windows.get("windows", []):
        cid = window.get("case_id")
        if not isinstance(cid, str) or not cid or cid in seen_windows:
            raise ValueError("Missing or repeated challenger case ID")
        seen_windows.add(cid)
        if window.get("media_fingerprint_sha256") != media or window.get("language") != language:
            continue  # Another source recording is allowed in the same manifest.
        wstart = _finite(window.get("start_s"), "crop start")
        wend = _finite(window.get("end_s"), "crop end")
        if not (0 <= wstart < wend) or not isinstance(window.get("wav_sha256"), str) or len(window["wav_sha256"]) != 64:
            raise ValueError("Invalid crop interval or fingerprint")
        item = model["cases"].get(cid)
        if not isinstance(item, dict) or item.get("language") != language:
            excluded.append({"case_id": cid, "reason": "model_record_missing_or_language_mismatch"})
            continue
        words = [w for s in item.get("segments", []) for w in s.get("words", [])]
        token_words: list[tuple[str, dict[str, Any]]] = []
        for word in words:
            if not isinstance(word, dict):
                raise ValueError("Malformed challenger word")
            start = _finite(word.get("abs_start_s"), "challenger word start")
            end = _finite(word.get("abs_end_s"), "challenger word end")
            if start < wstart - .25 or end > wend + .25 or start > end:
                raise ValueError("Challenger word timestamps outside authenticated WAV")
            canonical = _token(word.get("word"))
            if canonical:
                token_words.append((canonical, word))
        if len(token_words) < minimum_matched_words:
            excluded.append({"case_id": cid, "reason": "insufficient_challenger_lexical_content"})
            continue
        rt = [u[0] for u in token_words]
        right_grams = Counter(tuple(rt[i:i + 4]) for i in range(len(rt) - 3))
        matched_case = 0
        for idx, obs in sorted(asrs.items()):
            a, b = float(obs["infer_start_s"]), float(obs["infer_end_s"])
            if max(a, wstart) >= min(b, wend):
                continue
            chunk = source_by_chunk.get(idx, [])
            lt = [_token(u["text"]) for u in chunk]
            left_grams = Counter(tuple(lt[i:i + 4]) for i in range(len(lt) - 3))
            for block in SequenceMatcher(None, lt, rt, autojunk=False).get_matching_blocks():
                if block.size < minimum_matched_words:
                    continue
                left = chunk[block.a:block.a + block.size]
                right = token_words[block.b:block.b + block.size]
                tokens = [x[0] for x in right]
                if (len(set(tokens)) < 3 or sum(len(t) >= 4 for t in set(tokens)) < 2
                        or left_grams[tuple(tokens[:4])] != 1
                        or right_grams[tuple(tokens[:4])] != 1):
                    continue
                source_times = [
                    u.get("ctc_start_s") if u.get("ctc_start_s") is not None
                    else u.get("ctc_rejected_start_s") for u in left
                ]
                offsets = [float(r[1]["abs_start_s"]) - float(st)
                           for st, r in zip(source_times, right) if st is not None]
                if offsets:
                    offset = median(offsets)
                    median_abs_dev = median(abs(v-offset) for v in offsets)
                    classification = (
                        "displaced_ctc_candidate" if abs(offset) >= review_shift_s
                        else "concordant_timestamp_observations"
                    )
                    if median_abs_dev > 1.5:
                        classification = "nonuniform_ctc_conflict_requires_review"
                else:
                    offset = None
                    median_abs_dev = None
                    classification = "source_ctc_unlocalized"
                results.append({
                    "case_id": cid, "source_chunk_index": idx,
                    "source_raw_response_sha256": obs["raw_response_sha256"],
                    "source_asr_observation_id": obs["id"],
                    "source_unit_ids": [u["id"] for u in left],
                    "source_ordinals": [u["source_ordinal"] for u in left],
                    "matched_word_count": block.size,
                    "challenger_word_start_s": float(right[0][1]["abs_start_s"]),
                    "challenger_word_end_s": float(right[-1][1]["abs_end_s"]),
                    "challenger_crop_sha256": window["wav_sha256"],
                    "median_ctc_minus_challenger_s": -round(offset,3) if offset is not None else None,
                    "ctc_offset_mad_s": round(median_abs_dev,3) if median_abs_dev is not None else None,
                    "ctc_observation_count": len(offsets),
                    "classification": classification,
                    "timing_accepted": False,
                    "speaker_identity_accepted": False,
                    "transcription_accepted": False,
                    "requires_positive_acoustic_review_before_promotion": True,
                })
                matched_case += 1
        if not matched_case:
            excluded.append({"case_id": cid, "reason": "no_unique_corresponding_source_phrase"})
    return {
        "schema": SCHEMA, "media_fingerprint_sha256": media,
        "source_language": language,
        "independent_model": {"repo_id": model["repo_id"], "revision": model["revision"]},
        "candidates": results, "unmatched_cases": excluded,
        "diagnostics": {"candidates": len(results), "displaced": sum(
            r["classification"] == "displaced_ctc_candidate" for r in results),
            "nonuniform": sum(r["classification"] == "nonuniform_ctc_conflict_requires_review" for r in results),
            "no_automatic_source_mutation": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Produce private cross-ASR timing comparison; never change source")
    parser.add_argument("--source-evidence", required=True, type=Path)
    parser.add_argument("--frozen-windows", required=True, type=Path)
    parser.add_argument("--challenger-results", required=True, type=Path)
    parser.add_argument("--model-key", default="whisper_large_v3")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    data = compare_independent_word_timing(
        json.loads(args.source_evidence.read_text(encoding="utf-8")),
        json.loads(args.frozen_windows.read_text(encoding="utf-8")),
        json.loads(args.challenger_results.read_text(encoding="utf-8")),
        model_key=args.model_key,
    )
    payload = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    # Exclusive create, private by default, no change to historical output.
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
