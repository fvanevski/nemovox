"""Immutable ASR source-evidence ledger for transcript-first reconstruction.

This v1 sidecar is observational, not a canonical transcript or a subtitle
renderer input. A lexical match across model requests is never independent
acoustic evidence that either occurrence can be deleted.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Sequence

from nvidia_transcribe.utterances import join_multilingual_units
from nvidia_transcribe.source_occurrences import source_observation_id

SCHEMA_VERSION = "nemovox-source-evidence/v1"


def _stable_id(kind: str, material: Any) -> str:
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"{kind}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def _token_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(
        char for char in decomposed
        if char.isalnum() and not unicodedata.combining(char)
    )


def _trusted_numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _timing_hypothesis(unit: dict[str, Any], start: float, end: float) -> tuple[str, float | None, float | None, str]:
    """Structural plausibility only. No word time is certified against audio."""
    s = _trusted_numeric(unit.get("start"))
    e = _trusted_numeric(unit.get("end"))
    prior = str(unit.get("timing_evidence_status") or "")
    if not unit.get("valid_timing") or s is None or e is None:
        return ("unresolved", None, None,
                str(unit.get("timing_evidence_reason") or unit.get("timing_error") or "no_usable_word_timing"))
    if s > e or s < start - 0.15 or e > end + 0.15:
        return ("disputed", s, e, "timestamp_outside_inference_window")
    if prior and prior not in ("ctc_aligned", "accepted", "valid"):
        return ("disputed", s, e, str(unit.get("timing_evidence_reason") or prior))
    if unit.get("ctc_sequence_gap_s") is not None:
        return ("provisional_sequence_gap", s, e, "large_gap_requires_audio_review")
    return ("provisional_ctc", s, e, "ctc_not_acoustically_verified")


def _subsequence_occurrences(haystack: list[str], needle: list[str]) -> int:
    n = len(needle)
    if n == 0:
        return 0
    return sum(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def _stream_units(units: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Local tags are per-request sequence labels, never episode speaker identity."""
    streams: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        key = str(unit.get("local_speaker_id") or "unknown")
        if unit["normalized_text"]:
            streams.setdefault(key, []).append(unit)
    return streams


def _correspondences(
    chunk_specs: Sequence[Any],
    by_chunk: Sequence[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for li, left in enumerate(chunk_specs):
        for ri in range(li + 1, len(chunk_specs)):
            right = chunk_specs[ri]
            overlap_start = max(float(left.infer_start_s), float(right.infer_start_s))
            overlap_end = min(float(left.infer_end_s), float(right.infer_end_s))
            if overlap_end <= overlap_start:
                continue
            for l_tag, l_stream in _stream_units(by_chunk[li]).items():
                a = [u["normalized_text"] for u in l_stream]
                for r_tag, r_stream in _stream_units(by_chunk[ri]).items():
                    b = [u["normalized_text"] for u in r_stream]
                    matcher = SequenceMatcher(None, a, b, autojunk=False)
                    for match in matcher.get_matching_blocks():
                        if match.size < 4:
                            continue
                        l_group = l_stream[match.a:match.a + match.size]
                        r_group = r_stream[match.b:match.b + match.size]
                        words = a[match.a:match.a + match.size]
                        ambiguous_repeat = (
                            _subsequence_occurrences(a, words) > 1
                            or _subsequence_occurrences(b, words) > 1
                        )
                        l_global = {u.get("global_speaker_hypothesis") for u in l_group}
                        r_global = {u.get("global_speaker_hypothesis") for u in r_group}
                        disagreement = (
                            len(l_global) == len(r_global) == 1
                            and None not in l_global and None not in r_global
                            and l_global != r_global
                        )
                        left_starts = [u["ctc_start_s"] for u in l_group]
                        right_starts = [u["ctc_start_s"] for u in r_group]
                        offset = (
                            round(sum(float(x) - float(y) for x, y in zip(left_starts, right_starts))
                                  / len(left_starts), 4)
                            if all(x is not None for x in left_starts + right_starts)
                            else None
                        )
                        refs = {
                            "left": [u["id"] for u in l_group],
                            "right": [u["id"] for u in r_group],
                        }
                        candidates.append({
                            "id": _stable_id("corr", refs),
                            "left_chunk": int(left.chunk_index),
                            "right_chunk": int(right.chunk_index),
                            "shared_inference_window_s": [round(overlap_start, 6), round(overlap_end, 6)],
                            "left_local_tag": l_tag,
                            "right_local_tag": r_tag,
                            "left_observation_unit_ids": refs["left"],
                            "right_observation_unit_ids": refs["right"],
                            "normalized_token_count": match.size,
                            "ctc_relative_offset_hypothesis_s": offset,
                            "repeated_phrase_ambiguity": ambiguous_repeat,
                            "different_global_speaker_hypotheses": disagreement,
                            "status": "candidate_only_unverified",
                            "decision": "preserve_both_observations",
                            "reason": "lexical_correspondence_is_not_independent_acoustic_evidence",
                        })
    return sorted(candidates, key=lambda x: (x["left_chunk"], x["right_chunk"], x["id"]))


def build_source_evidence_ledger(
    *,
    media_fingerprint_sha256: str,
    source_language: str,
    chunk_specs: Sequence[Any],
    chunk_units: Sequence[list[dict[str, Any]]],
    raw_response_sha256: Sequence[str],
    raw_source_texts: Sequence[str],
    diarization_by_chunk: Sequence[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Emit a deterministic, lossless lexical-observation inventory.

    Caller provides units BEFORE legacy boundary reconciliation removes or
    promotes observations. All ctc times and speaker IDs remain hypotheses.
    No source event is marked accepted and no subtitle output is changed.
    """
    lengths = {
        len(chunk_specs), len(chunk_units), len(raw_response_sha256),
        len(raw_source_texts), len(diarization_by_chunk),
    }
    if len(lengths) != 1:
        raise ValueError("Source evidence inputs must have one entry per ASR chunk")
    if not media_fingerprint_sha256:
        raise ValueError("Missing immutable media fingerprint")
    if len({int(c.chunk_index) for c in chunk_specs}) != len(chunk_specs):
        raise ValueError("Duplicate source chunk indices")

    asr_observations: list[dict[str, Any]] = []
    lexical_observations: list[dict[str, Any]] = []
    turn_candidates: list[dict[str, Any]] = []
    activity_hypotheses: list[dict[str, Any]] = []
    grouped: list[list[dict[str, Any]]] = []

    for spec, units, raw_sha, raw_text, diarization in zip(
        chunk_specs, chunk_units, raw_response_sha256, raw_source_texts, diarization_by_chunk
    ):
        cidx = int(spec.chunk_index)
        start, end = float(spec.infer_start_s), float(spec.infer_end_s)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid inference window for chunk {cidx}")
        if len(raw_sha) != 64 or any(ch not in "0123456789abcdef" for ch in raw_sha.lower()):
            raise ValueError(f"Chunk {cidx} lacks a raw-response SHA-256")
        observation_id = source_observation_id(media_fingerprint_sha256, raw_sha, spec)
        chunk_rows: list[dict[str, Any]] = []
        asr_observations.append({
            "id": observation_id,
            "chunk_index": cidx,
            "raw_response_sha256": raw_sha,
            "recognized_text_projection": raw_text,
            "text_projection_method": "cleanup_generation_loops",
            "infer_start_s": start,
            "infer_end_s": end,
            "core_start_s": float(spec.core_start_s),
            "core_end_s": float(spec.core_end_s),
            "coverage_status": "unverified",
            "evidence_kind": "model_generated_observation",
        })
        for ordinal, unit in enumerate(units):
            content = str(unit.get("text") or "")
            if not content:
                continue
            status, ts, te, reason = _timing_hypothesis(unit, start, end)
            row = {
                "id": unit.get("source_unit_id") or _stable_id("unit", [observation_id, ordinal, content]),
                "recognized_turn_id": unit.get("recognized_turn_id"),
                "raw_recognized_token_ids": unit.get("raw_recognized_token_ids", []),
                "ctc_association_status": unit.get("ctc_association_status"),
                "ctc_word_index": unit.get("ctc_word_index"),
                "asr_observation_id": observation_id,
                "chunk_index": cidx,
                "source_ordinal": ordinal,
                "text": content,
                "normalized_text": _token_key(content),
                "unit_type": str(unit.get("unit_type") or "unknown"),
                "join_no_space_prev": unit.get("join_no_space_prev"),
                "local_speaker_id": unit.get("local_speaker_id"),
                "raw_speaker_tag": unit.get("raw_speaker_tag"),
                "global_speaker_hypothesis": (
                    unit.get("global_speaker_id")
                    if unit.get("global_speaker_id") not in (None, "SPK_UNK")
                    else None
                ),
                "ctc_start_s": ts,
                "ctc_end_s": te,
                "ctc_rejected_start_s": unit.get("ctc_hypothesis_start_s"),
                "ctc_rejected_end_s": unit.get("ctc_hypothesis_end_s"),
                "timing_status": status,
                "timing_reason": reason,
                "word_alignment_accepted": False,
                "speaker_identity_accepted": False,
            }
            chunk_rows.append(row)
            lexical_observations.append(row)
        grouped.append(chunk_rows)

        # An ASR-local same-tag run is a candidate, not proof of a real
        # speaker turn. Preserve order even if a tag is reused for two voices.
        def add_turn(run: list[dict[str, Any]]) -> None:
            if not run:
                return
            turn_candidates.append({
                "id": run[0].get("recognized_turn_id") or _stable_id("turncand", [row["id"] for row in run]),
                "asr_observation_id": observation_id,
                "local_speaker_id": run[0]["local_speaker_id"],
                "unit_ids": [row["id"] for row in run],
                "source_text_hypothesis": join_multilingual_units(run),
                "status": "unverified_local_speaker_run",
                "accepted_source_turn_id": None,
                "accepted_timing": None,
            })

        run: list[dict[str, Any]] = []
        for row in chunk_rows:
            if run and (row["recognized_turn_id"] != run[0]["recognized_turn_id"]
                        or row["local_speaker_id"] != run[0]["local_speaker_id"]):
                add_turn(run)
                run = []
            run.append(row)
        add_turn(run)

        for j, interval in enumerate(diarization):
            ds = _trusted_numeric(interval.get("start"))
            de = _trusted_numeric(interval.get("end"))
            valid = ds is not None and de is not None and ds <= de
            covered = bool(valid and any(
                row["ctc_start_s"] is not None and row["ctc_end_s"] is not None
                and min(float(row["ctc_end_s"]), de) > max(float(row["ctc_start_s"]), ds)
                for row in chunk_rows
            ))
            activity_hypotheses.append({
                "id": _stable_id("activity", [observation_id, j]),
                "asr_observation_id": observation_id,
                "speaker_tag_hypothesis": interval.get("speaker"),
                "start_s": ds if valid else None,
                "end_s": de if valid else None,
                "status": "model_activity_hypothesis_only",
                "has_overlapping_ctc_hypothesis": covered,
                "possible_unrepresented_speech": bool(valid and not covered),
                "must_not_infer_missing_words_from_this_flag": True,
            })

    correspondences = _correspondences(chunk_specs, grouped)
    status_counts = {
        name: sum(row["timing_status"] == name for row in lexical_observations)
        for name in sorted({row["timing_status"] for row in lexical_observations})
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "shadow_observation_only",
        "media_fingerprint_sha256": media_fingerprint_sha256,
        "source_language": source_language,
        "asr_observations": asr_observations,
        "lexical_observations": lexical_observations,
        "local_turn_candidates": turn_candidates,
        "cross_chunk_correspondence_candidates": correspondences,
        "diarization_activity_hypotheses": activity_hypotheses,
        "diagnostics": {
            "source_unit_observations": len(lexical_observations),
            "unverified_correspondence_candidates": len(correspondences),
            "timing_status_counts": status_counts,
            "activity_without_ctc_hypothesis": sum(
                a["possible_unrepresented_speech"] for a in activity_hypotheses
            ),
            "source_coverage_certified": False,
            "source_units_dropped_by_ledger": 0,
        },
    }
