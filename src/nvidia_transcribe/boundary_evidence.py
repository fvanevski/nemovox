"""Evidence-gated overlap reconciliation across neighboring ASR chunk excerpts.

Different chunks transcribe overlapping *audio*. Their CTC word times and
provisional speaker IDs are not reliable keys for de-duplication. Compare
locally ordered multiword sequences, but preserve ambiguous short repetitions
and simultaneous voices. This is not a phonetic aligner or a timestamp repair.
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Sequence


def _norm(token: str) -> str:
    return "".join(c for c in str(token).casefold() if c.isalnum())


def _speaker(unit: dict[str, Any]) -> str | None:
    v = unit.get("global_speaker_id") or unit.get("local_speaker_id") or unit.get("speaker")
    if (v is None or str(v) == "unknown"
            or str(v).startswith("SPK_UNK") or str(v).endswith(":unknown")):
        return None
    return str(v)


def _median_mid(units: Sequence[dict[str, Any]]) -> float:
    mids = sorted(float(u["_mid"]) for u in units if u.get("_mid") is not None)
    if not mids:
        raise ValueError("No timing observations in overlap segment")
    return mids[len(mids) // 2]


def _same_voice_within(units: Sequence[dict[str, Any]]) -> bool:
    known = {_speaker(u) for u in units if _speaker(u) is not None}
    return len(known) <= 1


def _continuation(
    seq: list[dict[str, Any]], after: int, reference: dict[str, Any],
    boundary: float, limit: int = 3,
) -> list[dict[str, Any]]:
    """Small contiguous same-voice suffix, never an arbitrary neighboring turn."""
    result: list[dict[str, Any]] = []
    prior = reference
    for unit in seq[after:after + limit]:
        if unit["_drop"] or unit.get("_mid") is None or _speaker(unit) != _speaker(reference):
            break
        if not (abs(float(unit["_mid"]) - boundary) <= 15.0
                and float(unit["start"]) - float(prior["end"]) <= 0.75
                and float(unit["start"]) >= float(prior["start"]) - 0.15):
            break
        if not _norm(unit.get("text", "")):
            break
        result.append(unit)
        prior = unit
    return result


def reconcile_displaced_sequences(
    tagged_by_chunk: list[list[dict[str, Any]]],
    chunks: Sequence[Any],
    *,
    lookaround_s: float = 45.0,
) -> dict[str, Any]:
    """Eliminate only independently repeated, distinctive boundary sequences.

    For long displaced phrases require >=6 matching tokens with >=3 distinct
    lexical forms, unique four-token anchors, and internally consistent speaker
    streams. For short 4-5-token phrases, require close timing plus same speaker,
    or a demonstrably more complete neighbor phrase. A short displaced phrase
    whose speakers disagree is *reported*, not automatically deleted.

    All decisions are bounded to the adjacent chunks' actual inference coverage.
    """
    deduped = 0
    promoted = 0
    matches: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    for bi in range(len(chunks) - 1):
        left_spec, right_spec = chunks[bi:bi + 2]
        boundary = float(left_spec.core_end_s)
        common_start = max(left_spec.infer_start_s, right_spec.infer_start_s)
        common_end = min(left_spec.infer_end_s, right_spec.infer_end_s)
        if common_end <= common_start:
            continue

        def candidate(units: list[dict[str, Any]], spec: Any) -> list[dict[str, Any]]:
            return [
                u for u in units if not u["_drop"] and u.get("_mid") is not None
                and _norm(u.get("text", ""))
                and abs(float(u["_mid"]) - boundary) <= lookaround_s
                and spec.infer_start_s - 0.15 <= float(u["_mid"]) <= spec.infer_end_s + 0.15
            ]

        left = candidate(tagged_by_chunk[bi], left_spec)
        right = candidate(tagged_by_chunk[bi + 1], right_spec)
        lt, rt = [_norm(u.get("text", "")) for u in left], [_norm(u.get("text", "")) for u in right]
        if len(lt) < 4 or len(rt) < 4:
            continue
        # Four-gram multiplicity guards repeated slogans or identical phrases
        # spoken twice *within* either chunk's acoustic excerpt.
        lgrams = Counter(tuple(lt[j:j + 4]) for j in range(len(lt) - 3))
        rgrams = Counter(tuple(rt[j:j + 4]) for j in range(len(rt) - 3))
        blocks = [b for b in SequenceMatcher(None, lt, rt, autojunk=False).get_matching_blocks()
                  if b.size >= 4]
        if not blocks:
            continue
        groups: list[tuple[int, int, int, int, int]] = []
        for b in blocks:
            a0, a1, z0, z1, count = b.a, b.a + b.size, b.b, b.b + b.size, b.size
            if groups and a0 - groups[-1][1] <= 2 and z0 - groups[-1][3] <= 2:
                pa0, _, pz0, _, pc = groups[-1]
                groups[-1] = (pa0, a1, pz0, z1, pc + count)
            else:
                groups.append((a0, a1, z0, z1, count))

        for a0, a1, z0, z1, exact_matches in groups:
            lu, ru = left[a0:a1], right[z0:z1]
            if any(u["_drop"] for u in (*lu, *ru)):
                continue
            shared = [v for v in lt[a0:a1] if v in set(rt[z0:z1])]
            uniqueness = len(set(shared))
            density = exact_matches / max(a1 - a0, z1 - z0)
            phrase = tuple(lt[a0:min(a0 + 4, a1)])
            if exact_matches < 4 or uniqueness < 3 or density < 0.72:
                continue
            left_mid, right_mid = _median_mid(lu), _median_mid(ru)
            delta = round(right_mid - left_mid, 3)
            signature = {"left_chunk": left_spec.chunk_index,
                         "right_chunk": right_spec.chunk_index,
                         "left_source_indices": [u["_unit_idx"] for u in lu],
                         "right_source_indices": [u["_unit_idx"] for u in ru],
                         "matched_tokens": exact_matches,
                         "relative_ctc_shift_s": delta}
            if (lgrams[phrase] != 1 or rgrams[phrase] != 1
                    or not _same_voice_within(lu) or not _same_voice_within(ru)):
                ambiguities.append({**signature, "reason": "repeated_or_mixed_voice_sequence"})
                continue
            # Character-level (e.g. Mandarin) CTC needs a different lexical
            # uniqueness model; keep the legacy near-time reconciliation only.
            # Unknown identities are not positive evidence for deleting speech.
            if any(u.get("unit_type") == "character" for u in (*lu, *ru)):
                ambiguities.append({**signature, "reason": "character_sequence_requires_review"})
                continue
            if any(_speaker(u) is None for u in (*lu, *ru)):
                ambiguities.append({**signature, "reason": "unknown_speaker_sequence"})
                continue
            same_speaker = _speaker(lu[0]) is not None and _speaker(lu[0]) == _speaker(ru[0])
            left_suffix = _continuation(left, a1, lu[-1], boundary)
            right_suffix = _continuation(right, z1, ru[-1], boundary)
            right_more_complete = (
                not left_suffix and bool(right_suffix) and abs(delta) <= 1.0
                and not str(lu[-1].get("text", "")).endswith((".", "!", "?"))
            )
            if abs(delta) > 3.0 and not same_speaker and exact_matches < 8:
                ambiguities.append({**signature, "reason": "insufficient_cross_speaker_sequence_evidence"})
                continue
            if (exact_matches < 6 and not (abs(delta) <= 1.0 and same_speaker)
                    and not right_more_complete):
                ambiguities.append({**signature, "reason": "short_unanchored_duplicate_candidate"})
                continue
            if (abs(delta) <= 1.0 and not same_speaker
                    and not right_more_complete):
                ambiguities.append({**signature, "reason": "possible_simultaneous_distinct_speakers"})
                continue
            # The preferred candidate is the original core observation unless
            # the adjacent overlap contains an independently more complete turn.
            keep_right = right_more_complete
            kept, dropped = (ru, lu) if keep_right else (lu, ru)
            for u in dropped:
                u["_drop"] = True
                deduped += 1
            promoted_indices: list[int] = []
            for u in (*kept, *(right_suffix if keep_right else left_suffix)):
                if not u["_in_core"]:
                    u["_in_core"] = True
                    u["_recovered"] = True
                    u["_recovery_kind"] = "sequence_alternate_recovered"
                    promoted += 1
                    promoted_indices.append(u["_unit_idx"])
                # The winning CTC hypothesis is not an acoustic verification.
                if abs(delta) > 3.0:
                    u["timing_evidence_status"] = "displaced_sequence_ctc_unverified"
                    u["timing_evidence_reason"] = "cross_chunk_ctc_disagreement"
            matches.append({
                **signature, "kept_chunk": right_spec.chunk_index if keep_right else left_spec.chunk_index,
                "discarded_chunk": left_spec.chunk_index if keep_right else right_spec.chunk_index,
                "selected_better_complete_turn": keep_right,
                "promoted_source_indices": promoted_indices,
                "decision": "sequence_duplicate_removed",
            })

    return {
        "schema": "nemovox-boundary-sequence-evidence/v1",
        "sequence_deduplicated_units": deduped,
        "sequence_promoted_units": promoted,
        "matches": matches,
        "ambiguities": ambiguities,
    }


def recover_extended_right_context(
    tagged_by_chunk: list[list[dict[str, Any]]],
    chunks: Sequence[Any],
) -> dict[str, Any]:
    """Recover coherent right-context phrases omitted by their owning chunk.

    Never merely widen the previous single-token 1.5s rescue: require >=3
    lexical tokens, consistent local speaker, valid plausible source timing,
    no right-chunk time coverage and no matching right-chunk lexical sequence
    anywhere within its inference excerpt. Keep an uncertainty label because
    absence in the second ASR pass is not itself proof of ground truth.
    """
    recovered = 0
    examples: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for bi in range(len(chunks) - 1):
        lspec, rspec = chunks[bi:bi + 2]
        boundary = lspec.core_end_s
        remaining = [
            u for u in tagged_by_chunk[bi]
            if not u["_drop"] and not u["_in_core"]
            and u.get("valid_timing") and u.get("_mid") is not None
            and boundary + 1.5 < float(u["_mid"]) <= lspec.infer_end_s
            and _norm(u.get("text", ""))
        ]
        remaining.sort(key=lambda u: u["_unit_idx"])
        groups: list[list[dict[str, Any]]] = []
        for u in remaining:
            if not groups:
                groups.append([u])
                continue
            prev = groups[-1][-1]
            if (_speaker(prev) != _speaker(u)
                    or u["_unit_idx"] != prev["_unit_idx"] + 1
                    or float(u["start"]) - float(prev["end"]) > 0.85
                    or float(u["start"]) < float(prev["start"]) - 0.15):
                groups.append([u])
            else:
                groups[-1].append(u)
        right = [u for u in tagged_by_chunk[bi + 1]
                 if not u["_drop"] and _norm(u.get("text", ""))]
        right_words = [_norm(u["text"]) for u in right]
        for group in groups:
            if (len(group) < 3 or not _same_voice_within(group)
                    or _speaker(group[0]) is None
                    or any(u.get("unit_type") == "character" for u in group)):
                continue
            if float(group[-1]["end"]) - float(group[0]["start"]) > 8.0:
                continue
            words = [_norm(u["text"]) for u in group]
            has_lexical_copy = SequenceMatcher(None, words, right_words, autojunk=False).find_longest_match(
                0, len(words), 0, len(right_words)).size >= min(3, len(words))
            # A different phrase in the same time window may be a genuine
            # overlapping speaker or an ASR substitution; do not guess.
            nearby = any(
                u.get("_mid") is not None and u.get("valid_timing")
                and float(u["end"]) >= float(group[0]["start"]) - 0.30
                and float(u["start"]) <= float(group[-1]["end"]) + 0.30
                for u in right
            )
            if has_lexical_copy or nearby:
                unresolved.append({
                    "boundary_s": boundary,
                    "left_source_indices": [u["_unit_idx"] for u in group],
                    "reason": "right_overlap_covered_or_lexically_ambiguous",
                })
                continue
            for u in group:
                u["_in_core"] = True
                u["_recovered"] = True
                u["_recovery_kind"] = "extended_right_context_recovered"
                u["timing_evidence_status"] = "single_chunk_context_not_independently_verified"
                recovered += 1
            examples.append({
                "boundary_s": boundary,
                "left_chunk": lspec.chunk_index,
                "right_chunk": rspec.chunk_index,
                "source_unit_indices": [u["_unit_idx"] for u in group],
                "start_s": float(group[0]["start"]),
                "end_s": float(group[-1]["end"]),
                "recovered_units": len(group),
            })
    return {
        "extended_recovered_units": recovered,
        "extended_recovery_examples": examples,
        "extended_ambiguities": unresolved,
    }
