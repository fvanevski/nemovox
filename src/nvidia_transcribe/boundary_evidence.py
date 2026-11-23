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
import unicodedata


def _norm(token: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(token).casefold()) if c.isalnum() and not unicodedata.combining(c))


def _speaker(unit: dict[str, Any]) -> str | None:
    v = unit.get("global_speaker_id") or unit.get("local_speaker_id") or unit.get("speaker")
    if (v is None or str(v) == "unknown"
            or str(v).startswith("SPK_UNK") or str(v).endswith(":unknown")):
        return None
    return str(v)


def _mid(unit: dict[str, Any]) -> float:
    return float(unit["_mid"])


def _stream_candidates(
    units: list[dict[str, Any]], spec: Any, boundary: float, lookaround_s: float,
) -> dict[str, list[dict[str, Any]]]:
    """Compare each voice in lexical order, not a mixed-speaker timeline."""
    streams: dict[str, list[dict[str, Any]]] = {}
    for unit in units:
        speaker = _speaker(unit)
        if (unit["_drop"] or speaker is None or unit.get("_mid") is None
                or unit.get("unit_type") == "character"
                or not _norm(unit.get("text", ""))):
            continue
        t = _mid(unit)
        if (abs(t - boundary) > lookaround_s
                or not spec.infer_start_s - 0.15 <= t <= spec.infer_end_s + 0.15):
            continue
        streams.setdefault(speaker, []).append(unit)
    return streams


def _consistent_runs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], tolerance_s: float = 2.0,
) -> list[list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Split matches at discontinuous CTC offsets or backward word movement.

    A median offset across a *whole* phrase previously hid the one-word
    ten-second W04c displacement and deleted its correctly timed copy.
    """
    runs: list[list[tuple[dict[str, Any], dict[str, Any]]]] = []
    for left, right in pairs:
        if not runs:
            runs.append([(left, right)])
            continue
        prev_l, prev_r = runs[-1][-1]
        d = _mid(right) - _mid(left)
        prev_d = _mid(prev_r) - _mid(prev_l)
        if (abs(d - prev_d) > tolerance_s
                or _mid(left) < _mid(prev_l) - 0.3
                or _mid(right) < _mid(prev_r) - 0.3):
            runs.append([(left, right)])
        else:
            runs[-1].append((left, right))
    return runs


def _same_voice_within(units: Sequence[dict[str, Any]]) -> bool:
    """Unknowns are never proof of one voice; known IDs must agree."""
    speakers = {_speaker(unit) for unit in units}
    return len(speakers) == 1 and None not in speakers


def _right_continuation(
    stream: list[dict[str, Any]], end: int,
    matched_last: dict[str, Any],
    boundary: float,
) -> list[dict[str, Any]]:
    """Accept only a coherent multiword continuation; not one noisy edge token."""
    suffix: list[dict[str, Any]] = []
    prev = matched_last
    for unit in stream[end:end + 4]:
        if unit["_drop"] or abs(_mid(unit) - boundary) > 15.0:
            break
        if (unit["_unit_idx"] != prev["_unit_idx"] + 1
                or float(unit["start"]) - float(prev["end"]) > 0.75
                or float(unit["start"]) < float(prev["start"]) - 0.15):
            break
        suffix.append(unit)
        prev = unit
    return suffix if len(suffix) >= 2 else []


def reconcile_displaced_sequences(
    tagged_by_chunk: list[list[dict[str, Any]]],
    chunks: Sequence[Any],
    *,
    lookaround_s: float = 45.0,
) -> dict[str, Any]:
    """Conservatively reconcile distinct multiword echoes across shared audio.

    Long cross-speaker matches need strong distinctive lexical evidence and
    coherent *per-word* relative offsets. Short or ambiguous repetitions are
    not deleted. Only a better, multiword right-hand continuation can replace
    a truncated core candidate; arbitrary unmatched suffix promotion is banned.
    """
    deduped = 0
    promoted = 0
    matches: list[dict[str, Any]] = []
    ambiguities: list[dict[str, Any]] = []
    for bi in range(len(chunks) - 1):
        lspec, rspec = chunks[bi:bi + 2]
        boundary = float(lspec.core_end_s)
        if min(lspec.infer_end_s, rspec.infer_end_s) <= max(
            lspec.infer_start_s, rspec.infer_start_s
        ):
            continue
        # Character-level CTC needs its own alignment model. Expose a
        # review-only candidate instead of silently deleting repeated glyphs.
        lc = [u for u in tagged_by_chunk[bi]
              if not u["_drop"] and u.get("unit_type") == "character"
              and u.get("_mid") is not None and abs(_mid(u) - boundary) <= lookaround_s]
        rc = [u for u in tagged_by_chunk[bi + 1]
              if not u["_drop"] and u.get("unit_type") == "character"
              and u.get("_mid") is not None and abs(_mid(u) - boundary) <= lookaround_s]
        if len(lc) >= 4 and len(rc) >= 4:
            match = SequenceMatcher(
                None, [_norm(u["text"]) for u in lc],
                [_norm(u["text"]) for u in rc], autojunk=False
            ).find_longest_match(0, len(lc), 0, len(rc))
            if match.size >= 4:
                ambiguities.append({
                    "left_chunk": lspec.chunk_index, "right_chunk": rspec.chunk_index,
                    "reason": "character_sequence_requires_review",
                    "matched_tokens": match.size, "resolution": "preserved_both",
                })
        ls = _stream_candidates(tagged_by_chunk[bi], lspec, boundary, lookaround_s)
        rs = _stream_candidates(tagged_by_chunk[bi + 1], rspec, boundary, lookaround_s)
        # The number of local speakers is bounded by the actual ASR response;
        # compare pairwise streams rather than merging interleaved voices.
        for lvoice, left in sorted(ls.items()):
            lt = [_norm(x["text"]) for x in left]
            if len(lt) < 4:
                continue
            left_grams = Counter(tuple(lt[i:i + 4]) for i in range(len(lt) - 3))
            for rvoice, right in sorted(rs.items()):
                rt = [_norm(x["text"]) for x in right]
                if len(rt) < 4:
                    continue
                right_grams = Counter(tuple(rt[i:i + 4]) for i in range(len(rt) - 3))
                blocks = SequenceMatcher(
                    None, lt, rt, autojunk=False
                ).get_matching_blocks()
                for block in blocks:
                    if block.size < 4:
                        continue
                    pairs = [(left[block.a + j], right[block.b + j])
                             for j in range(block.size)]
                    for run in _consistent_runs(pairs):
                        lu = [p[0] for p in run]
                        ru = [p[1] for p in run]
                        n = len(run)
                        if n < 4 or any(u["_drop"] for u in (*lu, *ru)):
                            continue
                        words = [_norm(u["text"]) for u in lu]
                        if len(set(words)) < 3:
                            continue
                        phrase = tuple(words[:4])
                        if left_grams[phrase] != 1 or right_grams[phrase] != 1:
                            continue
                        offsets = [_mid(b) - _mid(a) for a, b in run]
                        shift = round(sum(offsets) / n, 3)
                        same_voice = lvoice == rvoice
                        # A few repeated tokens cannot prove that two distinct
                        # speakers in the overlapping recording said one phrase.
                        reason: str | None = None
                        suffix: list[dict[str, Any]] = []
                        # A more complete right-hand observation is separate
                        # positive evidence for selecting it over a truncated core.
                        last_right_i = next(
                            (i for i, u in enumerate(right) if u is ru[-1]), -1
                        )
                        if last_right_i >= 0:
                            suffix = _right_continuation(
                                right, last_right_i + 1, ru[-1], boundary
                            )
                        right_more_complete = (
                            bool(suffix) and abs(shift) <= 1.0
                            and not str(lu[-1].get("text", "")).endswith((".", "!", "?"))
                            and not any(
                                # If the left chunk itself includes those words
                                # as a continuation, right is not more complete.
                                _norm(u.get("text", "")) == _norm(suffix[0]["text"])
                                for u in left
                                if not u["_drop"] and u["_unit_idx"] > lu[-1]["_unit_idx"]
                                and _mid(u) <= _mid(lu[-1]) + 1.2
                            )
                        )
                        if abs(shift) > 3.0 and (n < 8 if not same_voice else n < 6):
                            reason = "insufficient_cross_speaker_sequence_evidence"
                        elif abs(shift) <= 1.0 and not same_voice and not right_more_complete:
                            reason = "possible_simultaneous_distinct_speakers"
                        elif n < 6 and not (same_voice and abs(shift) <= 1.0) and not right_more_complete:
                            reason = "short_unanchored_duplicate_candidate"
                        if reason:
                            ambiguities.append({
                                "left_chunk": lspec.chunk_index, "right_chunk": rspec.chunk_index,
                                "left_source_indices": [u["_unit_idx"] for u in lu],
                                "right_source_indices": [u["_unit_idx"] for u in ru],
                                "matched_tokens": n, "relative_ctc_shift_s": shift,
                                "reason": reason, "resolution": "preserved_both",
                            })
                            continue
                        # In a close duplicate, honor core ownership. In a
                        # displaced duplicate, preserve only a single lexical
                        # occurrence, but do not claim either CTC time verified.
                        left_core = sum(bool(u["_in_core"]) for u in lu)
                        right_core = sum(bool(u["_in_core"]) for u in ru)
                        keep_right = right_more_complete or (
                            not right_more_complete and right_core > left_core
                        )
                        kept, discarded = (ru, lu) if keep_right else (lu, ru)
                        for u in discarded:
                            u["_drop"] = True
                            deduped += 1
                        promoted_indices: list[int] = []
                        to_promote = (*kept, *(suffix if right_more_complete else ()))
                        for u in to_promote:
                            if not u["_in_core"]:
                                u["_in_core"] = True
                                u["_recovered"] = True
                                u["_recovery_kind"] = "sequence_alternate_recovered"
                                promoted += 1
                                promoted_indices.append(u["_unit_idx"])
                            if abs(shift) > 3.0:
                                u["timing_evidence_status"] = "cross_chunk_ctc_disagreement"
                                u["timing_evidence_reason"] = "relative_sequence_shift_unverified"
                        matches.append({
                            "left_chunk": lspec.chunk_index,
                            "right_chunk": rspec.chunk_index,
                            "left_source_indices": [u["_unit_idx"] for u in lu],
                            "right_source_indices": [u["_unit_idx"] for u in ru],
                            "matched_tokens": n,
                            "relative_ctc_shift_s": shift,
                            "kept_chunk": rspec.chunk_index if keep_right else lspec.chunk_index,
                            "discarded_chunk": lspec.chunk_index if keep_right else rspec.chunk_index,
                            "selected_better_complete_turn": right_more_complete,
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


def recover_ownerless_straddled_overlap(
    tagged_by_chunk: list[list[dict[str, Any]]],
    chunks: Sequence[Any],
) -> dict[str, Any]:
    """Preserve corroborated words with UNKNOWN time when both cores lose them.

    Specifically, two overlapping inference requests can return a matching
    3+ word sequence but assign contradictory CTC positions on opposite
    sides of their shared core boundary. Neither observation belongs to
    its nominal core, so ordinary boundary filtering silently loses both.
    Preserve the more complete context phrase as an *untimed hypothesis*,
    with exact original CTC estimates and corroborating source indices.
    This is NOT proof of speech, an acoustic time or speaker identity.
    """
    recovered: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for i in range(len(chunks) - 1):
        left_spec, right_spec = chunks[i:i + 2]
        boundary = left_spec.core_end_s
        if max(left_spec.infer_start_s, right_spec.infer_start_s) >= min(
            left_spec.infer_end_s, right_spec.infer_end_s
        ):
            continue
        left_units = [
            u for u in tagged_by_chunk[i]
            if (not u["_drop"] and not u["_in_core"]
                and u.get("valid_timing") and u.get("_mid") is not None
                and boundary <= _mid(u) <= min(left_spec.infer_end_s, boundary + 10)
                and u.get("unit_type") != "character" and _norm(u.get("text", "")))
        ]
        right_units = [
            u for u in tagged_by_chunk[i + 1]
            if (not u["_drop"] and not u["_in_core"]
                and u.get("valid_timing") and u.get("_mid") is not None
                and max(right_spec.infer_start_s, boundary - 10) <= _mid(u) < boundary
                and u.get("unit_type") != "character" and _norm(u.get("text", "")))
        ]
        def phrases(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
            by_voice: dict[str, list[dict[str, Any]]] = {}
            for unit in items:
                key = _speaker(unit)
                if key is not None:
                    by_voice.setdefault(key, []).append(unit)
            groups: list[list[dict[str, Any]]] = []
            for stream in by_voice.values():
                stream.sort(key=lambda x: x["_unit_idx"])
                group: list[dict[str, Any]] = []
                for unit in stream:
                    if group and (unit["_unit_idx"] != group[-1]["_unit_idx"] + 1
                                  or float(unit["start"]) - float(group[-1]["end"]) > .9
                                  or _mid(unit) < _mid(group[-1]) - .2):
                        groups.append(group)
                        group = []
                    group.append(unit)
                if group:
                    groups.append(group)
            return [g for g in groups if 3 <= len(g) <= 30
                    and float(g[-1]["end"]) - float(g[0]["start"]) <= 8.0]
        left_phrases, right_phrases = phrases(left_units), phrases(right_units)
        candidates = []
        for left in left_phrases:
            lt = [_norm(u["text"]) for u in left]
            for right in right_phrases:
                rt = [_norm(u["text"]) for u in right]
                blocks = SequenceMatcher(None, lt, rt, autojunk=False).get_matching_blocks()
                for block in blocks:
                    if block.size < 3:
                        continue
                    shared = lt[block.a:block.a+block.size]
                    if len(set(shared)) < 3 or sum(len(w) >= 4 for w in shared) < 2:
                        continue
                    if lt.count(shared[0]) != 1 or rt.count(shared[0]) != 1:
                        continue
                    # The corroborating core match must not be an already
                    # owned occurrence elsewhere at this boundary.
                    owned = [
                        u for c in tagged_by_chunk[i:i+2] for u in c
                        if u["_in_core"] and not u["_drop"] and
                        u.get("_mid") is not None and
                        abs(_mid(u) - boundary) <= 10
                    ]
                    owned_words = [_norm(u.get("text", "")) for u in owned]
                    if SequenceMatcher(None, shared, owned_words, autojunk=False).find_longest_match(
                        0, len(shared), 0, len(owned_words)
                    ).size >= 3:
                        unresolved.append({"boundary_s": boundary,
                            "reason": "straddled_sequence_already_has_core_coverage",
                            "left_chunk": left_spec.chunk_index, "right_chunk": right_spec.chunk_index})
                        continue
                    candidates.append((left, right, block.size))
        # A cross-speaker or duplicated triple match can represent distinct
        # simultaneous speech; never pick one if multiple plausible pairs.
        pairs = {(tuple(u["_unit_idx"] for u in l), tuple(u["_unit_idx"] for u in r))
                 for l, r, _ in candidates}
        if len(pairs) != 1:
            if candidates:
                unresolved.append({"boundary_s": boundary,
                    "reason": "ambiguous_straddled_audio_event_correspondence",
                    "candidate_pairs": len(pairs)})
            continue
        left, right, shared_count = max(candidates, key=lambda x: x[2])
        # Preserve the more complete *one* model observation. In the equal
        # case prefer the right-side source, but no timestamp is promoted.
        chosen = right if len(right) >= len(left) else left
        if any(u["_drop"] or u["_in_core"] for u in chosen):
            continue
        for u in chosen:
            u["ctc_hypothesis_start_s"] = u.get("start")
            u["ctc_hypothesis_end_s"] = u.get("end")
            u["start"] = None
            u["end"] = None
            u["valid_timing"] = False
            u["_mid"] = None
            u["_in_core"] = True
            u["_recovered"] = True
            u["_recovery_kind"] = "straddled_overlap_source_unlocalized"
            u["alignment_status"] = "ctc_timing_suspect"
            u["timing_evidence_status"] = "unverified_cross_boundary_ctc_conflict"
            u["timing_evidence_reason"] = "both_chunks_outside_owning_core"
            u["speaker_identity_accepted"] = False
            u["global_speaker_id"] = f"SPK_UNK_BOUNDARY_{left_spec.chunk_index}_{right_spec.chunk_index}"
        recovered.append({
            "boundary_s": boundary,
            "left_chunk": left_spec.chunk_index, "right_chunk": right_spec.chunk_index,
            "left_source_unit_indices": [u["_unit_idx"] for u in left],
            "right_source_unit_indices": [u["_unit_idx"] for u in right],
            "retained_chunk": chosen[0]["_chunk_idx"],
            "retained_source_indices": [u["_unit_idx"] for u in chosen],
            "source_units_preserved_unlocalized": len(chosen),
            "matching_lexical_tokens": shared_count,
            "timing_truth_accepted": False,
            "speaker_identity_accepted": False,
        })
    return {
        "schema": "nemovox-straddled-core-source-evidence/v1",
        "source_units_preserved_unlocalized": sum(
            row["source_units_preserved_unlocalized"] for row in recovered
        ),
        "recovery_candidates": recovered,
        "ambiguities": unresolved,
        "no_new_acoustic_timestamp": True,
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
                0, len(words), 0, len(right_words)).size >= max(3, (3 * len(words) + 3) // 4)
            # A different phrase in the same time window may be a genuine
            # overlapping speaker or an ASR substitution; do not guess.
            nearby = any(
                u.get("_mid") is not None and u.get("valid_timing")
                and float(u["end"]) > float(group[0]["start"])
                and float(u["start"]) < float(group[-1]["end"])
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
