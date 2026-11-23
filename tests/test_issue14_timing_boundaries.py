"""Issue #14 timing and coverage invariants (offline synthetic evidence).

No copyrighted audio, series dialogue or model execution required.
"""

from __future__ import annotations

from nvidia_transcribe.chunking import ChunkSpec, reconcile_chunk_boundaries
from nvidia_transcribe.timing_evidence import audit_ctc_timing_units


def _chunks(at: float) -> list[ChunkSpec]:
    return [
        ChunkSpec(0, at - 90, at, at - 100, at + 10),
        ChunkSpec(1, at, at + 90, at - 10, at + 100),
    ]


def _word(text: str, at: float, spk: str = "SPK_00") -> dict:
    return {"text": text, "start": at, "end": at + 0.21,
            "speaker": spk, "global_speaker_id": spk, "valid_timing": True,
            "alignment_status": "ctc_aligned", "unit_type": "word"}


def _phrase(tokens: list[str], at: float, speaker: str = "SPK_00", step: float = .37) -> list[dict]:
    return [_word(word, at + i * step, speaker) for i, word in enumerate(tokens)]


def test_ctc_audit_only_quarantines_impossible_or_flank_confirmed_outliers() -> None:
    units = _phrase(["here", "comes", "a", "quiet", "speech"], 20)
    units[2]["start"], units[2]["end"] = 74.0, 74.21
    audited, diag = audit_ctc_timing_units(units, infer_start_s=0, infer_end_s=90)
    assert diag["quarantined_units"] == 1
    assert diag["quarantine_reasons"]["ctc_isolated_sequence_outlier"] == 1
    assert audited[2]["start"] is None and audited[2]["end"] is None
    assert audited[2]["ctc_hypothesis_start_s"] == 74.0
    assert audited[2]["valid_timing"] is False
    assert [x["text"] for x in audited] == [x["text"] for x in units]
    assert units[2]["start"] == 74.0  # original never modified

    natural_pause = _phrase(["first", "sentence", "second", "sentence"], 1)
    natural_pause[2]["start"], natural_pause[2]["end"] = 42.0, 42.21
    natural_pause[3]["start"], natural_pause[3]["end"] = 42.37, 42.58
    valid, separate = audit_ctc_timing_units(natural_pause, infer_start_s=0, infer_end_s=90)
    assert separate["quarantined_units"] == 0  # large gap alone not proof
    assert all(u["valid_timing"] for u in valid)
    impossible = _phrase(["edge"], 99)
    _, diag2 = audit_ctc_timing_units(impossible, infer_start_s=0, infer_end_s=90)
    assert diag2["quarantine_reasons"]["ctc_outside_inference_excerpt"] == 1


def test_long_displaced_duplicate_from_adjacent_excerpts_is_not_rendered_twice() -> None:
    boundary = 270.0
    phrase = ["silent", "valley", "ancient", "garden", "golden", "river",
              "hidden", "tower", "silver", "bridge", "stone", "door"]
    # The *same* long speech appears at disjoint raw CTC times in
    # overlapping chunks. It must not generate two separate utterances.
    left = _phrase(phrase, 242.8, "SPK_03", .41)
    right = _phrase(phrase, 282.1, "SPK_12", .41)
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(boundary))
    assert [u["text"] for u in merged] == phrase
    assert diag["sequence_evidence"]["sequence_deduplicated_units"] == len(phrase)
    assert diag["sequence_evidence"]["matches"][0]["relative_ctc_shift_s"] > 30
    assert all(
        u["timing_evidence_status"] == "cross_chunk_ctc_disagreement"
        and u["timing_evidence_reason"] == "relative_sequence_shift_unverified"
        and not u["valid_timing"]
        and u["start"] is None and u["end"] is None
        and u["ctc_hypothesis_start_s"] is not None
        and u["alignment_status"] == "ctc_timing_suspect"
        for u in merged
    )
    assert diag["sequence_evidence"]["sequence_ctc_unverified_units"] == len(phrase)
    # The raw inputs remain available with original CTC hypotheses.
    assert left[0]["start"] == 242.8 and right[0]["start"] == 282.1


def test_more_complete_right_excerpt_replaces_truncated_core_without_losing_tail() -> None:
    boundary = 1260.0
    phrase = ["the", "city", "has", "been", "saved"]
    left = _phrase(phrase, 1252.0, "SPK_A", .30)
    right = _phrase(phrase + ["today", "indeed"], 1252.0, "SPK_B", .30)
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(boundary))
    assert [u["text"] for u in merged] == phrase + ["today", "indeed"]
    assert diag["sequence_evidence"]["matches"][0]["selected_better_complete_turn"] is True
    assert all(u["boundary_provenance"] == "sequence_alternate_recovered" for u in merged)
    assert diag["sequence_evidence"]["sequence_promoted_units"] == 7


def test_same_six_word_phrase_by_distinct_speakers_far_apart_is_not_auto_deleted() -> None:
    tokens = ["wait", "for", "the", "ancient", "golden", "river"]
    left = _phrase(tokens, 985.0, "SPK_A")
    right = _phrase(tokens, 1003.0, "SPK_B")
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(990))
    assert len(merged) == 12
    assert diag["sequence_evidence"]["ambiguities"][0]["reason"] == (
        "insufficient_cross_speaker_sequence_evidence"
    )


def test_short_displaced_distinct_speaker_repetition_is_preserved_as_ambiguous() -> None:
    tokens = ["stop", "the", "river", "now"]
    left = _phrase(tokens, 1073.0, "SPK_A")
    right = _phrase(tokens, 1085.5, "SPK_B")
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(1080))
    assert len(merged) == 8
    assert diag["sequence_evidence"]["ambiguities"][0]["reason"] == "insufficient_cross_speaker_sequence_evidence"


def test_actual_repeated_interjection_in_one_chunk_survives() -> None:
    repeated = _phrase(["again", "again", "please", "now",
                        "again", "again", "please", "now"], 1022.0, "SPK_A")
    other = _phrase(["different", "dialogue", "not", "repeated"], 1082.0, "SPK_B")
    merged, _ = reconcile_chunk_boundaries([repeated, other], _chunks(1080))
    assert [u["text"] for u in merged].count("again") == 4


def test_mandarin_character_spans_are_not_deleted_or_extended_by_word_matching() -> None:
    left = _phrase(list("古老城市仍在这里"), 985.0, "SPK_A", .21)
    right = _phrase(list("古老城市仍在这里"), 1003.0, "SPK_B", .21)
    for unit in (*left, *right):
        unit["unit_type"] = "character"
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(990))
    assert len(merged) == len(left) + len(right)
    assert diag["sequence_evidence"]["sequence_deduplicated_units"] == 0
    assert diag["sequence_evidence"]["ambiguities"][0]["reason"] == (
        "character_sequence_requires_review"
    )


def test_missed_right_context_6_words_recovered_beyond_legacy_cap() -> None:
    tokens = ["small", "stones", "are", "falling", "over", "there"]
    left = _phrase(tokens, 1352.05, "SPK_CHILD", .25)
    right = [
        _word("before", 1351.15, "SPK_NARRATOR"),
        _word("after", 1354.7, "SPK_NARRATOR"),
    ]
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(1350.0))
    assert [u["text"] for u in merged if u["global_speaker_id"] == "SPK_CHILD"] == tokens
    assert diag["extended_recovery"]["extended_recovered_units"] == 6
    assert all(u["boundary_provenance"] == "extended_right_context_recovered"
               for u in merged if u["global_speaker_id"] == "SPK_CHILD")


def test_right_context_recovery_does_not_duplicate_shifted_lexical_copy() -> None:
    tokens = ["small", "stones", "are", "falling", "over", "there"]
    left = _phrase(tokens, 1352.05, "SPK_CHILD", .25)
    right = _phrase(tokens, 1367.0, "SPK_OTHER", .25)
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(1350))
    assert diag["extended_recovery"]["extended_recovered_units"] == 0
    assert len(merged) == 6  # right already has it; never add left as a third copy


def test_extended_recovery_does_not_override_alternate_right_speech() -> None:
    tokens = ["small", "stones", "are", "falling", "over", "there"]
    left = _phrase(tokens, 1352.05, "SPK_CHILD", .25)
    right = [_word("narrator", 1352.7, "SPK_ADULT")]
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(1350))
    assert [u["text"] for u in merged] == ["narrator"]
    assert diag["extended_recovery"]["extended_recovered_units"] == 0
    assert diag["extended_recovery"]["extended_ambiguities"]


def test_one_word_or_unknown_voice_cannot_trigger_sequence_based_deletion() -> None:
    left = [_word("yes", 89.80, "SPK_A")]
    right = [_word("yes", 90.03, "SPK_B")]
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(90))
    assert len(merged) == 2
    assert diag["sequence_evidence"]["sequence_deduplicated_units"] == 0


def test_displaced_matching_skips_interleaved_other_voice_without_bridge_deletion() -> None:
    boundary = 990.0
    words = ["ancient", "arches", "stand", "beyond", "the", "silent",
             "mountain", "where", "golden", "water", "flows", "again"]
    left = _phrase(words, 985.0, "SPK_A", .20)
    right_words = _phrase(words, 997.0, "SPK_B", .20)
    right = right_words[:5] + [
        _word("wait", 998.03, "SPK_INTERJECT"),
        _word("listen", 998.21, "SPK_INTERJECT"),
    ] + right_words[5:]
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(boundary))
    counts = {word: sum(u["text"] == word for u in merged) for word in words}
    assert all(n == 1 for n in counts.values())
    assert "wait" in [u["text"] for u in merged]
    assert "listen" in [u["text"] for u in merged]
    assert diag["sequence_evidence"]["sequence_deduplicated_units"] == len(words)
    assert not diag["sequence_evidence"]["sequence_promoted_units"]


def test_w04c_discontinuous_offset_does_not_delete_correctly_timed_first_word() -> None:
    boundary = 1260.0
    words = ["we", "return", "from", "the", "valley"]
    left = _phrase(words, 1264.45, "SPK_A", .32)
    left[0]["start"], left[0]["end"] = 1254.61, 1254.83
    right = _phrase(words, 1264.45, "SPK_A", .32)
    for u in left:
        u["chunk_index"] = 0
    for u in right:
        u["chunk_index"] = 1
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(boundary))
    assert any(u["text"] == "we" and u["chunk_index"] == 1
               and abs(u["start"] - 1264.45) < .01 for u in merged)
    assert not any(0 in m["right_source_indices"] and 0 in m["left_source_indices"]
                   for m in diag["sequence_evidence"]["matches"])
    assert [u["text"] for u in merged if u["chunk_index"] == 1] == words
    assert diag["sequence_evidence"]["sequence_deduplicated_units"] <= 4


def test_diacritic_variant_selects_complete_neighbor_phrase_without_noise() -> None:
    words = ["we", "have", "saved", "tao", "today"]
    left = _phrase(words, 1252.0, "SPK_A", .30)
    right = _phrase(words[:3] + ["táo"] + words[4:] + ["for", "everyone"],
                    1252.0, "SPK_B", .30)
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(1260))
    assert [u["text"] for u in merged] == words[:3] + ["táo"] + words[4:] + ["for", "everyone"]
    assert diag["sequence_evidence"]["matches"][0]["selected_better_complete_turn"]
    assert diag["sequence_evidence"]["sequence_promoted_units"] == 7


def test_exact_ended_preceding_turn_does_not_block_six_word_boundary_recovery() -> None:
    words = ["small", "stones", "are", "falling", "over", "there"]
    left = _phrase(words, 1352.01, "SPK_CHILD", .25)
    right = [_word("preceding", 1351.74, "SPK_OTHER"),
             _word("different", 1355.2, "SPK_OTHER")]
    right[0]["end"] = 1351.95
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(1350))
    assert [u["text"] for u in merged if u["global_speaker_id"] == "SPK_CHILD"] == words
    assert diag["extended_recovery"]["extended_recovered_units"] == 6
    assert diag["extended_recovery"]["extended_ambiguities"] == []


def test_same_long_statement_genuinely_repeated_twice_in_each_chunk_stays_distinct() -> None:
    words = ["beyond", "the", "mountains", "we", "see",
             "an", "ancient", "tower"]
    first = _phrase(words, 986.0, "SPK_A", .20)
    second = _phrase(words, 992.0, "SPK_A", .20)
    left = first + second
    right = _phrase(words, 986.1, "SPK_B", .20) + _phrase(words, 992.1, "SPK_B", .20)
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(990))
    assert diag["sequence_evidence"]["sequence_deduplicated_units"] == 0
    assert not diag["sequence_evidence"]["matches"]
    assert [u["text"] for u in merged].count("beyond") >= 2


def test_ownerless_boundary_s02_source_is_retained_unlocated_not_given_false_time():
    # Boundary 360: chunk3 says a short negation *after* its core, chunk4
    # has the longer corroborated phrase *before* its own core. Under plain
    # core ownership BOTH observations are dropped.
    left = [
        _word("C'est", 364.87, "SPK_LEFT"),
        _word("Edward.", 364.88, "SPK_OTHER"),
        _word("pas", 364.98, "SPK_LEFT"),
        _word("notre", 365.08, "SPK_LEFT"),
    ]
    right = _phrase(
        ["C'est", "pas", "notre", "nacale", "qui", "aurait", "pensé", "à", "ça!"],
        357.32, "SPK_RIGHT", .27,
    )
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(360.0))
    ours = [u for u in merged if u.get("boundary_provenance") ==
            "straddled_overlap_source_unlocalized"]
    assert [u["text"] for u in ours] == [u["text"] for u in right]
    assert diag["straddled_overlap_source"]["source_units_preserved_unlocalized"] == 9
    assert all(
        u["start"] is None and u["end"] is None
        and not u["valid_timing"] and u["ctc_hypothesis_start_s"] is not None
        and u["alignment_status"] == "ctc_timing_suspect"
        and u["speaker_identity_accepted"] is False
        and u["global_speaker_id"].startswith("SPK_UNK_BOUNDARY_")
        for u in ours
    )
    assert any(u["text"] == "Edward." for u in merged) is False
    # The original source lists remain untouched.
    assert left[0]["start"] == 364.87 and right[0]["start"] == 357.32


def test_straddled_single_word_or_nonmatching_overlap_is_not_promoted():
    left = _phrase(["C'est", "pas", "notre"], 364.87, "SPK_LEFT", .15)
    right = _phrase(["What", "a", "different", "phrase"], 357.3, "SPK_RIGHT", .18)
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(360))
    assert diag["straddled_overlap_source"]["source_units_preserved_unlocalized"] == 0
    assert not merged


def test_straddled_duplicate_does_not_compete_with_existing_core_speech():
    left = _phrase(["C'est", "pas", "notre"], 364.87, "SPK_LEFT", .15)
    right = _phrase(["C'est", "pas", "notre", "phrase"], 357.3, "SPK_RIGHT", .18)
    owned = _phrase(["C'est", "pas", "notre", "phrase"], 360.5, "SPK_CORE", .15)
    merged, diag = reconcile_chunk_boundaries(
        [left, right + owned], _chunks(360)
    )
    assert diag["straddled_overlap_source"]["source_units_preserved_unlocalized"] == 0
    assert any(u["text"] == "phrase" and u.get("global_speaker_id") == "SPK_CORE" for u in merged)


def test_clean_mid_core_29_token_control_remains_intact() -> None:
    tokens = [f"word{i:02d}" for i in range(29)]
    left = _phrase(tokens, 680.0, "SPK_07", .31)
    right = _phrase(["another", "speaker", "says", "something"], 721, "SPK_08")
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(720))
    assert [u["text"] for u in merged[:29]] == tokens
    assert not diag["sequence_evidence"]["matches"]
    assert not diag["extended_recovery"]["extended_recovered_units"]
