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
    assert all(u["timing_evidence_status"] == "displaced_sequence_ctc_unverified" for u in merged)


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


def test_clean_mid_core_29_token_control_remains_intact() -> None:
    tokens = [f"word{i:02d}" for i in range(29)]
    left = _phrase(tokens, 680.0, "SPK_07", .31)
    right = _phrase(["another", "speaker", "says", "something"], 721, "SPK_08")
    merged, diag = reconcile_chunk_boundaries([left, right], _chunks(720))
    assert [u["text"] for u in merged[:29]] == tokens
    assert not diag["sequence_evidence"]["matches"]
    assert not diag["extended_recovery"]["extended_recovered_units"]
