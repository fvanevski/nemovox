"""Source-to-translation/ASS retention: no similarity-driven deletion of speech."""
from nvidia_transcribe.subtitles import (
    SubtitleCue, assign_subtitle_lanes,
    build_display_cues,
)
from nvidia_transcribe.translation import _apply_safe_adjacent_regrouping


def _cue(uid, speaker, start=12.0, end=14.0, source_ids=()):
    return SubtitleCue(
        utterance_id=uid, start=start, end=end,
        raw_start=start, raw_end=end, speaker=speaker,
        text="The interior of the continent is still unexplored.",
        source_turn_ids=list(source_ids),
    )


def test_identical_english_from_different_source_turns_survives_srt_ass_lane_input():
    a = _cue("utt-01", "SPK_A", source_ids=["source-turn-a"])
    b = _cue("utt-02", "SPK_B", start=12.02, end=13.98, source_ids=["source-turn-b"])
    displayed, overflow, unrenderable = assign_subtitle_lanes([a, b], max_lanes=3)
    assert len(displayed) == 2 and not overflow and not unrenderable
    assert {tuple(c.source_turn_ids) for c in displayed} == {
        ("source-turn-a",), ("source-turn-b",)
    }


def test_exact_same_source_event_replayed_cue_may_be_deduplicated():
    a = _cue("utt-01", "SPK_A", source_ids=["source-turn-one"])
    b = _cue("utt-01", "SPK_A", source_ids=["source-turn-one"])
    kept, overflow, suppressed = assign_subtitle_lanes([a, b])
    assert len(kept) == 1 and not overflow and len(suppressed) == 1
    assert suppressed[0].source_turn_ids == ["source-turn-one"]
    assert "duplicate_display_replay" in suppressed[0].conflicts


def test_same_source_event_phrase_repeated_at_different_times_is_preserved():
    a = _cue("utt-01", "SPK_A", source_ids=["source-turn-one"])
    b = _cue("utt-01", "SPK_A", start=14.2, end=16.2, source_ids=["source-turn-one"])
    kept, overflow, suppressed = assign_subtitle_lanes([a, b])
    assert len(kept) == 2 and not overflow and not suppressed


def test_legacy_distinct_utterance_ids_also_cannot_be_silently_deleted():
    a = _cue("utt-01", "SPK_A")
    b = _cue("utt-02", "SPK_B", start=12.1, end=13.9)
    kept, overflow, suppressed = assign_subtitle_lanes([a, b])
    assert len(kept) == 2 and not overflow and not suppressed


def test_source_turn_ids_reach_unrenderable_and_normal_cues():
    base = {
        "id": "utt-1", "source_turn_ids": ["src-1", "src-2"],
        "speaker_id": "SPK_A",
        "source_text": "Bonjour", "translated_text": "Hello",
        "source_start": 2.0, "source_end": 3.0,
    }
    normal = build_display_cues([base])
    assert normal and all(c.source_turn_ids == ["src-1", "src-2"] for c in normal)
    untimed = {**base, "source_start": None, "source_end": None}
    unresolved = build_display_cues([untimed])
    assert unresolved[0].unrenderable
    assert unresolved[0].source_turn_ids == ["src-1", "src-2"]


def test_translation_regrouping_cannot_merge_distinct_source_turns():
    utterances = [
        {"id": "u1", "speaker_id": "SPK_A", "source_start": 1,
         "source_end": 2, "source_turn_id": "src-A",
         "source_text": "Oui", "translated_text": "Yes."},
        {"id": "u2", "speaker_id": "SPK_A", "source_start": 2.1,
         "source_end": 3, "source_turn_id": "src-B",
         "source_text": "Oui", "translated_text": "Yes.",
         "_grouped_with_prev": True},
    ]
    count = _apply_safe_adjacent_regrouping(utterances)
    assert count == 0
    assert not utterances[1].get("merged_into")
    assert utterances[0]["translated_text"] == utterances[1]["translated_text"] == "Yes."
