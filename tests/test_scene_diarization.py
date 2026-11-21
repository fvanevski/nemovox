"""CPU unit tests for experimental scene-bounded diarization adapter (Phase D)."""

from __future__ import annotations

from nvidia_transcribe.scene_diarization import (
    EVIDENCE_SOURCE,
    build_scene_speaker_evidence,
    plan_bounded_scenes,
)


def test_plan_bounded_scenes_merges_short_tail() -> None:
    scenes = plan_bounded_scenes(255.0, scene_seconds=120.0, min_tail_seconds=30.0)
    assert [(s.scene_id, s.start_s, s.end_s) for s in scenes] == [
        ("scene_000", 0.0, 120.0),
        ("scene_001", 120.0, 255.0),
    ]


def test_scene_evidence_consistent_and_unmatched_music() -> None:
    diar = [
        {"speaker": 0, "start": 10.0, "end": 14.0},
        {"speaker": 1, "start": 16.0, "end": 18.5},  # music FP with zero canonical speech
    ]
    canon = [
        {"global_speaker_id": "SPK_000", "source_start": 10.1, "source_end": 13.9},
    ]
    ev = build_scene_speaker_evidence(
        scene_id="scene_test",
        scene_start_s=0.0,
        scene_end_s=30.0,
        diar_segments=diar,
        canonical_utterances=canon,
        model_revision="local-test-rev",
    )
    assert ev["evidence_source"] == EVIDENCE_SOURCE
    by_ch = {c["local_speaker_channel"]: c for c in ev["channels"]}

    c0 = by_ch["diar_spk_0"]
    assert c0["unresolved_status"] == "consistent_support"
    assert c0["supported_identity_candidates"][0]["global_speaker_id"] == "SPK_000"
    assert c0["contradictions"] == []
    assert c0["authoritative_override_applied"] is False

    c1 = by_ch["diar_spk_1"]
    assert c1["unresolved_status"] == "unmatched_acoustic_activity"
    assert c1["contradictions"][0]["type"] == "no_canonical_speech_overlap"


def test_scene_evidence_detects_collapsed_child_voices_and_never_forces_merge() -> None:
    # Two distinct diarization channels overlap different turns collapsed under SPK_000.
    diar = [
        {"speaker": 0, "start": 200.0, "end": 204.0},
        {"speaker": 1, "start": 205.0, "end": 209.0},
    ]
    canon = [
        {"global_speaker_id": "SPK_000", "source_start": 200.1, "source_end": 203.9},
        {"global_speaker_id": "SPK_000", "source_start": 205.1, "source_end": 208.9},
    ]
    ev = build_scene_speaker_evidence(
        scene_id="F1_child_voices",
        scene_start_s=195.0,
        scene_end_s=235.0,
        diar_segments=diar,
        canonical_utterances=canon,
        model_revision="local-test-rev",
    )
    for ch in ev["channels"]:
        assert ch["unresolved_status"] == "ambiguous_many_to_one"
        assert any(c["type"] == "multiple_channels_for_canonical_speaker" for c in ch["contradictions"])
        assert ch["authoritative_override_applied"] is False
    assert ev["advisory_flags"][0]["type"] == "inspect_possible_collapsed_speaker"
    assert ev["advisory_flags"][0]["action_taken"] == "none_experimental_evidence_only"


def test_scene_evidence_detects_one_channel_spanning_two_speakers() -> None:
    diar = [{"speaker": 0, "start": 1.0, "end": 6.0}]
    canon = [
        {"global_speaker_id": "SPK_000", "source_start": 1.0, "source_end": 3.4},
        {"global_speaker_id": "SPK_001", "source_start": 3.5, "source_end": 6.0},
    ]
    ev = build_scene_speaker_evidence(
        scene_id="rapid_turns",
        scene_start_s=0.0,
        scene_end_s=10.0,
        diar_segments=diar,
        canonical_utterances=canon,
        model_revision="local-test-rev",
    )
    ch0 = ev["channels"][0]
    assert ch0["unresolved_status"] == "ambiguous_one_to_many"
    assert ch0["contradictions"][0]["type"] == "channel_spans_multiple_canonical_speakers"
