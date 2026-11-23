"""Issue #14 AudioRegion planning: no cross-voice merges, no word invention."""
import pytest
from nvidia_transcribe.audio_regions import plan_audio_regions


def audio():
    return {
        "schema": "nemovox-audio-activity-evidence/v1",
        "media_fingerprint_sha256": "original",
        "intervals": [
            {"id": "speech", "source": "FireRedAED", "kind": "speech",
             "start_s": 1131.5, "end_s": 1136},
            {"id": "song", "source": "FireRedAED", "kind": "singing",
             "start_s": 25, "end_s": 40},
            {"id": "music", "source": "FireRedAED", "kind": "music",
             "start_s": 1100, "end_s": 1140},
        ],
    }


def plan(**kwargs):
    return plan_audio_regions(media_fingerprint_sha256="original",
                              source_language="fr", media_duration_s=1380,
                              activity=audio(), **kwargs)


def test_source_music_not_promoted_and_c28_proposal_carries_risk():
    result = plan()
    assert len(result["regions"]) == 1
    r = result["regions"][0]
    assert r["speaker_identity_accepted"] is False
    assert r["model_text_accepted"] is False
    assert r["speech_activity_ids"] == ["speech"]
    assert "music" in r["non_speech_activity_ids"]
    assert (r["bounded_crop_start_s"], r["bounded_crop_end_s"]) == (1128.5, 1139.0)


def test_independent_auditioned_vad_negative_event_is_a_separate_proposal():
    audits = {
        "schema": "nemovox-audited-source-events/v1",
        "media_fingerprint_sha256": "original",
        "events": [{"id": "W04c", "start_s": 1261.5, "end_s": 1263.5,
                    "kind": "speech", "acoustic_truth": "independently_auditioned"}]
    }
    out = plan(audited_events=audits)
    assert any(r["audio_event_id"] == "W04c" for r in out["regions"])
    assert out["diagnostics"]["auditioned_source_events"] == 1


def test_scene_overlap_keeps_two_channels_and_no_global_speaker_identity():
    scene = {
        "schema": "nemovox-scene-turn-evidence/v1",
        "media_fingerprint_sha256": "original",
        "turns": [
            {"id": "elder_1", "speaker_channel": "local_A",
             "start_s": 518, "end_s": 522},
            {"id": "younger", "speaker_channel": "local_B",
             "start_s": 520, "end_s": 524},
        ],
    }
    out = plan(scene_tracks=scene)
    assert len(out["regions"]) == 2
    assert out["regions"][0]["speaker_channel_hypothesis"] == "local_A"
    assert out["regions"][1]["speaker_channel_hypothesis"] == "local_B"
    assert all(r["audio_event_id"] is None for r in out["regions"])


def test_challenger_full_window_text_not_split_into_fake_words():
    dossier = {
        "schema": "nemovox-source-reconciliation/v1",
        "media_fingerprint_sha256": "original",
        "model_windows": [{
            "id": "qwen_57", "start_s": 1254, "end_s": 1278,
            "status": "window_only_unlocalized_hypothesis",
            "source_content_accepted": False,
            "source_text": "Some hypothetical recognized sentence",
        }],
    }
    out = plan(dossier=dossier)
    assert len(out["unlocalized_model_windows"]) == 1
    assert out["unlocalized_model_windows"][0]["model_context_window_id"] == "qwen_57"
    assert all(r["model_text_accepted"] is False for r in out["regions"])


def test_audio_event_too_long_requires_split_not_false_single_turn():
    evidence = audio()
    evidence["intervals"].append({
        "id": "long", "source": "Silero", "kind": "speech",
        "start_s": 50, "end_s": 88,
    })
    out = plan_audio_regions(media_fingerprint_sha256="original",
                             source_language="fr", media_duration_s=1380,
                             activity=evidence)
    long = next(r for r in out["regions"] if r["source_id"] == "long")
    assert long["needs_subsegmentation"] is True
    assert long["bounded_crop_start_s"] is None


def test_different_media_evidence_fail_closed():
    wrong = audio()
    wrong["media_fingerprint_sha256"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        plan_audio_regions(media_fingerprint_sha256="original",
                           source_language="fr", media_duration_s=1380,
                           activity=wrong)
