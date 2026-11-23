"""CPU-only regression tests for the Issue #14 model-challenge scorer."""

import pytest

from evaluations.issue14.model_challenge.score import score


def ref():
    return {
        "schema": "nemovox-issue14-acoustic-reference/v1",
        "media_fingerprint_sha256": "media-a",
        "regions": [{
            "id": "H2",
            "start_s": 360,
            "end_s": 380,
            "acoustic_truth": "independently_auditioned",
            "events": [
                {"start_s": 362, "end_s": 368, "kind": "speech"},
                {"start_s": 370, "end_s": 374, "kind": "sfx"},
            ],
        }, {
            "id": "song",
            "start_s": 25,
            "end_s": 40,
            "acoustic_truth": "independently_auditioned",
            "events": [{"start_s": 26, "end_s": 39, "kind": "singing"}],
        }],
    }


def pred():
    return {
        "schema": "nemovox-issue14-model-predictions/v1",
        "media_fingerprint_sha256": "media-a",
        "model": {"repo_id": "test/model", "revision": "a" * 40},
        "regions": [{
            "id": "H2", "status": "ok",
            "observations": [
                {"start_s": 364, "end_s": 367, "kind": "speech"},
                {"start_s": 371, "end_s": 373, "kind": "speech"},
            ],
        }, {
            "id": "song", "status": "ok",
            "observations": [{"start_s": 27, "end_s": 38, "kind": "singing"}],
        }],
    }


def test_event_proxy_measures_missing_coverage_and_sfx_false_alarm_separately():
    result = score(ref(), pred())
    assert result["scored_regions"] == 2
    assert result["summary"]["reference_speech_events"] == 1
    assert result["summary"]["event_hit_any_overlap"] == 1
    assert result["summary"]["reference_speech_covered_s"] == 3
    assert result["summary"]["non_speech_only_predicted_speech_s"] == 2
    assert result["regions"][0]["event_coverage_fractions"] == [0.5]
    assert not result["regions"][0]["source_word_accuracy_validated"]


def test_unavailable_region_is_not_implicitly_silence():
    p = pred()
    p["regions"][0] = {"id": "H2", "status": "blocked", "reason": "gpu_ownership"}
    result = score(ref(), p)
    assert result["scored_regions"] == 1
    assert result["unscored_regions"] == 1
    assert result["summary"]["reference_speech_events"] == 0


def test_identity_and_schema_fail_closed():
    p = pred()
    p["media_fingerprint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        score(ref(), p)
    p = pred()
    p["regions"].pop()
    with pytest.raises(ValueError, match="each reference region"):
        score(ref(), p)


def test_reject_outside_window_and_unverified_reference():
    p = pred()
    p["regions"][0]["observations"][0]["start_s"] = 355
    with pytest.raises(ValueError, match="outside region"):
        score(ref(), p)
    p = pred()
    r = ref()
    r["regions"][0]["acoustic_truth"] = "unverified"
    result = score(r, p)
    assert result["regions"][0]["status"] == "reference_unverified"
    assert result["scored_regions"] == 1


def test_double_counting_overlapping_intervals_is_prevented():
    p = pred()
    p["regions"][0]["observations"] = [
        {"start_s": 362, "end_s": 367, "kind": "speech"},
        {"start_s": 365, "end_s": 368, "kind": "speech"},
    ]
    result = score(ref(), p)
    assert result["summary"]["reference_speech_covered_s"] == 6
    assert result["summary"]["non_speech_only_predicted_speech_s"] == 0
