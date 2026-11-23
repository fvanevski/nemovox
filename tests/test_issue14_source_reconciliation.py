"""Positive audio-event identity, model hallucination and no-silent-deletion gates."""
import copy

import pytest

from nvidia_transcribe.source_reconciliation import reconcile_source_observations
from nvidia_transcribe.source_recovery import (
    plan_source_recovery, project_source_turn_to_v1_utterance,
)


def source():
    return {
        "schema_version": "nemovox-source-evidence/v1",
        "source_language": "fr",
        "media_fingerprint_sha256": "fingerprint",
        "asr_observations": [
            {"id": "asr_orig", "infer_start_s": 25, "infer_end_s": 45,
             "raw_response_sha256": "b" * 64},
            {"id": "asr_c28", "infer_start_s": 1100, "infer_end_s": 1154,
             "raw_response_sha256": "c" * 64},
        ],
        "lexical_observations": [],
        "cross_chunk_correspondence_candidates": [{
            "id": "corr_false", "left_observation_unit_ids": ["unit_a"],
            "right_observation_unit_ids": ["unit_b"],
            "ctc_relative_offset_hypothesis_s": -15.3643,
            "repeated_phrase_ambiguity": False,
        }],
    }


def setup():
    base = source()
    plan = plan_source_recovery(base, coverage_floor={"start_s": 0, "end_s": 1200}, max_requests=70)
    req = plan["requests"][1]
    data = {
        "schema": "nemovox-challenger-observations/v1",
        "media_fingerprint_sha256": "fingerprint",
        "source_language": "fr",
        "model": {"repo_id": "Qwen/Qwen3-ASR-1.7B", "revision": "a"*40,
                  "pinned_language": "French"},
        "windows": [{
            "request_id": req["id"],
            "start_s": req["start_s"], "end_s": req["end_s"],
            "audio_crop_sha256": "d"*64, "raw_response_sha256": "e"*64,
            "source_text": "Paroles inventées", "finish_status": "completed",
        }],
    }
    audited = {
        "schema": "nemovox-audited-source-events/v1",
        "media_fingerprint_sha256": "fingerprint",
        "events": [{
            "id": "music_only", "start_s": 25, "end_s": 40, "kind": "singing",
            "acoustic_truth": "independently_auditioned",
            "source_text_verified": False,
        }, {
            "id": "c28_heard", "start_s": 1131.5, "end_s": 1136,
            "kind": "speech", "acoustic_truth": "independently_auditioned",
            "source_text_verified": False,
        }],
    }
    return base, plan, data, audited


def test_qwen_window_is_context_not_source_turn_and_false_song_stays_unaccepted():
    src, plan, q, audited = setup()
    result = reconcile_source_observations(src, plan, q, audited_events=audited)
    assert result["diagnostics"]["model_only_source_turns_automatically_accepted"] == 0
    assert result["source_turns_audio_review_attested"] == []
    assert result["model_windows"][0]["source_turn_id"] is None
    assert not result["model_windows"][0]["source_content_accepted"]
    assert result["echo_correspondence_hypotheses"][0]["decision"].startswith("preserve_both")
    assert len(result["unexecuted_request_ids"]) == len(plan["requests"]) - 1
    assert result["policy"]["source_fidelity_certified"] is False


def test_verbatim_independent_audio_event_may_be_projected_but_not_word_aligned():
    src, plan, q, audited = setup()
    event = audited["events"][1]
    event.update({
        "source_text_verified": True, "source_text": "Attention !",
        "source_language": "fr",
        "review_method": "direct_audio_listening", "review_evidence_sha256": "f"*64,
    })
    out = reconcile_source_observations(src, plan, q, audited_events=audited)
    assert len(out["source_turns_audio_review_attested"]) == 1
    turn = out["source_turns_audio_review_attested"][0]
    assert turn["source_status"] == "audio_review_attested_source"
    assert turn["word_alignment_status"] == "unverified"
    projection = project_source_turn_to_v1_utterance(turn)
    assert projection["source_turn_id"] == turn["source_turn_id"]
    assert projection["speaker_id"] == "SPK_UNK"
    assert projection["source_start"] == pytest.approx(1131.5)


def test_stable_source_event_id_across_observation_text_revision():
    src, plan, q, audited = setup()
    ev = audited["events"][1]
    ev.update({
        "source_text_verified": True, "source_text": "Attention !",
        "source_language": "fr",
        "review_method": "direct_audio_listening", "review_evidence_sha256": "f"*64,
    })
    a = reconcile_source_observations(src, plan, q, audited_events=audited)
    ev["source_text"] = "Attention, vite !"
    b = reconcile_source_observations(src, plan, q, audited_events=audited)
    assert a["source_turns_audio_review_attested"][0]["source_turn_id"] == b["source_turns_audio_review_attested"][0]["source_turn_id"]
    assert a["source_turns_audio_review_attested"][0]["source_revision_id"] != b["source_turns_audio_review_attested"][0]["source_revision_id"]


def test_no_human_review_attestation_accepted_from_model_confidence():
    src, plan, q, audited = setup()
    event = audited["events"][1]
    event.update({"source_text_verified": True, "source_text": "Attention !",
                  "review_evidence_sha256": "f"*64})
    with pytest.raises(ValueError, match="direct audio review"):
        reconcile_source_observations(src, plan, q, audited_events=audited)


def test_reject_wrong_language_and_media():
    src, plan, q, audited = setup()
    q["model"]["pinned_language"] = "Chinese"
    with pytest.raises(ValueError, match="pinned"):
        reconcile_source_observations(src, plan, q, audited_events=audited)
    src, plan, q, audited = setup()
    q["media_fingerprint_sha256"] = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        reconcile_source_observations(src, plan, q, audited_events=audited)


def test_reject_responses_for_deferred_or_wrong_crop():
    src, plan, q, audited = setup()
    p = plan["requests"][1]
    q["windows"][0]["end_s"] = p["end_s"] + 1
    with pytest.raises(ValueError, match="crop"):
        reconcile_source_observations(src, plan, q, audited_events=audited)
    src, plan, q, audited = setup()
    plan["unplanned_due_to_budget"] = [plan["requests"].pop()]
    q["windows"][0]["request_id"] = plan["unplanned_due_to_budget"][0]["id"]
    with pytest.raises(ValueError, match="deferred"):
        reconcile_source_observations(src, plan, q, audited_events=audited)


def test_incomplete_model_generation_does_not_promote_partial_words():
    src, plan, q, audited = setup()
    q["windows"][0]["finish_status"] = "incomplete"
    out = reconcile_source_observations(src, plan, q, audited_events=audited)
    assert out["model_windows"][0]["status"] == "incomplete_generation_do_not_accept"
    assert out["diagnostics"]["challenger_windows_incomplete"] == 1


def test_audio_event_ids_not_deduced_from_similar_model_text():
    src, plan, q, audited = setup()
    e = copy.deepcopy(audited["events"][1])
    e["id"] = "another_voice"
    e["start_s"] = 1132.0
    e["end_s"] = 1134.0
    audited["events"].append(e)
    out = reconcile_source_observations(src, plan, q, audited_events=audited)
    assert len(out["audited_audio_regions"]) == 3
    assert out["audited_audio_regions"][-2]["audio_event_id"] != out["audited_audio_regions"][-1]["audio_event_id"]
    assert all(r["cross_chunk_echo_decision"].startswith("unresolved") for r in out["audited_audio_regions"])
