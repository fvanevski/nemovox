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
            {"id": "asr_c28_context", "infer_start_s": 1122, "infer_end_s": 1168,
             "raw_response_sha256": "f" * 64},
        ],
        "lexical_observations": [
            {"id": "unit_a", "asr_observation_id": "asr_c28"},
            {"id": "unit_b", "asr_observation_id": "asr_c28_context"},
            {"id": "unit_wrong_audio", "asr_observation_id": "asr_orig"},
        ],
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


def test_reanchored_audio_review_requires_fresh_correct_audio_crop():
    src, plan, q, audited = setup()
    event = audited["events"][1]
    event.update({
        "source_text_verified": True, "source_text": "Faites voir.",
        "source_language": "fr",
        "review_method": "direct_audio_listening",
        "review_evidence_sha256": "f" * 64,
        # Prior auditioned crop covers the OLD event interval.
        "review_audio_crop_start_s": 1131.3,
        "review_audio_crop_end_s": 1136.2,
        "review_audio_crop_sha256": "c" * 64,
    })
    before = reconcile_source_observations(src, plan, q, audited_events=audited)
    previous_id = before["source_turns_audio_review_attested"][0]["source_turn_id"]
    previous_revision = before["source_turns_audio_review_attested"][0]["source_revision_id"]
    # The same acoustic event may have corrected bounds, but an old
    # auditioned WAV cannot substantiate a later source event.
    event["start_s"], event["end_s"] = 1136.8, 1138.0
    with pytest.raises(ValueError, match="outside the review audio crop"):
        reconcile_source_observations(src, plan, q, audited_events=audited)
    event["review_audio_crop_start_s"] = 1136.5
    event["review_audio_crop_end_s"] = 1138.4
    event["review_audio_crop_sha256"] = "d" * 64
    after = reconcile_source_observations(src, plan, q, audited_events=audited)
    changed = after["source_turns_audio_review_attested"][0]
    assert changed["source_turn_id"] == previous_id
    assert changed["source_revision_id"] != previous_revision
    assert changed["review_audio_crop_provenance"]["event_fully_covered_by_review_audio"]
    assert after["diagnostics"]["reviewed_source_turns_with_bounded_audio_crop"] == 1


def test_audio_review_rejects_partial_crop_provenance():
    src, plan, q, audited = setup()
    ev = audited["events"][1]
    ev["review_audio_crop_start_s"] = 1131.0
    with pytest.raises(ValueError, match="requires complete bounds"):
        reconcile_source_observations(src, plan, q, audited_events=audited)


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


def test_cross_chunk_candidate_only_resolves_with_positive_audio_event_unit_provenance():
    src, plan, q, audited = setup()
    reviewed = audited["events"][1]
    reviewed.update({
        "linked_source_unit_ids": ["unit_a", "unit_b"],
        "same_audio_event_verified": True,
        "review_method": "direct_audio_listening",
        "review_evidence_sha256": "f" * 64,
    })
    dossier = reconcile_source_observations(src, plan, q, audited_events=audited)
    corr = dossier["echo_correspondence_hypotheses"][0]
    assert corr["decision"] == "same_audio_event_reviewed_keep_observation_lineage"
    assert corr["positively_verified_event_id"] == "c28_heard"
    assert corr["source_unit_observations_deleted"] == 0

    reviewed.pop("same_audio_event_verified")
    with pytest.raises(ValueError, match="positive independent"):
        reconcile_source_observations(src, plan, q, audited_events=audited)


def test_same_lexical_phrase_reviewed_in_two_distinct_audio_events_is_not_deleted():
    src, plan, q, audited = setup()
    first = audited["events"][1]
    second = copy.deepcopy(first)
    second["id"] = "second_voice"
    second["start_s"], second["end_s"] = 1136, 1140
    first.update({
        "linked_source_unit_ids": ["unit_a"], "same_audio_event_verified": True,
        "review_method": "direct_audio_listening", "review_evidence_sha256": "f" * 64,
    })
    second.update({
        "linked_source_unit_ids": ["unit_b"], "same_audio_event_verified": True,
        "review_method": "direct_audio_listening", "review_evidence_sha256": "e" * 64,
    })
    audited["events"].append(second)
    result = reconcile_source_observations(src, plan, q, audited_events=audited)
    assert result["echo_correspondence_hypotheses"][0]["decision"] == "distinct_audio_events_reviewed_preserve_both"
    assert result["diagnostics"]["source_unit_observations_deleted"] == 0


def test_reviewed_unit_cannot_be_assigned_to_an_event_outside_its_audio_input():
    src, plan, q, audited = setup()
    audited["events"][1].update({
        "linked_source_unit_ids": ["unit_wrong_audio"],
        "same_audio_event_verified": True,
        "review_method": "direct_audio_listening",
        "review_evidence_sha256": "f" * 64,
    })
    with pytest.raises(ValueError, match="no parent audio-window overlap"):
        reconcile_source_observations(src, plan, q, audited_events=audited)


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
