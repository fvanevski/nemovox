"""Issue #14 source-recovery contracts: audibility, coverage and lineage."""
import pytest

from nvidia_transcribe.source_recovery import (
    assemble_source_turn_revision, inspect_forced_alignment,
    plan_source_recovery, project_source_turn_to_v1_utterance,
)


def ledger():
    return {
        "schema_version": "nemovox-source-evidence/v1",
        "media_fingerprint_sha256": "media",
        "source_language": "fr",
        "asr_observations": [
            {"id": "asr_1", "infer_start_s": 1100.0, "infer_end_s": 1154.0}
        ],
        "lexical_observations": [{
            "asr_observation_id": "asr_1",
            "ctc_start_s": 1130.0, "ctc_end_s": 1131.0,
            "timing_status": "provisional_ctc",
        }],
    }


def activity():
    return {
        "schema": "nemovox-audio-activity-evidence/v1",
        "media_fingerprint_sha256": "media",
        "intervals": [
            {"id": "fire_red_01", "source": "FireRedVAD-AED", "start_s": 1131.5, "end_s": 1136, "kind": "speech"},
            {"id": "song", "source": "FireRedVAD-AED", "start_s": 25, "end_s": 40, "kind": "singing"},
        ],
    }


def test_c28_incomplete_chunk_is_tiled_without_vad_and_never_accepted():
    plan = plan_source_recovery(
        ledger(), incomplete=[{
            "infer_start_s": 1100.14, "infer_end_s": 1154.14,
            "reason": "generation_limit_exhausted", "raw_response_sha256": "a" * 64,
        }]
    )
    req = [r for r in plan["requests"] if r["reason"] == "generation_incomplete"]
    assert len(req) == 3
    assert req[0]["start_s"] == pytest.approx(1100.14)
    assert req[-1]["end_s"] == pytest.approx(1154.14)
    assert all(r["primary_retry_required"] and not r["source_content_accepted"] for r in req)
    assert all(r["challenger_language"] == "French" for r in req)


def test_independently_heard_vad_negative_voice_has_recovery_request():
    plan = plan_source_recovery(
        ledger(), activity=activity(),
        independently_auditioned_misses=[{
            "id": "W04c_heard", "start_s": 1261.5, "end_s": 1263.5,
            "acoustic_truth": "independently_auditioned",
        }],
    )
    assert any(r["reason"] == "acoustic_reference_omission"
               and r["start_s"] == 1261.5 for r in plan["requests"])
    assert not any(r["start_s"] == 25 for r in plan["requests"])
    assert any(r["reason"] == "activity_without_timed_source_hypothesis"
               and r["start_s"] == 1131.5 for r in plan["requests"])


def test_model_timing_hypothesis_is_not_verified_word_coverage():
    a = activity()
    a["intervals"][0]["start_s"] = 1130.0
    a["intervals"][0]["end_s"] = 1131.0
    plan = plan_source_recovery(ledger(), activity=a)
    assert plan["requests"] == []
    assert not plan["diagnostics"]["source_fidelity_certified"]


def test_budget_limit_reports_unserved_gap_instead_of_silently_dropping():
    plan = plan_source_recovery(
        ledger(), max_requests=1,
        incomplete=[{
            "infer_start_s": 1100.0, "infer_end_s": 1154.0,
            "reason": "generation_limit_exhausted", "raw_response_sha256": "a" * 64,
        }],
    )
    assert plan["status"] == "partial_budget"
    assert plan["diagnostics"]["pending_requests"] == 2
    assert len(plan["unplanned_due_to_budget"]) == 2


def test_independent_coverage_floor_catches_joint_vad_false_negative_without_inventing_words():
    plan = plan_source_recovery(ledger(), coverage_floor={"start_s": 1250, "end_s": 1274})
    assert len(plan["requests"]) == 1
    request = plan["requests"][0]
    assert request["start_s"] == 1250
    assert request["end_s"] == 1274
    assert request["reason"] == "independent_coverage_probe"
    assert not request["source_content_accepted"]
    assert not plan["diagnostics"]["source_fidelity_certified"]


def test_full_media_coverage_floor_reports_budget_unserved_regions():
    plan = plan_source_recovery(
        ledger(), coverage_floor={"start_s": 0, "end_s": 1379.9},
        max_requests=30,
    )
    assert plan["status"] == "partial_budget"
    assert plan["diagnostics"]["pending_requests"] > 0
    assert plan["requests"][0]["start_s"] == 0
    assert plan["unplanned_due_to_budget"][-1]["end_s"] == pytest.approx(1379.9)


def test_floor_precedes_activity_fanout_and_coalesces_multiple_detector_proposals():
    base = ledger()
    base["lexical_observations"] = []
    evidence = {
        "schema": "nemovox-audio-activity-evidence/v1",
        "media_fingerprint_sha256": "media",
        "intervals": [
            {"id": f"detector_{i}", "source": f"model_{i}",
             "kind": "speech", "start_s": i * 0.2, "end_s": i * 0.2 + 3}
            for i in range(50)
        ],
    }
    result = plan_source_recovery(
        base, activity=evidence, coverage_floor={"start_s": 0, "end_s": 100},
        max_requests=6
    )
    # Five contiguous independent floor windows and one coalesced activity
    # region, rather than 50 detector requests crowding out the floor.
    assert result["diagnostics"]["coverage_floor_requests"] == 5
    assert result["diagnostics"]["coverage_floor_selected"] == 5
    assert result["diagnostics"]["coverage_floor_pending"] == 0
    assert result["diagnostics"]["activity_requests"] == 1
    assert result["status"] == "complete_proposals"
    activity_req = next(r for r in result["requests"]
                        if r["reason"] == "activity_without_timed_source_hypothesis")
    assert len(activity_req["trigger_evidence_ids"]) == 50


def test_floor_short_budget_reports_missing_audio_windows_before_activity():
    base = ledger()
    result = plan_source_recovery(
        base, coverage_floor={"start_s": 0, "end_s": 100}, max_requests=3
    )
    assert result["status"] == "partial_budget"
    assert result["diagnostics"]["coverage_floor_pending"] == 2
    assert all(r["reason"] == "independent_coverage_probe"
               for r in result["unplanned_due_to_budget"])


def test_different_heard_events_are_not_coalesced_by_overlap():
    result = plan_source_recovery(
        ledger(), independently_auditioned_misses=[
            {"id": "first_voice", "acoustic_truth": "independently_auditioned",
             "start_s": 10, "end_s": 14},
            {"id": "second_voice", "acoustic_truth": "independently_auditioned",
             "start_s": 12, "end_s": 16},
        ]
    )
    assert len(result["requests"]) == 2
    assert {tuple(r["trigger_evidence_ids"]) for r in result["requests"]} == {
        ("first_voice",), ("second_voice",)
    }


def test_activity_media_mismatch_fails_closed():
    a = activity()
    a["media_fingerprint_sha256"] = "other"
    with pytest.raises(ValueError, match="fingerprint"):
        plan_source_recovery(ledger(), activity=a)


def test_forced_aligner_false_text_on_sfx_is_never_acoustic_verification():
    result = inspect_forced_alignment([
        {"start_s": 1200.0, "end_s": 1207.36},
        {"start_s": 1207.36, "end_s": 1207.6},
    ], source_region={"start_s": 1200.0, "end_s": 1216.0})
    assert result["status"] == "rejected"
    assert not result["text_verified"]


def test_w04c_bounded_alignment_can_only_be_structurally_provisional():
    result = inspect_forced_alignment([
        {"start_s": 1261, "end_s": 1261.4},
        {"start_s": 1261.5, "end_s": 1261.9},
        {"start_s": 1262.0, "end_s": 1262.3},
    ], source_region={"start_s": 1261.0, "end_s": 1264.0},
    acoustic_pauses=[{"start_s": 1262.9, "end_s": 1264.0}])
    assert result["status"] == "structurally_provisional"
    assert result["timestamp_is_acoustic_truth"] is False


def test_false_mandarin_characters_can_be_structurally_provisional():
    result = inspect_forced_alignment([
        {"start_s": 1200+i*.2, "end_s": 1200+(i+1)*.2} for i in range(8)
    ], source_region={"start_s": 1200, "end_s": 1216})
    assert result["status"] == "structurally_provisional"
    assert result["text_verified"] is False


def test_source_turn_ids_audio_event_bound_and_source_lineage_preserved():
    r1 = assemble_source_turn_revision(
        ledger(), audio_event_id="heard_c1", chosen_observation_id="asr_1",
        candidate_source_text="Encore !", selection_basis="multi_asr_provisional",
    )
    r2 = assemble_source_turn_revision(
        ledger(), audio_event_id="heard_c2", chosen_observation_id="asr_1",
        candidate_source_text="Encore !", selection_basis="multi_asr_provisional",
    )
    assert r1["source_turn_id"] != r2["source_turn_id"]
    assert r1["source_observation_ids"] == ["asr_1"]
    assert r1["source_status"] == "model_source_candidate_provisional"
    assert r1["source_start"] is None


def test_zero_duration_forced_alignment_rejected_not_crashed():
    result = inspect_forced_alignment([
        {"start_s": 1261.1, "end_s": 1261.1},
        {"start_s": 1261.2, "end_s": 1261.2},
        {"start_s": 1261.4, "end_s": 1261.6},
    ], source_region={"start_s": 1261, "end_s": 1264})
    assert result["status"] == "rejected"
    assert "excessive_zero_duration" in result["reasons"]


def test_distinct_candidate_texts_have_same_event_id_but_different_revision_ids():
    old = assemble_source_turn_revision(
        ledger(), audio_event_id="event_1", chosen_observation_id="asr_1",
        candidate_source_text="Bonjour", selection_basis="legacy_provisional"
    )
    revised = assemble_source_turn_revision(
        ledger(), audio_event_id="event_1", chosen_observation_id="asr_1",
        candidate_source_text="Bonsoir", selection_basis="multi_asr_provisional"
    )
    assert old["source_turn_id"] == revised["source_turn_id"]
    assert old["source_revision_id"] != revised["source_revision_id"]


def test_v1_projection_keeps_source_revision_and_explicit_unknown_timing():
    revision = assemble_source_turn_revision(
        ledger(), audio_event_id="heard_original", chosen_observation_id="asr_1",
        candidate_source_text="Le capitaine arrive", selection_basis="multi_asr_provisional"
    )
    utt = project_source_turn_to_v1_utterance(revision)
    assert utt["source_turn_id"] == revision["source_turn_id"]
    assert utt["source_revision_id"] == revision["source_revision_id"]
    assert utt["source_observation_ids"] == ["asr_1"]
    assert utt["source_start"] is None and utt["source_end"] is None
    assert utt["speaker_id"] == "SPK_UNK"
    assert utt["source_acceptance_status"] == "model_source_candidate_provisional"


def test_source_selection_cannot_use_unproven_observation_identity():
    with pytest.raises(ValueError, match="not present"):
        assemble_source_turn_revision(
            ledger(), audio_event_id="x", chosen_observation_id="asr_other",
            candidate_source_text="Oui", selection_basis="legacy_provisional",
        )
