"""Issue #14 candidate promotion into recording-v1 and subtitle pipeline.

Invariants: v1 unchanged by default, bounded source only, model output remains
provisional, false song/SFX not promoted, same-event lineage required for
supersession, and distinct repeated real events cannot collapse by text.
"""
from copy import deepcopy
from pathlib import Path
import hashlib

import pytest

from nvidia_transcribe.source_promotion import promote_bounded_source_turns
from nvidia_transcribe.source_recovery import _id
from nvidia_transcribe.subtitles import render_subtitles
from nvidia_transcribe.translation import TranslationConfig, translate_utterances


MEDIA = "same-media"
LANGUAGE = "fr"
SHA = "a" * 64


def baseline():
    return {
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"fingerprint_sha256": MEDIA, "duration_s": 180.0},
        "source_language": LANGUAGE,
        "utterances": [
            {"id": "legacy_1", "source_text": "Original source words",
             "source_start": 5.0, "source_end": 7.0,
             "speaker_id": "SPK_001", "alignment_status": "ctc_aligned"},
        ],
        "processing": {"chunk_count": 1},
        "events": [],
    }


def dossier():
    reviewed_id = _id("src_turn", [MEDIA, "heard_turn"])
    return {
        "schema": "nemovox-source-reconciliation/v1",
        "media_fingerprint_sha256": MEDIA,
        "source_language": LANGUAGE,
        "source_turns_audio_review_attested": [{
            "schema": "nemovox-source-turn-revision/v1",
            "audio_event_id": "heard_turn",
            "source_turn_id": reviewed_id,
            "source_revision_id": _id("src_revision", ["review", "heard_turn"]),
            "source_start": 30.0, "source_end": 32.0,
            "source_text": "Attention !",
            "source_language": LANGUAGE,
            "selection_basis": "independent_audio_review",
            "source_status": "audio_review_attested_source",
            "reviewer_has_attested_direct_audio_listening": True,
            "review_evidence_sha256": SHA,
            "source_observation_ids": ["asr_a"],
        }],
        "audited_audio_regions": [
            {"audio_event_id": "heard_turn", "id": "audited_speech",
             "start_s": 30.0, "end_s": 32.0, "kind": "speech"},
            {"audio_event_id": "song_only", "id": "audited_song",
             "start_s": 80.0, "end_s": 95.0, "kind": "singing"},
        ],
        "challenger_model": {"repo_id": "Qwen/Qwen3-ASR-1.7B",
                             "revision": "b" * 40, "pinned_language": "French"},
        "model_windows": [
            {"id": "qwen_window_01", "start_s": 22.0, "end_s": 46.0},
            {"id": "qwen_window_02", "start_s": 44.0, "end_s": 68.0},
            {"id": "qwen_window_03", "start_s": 74.0, "end_s": 98.0},
        ],
    }


def activity():
    return {
        "schema": "nemovox-audio-activity-evidence/v1",
        "media_fingerprint_sha256": MEDIA,
        "intervals": [
            {"id": "fr_speech", "source": "FireRedVAD-AED", "kind": "speech",
             "start_s": 49.5, "end_s": 53.0},
            {"id": "fr_music", "source": "FireRedVAD-AED", "kind": "singing",
             "start_s": 80.0, "end_s": 95.0},
        ],
    }


def candidate(eid, start, end, text, **extras):
    return {
        "id": "candidate_" + eid, "audio_event_id": eid,
        "start_s": start, "end_s": end, "kind": "speech",
        "source_text": text, **extras,
    }


def segments(*items):
    return {
        "schema": "nemovox-bounded-source-segments/v1",
        "media_fingerprint_sha256": MEDIA,
        "source_language": LANGUAGE,
        "segments": list(items),
    }


def model_evidence(crop_start, crop_end, text="À demain !", **extras):
    return {
        "model_repo_id": "Qwen/Qwen3-ASR-1.7B",
        "model_revision": "b" * 40,
        "crop_start_s": crop_start, "crop_end_s": crop_end,
        "audio_crop_sha256": "c" * 64,
        "raw_response_sha256": "d" * 64,
        "source_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "finish_status": "completed", "language": "French",
        **extras,
    }


def reviewed_candidate(**extras):
    return candidate("heard_turn", 30, 32, "Attention !",
                     context_window_ids=["qwen_window_01"], **extras)


def provisional_candidate(**extras):
    return candidate(
        "new_model_turn", 50, 52, "À demain !",
        context_window_ids=["qwen_window_02"],
        bounded_asr=model_evidence(48, 56),
        speech_activity_ids=["fr_speech"],
        **extras,
    )


def test_v1_original_unmodified_and_reviewed_source_turn_with_stable_lineage():
    base = baseline()
    original = deepcopy(base)
    doc = promote_bounded_source_turns(
        base, dossier(), segments(reviewed_candidate()), activity=activity()
    )
    assert base == original
    assert doc["schema_version"] == "nemotron-recording-contract/v1"
    assert len(doc["utterances"]) == 2
    assert doc["source_promotion"]["new_audio_review_attested_turns"] == 1
    assert doc["source_promotion"]["source_fidelity_certified"] is False
    row = next(u for u in doc["utterances"] if u["id"] != "legacy_1")
    assert row["source_turn_id"] == _id("src_turn", [MEDIA, "heard_turn"])
    assert row["source_start"] == 30.0 and row["source_end"] == 32.0
    assert row["speaker_id"] == "SPK_UNK"
    assert row["source_words"] == []
    assert row["alignment_status"] == "turn_bounded_word_timing_unverified"


def test_positive_speech_plus_bounded_model_ASR_is_renderable_but_provisional(tmp_path: Path):
    doc = promote_bounded_source_turns(
        baseline(), dossier(), segments(provisional_candidate()),
        activity=activity(),
    )
    assert doc["source_promotion"]["new_model_provisional_turns"] == 1
    added = next(u for u in doc["utterances"] if u["id"] != "legacy_1")
    assert added["source_text"] == "À demain !"
    assert added["source_text_review_attested"] is False
    assert added["source_text_human_reference_verified"] is False
    assert added["source_acceptance_status"] == "model_source_candidate_provisional"
    assert added["speaker_id"] == "SPK_UNK"
    translated, _ = translate_utterances(
        doc["utterances"], source_language="fr",
        config=TranslationConfig(backend="none")
    )
    out = render_subtitles(
        {"utterances": translated}, output_dir=tmp_path, episode_stem="test",
        target_language="en",
    )
    assert out["stats"]["renderable_cues"] >= 2
    assert "À demain !" in (tmp_path / "test.en.srt").read_text(encoding="utf-8")
    assert "À demain !" in (tmp_path / "test.en.ass").read_text(encoding="utf-8")


def test_song_only_lyrics_not_accepted_even_with_qwen_model_text():
    song = candidate("song_model_guess", 80, 90, "Lyrics invented by model",
                     context_window_ids=["qwen_window_03"],
                     bounded_asr=model_evidence(78, 94, "Lyrics invented by model"),
                     speech_activity_ids=[])
    out = promote_bounded_source_turns(
        baseline(), dossier(), segments(song), activity=activity()
    )
    assert out["source_promotion"]["new_source_turns"] == 0
    assert "no_positive_speech_activity_for_model_only_text" in out["source_promotion"]["quarantined"][0]["reasons"]
    assert all("Lyrics" not in u["source_text"] for u in out["utterances"])


def test_false_sfx_voice_activation_plus_qwen_exclamation_is_quarantined():
    candidate_sfx = candidate(
        "sfx_guess", 80, 82, "Ah !",
        context_window_ids=["qwen_window_03"],
        bounded_asr=model_evidence(78, 86, "Ah !"),
        speech_activity_ids=["false_sfx_speech"],
    )
    acoustic = activity()
    acoustic["intervals"].extend([
        {"id": "false_sfx_speech", "source": "FireRedAED", "kind": "speech",
         "start_s": 80, "end_s": 82},
        {"id": "nonlexical", "source": "FireRedAED", "kind": "sfx",
         "start_s": 80, "end_s": 82},
    ])
    out = promote_bounded_source_turns(
        baseline(), dossier(), segments(candidate_sfx), activity=acoustic
    )
    assert out["source_promotion"]["new_source_turns"] == 0
    assert ("high_nonlexical_risk_without_independent_speech_corroboration"
            in out["source_promotion"]["quarantined"][0]["reasons"])
    # Independent corroborating source may remain PROVISIONAL, never human truth.
    acoustic["intervals"].append({
        "id": "silero_second_source", "source": "SileroVAD", "kind": "speech",
        "start_s": 80.2, "end_s": 81.5,
    })
    candidate_sfx["speech_activity_ids"].append("silero_second_source")
    out = promote_bounded_source_turns(
        baseline(), dossier(), segments(candidate_sfx), activity=acoustic
    )
    assert out["source_promotion"]["new_model_provisional_turns"] == 1
    assert out["utterances"][-1]["source_text_human_reference_verified"] is False


def test_audited_singing_kind_conflict_is_rejected():
    lying = candidate("song_only", 80, 95, "False spoken lyric",
                      context_window_ids=["qwen_window_03"])
    with pytest.raises(ValueError, match="reviewed non-speech event"):
        promote_bounded_source_turns(
            baseline(), dossier(), segments(lying), activity=activity()
        )


def test_window_wide_text_without_bounded_asr_is_not_promoted():
    unlocalized = candidate("new_model_turn", 50, 52, "A guessed utterance",
                            context_window_ids=["qwen_window_02"],
                            speech_activity_ids=["fr_speech"])
    out = promote_bounded_source_turns(
        baseline(), dossier(), segments(unlocalized), activity=activity()
    )
    assert out["source_promotion"]["new_source_turns"] == 0
    assert "missing_bounded_asr_for_model_only_text" in out["source_promotion"]["quarantined"][0]["reasons"]


def test_model_only_candidate_cannot_suppress_overlapping_legacy_utterance():
    a = candidate("overlap", 5.2, 6.2, "Different sound",
                  bounded_asr=model_evidence(4, 10, "Different sound"),
                  speech_activity_ids=["fr_overlap"])
    ac = activity()
    ac["intervals"].append({
        "id": "fr_overlap", "source": "Silero", "kind": "speech",
        "start_s": 5.0, "end_s": 7.0,
    })
    result = promote_bounded_source_turns(
        baseline(), dossier(), segments(a), activity=ac
    )
    assert result["source_promotion"]["new_source_turns"] == 0
    assert "model_candidate_collides_with_existing_legacy_speech" in result["source_promotion"]["quarantined"][0]["reasons"]


def test_similar_text_distinct_audio_event_ids_are_both_preserved():
    heard = reviewed_candidate()
    model = provisional_candidate()
    model["source_text"] = heard["source_text"]
    model["bounded_asr"]["source_text_sha256"] = hashlib.sha256(model["source_text"].encode("utf-8")).hexdigest()
    out = promote_bounded_source_turns(
        baseline(), dossier(), segments(heard, model), activity=activity()
    )
    ids = [u.get("source_turn_id") for u in out["utterances"] if u.get("source_turn_id")]
    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert out["source_promotion"]["new_source_turns"] == 2


def test_divergent_asr_candidates_for_same_audio_event_must_not_select_one():
    a = provisional_candidate()
    b = deepcopy(a)
    b["id"] = "candidate_alt"
    b["source_text"] = "Another plausible sentence"
    b["bounded_asr"]["source_text_sha256"] = hashlib.sha256(b["source_text"].encode("utf-8")).hexdigest()
    out = promote_bounded_source_turns(
        baseline(), dossier(), segments(a, b), activity=activity()
    )
    assert out["source_promotion"]["new_source_turns"] == 0
    assert "conflicting_candidates_same_audio_event" in out["source_promotion"]["quarantined"][0]["reasons"]


def test_wrong_model_language_and_audio_crop_are_rejected():
    a = provisional_candidate()
    a["bounded_asr"]["language"] = "Chinese"
    with pytest.raises(ValueError, match="language must be pinned"):
        promote_bounded_source_turns(
            baseline(), dossier(), segments(a), activity=activity()
        )
    a = provisional_candidate()
    a["bounded_asr"]["crop_start_s"] = 51.0
    with pytest.raises(ValueError, match="tightly contain"):
        promote_bounded_source_turns(
            baseline(), dossier(), segments(a), activity=activity()
        )


def test_unbounded_full_context_cannot_impersonate_short_utterance():
    a = provisional_candidate()
    a["bounded_asr"]["crop_start_s"] = 44.0
    a["bounded_asr"]["crop_end_s"] = 68.0
    with pytest.raises(ValueError, match="tightly contain"):
        promote_bounded_source_turns(
            baseline(), dossier(), segments(a), activity=activity()
        )


def test_model_text_edit_without_authentic_transcript_digest_rejected():
    a = provisional_candidate()
    a["source_text"] = "Other model answer"
    with pytest.raises(ValueError, match="authenticated digest"):
        promote_bounded_source_turns(
            baseline(), dossier(), segments(a), activity=activity()
        )


def test_new_model_checkpoint_not_in_reconciliation_dossier_rejected():
    a = provisional_candidate()
    a["bounded_asr"]["model_revision"] = "changed_revision"
    with pytest.raises(ValueError, match="authenticated challenger"):
        promote_bounded_source_turns(
            baseline(), dossier(), segments(a), activity=activity()
        )


def test_opt_out_and_incomplete_model_asr_do_not_render():
    a = provisional_candidate()
    result = promote_bounded_source_turns(
        baseline(), dossier(), segments(a), activity=activity(),
        include_model_provisional=False,
    )
    assert result["source_promotion"]["new_source_turns"] == 0
    assert "model_provisional_not_enabled" in result["source_promotion"]["quarantined"][0]["reasons"]
    a = provisional_candidate()
    a["bounded_asr"]["finish_status"] = "incomplete"
    result = promote_bounded_source_turns(
        baseline(), dossier(), segments(a), activity=activity()
    )
    assert result["source_promotion"]["new_source_turns"] == 0
    assert "bounded_asr_not_complete" in result["source_promotion"]["quarantined"][0]["reasons"]


def test_no_supersession_without_positive_same_event_id():
    a = reviewed_candidate(supersedes_legacy_utterance_ids=["legacy_1"])
    result = promote_bounded_source_turns(
        baseline(), dossier(), segments(a), activity=activity()
    )
    assert result["source_promotion"]["new_source_turns"] == 0
    assert "legacy_turn_lacks_positive_same_event_lineage" in result["source_promotion"]["quarantined"][0]["reasons"]
    assert len(result["utterances"]) == 1


def test_positive_event_lineage_allows_explicit_same_event_supersession():
    original = baseline()
    event_id = _id("src_turn", [MEDIA, "heard_turn"])
    original["utterances"][0]["source_turn_id"] = event_id
    a = reviewed_candidate(supersedes_legacy_utterance_ids=["legacy_1"])
    result = promote_bounded_source_turns(
        original, dossier(), segments(a), activity=activity()
    )
    assert result["source_promotion"]["new_source_turns"] == 1
    assert result["source_promotion"]["superseded_legacy_utterance_count"] == 1
    assert len(result["utterances"]) == 1
    assert result["utterances"][0]["source_turn_id"] == event_id
    assert original["utterances"][0]["id"] == "legacy_1"


def test_mismatched_media_and_reviewed_attestation_rejected():
    doc = dossier()
    doc["media_fingerprint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="media fingerprint mismatch"):
        promote_bounded_source_turns(
            baseline(), doc, segments(reviewed_candidate()), activity=activity()
        )
    doc = dossier()
    doc["source_turns_audio_review_attested"][0]["reviewer_has_attested_direct_audio_listening"] = False
    with pytest.raises(ValueError, match="positive direct audio-review"):
        promote_bounded_source_turns(
            baseline(), doc, segments(reviewed_candidate()), activity=activity()
        )
