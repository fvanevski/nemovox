"""Issue #14: immutable unit identity is required before replacing legacy speech."""
from copy import deepcopy

import pytest

from nvidia_transcribe.source_promotion import promote_bounded_source_turns
from nvidia_transcribe.source_recovery import _id
from nvidia_transcribe.source_unit_lineage import bind_legacy_source_lineage


MEDIA = "same_recording"
SHA = "a" * 64


def unit(word, idx):
    return {
        "text": word, "chunk_index": 0,
        "source_unit_index": idx, "unit_type": "word",
    }


def fixtures():
    recording = {
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"fingerprint_sha256": MEDIA, "duration_s": 100.0},
        "source_language": "fr",
        "processing": {"chunks": [
            {"chunk_index": 0, "raw_response_sha256": SHA,
             "infer_start_s": 0.0, "infer_end_s": 40.0},
        ]},
        "utterances": [{
            "id": "legacy_a", "source_text": "Bonjour monde",
            "source_start": 10.0, "source_end": 12.0,
            "source_words": [unit("Bonjour", 0), unit("monde", 1)],
        }, {
            "id": "legacy_b", "source_text": "Other dialogue",
            "source_start": 20.0, "source_end": 22.0,
            "source_words": [unit("Other", 2), unit("dialogue", 3)],
        }],
    }
    lexical = []
    for idx, word in enumerate(["Bonjour", "monde", "Other", "dialogue"]):
        lexical.append({
            "id": "lex_" + str(idx),
            "chunk_index": 0, "source_ordinal": idx,
            "unit_type": "word", "text": word,
        })
    ledger = {
        "schema_version": "nemovox-source-evidence/v1",
        "media_fingerprint_sha256": MEDIA,
        "source_language": "fr",
        "asr_observations": [{
            "id": "asr_a", "chunk_index": 0,
            "infer_start_s": 0.0, "infer_end_s": 40.0,
            "raw_response_sha256": SHA,
        }],
        "lexical_observations": lexical,
    }
    turn_id = _id("src_turn", [MEDIA, "event_a"])
    dossier = {
        "schema": "nemovox-source-reconciliation/v1",
        "media_fingerprint_sha256": MEDIA,
        "source_language": "fr",
        "challenger_model": {"repo_id": "Qwen/Qwen3-ASR-1.7B",
                             "revision": "b" * 40, "pinned_language": "French"},
        "model_windows": [],
        "source_turns_audio_review_attested": [{
            "schema": "nemovox-source-turn-revision/v1",
            "audio_event_id": "event_a",
            "source_turn_id": turn_id,
            "source_revision_id": _id("src_revision", ["review", "event_a"]),
            "source_start": 10.0, "source_end": 12.0,
            "source_text": "Bonjour monde",
            "source_language": "fr",
            "selection_basis": "independent_audio_review",
            "source_status": "audio_review_attested_source",
            "reviewer_has_attested_direct_audio_listening": True,
            "review_evidence_sha256": SHA,
            "source_observation_ids": ["asr_a"],
        }],
        "audited_audio_regions": [{
            "audio_event_id": "event_a", "id": "review_a",
            "reviewed_source_turn_id": turn_id,
            "start_s": 10.0, "end_s": 12.0,
            "kind": "speech", "source_status": "audio_review_attested",
            "positive_audio_event_unit_ids": ["lex_0", "lex_1"],
        }],
    }
    segments = {
        "schema": "nemovox-bounded-source-segments/v1",
        "media_fingerprint_sha256": MEDIA,
        "source_language": "fr",
        "segments": [{
            "id": "seg_a", "audio_event_id": "event_a",
            "start_s": 10.0, "end_s": 12.0,
            "kind": "speech", "source_text": "Bonjour monde",
            "context_window_ids": [],
        }],
    }
    return recording, ledger, dossier, segments


def test_same_event_source_unit_mapping_all_words_and_source_identity():
    recording, ledger, dossier, _ = fixtures()
    base = deepcopy(recording)
    out = bind_legacy_source_lineage(recording, ledger, dossier)
    assert recording == base
    a, b = out["legacy_utterances"]
    assert a["status"] == "fully_same_event_time_plausible"
    assert a["source_turn_id"] == _id("src_turn", [MEDIA, "event_a"])
    assert a["positively_reviewed_unit_count"] == 2
    assert a["eligible_for_reviewed_same_event_supersession"]
    assert b["status"] == "no_positive_same_event_assignment"
    assert out["diagnostics"]["matched_lexical_units"] == 4
    assert out["diagnostics"]["source_unit_observations_deleted"] == 0


def test_duplicate_reviewed_event_not_added_without_explicit_safe_replacement():
    recording, ledger, dossier, segments = fixtures()
    out = promote_bounded_source_turns(
        recording, dossier, segments, source_evidence=ledger
    )
    assert out["source_promotion"]["new_source_turns"] == 0
    assert len(out["utterances"]) == 2
    assert ("reviewed_same_event_requires_complete_explicit_supersession"
            in out["source_promotion"]["quarantined"][0]["reasons"])


def test_explicit_fully_attributed_legacy_turn_can_be_replaced_once():
    recording, ledger, dossier, segments = fixtures()
    segments["segments"][0]["supersedes_legacy_utterance_ids"] = ["legacy_a"]
    out = promote_bounded_source_turns(
        recording, dossier, segments, source_evidence=ledger
    )
    assert out["source_promotion"]["new_audio_review_attested_turns"] == 1
    assert out["source_promotion"]["superseded_legacy_utterance_count"] == 1
    assert len(out["utterances"]) == 2
    assert "legacy_a" not in {u["id"] for u in out["utterances"]}
    assert "legacy_b" in {u["id"] for u in out["utterances"]}
    assert recording["utterances"][0]["id"] == "legacy_a"


def test_unmapped_or_partial_unit_must_remain_in_legacy_output():
    recording, ledger, dossier, segments = fixtures()
    dossier["audited_audio_regions"][0]["positive_audio_event_unit_ids"] = ["lex_0"]
    report = bind_legacy_source_lineage(recording, ledger, dossier)
    assert report["legacy_utterances"][0]["status"] == "partial_or_multiple_reviewed_audio_events"
    segments["segments"][0]["supersedes_legacy_utterance_ids"] = ["legacy_a"]
    out = promote_bounded_source_turns(
        recording, dossier, segments, source_evidence=ledger
    )
    assert out["source_promotion"]["new_source_turns"] == 0
    assert out["source_promotion"]["preserved_legacy_utterance_count"] == 2
    assert ("reviewed_event_has_partially_mapped_legacy_turns"
            in out["source_promotion"]["quarantined"][0]["reasons"])


def test_cross_run_raw_asr_digest_and_inference_window_cannot_reuse_ordinal():
    recording, ledger, dossier, _ = fixtures()
    other_ledger = deepcopy(ledger)
    other_ledger["asr_observations"][0]["raw_response_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="raw ASR response"):
        bind_legacy_source_lineage(recording, other_ledger, dossier)
    other_ledger = deepcopy(ledger)
    other_ledger["asr_observations"][0]["infer_end_s"] = 39.0
    with pytest.raises(ValueError, match="input window"):
        bind_legacy_source_lineage(recording, other_ledger, dossier)


def test_source_word_text_or_index_mismatch_never_claims_positive_event():
    recording, ledger, dossier, _ = fixtures()
    recording["utterances"][0]["source_words"][1]["text"] = "unrelated"
    out = bind_legacy_source_lineage(recording, ledger, dossier)
    row = out["legacy_utterances"][0]
    assert row["status"] == "incomplete_or_unmatched_legacy_source_units"
    assert not row["eligible_for_reviewed_same_event_supersession"]
    assert row["missing_or_unmatched_unit_count"] == 1


def test_speaker_overlap_or_displaced_ctc_never_proves_same_event_deletion():
    recording, ledger, dossier, _ = fixtures()
    recording["utterances"][0]["source_start"] = 50.0
    recording["utterances"][0]["source_end"] = 55.0
    out = bind_legacy_source_lineage(recording, ledger, dossier)
    row = out["legacy_utterances"][0]
    assert row["status"] == "fully_same_event_timing_displaced_or_missing"
    assert not row["eligible_for_reviewed_same_event_supersession"]


def test_source_media_language_and_ambiguous_raw_id_rejected():
    recording, ledger, dossier, _ = fixtures()
    ledger["source_language"] = "zh"
    with pytest.raises(ValueError, match="source language mismatch"):
        bind_legacy_source_lineage(recording, ledger, dossier)
    recording, ledger, dossier, _ = fixtures()
    ledger["lexical_observations"][1]["source_ordinal"] = 0
    with pytest.raises(ValueError, match="Ambiguous"):
        bind_legacy_source_lineage(recording, ledger, dossier)
