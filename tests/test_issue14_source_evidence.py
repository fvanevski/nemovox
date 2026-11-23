"""Transcript-first observation-ledger invariants (CPU only)."""

from dataclasses import dataclass

import pytest

from nvidia_transcribe.source_events import build_source_evidence_ledger


@dataclass(frozen=True)
class Spec:
    chunk_index: int
    core_start_s: float
    core_end_s: float
    infer_start_s: float
    infer_end_s: float


def word(text, start=None, speaker="chunk_000:spk_0", *, valid=True, status=None):
    return {
        "text": text,
        "start": start,
        "end": start + 0.3 if start is not None else None,
        "valid_timing": valid and start is not None,
        "local_speaker_id": speaker,
        "global_speaker_id": "SPK_01",
        "unit_type": "character" if len(text) == 1 and ord(text[0]) > 0x3000 else "word",
        **({"timing_evidence_status": status} if status else {}),
    }


def ledger(specs, units, *, diar=None, sha="a" * 64):
    return build_source_evidence_ledger(
        media_fingerprint_sha256="media-immutable",
        source_language="fr",
        chunk_specs=specs,
        chunk_units=units,
        raw_response_sha256=[sha] * len(specs),
        raw_source_texts=["raw recognized text"] * len(specs),
        diarization_by_chunk=diar if diar is not None else [[] for _ in specs],
    )


def test_multiword_overlap_is_only_a_hypothesis_and_never_deletes_source():
    specs = [Spec(0, 0, 10, 0, 14), Spec(1, 10, 20, 6, 20)]
    phrase = ["the", "silver", "bridge", "returns", "today"]
    left = [word(t, 8 + i * .3) for i, t in enumerate(phrase)]
    right = [word(t, 8 + i * .3, "chunk_001:spk_0") for i, t in enumerate(phrase)]
    result = ledger(specs, [left, right])
    assert len(result["lexical_observations"]) == 10
    assert result["diagnostics"]["source_units_dropped_by_ledger"] == 0
    assert len(result["cross_chunk_correspondence_candidates"]) == 1
    match = result["cross_chunk_correspondence_candidates"][0]
    assert match["decision"] == "preserve_both_observations"
    assert match["status"] == "candidate_only_unverified"
    assert len(set(match["left_observation_unit_ids"] + match["right_observation_unit_ids"])) == 10


def test_no_cross_chunk_duplicate_inference_without_shared_audio():
    specs = [Spec(0, 0, 10, 0, 10), Spec(1, 10, 20, 10, 20)]
    phrase = ["tell", "them", "we", "return"]
    result = ledger(specs, [
        [word(t, 8 + i*.3) for i, t in enumerate(phrase)],
        [word(t, 13 + i*.3) for i, t in enumerate(phrase)],
    ])
    assert result["cross_chunk_correspondence_candidates"] == []
    assert len(result["lexical_observations"]) == 8


def test_displaced_ctc_never_becomes_certified_acoustic_timing():
    specs = [Spec(0, 0, 10, 0, 14), Spec(1, 10, 20, 6, 20)]
    phrase = ["there", "were", "some", "hidden", "voices"]
    left = [word(t, 8 + i*.3) for i, t in enumerate(phrase)]
    right = [word(t, 45 + i*.3, "chunk_001:spk_1") for i, t in enumerate(phrase)]
    result = ledger(specs, [left, right])
    assert len(result["cross_chunk_correspondence_candidates"]) == 1
    assert all(not w["word_alignment_accepted"] for w in result["lexical_observations"])
    assert all(w["timing_status"] == "disputed" for w in result["lexical_observations"][5:])
    assert result["diagnostics"]["source_coverage_certified"] is False


def test_unknown_time_preserves_source_and_does_not_invent_alignment():
    spec = Spec(0, 0, 10, 0, 10)
    result = ledger([spec], [[word("encore", None, valid=False)]])
    unit = result["lexical_observations"][0]
    assert unit["text"] == "encore"
    assert unit["ctc_start_s"] is None and unit["ctc_end_s"] is None
    assert unit["timing_status"] == "unresolved"


def test_cjk_characters_preserve_sequence_and_text():
    spec = Spec(0, 0, 10, 0, 10)
    result = ledger([spec], [[word(t, 1+i*.3) for i, t in enumerate("古老城市")]])
    assert len(result["lexical_observations"]) == 4
    assert result["local_turn_candidates"][0]["source_text_hypothesis"] == "古老城市"


def test_false_speech_activity_is_only_a_model_hypothesis():
    spec = Spec(0, 0, 10, 0, 10)
    result = ledger([spec], [[]], diar=[[{"speaker": "0", "start": 4.0, "end": 6.0}]])
    activity = result["diarization_activity_hypotheses"][0]
    assert activity["possible_unrepresented_speech"]
    assert activity["must_not_infer_missing_words_from_this_flag"]
    assert result["diagnostics"]["source_unit_observations"] == 0


def test_stable_ids_for_same_raw_and_distinct_ids_after_source_change():
    spec = Spec(0, 0, 10, 0, 10)
    units = [[word("repeat", 1), word("repeat", 1.3)]]
    old = ledger([spec], units)
    again = ledger([spec], units)
    changed = ledger([spec], units, sha="b" * 64)
    ids = [u["id"] for u in old["lexical_observations"]]
    assert ids == [u["id"] for u in again["lexical_observations"]]
    assert len(set(ids)) == 2
    assert ids != [u["id"] for u in changed["lexical_observations"]]


def test_inputs_fail_closed_on_missing_provenance_or_chunk_mismatch():
    spec = Spec(0, 0, 10, 0, 10)
    with pytest.raises(ValueError, match="SHA-256"):
        ledger([spec], [[]], sha="")
    with pytest.raises(ValueError, match="one entry per"):
        build_source_evidence_ledger(
            media_fingerprint_sha256="media",
            source_language="fr",
            chunk_specs=[spec],
            chunk_units=[],
            raw_response_sha256=["a" * 64],
            raw_source_texts=["text"],
            diarization_by_chunk=[[]],
        )
