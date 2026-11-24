"""Current source quality: functional invariants, not historical episode fixtures."""
from copy import deepcopy

from nvidia_transcribe.source_quality import (
    SourceQualityPolicy, corroborated_events, plan_windows, select_source_events,
)


def _row(uid, text, a, b, chunks=None):
    return {"id": uid, "source_text": text, "source_start": a, "source_end": b,
            "speaker_id": "SPK_A", "chunk_indices": chunks or [1]}


def _event(text, a, b, n=0):
    return {"source_text": text, "source_start": a, "source_end": b, "chunks": [1],
            "supporting_observation_ids": [f"probe-0:{n}", f"probe-1:{n}"],
            "audio_sha256": ["a", "b"], "raw_response_sha256": ["c", "d"]}


def _recording(rows):
    return {"source": {"fingerprint_sha256": "f", "duration_s": 100.},
            "source_language": "fr", "utterances": rows}


def _probe(crop, sha, rows):
    return {"crop": crop, "audio_sha256": sha, "raw_response_sha256": sha,
            "utterances": [{**u, "observation_id": sha + ":" + u["id"]} for u in rows]}


def test_positive_corroboration_uses_distinct_audio_and_observed_event_bounds():
    a = _probe((0, 12), "a", [_row("1", "Je ne sais pas", 4, 6)])
    b = _probe((2, 14), "b", [_row("2", "Je ne sais pas.", 4.1, 6.2)])
    result = corroborated_events(a, b, (3, 7), .85)
    assert len(result) == 1
    assert result[0]["source_start"] == 4
    assert result[0]["source_end"] == 6
    assert result[0]["audio_sha256"] == ["a", "b"]
    assert not corroborated_events(a, {**b, "audio_sha256": "a"}, (3, 7), .85)
    assert not corroborated_events(a, {**b, "crop": a["crop"]}, (3, 7), .85)


def test_repetition_separated_by_time_is_not_deduplicated_in_observations():
    a = _probe((0, 12), "a", [_row("1", "Allons vite", 4, 5),
                              _row("2", "Allons vite", 7, 8)])
    b = _probe((2, 14), "b", [_row("3", "Allons vite", 4.1, 5.1),
                              _row("4", "Allons vite", 7.1, 8.1)])
    assert len(corroborated_events(a, b, (3, 9), .85)) == 2


def test_corroborated_gap_insertion_preserves_existing_source_and_unknown_voice():
    recording = _recording([_row("old1", "Bonjour mon ami", 1, 2),
                            _row("old2", "Il fait beau", 8, 9)])
    before = deepcopy(recording)
    rows, decisions = select_source_events(recording, [_event("Je ne sais pas", 4, 6)], SourceQualityPolicy())
    assert recording == before
    assert rows[0] == before["utterances"][0] and rows[-1] == before["utterances"][1]
    assert rows[1]["source_text"] == "Je ne sais pas"
    assert rows[1]["speaker_id"] == "SPK_UNK"
    assert rows[1]["source_words"] == []
    assert not rows[1]["timestamp_provenance"]["human_verified"]
    assert decisions[0]["status"] == "recovered_corroborated_turn"


def test_no_duplicate_insertion_or_destruction_of_simultaneous_speech():
    first = _row("a", "Je ne sais pas", 10, 11)
    second = _row("b", "Viens par ici", 4, 5)
    recording = _recording([second, first])
    rows, decisions = select_source_events(recording, [
        _event("Je ne sais pas", 3, 4, 1),
        _event("Une autre phrase", 4, 5, 2),
    ], SourceQualityPolicy())
    assert rows == recording["utterances"]
    assert all(d["status"].startswith("unresolved") for d in decisions)


def test_single_turn_extension_preserves_entire_original_lexical_hypothesis():
    recording = _recording([_row("a", "ne sais pas", 4.2, 5.8)])
    rows, decisions = select_source_events(recording, [_event("Je ne sais pas", 4, 6)], SourceQualityPolicy())
    assert len(rows) == 1 and rows[0]["id"] == "a"
    assert rows[0]["source_text"] == "Je ne sais pas"
    assert rows[0]["prior_source_hypothesis"] == recording["utterances"][0]
    assert decisions[0]["status"] == "extended_corroborated_turn"


def test_incompatible_polarity_and_ambiguous_correspondence_do_not_authorize_change():
    a = _probe((0, 12), "a", [_row("1", "Je ne sais pas", 4, 6)])
    b = _probe((2, 14), "b", [_row("2", "Je sais", 4, 6)])
    assert not corroborated_events(a, b, (3, 7), .85)
    b["utterances"] += [{**a["utterances"][0], "observation_id": "b:3"},
                         {**a["utterances"][0], "observation_id": "b:4"}]
    assert not corroborated_events(a, b, (3, 7), .85)


def test_window_budget_preserves_deferred_work_and_text_only_is_not_authority():
    request = {"cases": [
        {"case_id": "x", "context_window_s": [10, 13], "source_chunks": [1]},
        {"case_id": "y", "context_window_s": [40, 43], "source_chunks": [2]},
    ]}
    ledger = {"asr_observations": [], "diarization_activity_hypotheses": []}
    advisory = {"outcomes": [{"case_id": "x", "status": "advisory_only",
                              "assessment": {"decision": "retain_core"}}]}
    windows, deferred = plan_windows(_recording([]), ledger, request, advisory,
                                     SourceQualityPolicy(max_windows=1))
    assert windows[0]["evidence_ids"] == ["y"]
    assert deferred[0]["evidence_ids"] == ["x"]


def test_recovery_of_two_separately_corroborated_repetitions():
    events = [_event("Allons vite", 4, 5, 1), _event("Allons vite", 7, 8, 2)]
    rows, decisions = select_source_events(_recording([]), events, SourceQualityPolicy())
    assert [r["source_start"] for r in rows] == [4, 7]
    assert len({r["id"] for r in rows}) == 2
    assert all(d["status"] == "recovered_corroborated_turn" for d in decisions)
