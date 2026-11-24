"""Current source quality: functional invariants, not historical episode fixtures."""
from copy import deepcopy

import pytest

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


def test_current_recording_ledger_rejects_tampered_observation_sidecar(tmp_path):
    import hashlib
    import json
    import pytest
    from nvidia_transcribe.source_quality import load_current_ledger

    recording = _recording([])
    recording["processing"] = {"chunks": [{"chunk_index": 0, "raw_response_sha256": "raw"}]}
    ledger = {"schema_version": "nemovox-source-evidence/v1",
              "media_fingerprint_sha256": "f", "source_language": "fr",
              "asr_observations": [{"chunk_index": 0, "raw_response_sha256": "raw"}]}
    data = json.dumps(ledger).encode()
    sidecar = tmp_path / "source-evidence.v1.json"
    sidecar.write_bytes(data)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "asr_run_status": "completed",
        "source_evidence": {"sha256": hashlib.sha256(data).hexdigest()},
    }))
    assert load_current_ledger(recording, tmp_path / "recording.contract.json") == ledger
    sidecar.write_bytes(data + b" ")
    with pytest.raises(ValueError, match="stale"):
        load_current_ledger(recording, tmp_path / "recording.contract.json")


def test_shared_pipeline_translates_canonical_source_not_prior_asr(tmp_path, monkeypatch):
    import json
    from contextlib import contextmanager
    from types import SimpleNamespace
    from nvidia_transcribe import subtitle_pipeline as pipeline
    from nvidia_transcribe.translation import TranslationConfig
    from nvidia_transcribe.translation_review import TranslationReviewConfig

    original = _recording([_row("a", "ne sais pas", 4.2, 5.8)])
    recording_path = tmp_path / "recording.contract.json"
    recording_path.write_text(json.dumps(original))
    media = tmp_path / "media.wav"
    media.write_bytes(b"synthetic; inference is mocked")
    order = []
    resource = SimpleNamespace(name="test", credential=lambda: None)
    resources = {name: resource for name in ("asr", "source_refinement", "text_translation",
                                             "riva_translation", "translation_review")}

    @contextmanager
    def text_scope(*args, **kwargs):
        if kwargs.get("wire_protocol") == "ninfer_messages":
            order.append("messages")
        yield object()

    @contextmanager
    def asr_scope(*args, **kwargs):
        order.append("acoustic")
        yield object()

    def quality(recording, **kw):
        assert kw["media_path"] == media
        with kw["text_scope"]():
            pass
        with kw["asr_scope"]():
            pass
        rows, _ = select_source_events(recording, [_event("Je ne sais pas", 4, 6)],
                                       SourceQualityPolicy())
        return {**recording, "utterances": rows, "source_quality": {"status": "complete"}}, {"status": "complete"}

    def translate(rows, **kw):
        order.append("translate")
        assert rows[0]["source_text"] == "Je ne sais pas"
        return [{**rows[0], "translated_text": "I don't know."}], {"backend": "riva", "status": "complete"}

    def english_review(rows, **kw):
        order.append("review")
        assert kw["config"].wire_protocol == "ninfer_messages"
        assert kw["config"].thinking_budget == 0
        assert rows[0]["translated_text"] == "I don't know."
        return [{**rows[0], "machine_translated_text": rows[0]["translated_text"]}], {"status": "complete"}

    monkeypatch.setattr(pipeline, "managed_text_client", text_scope)
    monkeypatch.setattr(pipeline, "managed_asr_client", asr_scope)
    monkeypatch.setattr(pipeline, "review_current_source", quality)
    monkeypatch.setattr(pipeline, "translate_utterances", translate)
    monkeypatch.setattr(pipeline, "review_translated_utterances", english_review)
    monkeypatch.setattr(pipeline, "render_subtitles", lambda *a, **k: {
        "outputs": {}, "stats": {}, "conflicts": [], "sidecars": {}})
    out = tmp_path / "output"
    pipeline.generate_subtitles(
        media, recording_contract_path=recording_path, output_dir=out,
        config=pipeline.SubtitlePipelineConfig(
            source_quality="on", translation=TranslationConfig(backend="riva"),
            translation_review=TranslationReviewConfig(profile="balanced")),
        lifecycle=object(), lifecycle_resources=resources,
    )
    assert order == ["messages", "acoustic", "translate", "messages", "review"]
    assert json.loads(recording_path.read_text()) == original
    assert json.loads((out / "canonical-source.contract.json").read_text())["utterances"][0]["source_text"] == "Je ne sais pas"


@pytest.mark.parametrize("challenge_bounds", [(11.8, 13.8), (9.8, 13.1), (9.0, 12.1)])
def test_equal_text_with_materially_different_timing_remains_unresolved(challenge_bounds):
    """Even a candidate which covers the old turn cannot masquerade as an extension."""
    earlier = _row("original", "On y va !", 10.0, 12.0)
    between = {**_row("between", "Attends !", 11.3, 11.6), "speaker_id": "SPK_B"}
    recording = _recording([earlier, between])
    before = deepcopy(recording)
    a, b = challenge_bounds
    first = _probe((7, 18), "A", [_row("0", "On y va !", a, b)])
    second = _probe((5, 16), "B", [_row("0", "On y va !", a + .1, b + .1)])
    policy = SourceQualityPolicy()
    events = corroborated_events(first, second, (8, 15), policy.timing_tolerance_s)
    assert len(events) == 1
    events[0]["chunks"] = [1]
    rows, decisions = select_source_events(recording, events, policy)
    assert rows == before["utterances"] and recording == before
    decision, = decisions
    assert decision["status"] == "unresolved_possible_displaced_or_repeated_turn"
    assert decision["observations"] == ["A:0", "B:0"]
    assert decision["window"] == challenge_bounds
    assert decision["competing_source_hypotheses"] == [
        {"source_id": "original", "window": (10.0, 12.0)}]
    assert decision["superseded_source_id"] is None
    if a == 11.8:
        assert decision["crossed_source_ids"] == ["between"]


def test_small_timing_jitter_with_no_competing_occurrence_is_already_represented():
    recording = _recording([_row("a", "On y va !", 10, 12),
                            _row("b", "Attends !", 13, 14)])
    rows, decisions = select_source_events(
        recording, [_event("On y va !", 10.2, 12.1)], SourceQualityPolicy())
    assert rows == recording["utterances"]
    assert decisions[0]["status"] == "already_represented"


def test_tolerance_does_not_hide_crossing_an_entire_intervening_turn():
    recording = _recording([_row("a", "On y va !", 10, 12),
                            {**_row("b", "Attends !", 10.3, 10.5), "speaker_id": "SPK_B"}])
    rows, decisions = select_source_events(
        recording, [_event("On y va !", 10.7, 12.7)], SourceQualityPolicy())
    assert rows == recording["utterances"]
    assert decisions[0]["status"].startswith("unresolved")
    assert decisions[0]["crossed_source_ids"] == ["b"]


def test_two_overlapping_equal_text_source_candidates_do_not_establish_identity():
    recording = _recording([_row("a", "On y va !", 10, 12),
                            {**_row("b", "On y va !", 10.1, 12.1), "speaker_id": "SPK_B"}])
    rows, decisions = select_source_events(
        recording, [_event("On y va !", 10.2, 12.2)], SourceQualityPolicy())
    assert rows == recording["utterances"]
    assert decisions[0]["status"].startswith("unresolved")
    assert {h["source_id"] for h in decisions[0]["competing_source_hypotheses"]} == {"a", "b"}


def test_unresolved_timing_reaches_current_quality_report(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from nvidia_transcribe import source_quality as quality

    recording = _recording([_row("a", "On y va !", 10, 12),
                            {**_row("b", "Attends !", 11.3, 11.6), "speaker_id": "SPK_B"}])
    recording["source"]["selected_audio_stream"] = 0
    recording["processing"] = {
        "boundary_reconciliation": {"semantic_review_candidates": {"cases": []}}}
    monkeypatch.setattr(quality, "load_current_ledger", lambda *args: {})
    monkeypatch.setattr(quality, "probe_media", lambda *args, **kwargs: SimpleNamespace(
        path=tmp_path / "media.wav", fingerprint_sha256="f",
        duration_s=100., selected_audio_stream=0))
    monkeypatch.setattr(quality, "plan_windows", lambda *args: (
        [{"window": (10, 15), "chunks": [1], "evidence_ids": ["test"]}], []))
    monkeypatch.setattr(quality, "_observe", lambda client, media, crop, root, i:
                        _probe(crop, str(i), [_row("0", "On y va !",
                                                 11.8 + i * .1, 13.8 + i * .1)]))
    canonical, report = quality.review_current_source(
        recording, recording_path=tmp_path / "recording.contract.json",
        media_path=tmp_path / "media.wav", output_dir=tmp_path,
        text_scope=lambda: pytest.fail("No text candidates"),
        asr_scope=lambda: nullcontext(object()))
    assert report["audio_requests"] == 2
    assert not report["unresolved_windows"]  # Acoustic pairing itself succeeded.
    assert report["status"] == canonical["source_quality"]["status"] == "partial"
    assert report["decisions"][0]["status"].startswith("unresolved")
    assert canonical["utterances"] == recording["utterances"]
