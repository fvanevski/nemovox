"""CPU-only tests for independent machine-translation post-edit."""
from __future__ import annotations

import json

from nvidia_transcribe.translation_review import TranslationReviewConfig, review_translated_utterances
from nvidia_transcribe.refinement import GlossaryConfig


class FakeClient:
    def __init__(self, *batches):
        self.responses = list(batches)
        self.calls = []

    def complete(self, *, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return {"choices": [{"message": {"content": json.dumps(item)}}]}


def _utts():
    return [
        {"id": "u0", "speaker_id": "A", "source_text": "Bonjour, Esteban.",
         "translated_text": "Hello, Estban.", "source_start": 1.0, "source_end": 2.0},
        {"id": "u1", "speaker_id": "B", "source_text": "Salut!",
         "translated_text": "Hi!", "source_start": 2.0, "source_end": 3.0},
    ]


def _cfg(**kw):
    return TranslationReviewConfig(**{"profile": "balanced", "resource_fingerprint": "text-one", **kw})


def test_accepts_supported_correction_only_and_preserves_identity():
    client = FakeClient({"reviews": [
        {"utterance_id": "u0", "changed": True, "corrected_text": "Hello, Esteban.", "reason": "Source name"},
        {"utterance_id": "u1", "changed": False, "corrected_text": "Hi!", "reason": "No change"},
    ]})
    original = _utts()
    result, summary = review_translated_utterances(
        original, source_language="fr", config=_cfg(),
        glossary=GlossaryConfig(characters=("Esteban",)), text_client=client)
    assert (summary["changed"], summary["reviewed"]) == (1, 2)
    assert result[0]["machine_translated_text"] == "Hello, Estban."
    assert result[0]["translated_text"] == "Hello, Esteban."
    assert result[1]["translation_review_status"] == "unchanged"
    for old, new in zip(original, result):
        for field in ("id", "speaker_id", "source_text", "source_start", "source_end"):
            assert old[field] == new[field]
    assert "Esteban" in client.calls[0]["messages"][1]["content"]


def test_missing_or_duplicate_review_ids_preserve_primary_translation():
    client = FakeClient({"reviews": [
        {"utterance_id": "u0", "changed": True, "corrected_text": "invented", "reason": "none"},
    ]})
    result, summary = review_translated_utterances(_utts(), source_language="fr",
                                                     config=_cfg(), text_client=client)
    assert summary["changed"] == 0 and summary["errors"]
    assert [x["translated_text"] for x in result] == ["Hello, Estban.", "Hi!"]
    assert all(x["translation_review_status"] == "review_error_preserved_machine_translation" for x in result)


def test_invalid_edits_and_flag_contradictions_fail_closed():
    client = FakeClient({"reviews": [
        {"utterance_id": "u0", "changed": True, "corrected_text": "", "reason": "delete"},
        {"utterance_id": "u1", "changed": True, "corrected_text": "Hi!", "reason": "mismatch"},
    ]})
    result, summary = review_translated_utterances(_utts(), source_language="fr",
                                                     config=_cfg(), text_client=client)
    assert len(summary["errors"]) == 2 and summary["changed"] == 0
    assert all(x["translation_review_status"] == "review_invalid_preserved_machine_translation" for x in result)


def test_review_cache_is_bound_to_model_style_glossary_and_machine_translation(tmp_path):
    cache = tmp_path / "review.json"
    unchanged = {"reviews": [
        {"utterance_id": "u0", "changed": False, "corrected_text": "Hello, Estban.", "reason": "fine"},
        {"utterance_id": "u1", "changed": False, "corrected_text": "Hi!", "reason": "fine"},
    ]}
    client = FakeClient(unchanged, unchanged, unchanged, unchanged)
    for cfg, gl in [
        (_cfg(), GlossaryConfig()),
        (_cfg(), GlossaryConfig()),
        (_cfg(style="literal"), GlossaryConfig()),
        (_cfg(resource_fingerprint="text-two"), GlossaryConfig()),
        (_cfg(), GlossaryConfig(characters=("Esteban",))),
    ]:
        review_translated_utterances(_utts(), source_language="fr", config=cfg,
                                    glossary=gl, text_client=client, checkpoint_path=cache)
    assert len(client.calls) == 4
    altered = _utts()
    altered[0]["translated_text"] = "Different."
    client.responses.append({"reviews": [
        {"utterance_id": "u0", "changed": False, "corrected_text": "Different.", "reason": "fine"},
        {"utterance_id": "u1", "changed": False, "corrected_text": "Hi!", "reason": "fine"},
    ]})
    review_translated_utterances(altered, source_language="fr", config=_cfg(),
                                  text_client=client, checkpoint_path=cache)
    assert len(client.calls) == 5


def test_off_does_not_contact_text_model():
    result, summary = review_translated_utterances(_utts(), source_language="fr")
    assert summary["status"] == "disabled"
    assert result[0]["translated_text"] == result[0]["machine_translated_text"]



def test_programmatic_postedit_rejects_legacy_primary_translator(tmp_path):
    import pytest
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
    from nvidia_transcribe.translation import TranslationConfig
    config = SubtitlePipelineConfig(
        translation=TranslationConfig(backend="openai_chat"),
        translation_review=TranslationReviewConfig(profile="balanced"),
    )
    with pytest.raises(ValueError, match="requires dedicated Riva"):
        generate_subtitles(recording_contract_path=tmp_path / "not-needed",
                           output_dir=tmp_path / "new-output", config=config)
    assert not (tmp_path / "new-output").exists()



def test_riva_first_pipeline_runs_separate_postedit_and_preserves_raw_mt(tmp_path, monkeypatch):
    """No actual model or media file: verify stage order, saved contracts and rerender."""
    from pathlib import Path
    from types import SimpleNamespace
    from nvidia_transcribe import translation
    from nvidia_transcribe.translation import TranslationConfig
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles

    contract = {
        "schema_version": "nemotron-recording-contract/v1",
        "source": {"path": "/sample/French_episode.mkv", "duration_s": 10.0},
        "source_language": "fr",
        "utterances": [{"id": "utt-0000", "speaker_id": "SPK_00",
                        "source_start": 1.0, "source_end": 3.0,
                        "source_text": "Bonjour, Esteban."}],
    }
    recording = tmp_path / "recording.contract.json"
    recording.write_text(json.dumps(contract), encoding="utf-8")
    worker_calls = []

    def fake_riva(command, **kwargs):
        req = json.loads(kwargs["input"])
        worker_calls.append(req)
        assert req["device"] == "cuda:0"
        assert req["items"][0]["lang_pair"] == "fr-en"
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "ok", "device_used": "cuda:0", "results": [
                {"utterance_id": "utt-0000", "translated_text": "Hello, Estban."},
            ],
        }))
    monkeypatch.setattr(translation.subprocess, "run", fake_riva)

    review_client = FakeClient({"reviews": [
        {"utterance_id": "utt-0000", "changed": True,
         "corrected_text": "Hello, Esteban.", "reason": "Correct proper name from French source."},
    ]})
    outdir = tmp_path / "subtitle-output"
    config = SubtitlePipelineConfig(
        source_language="fr",
        translation=TranslationConfig(backend="riva", riva_device="cuda:0"),
        translation_review=_cfg(),
        condense_high_cps=False,
        formats=("srt",),
    )
    meta = generate_subtitles(
        recording_contract_path=recording, output_dir=outdir,
        config=config, text_client=review_client,
    )
    assert len(worker_calls) == 1
    assert len(review_client.calls) == 1
    assert meta["translation"]["backend"] == "riva"
    assert meta["translation_review"]["changed"] == 1
    assert meta["refinement"]["profile"] == "off"  # no source-text model call in the normal pipeline
    translated = json.loads((outdir / "translated.contract.json").read_text(encoding="utf-8"))
    u = translated["utterances"][0]
    assert u["machine_translated_text"] == "Hello, Estban."
    assert u["translated_text"] == "Hello, Esteban."
    assert u["translation_review_status"] == "corrected"
    assert u["source_text"] == "Bonjour, Esteban."
    assert u["speaker_id"] == "SPK_00" and (u["source_start"], u["source_end"]) == (1.0, 3.0)
    assert "Hello, Esteban." in Path(meta["files"]["srt"]).read_text(encoding="utf-8")

    # Rerender must not touch either Riva worker or reviewer.
    offline = generate_subtitles(output_dir=outdir,
                                 config=SubtitlePipelineConfig(re_render_only=True, formats=("srt",)))
    assert offline["translation_review"]["status"] == "reused_translated_contract"
    assert len(worker_calls) == 1 and len(review_client.calls) == 1


def test_riva_rejects_missing_or_duplicate_worker_items(tmp_path, monkeypatch):
    """The dedicated translator must not pass malformed MT through to review."""
    import pytest
    from types import SimpleNamespace
    from nvidia_transcribe import translation
    from nvidia_transcribe.translation import TranslationConfig, translate_utterances

    def bad_worker(command, **kwargs):
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "status": "ok", "device_used": "cuda:0", "results": [
                {"utterance_id": "wrong-id", "translated_text": "Incorrect"},
            ],
        }))
    monkeypatch.setattr(translation.subprocess, "run", bad_worker)
    with pytest.raises(RuntimeError, match="invalid or duplicate utterance IDs"):
        translate_utterances(
            [{"id": "id1", "source_text": "Bonjour"}], source_language="fr",
            config=TranslationConfig(backend="riva", riva_device="cuda:0"),
        )



def test_reviewer_uses_original_asr_and_uncorrected_english_neighbors_not_legacy_refined_text():
    initial = _utts()
    initial[0]["refined_text"] = "Invented altered source-language hypothesis"
    initial[1]["refined_text"] = "Another invented hypothesis"
    client = FakeClient({"reviews": [
        {"utterance_id": "u0", "changed": False,
         "corrected_text": "Hello, Estban.", "reason": "No supported correction"},
        {"utterance_id": "u1", "changed": False,
         "corrected_text": "Hi!", "reason": "No supported correction"},
    ]})
    out, summary = review_translated_utterances(
        initial, source_language="fr", config=_cfg(), text_client=client,
    )
    assert summary["changed"] == 0
    assert [u["translated_text"] for u in out] == [u["translated_text"] for u in initial]
    prompt = json.loads(client.calls[0]["messages"][1]["content"])
    targets = prompt["target_utterances"]
    assert [u["source_text"] for u in targets] == [
        "Bonjour, Esteban.", "Salut!",
    ]
    assert [u["machine_translation"] for u in targets] == ["Hello, Estban.", "Hi!"]
    assert "Invented altered source-language hypothesis" not in client.calls[0]["messages"][1]["content"]
    assert "Another invented hypothesis" not in client.calls[0]["messages"][1]["content"]
    instructions = client.calls[0]["messages"][0]["content"]
    assert "Only edit TARGET translated English cues" in instructions
    assert "NOT a fresh translation request" in instructions
    assert "Stylistic preferences alone do not justify edits" in instructions


def test_english_source_skips_riva_worker_and_text_review(tmp_path, monkeypatch):
    """No en→en translation or source/English-review rewrite in default subtitle jobs."""
    from pathlib import Path
    from nvidia_transcribe import translation
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
    from nvidia_transcribe.translation import TranslationConfig, is_english_source_to_english

    assert is_english_source_to_english("en-US", "en")
    assert is_english_source_to_english("eng", "English")
    assert not is_english_source_to_english("fr", "en")
    assert not is_english_source_to_english(None, "en")

    def forbidden_worker(*args, **kwargs):
        raise AssertionError("Already-English speech must never invoke Riva")
    monkeypatch.setattr(translation.subprocess, "run", forbidden_worker)

    class ForbiddenTextClient:
        def complete(self, **kwargs):
            raise AssertionError("Already-English speech must not invoke a text model")

    record = tmp_path / "recording.contract.json"
    record.write_text(json.dumps({
        "source_language": "en", "source": {"path": "/sample/english.mkv", "duration_s": 6.0},
        "utterances": [{"id": "eng-1", "source_text": "I can tell you what happened.",
                        "source_start": 1.0, "source_end": 3.0,
                        "speaker_id": "SPK_00"}],
    }), encoding="utf-8")
    outdir = tmp_path / "english-output"
    config = SubtitlePipelineConfig(
        translation=TranslationConfig(backend="riva"),
        translation_review=_cfg(),
        formats=("srt",),
    )
    meta = generate_subtitles(
        recording_contract_path=record, output_dir=outdir,
        config=config, text_client=ForbiddenTextClient(),
    )
    assert meta["translation"]["status"] == "bypassed_same_language"
    assert meta["translation_review"]["status"] == "skipped_same_language"
    assert meta["condensation"]["status"] == "disabled"
    contract = json.loads((outdir / "translated.contract.json").read_text(encoding="utf-8"))
    u = contract["utterances"][0]
    assert u["source_text"] == u["translated_text"] == "I can tell you what happened."
    assert u["translation_status"] == "bypassed_same_language"
    assert u["translation_review_status"] == "skipped_same_language"
    assert "machine_translated_text" not in u
    assert "I can tell you what happened." in Path(meta["files"]["srt"]).read_text(encoding="utf-8")
    offline = generate_subtitles(output_dir=outdir, config=SubtitlePipelineConfig(
        re_render_only=True, formats=("srt",),
    ))
    assert offline["subtitles"]["renderable_cues"] == 1


def test_cli_riva_english_contract_does_not_require_local_worker(tmp_path, monkeypatch, capsys):
    """Riva path preflight cannot reject an English contract with a missing worker."""
    from nvidia_transcribe import cli
    from nvidia_transcribe import translation

    def forbidden_worker(*args, **kwargs):
        raise AssertionError("English contract must not invoke a Riva subprocess")
    monkeypatch.setattr(translation.subprocess, "run", forbidden_worker)
    # The standard resource registry's Riva checkpoint is intentionally not
    # required. No configured text-model credentials or live endpoint are used.
    recording = tmp_path / "recording.contract.json"
    recording.write_text(json.dumps({
        "source_language": "eng", "source": {"path": "/example/en.webm", "duration_s": 5.0},
        "utterances": [{"id": "one", "source_text": "We should go.",
                        "source_start": 1.0, "source_end": 2.0}],
    }), encoding="utf-8")
    status = cli.main([
        "generate-subtitles", "--recording-contract", str(recording),
        "--output-dir", str(tmp_path / "subtitles"),
        "--translator", "riva", "--translation-review", "balanced",
        "--formats", "srt",
    ])
    assert status == 0, capsys.readouterr().err
    contract = json.loads((tmp_path / "subtitles" / "translated.contract.json").read_text())
    assert contract["translation"]["status"] == "bypassed_same_language"
