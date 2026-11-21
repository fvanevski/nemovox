"""Resource-role CLI and text-cache integration tests (CPU, no inference host)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nvidia_transcribe import cli
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.refinement import (
    OpenAIChatTextClient, RefinementConfig, refine_utterances_with_text_model,
    refine_utterances_with_swiftu,
)
from nvidia_transcribe.translation import (
    TranslationConfig, translate_utterances, condense_high_cps_cues_with_text_model,
    condense_high_cps_cues_with_swiftu,
)


def _resource_file(tmp_path: Path) -> Path:
    doc = {
        "version": 1,
        "resources": {
            "correction": {"adapter": "openai_chat", "endpoint": "http://localhost:8201/v1", "model": "corrector"},
            "translation": {"adapter": "openai_chat", "endpoint": "http://localhost:8202/v1", "model": "translator"},
            "condensation": {"adapter": "openai_chat", "endpoint": "http://localhost:8203/v1", "model": "condensor"},
        },
        "roles": {
            "source_refinement": "correction", "text_translation": "translation",
            "subtitle_condensation": "condensation",
        },
    }
    path = tmp_path / "models.local.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_cli_distinct_text_resources_and_specialized_adapter_settings(tmp_path, monkeypatch, capsys):
    path = _resource_file(tmp_path)
    captured = {}

    def fake_generate(*args, **kwargs):
        captured["config"] = kwargs["config"]
        return {"fake": "no-model-inference"}

    monkeypatch.setattr(cli, "generate_subtitles", fake_generate)
    result = cli.main([
        "generate-subtitles", "--recording-contract", "not-read-by-fake",
        "--output-dir", str(tmp_path / "output"), "--model-resources", str(path),
        "--condense-high-cps",
    ])
    assert result == 0, capsys.readouterr()
    cfg = captured["config"]
    assert cfg.refinement.model == "corrector"
    assert cfg.translation.model == "translator"
    assert cfg.condensation.model == "condensor"
    assert cfg.refinement.base_url.endswith("8201/v1")
    assert cfg.translation.base_url.endswith("8202/v1")
    assert cfg.condensation.base_url.endswith("8203/v1")
    assert len({cfg.refinement.resource_fingerprint, cfg.translation.resource_fingerprint,
                cfg.condensation.resource_fingerprint}) == 3
    assert cfg.qwen_device == "cuda:0"
    assert cfg.qwen_gpu_mode == "available"
    assert cfg.speaker_embedding_model == "titanet_small"


def test_cli_legacy_swiftu_options_and_backend_are_compatible(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "generate_subtitles", lambda *args, **kw:
                        captured.setdefault("cfg", kw["config"]) and {"ok": True})
    status = cli.main([
        "generate-subtitles", "--recording-contract", "none",
        "--output-dir", str(tmp_path / "out"), "--translator", "swiftu",
        "--swiftu-url", "http://localhost:9201/v1", "--swiftu-model", "legacy-text",
    ])
    assert status == 0
    cfg = captured["cfg"]
    assert cfg.translation.backend == "swiftu"
    assert cfg.refinement.model == cfg.translation.model == cfg.condensation.model == "legacy-text"
    assert cfg.refinement.base_url == cfg.translation.base_url == cfg.condensation.base_url == "http://localhost:9201/v1"
    assert refine_utterances_with_swiftu is refine_utterances_with_text_model
    assert condense_high_cps_cues_with_swiftu is condense_high_cps_cues_with_text_model


def test_cli_invalid_bindings_fail_before_pipeline(tmp_path, monkeypatch, capsys):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"version": 1, "roles": {"source_refinement": "asr_local"}}))
    monkeypatch.setattr(cli, "generate_subtitles", lambda *args, **kw:
                        pytest.fail("Inference must not run for invalid resource config"))
    status = cli.main([
        "generate-subtitles", "--recording-contract", "not-used",
        "--output-dir", str(tmp_path / "out"), "--model-resources", str(config),
    ])
    assert status == 2
    assert "CONFIG_ERROR" in capsys.readouterr().err


def test_text_client_wire_payload_remains_openai_compatible(monkeypatch):
    import httpx
    sent = []

    class Response:
        status_code = 200
        def raise_for_status(self):
            return None
        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    class FakeHTTP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def post(self, url, json):
            sent.append((url, json, self.kwargs))
            return Response()

    monkeypatch.setattr(httpx, "Client", FakeHTTP)
    c = OpenAIChatTextClient(base_url="http://localhost:8001/v1", model="chat", reasoning_effort="low")
    c.complete(messages=[{"role": "user", "content": "Test"}],
               response_format={"type": "json_object"}, max_tokens=20)
    assert len(sent) == 1
    url, payload, kw = sent[0]
    assert url == "http://localhost:8001/v1/chat/completions"
    assert payload == {
        "model": "chat", "messages": [{"role": "user", "content": "Test"}],
        "temperature": 0.0, "max_tokens": 20, "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }


class FakeText:
    def __init__(self, item_key, item):
        self.item_key = item_key
        self.item = item
        self.calls = 0

    def complete(self, **kwargs):
        self.calls += 1
        return {"choices": [{"message": {"content": json.dumps({self.item_key: [self.item]}
                        if self.item_key != "condensation" else self.item)}}]}


def test_refinement_cache_respects_model_identity(tmp_path):
    cache = tmp_path / "refinement.json"
    text = "Esteban revient demain."
    utts = [{"utterance_id": "u1", "source_text": text, "speaker_id": "speaker_0"}]
    fake = FakeText("corrections", {"utterance_id": "u1", "original_text": text,
        "refined_text": text, "changed": False, "edit_type": "none",
        "rationale": "", "confidence": 1.0})
    for token in ("identity-one", "identity-one", "identity-two"):
        cfg = RefinementConfig(profile="balanced", batch_size=1, resource_fingerprint=token)
        refine_utterances_with_text_model(utts, source_language="fr", config=cfg,
                                          text_client=fake, checkpoint_path=cache)
    assert fake.calls == 2


def test_translation_cache_respects_model_identity(tmp_path):
    cache = tmp_path / "translation.json"
    utts = [{"utterance_id": "u1", "source_text": "Bonjour", "speaker_id": "speaker_0",
             "source_start": 0.0, "source_end": 2.0}]
    fake = FakeText("translations", {"utterance_id": "u1", "translated_text": "Hello",
                                      "grouped_with_prev": False})
    for token in ("identity-one", "identity-one", "identity-two"):
        cfg = TranslationConfig(batch_size=1, resource_fingerprint=token)
        translate_utterances(utts, source_language="fr", config=cfg,
                             text_client=fake, checkpoint_path=cache)
    assert fake.calls == 2


def test_condensation_cache_respects_model_identity(tmp_path):
    cache = tmp_path / "condensation.json"
    long_text = "She reminded everyone that keeping the story's names, important events, and emotional context unchanged is essential. " * 3
    utts = [{"utterance_id": "u1", "source_text": "Bonjour", "translated_text": long_text,
             "source_start": 0.0, "source_end": 1.5}]
    class FakeCondense:
        calls = 0
        def complete(self, **kwargs):
            self.calls += 1
            return {"choices": [{"message": {"content": json.dumps({
                "condensed_text": "Keep the names and important events intact.",
                "changed": True, "rationale": "Shorter dialogue",
            })}}]}
    fake = FakeCondense()
    for token in ("identity-one", "identity-one", "identity-two"):
        cfg = TranslationConfig(resource_fingerprint=token)
        condense_high_cps_cues_with_text_model(
            utts, source_language="fr", config=cfg, text_client=fake,
            checkpoint_path=cache,
        )
    assert fake.calls == 2


def test_no_translation_skips_unused_condensation_credentials(tmp_path, monkeypatch, capsys):
    path = _resource_file(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["resources"]["condensation"]["api_key_env"] = "NEMOVOX_UNSET_CONDENSE_TEST_TOKEN"
    path.write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.delenv("NEMOVOX_UNSET_CONDENSE_TEST_TOKEN", raising=False)
    captured = {}
    monkeypatch.setattr(cli, "generate_subtitles",
                        lambda *args, **kwargs: captured.setdefault("cfg", kwargs["config"]) and {"ok": True})
    code = cli.main([
        "generate-subtitles", "--recording-contract", "unused",
        "--output-dir", str(tmp_path / "output"),
        "--model-resources", str(path), "--refinement", "off",
        "--translator", "none", "--condense-high-cps",
    ])
    assert code == 0, capsys.readouterr()
    assert captured["cfg"].translation.backend == "none"
    assert captured["cfg"].condensation.api_key is None


def test_programmatic_riva_condensation_requires_explicit_text_resource(tmp_path):
    from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
    cfg = SubtitlePipelineConfig(
        translation=TranslationConfig(backend="riva"), condense_high_cps=True)
    with pytest.raises(ValueError, match="explicit OpenAI-compatible condensation resource"):
        generate_subtitles(
            recording_contract_path=tmp_path / "not_needed",
            output_dir=tmp_path / "out",
            config=cfg,
        )
    assert not (tmp_path / "out").exists()


def test_condensation_rejects_specialized_riva_adapter():
    with pytest.raises(ValueError, match="OpenAI-compatible text resource"):
        condense_high_cps_cues_with_text_model(
            [], source_language="fr", config=TranslationConfig(backend="riva"))
