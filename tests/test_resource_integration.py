"""Resource-role CLI and text-cache integration tests (CPU, no inference host)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nvidia_transcribe import cli
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe.refinement import (
    OpenAIChatTextClient, RefinementConfig, refine_utterances_with_text_model,
)
from nvidia_transcribe.translation import (
    TranslationConfig, translate_utterances,
)


def _resource_file(tmp_path: Path) -> Path:
    doc = {
        "version": 1,
        "resources": {
            "correction": {"adapter": "openai_chat", "endpoint": "http://localhost:8201/v1", "model": "corrector"},
            "translation": {"adapter": "openai_chat", "endpoint": "http://localhost:8202/v1", "model": "translator"},
            "review": {"adapter": "openai_chat", "endpoint": "http://localhost:8203/v1", "model": "reviewer"},
        },
        "roles": {
            "source_refinement": "correction", "text_translation": "translation",
            "translation_review": "review",
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
        "generate-subtitles", "--translator", "openai_chat", "--recording-contract", "not-read-by-fake",
        "--output-dir", str(tmp_path / "output"), "--model-resources", str(path),
    ])
    assert result == 0, capsys.readouterr()
    cfg = captured["config"]
    assert cfg.refinement.model == "corrector"
    assert cfg.translation.model == "translator"
    assert cfg.translation_review.model == "reviewer"
    assert cfg.refinement.base_url.endswith("8201/v1")
    assert cfg.translation.base_url.endswith("8202/v1")
    assert cfg.translation_review.base_url.endswith("8203/v1")
    assert len({cfg.refinement.resource_fingerprint, cfg.translation.resource_fingerprint,
                cfg.translation_review.resource_fingerprint}) == 3


def test_cli_generic_text_options_override_all_text_roles(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "generate_subtitles", lambda *args, **kw:
                        captured.setdefault("cfg", kw["config"]) and {"ok": True})
    status = cli.main([
        "generate-subtitles", "--recording-contract", "none",
        "--output-dir", str(tmp_path / "out"), "--translator", "openai_chat",
        "--text-url", "http://localhost:9201/v1", "--text-model", "override-text",
    ])
    assert status == 0
    cfg = captured["cfg"]
    assert cfg.translation.backend == "openai_chat"
    assert cfg.refinement.model == cfg.translation.model == cfg.translation_review.model == "override-text"
    assert cfg.refinement.base_url == cfg.translation.base_url == cfg.translation_review.base_url == "http://localhost:9201/v1"


@pytest.mark.parametrize("retired", [
    ["--translator", "swiftu"],
    ["--swiftu-url", "http://localhost:9201/v1"],
    ["--swiftu-model", "old-model"],
])
def test_cli_rejects_retired_text_aliases(tmp_path, retired):
    with pytest.raises(SystemExit) as error:
        cli.build_parser().parse_args([
            "generate-subtitles", "--recording-contract", "none",
            "--output-dir", str(tmp_path / "out"), *retired,
        ])
    assert error.value.code == 2


def test_cli_invalid_bindings_fail_before_pipeline(tmp_path, monkeypatch, capsys):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps({"version": 1, "roles": {"source_refinement": "moss_local"}}))
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
        return {"choices": [{"message": {"content": json.dumps({self.item_key: [self.item]})}}]}


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


def test_retired_condensation_cli_options_rejected():
    for flag in ("--condense-high-cps", "--condense-threshold", "--condensation-resource"):
        with pytest.raises(SystemExit) as exc:
            cli.build_parser().parse_args(["generate-subtitles", "--output-dir", "unused", flag])
        assert exc.value.code == 2


def test_generate_subtitles_exposes_audio_stream_and_source_language(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("NEMOVOX_MODEL_RESOURCES", raising=False)
    monkeypatch.setattr(cli, "generate_subtitles",
                        lambda *a, **kw: captured.append(kw["config"]) or {"ok": True})
    assert cli.main(["generate-subtitles", "--translator", "openai_chat",
                     "--recording-contract", "unused", "--output-dir", str(tmp_path),
                     "--audio-stream", "2", "--source-language", "fr"]) == 0
    assert captured[-1].audio_stream_index == 2
    assert captured[-1].source_language == "fr"
