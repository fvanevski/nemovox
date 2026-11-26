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
    TranslationConfig, translate_utterances, condense_high_cps_cues_with_text_model,
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
        "generate-subtitles", "--translator", "openai_chat", "--recording-contract", "not-read-by-fake",
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
    assert cfg.refinement.model == cfg.translation.model == cfg.condensation.model == "override-text"
    assert cfg.refinement.base_url == cfg.translation.base_url == cfg.condensation.base_url == "http://localhost:9201/v1"


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


def test_single_clip_asr_role_default_and_legacy_explicit_endpoint(tmp_path, monkeypatch, capsys):
    from nvidia_transcribe import resource_lifecycle

    for key in ("ASR_BASE_URL", "ASR_MODEL", "NEMOVOX_MODEL_RESOURCES",
                "NEMOVOX_LIFECYCLE_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEMOVOX_LIFECYCLE_LOCK", str(tmp_path / "model-stage.lock"))

    # This CPU case tests resource-binding precedence, not GPU inference.
    # Simulate an exact-model, healthy *unowned* endpoint at the HTTP
    # readiness boundary. The real lifecycle must borrow, never start/stop,
    # and still reject unavailable endpoints in the negative test below.
    readiness = []
    def healthy(resource):
        readiness.append((resource.name, resource.endpoint, resource.model))
        return True

    monkeypatch.setattr(resource_lifecycle, "_models_healthy", healthy)
    monkeypatch.setattr(resource_lifecycle, "_unit", lambda _unit: {
        "LoadState": "not-found", "FragmentPath": "", "Environment": "",
        "ActiveState": "inactive",
    })
    monkeypatch.setattr(resource_lifecycle, "_systemctl",
                        lambda *args: pytest.fail("Unowned resource must never be mutated"))
    constructed = []

    class FakeASR:
        def __init__(self, *, base_url, model, api_key=None):
            constructed.append((base_url, model))

        def transcribe(self, audio, config):
            return {"contract": {"raw_text": "synthetic"}, "raw_response": {}}

    monkeypatch.setattr(cli, "NemotronVLLMClient", FakeASR)
    assert cli.main(["transcribe", "synthetic.wav"]) == 0, capsys.readouterr()
    assert constructed[-1] == (
        "http://127.0.0.1:8190/v1",
        "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
    )

    legacy_model = "/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
    assert cli.main([
        "transcribe", "synthetic.wav",
        "--base-url", "http://127.0.0.1:8000/v1", "--model", legacy_model,
    ]) == 0, capsys.readouterr()
    assert constructed[-1] == ("http://127.0.0.1:8000/v1", legacy_model)
    assert readiness == [
        ("asr_local", "http://127.0.0.1:8190/v1",
         "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"),
        ("asr_local", "http://127.0.0.1:8000/v1", legacy_model),
    ]


def test_single_clip_cli_rejects_unavailable_unowned_asr(tmp_path, monkeypatch, capsys):
    from nvidia_transcribe import resource_lifecycle

    for key in ("ASR_BASE_URL", "ASR_MODEL", "NEMOVOX_MODEL_RESOURCES",
                "NEMOVOX_LIFECYCLE_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NEMOVOX_LIFECYCLE_LOCK", str(tmp_path / "model-stage.lock"))
    monkeypatch.setattr(resource_lifecycle, "_models_healthy", lambda _resource: False)
    monkeypatch.setattr(resource_lifecycle, "_unit", lambda _unit: {
        "LoadState": "not-found", "FragmentPath": "", "Environment": "",
        "ActiveState": "inactive",
    })
    monkeypatch.setattr(resource_lifecycle, "_systemctl",
                        lambda *args: pytest.fail("Unowned service must not be started"))

    class FakeASR:
        def __init__(self, *, base_url, model, api_key=None):
            pass
        def transcribe(self, audio, config):
            pytest.fail("Inference must never run before resource acquisition")

    monkeypatch.setattr(cli, "NemotronVLLMClient", FakeASR)
    assert cli.main(["transcribe", "synthetic.wav"]) == 1
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "RESOURCE_BLOCKED"
    assert error["reason"] == "managed_unit_not_configured"


def test_alternative_asr_checkpoint_is_not_fabricated_in_cli_configs(tmp_path, monkeypatch, capsys):
    path = _resource_file(tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["resources"]["other_asr"] = {
        "adapter": "nemotron_ctc", "endpoint": "http://localhost:8191/v1",
        "model": "other-ctc-model",
    }
    doc["roles"]["asr"] = "other_asr"
    path.write_text(json.dumps(doc), encoding="utf-8")

    class StopAtRecording(Exception):
        pass

    captured = {}
    def intercept_recording(media, *, output_dir, client, config):
        captured["model_path"] = config.model_path
        captured["client_model"] = client.model
        raise StopAtRecording()

    monkeypatch.setattr(cli, "process_media", intercept_recording)
    with pytest.raises(StopAtRecording):
        cli.main(["process-media", "synthetic-media.wav", "--output-dir",
                  str(tmp_path / "out"), "--model-resources", str(path)])
    assert captured == {"model_path": None, "client_model": "other-ctc-model"}

    with pytest.raises(StopAtRecording):
        cli.main(["process-media", "synthetic-media.wav", "--output-dir",
                  str(tmp_path / "out"), "--model-resources", str(path),
                  "--model-path", "/matching/other-checkpoint"])
    assert captured["model_path"] == "/matching/other-checkpoint"

    captured.clear()
    monkeypatch.setattr(cli, "generate_subtitles",
                        lambda *args, **kwargs: captured.setdefault("cfg", kwargs["config"]) and {"ok": True})
    code = cli.main([
        "generate-subtitles", "--translator", "openai_chat", "--recording-contract", "not-opened-by-fake",
        "--output-dir", str(tmp_path / "subs"), "--model-resources", str(path),
    ])
    assert code == 0, capsys.readouterr()
    assert captured["cfg"].asr_model_path is None


def test_ctc_without_checkpoint_disables_vocabulary_claims(monkeypatch):
    from nvidia_transcribe import alignment

    def must_not_load(_checkpoint):
        pytest.fail("No CTC vocabulary checkpoint was configured")

    monkeypatch.setattr(alignment, "load_ctc_vocabulary_chars", must_not_load)
    result = alignment.recover_chunk_alignment(
        raw_text="alternate language vocab", ctc_payload=None,
        source_offset_s=0.0, chunk_index=0, model_path=None)
    diagnostic = result["diagnostic"]
    assert diagnostic["ctc_vocabulary_status"] == "not_configured"
    assert diagnostic["oov_characters"] == []
    assert diagnostic["failure_cause"] != "ctc_vocab_oov"


def test_generate_subtitles_exposes_audio_stream_and_embedding_policy(tmp_path, monkeypatch):
    captured = []
    monkeypatch.delenv("NEMOVOX_MODEL_RESOURCES", raising=False)
    monkeypatch.setattr(cli, "generate_subtitles",
                        lambda *args, **kw: captured.append(kw["config"]) or {"ok": True})
    args = ["generate-subtitles", "--translator", "openai_chat", "--recording-contract",
            "not-read-by-fake", "--output-dir", str(tmp_path)]
    assert cli.main(args) == 0
    assert captured[-1].speaker_embeddings == "none"
    assert captured[-1].audio_stream_index == 0
    assert cli.main(args + ["--audio-stream", "2", "--speaker-embeddings", "titanet"]) == 0
    assert captured[-1].audio_stream_index == 2
    assert captured[-1].speaker_embeddings == "titanet"
