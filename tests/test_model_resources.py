"""CPU coverage of role/resource validation, precedence, and identity."""
import json

import pytest

from nvidia_transcribe.model_resources import (
    ResourceConfigError, load_model_resources,
)


def _file(tmp_path, payload):
    path = tmp_path / "model-resources.local.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_default_resource_roles_preserve_current_local_services():
    reg = load_model_resources(environ={})
    assert reg.resolve("asr").endpoint == "http://127.0.0.1:8190/v1"
    assert reg.resolve("source_refinement").model == "chat"
    assert reg.resolve("source_refinement") == reg.resolve("text_translation")
    assert reg.resolve("text_translation") == reg.resolve("subtitle_condensation")
    assert reg.resolve("forced_alignment").device == "cuda:0"


def test_host_bindings_support_distinct_text_roles(tmp_path):
    path = _file(tmp_path, {
        "version": 1,
        "resources": {
            "translator": {"adapter": "openai_chat", "endpoint": "http://localhost:8200/v1", "model": "translation-v2"},
            "condensor": {"adapter": "openai_chat", "endpoint": "http://localhost:8300/v1", "model": "concise"},
        },
        "roles": {"text_translation": "translator", "subtitle_condensation": "condensor"},
    })
    reg = load_model_resources(path, environ={})
    assert reg.resolve("source_refinement").name == "text_local"
    assert reg.resolve("text_translation").model == "translation-v2"
    assert reg.resolve("subtitle_condensation").model == "concise"
    assert len({reg.resolve(k).fingerprint() for k in
                ("source_refinement", "text_translation", "subtitle_condensation")}) == 3


def test_env_overrides_file_and_command_override_last(tmp_path):
    path = _file(tmp_path, {"version": 1, "resources": {
        "text_local": {"model": "file-model", "endpoint": "http://host.test:9000/v1"}}})
    reg = load_model_resources(path, environ={
        "TEXT_MODEL": "new", "TEXT_BASE_URL": "http://override.test/v1"})
    assert reg.resolve("source_refinement").model == "new"
    assert reg.resolve("source_refinement").endpoint == "http://override.test/v1"
    chosen = reg.resolve("source_refinement", model="cli-model")
    assert chosen.model == "cli-model"
    assert chosen.fingerprint() != reg.resolve("source_refinement").fingerprint()


@pytest.mark.parametrize("document", [
    {"version": 2},
    {"version": 1, "roles": {"unexpected": "text_local"}},
    {"version": 1, "roles": {"source_refinement": "asr_local"}},
    {"version": 1, "roles": {"text_translation": "missing"}},
    {"version": 1, "roles": {"text_translation": ["unhashable"]}},
    {"version": 1, "roles": {"text_translation": ""}},
    {"version": 1, "resources": {"malformed": {"adapter": "openai_chat", "endpoint": "file:///x", "model": "a"}}},
    {"version": 1, "resources": {"x": {"adapter": "openai_chat", "endpoint": "http://user:pass@host/v1", "model": "a"}}},
    {"version": 1, "resources": {"x": {"adapter": "openai_chat", "endpoint": "http://host:bad/v1", "model": "a"}}},
    {"version": 1, "resources": {"x": {"adapter": "openai_chat", "endpoint": "http://host/v1", "model": "a", "token": "secret"}}},
    {"version": 1, "resources": {"x": {"adapter": "qwen_forced_aligner", "checkpoint_path": "/nope", "worker_python": "/nope", "device": "cuda:2"}}},
])
def test_invalid_resource_config_fails_closed(tmp_path, document):
    with pytest.raises(ResourceConfigError):
        load_model_resources(_file(tmp_path, document), environ={})


def test_local_path_check_is_lazy_and_explicit(tmp_path):
    path = _file(tmp_path, {"version": 1, "resources": {
        "qwen_local": {"checkpoint_path": str(tmp_path / "missing")}}})
    reg = load_model_resources(path, environ={})
    assert reg.resolve("forced_alignment").checkpoint_path.endswith("missing")
    with pytest.raises(ResourceConfigError, match="unavailable"):
        reg.resolve("forced_alignment", require_local_paths=True)


def test_safe_bindings_and_credentials_never_expose_values():
    reg = load_model_resources(environ={})
    for value in reg.public_bindings().values():
        assert set(value) == {"resource", "adapter", "fingerprint"}
        assert "workspace" not in str(value)
    spec = reg.resolve("source_refinement")
    assert spec.credential(environ={"TEXT_TOKEN": "secret"}) is None
    with pytest.raises(ResourceConfigError, match="Unknown task role"):
        reg.resolve("make_coffee")


def test_overrides_do_not_mutate_registry():
    reg = load_model_resources(environ={})
    orig = reg.resolve("source_refinement")
    changed = reg.resolve("source_refinement", endpoint="http://localhost:8444/v1")
    assert changed.endpoint != orig.endpoint
    assert reg.resolve("source_refinement") == orig


@pytest.mark.parametrize("key", ["TEXT_BASE_URL", "TEXT_MODEL"])
def test_explicit_empty_env_override_fails_closed(key):
    with pytest.raises(ResourceConfigError, match="Empty environment override"):
        load_model_resources(environ={key: " "})


@pytest.mark.parametrize("key", ["SWIFTU_BASE_URL", "SWIFTU_MODEL"])
def test_retired_text_environment_alias_fails_closed(key):
    with pytest.raises(ResourceConfigError, match="Retired SWIFTU"):
        load_model_resources(environ={key: "old-value"})


@pytest.mark.parametrize("adapter,fields", [
    ("titanet", {"model": "titanet_small", "device": "cuda:0"}),
    ("titanet", {"model": "titanet_small", "checkpoint_path": "/some/checkpoint"}),
    ("openai_chat", {"model": "chat", "endpoint": "http://localhost:8001/v1", "device": "cuda:0"}),
    ("openai_chat", {"model": "chat", "endpoint": "http://localhost:8001/v1", "worker_python": "/tmp/python"}),
    ("riva_worker", {"checkpoint_path": "/tmp/riva", "worker_python": "/tmp/python", "device": "not-a-device"}),
    ("qwen_forced_aligner", {"checkpoint_path": "/tmp/qwen", "worker_python": "/tmp/python", "api_key_env": "TOKEN"}),
    ("nemotron_ctc", {"model": "asr", "endpoint": "http://localhost:8190/v1", "device": "cpu"}),
])
def test_rejects_unsupported_adapter_fields_and_silent_device_fallback(tmp_path, adapter, fields):
    path = _file(tmp_path, {"version": 1, "resources": {"invalid_resource": {"adapter": adapter, **fields}}})
    with pytest.raises(ResourceConfigError):
        load_model_resources(path, environ={})


def test_configured_cpu_titanet_is_valid(tmp_path):
    path = _file(tmp_path, {"version": 1, "resources": {
        "titanet_cpu": {"adapter": "titanet", "model": "titanet_small", "device": "cpu"}}})
    assert load_model_resources(path, environ={}).resolve("speaker_embedding").device == "cpu"



def test_alternate_asr_resource_without_checkpoint_has_no_borrowed_vocabulary(tmp_path):
    path = _file(tmp_path, {"version": 1, "resources": {
        "alternative": {"adapter": "nemotron_ctc", "endpoint": "http://localhost:8191/v1",
                        "model": "served-alternative"}},
        "roles": {"asr": "alternative"}})
    reg = load_model_resources(path, environ={})
    assert reg.resolve("asr").checkpoint_path is None
    assert reg.resolve("asr", model="another-model").checkpoint_path is None
    assert reg.resolve("asr").model == "served-alternative"


def test_asr_model_override_clears_inherited_checkpoint_unless_explicit(tmp_path):
    original = load_model_resources(environ={}).resolve("asr")
    assert original.checkpoint_path is not None
    assert load_model_resources(environ={"ASR_MODEL": "different"}).resolve("asr").checkpoint_path is None
    assert load_model_resources(environ={}).resolve("asr", model="different").checkpoint_path is None

    file_override = _file(tmp_path, {"version": 1, "resources": {
        "asr_local": {"model": "other-model"}}})
    assert load_model_resources(file_override, environ={}).resolve("asr").checkpoint_path is None

    explicit_checkpoint = "/matching/checkpoint"
    file_override = _file(tmp_path, {"version": 1, "resources": {
        "asr_local": {"model": "other-model", "checkpoint_path": explicit_checkpoint}}})
    assert load_model_resources(file_override, environ={}).resolve("asr").checkpoint_path == explicit_checkpoint
    assert load_model_resources(file_override, environ={"ASR_MODEL": "override"}).resolve("asr").checkpoint_path is None


def test_asr_endpoint_only_override_retains_default_checkpoint():
    reg = load_model_resources(environ={"ASR_BASE_URL": "http://localhost:8200/v1"})
    assert reg.resolve("asr").checkpoint_path == load_model_resources(environ={}).resolve("asr").checkpoint_path



def test_translation_review_role_defaults_to_text_and_rejects_specialized_adapter(tmp_path):
    registry = load_model_resources(environ={})
    review = registry.resolve("translation_review")
    assert review.adapter == "openai_chat"
    assert review.name == "text_local"
    assert registry.resolve("riva_translation").device == "cuda:0"



def test_optional_independent_challenger_requires_explicit_local_cuda_binding(tmp_path):
    assert "source_challenger" not in load_model_resources(environ={}).roles
    spec={"adapter":"whisper_ct2","checkpoint_path":"/local/model",
          "worker_python":"/local/python","device":"cuda:0"}
    path=_file(tmp_path,{"version":1,"resources":{"whisper":spec},
                         "roles":{"source_challenger":"whisper"}})
    assert load_model_resources(path,environ={}).resolve("source_challenger").device=="cuda:0"
    with pytest.raises(ResourceConfigError,match="explicitly configured cuda:0"):
        load_model_resources(_file(tmp_path,{"version":1,"resources":{"whisper":{**spec,"device":"cpu"}},
                                             "roles":{"source_challenger":"whisper"}}),environ={})
