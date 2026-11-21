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


def test_env_new_overrides_legacy_and_command_override_last(tmp_path):
    path = _file(tmp_path, {"version": 1, "resources": {
        "text_local": {"model": "file-model", "endpoint": "http://host.test:9000/v1"}}})
    reg = load_model_resources(path, environ={
        "SWIFTU_MODEL": "legacy", "TEXT_MODEL": "new",
        "SWIFTU_BASE_URL": "http://legacy.test/v1"})
    assert reg.resolve("source_refinement").model == "new"
    assert reg.resolve("source_refinement").endpoint == "http://legacy.test/v1"
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


def test_explicit_empty_new_env_does_not_silently_fall_back_to_legacy():
    with pytest.raises(ResourceConfigError, match="Empty environment override"):
        load_model_resources(environ={
            "SWIFTU_BASE_URL": "http://127.0.0.1:8001/v1",
            "TEXT_BASE_URL": "",
        })
    with pytest.raises(ResourceConfigError, match="Empty environment override"):
        load_model_resources(environ={
            "SWIFTU_MODEL": "chat",
            "TEXT_MODEL": " ",
        })
