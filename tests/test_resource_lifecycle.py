"""Deterministic CPU tests for automatic resource lifecycle and ownership."""
from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe import resource_lifecycle as lifecycle


@pytest.fixture
def asr():
    return load_model_resources(environ={}).resolve("asr")


@pytest.fixture
def manager(tmp_path):
    return lifecycle.ModelLifecycle(lock_path=tmp_path / "locks" / "models.lock")


def _unit(*, active=False, name="asr_local", valid=True):
    return {
        "LoadState": "loaded",
        "FragmentPath": "/home/operator/.config/systemd/user/nemovox-asr.service",
        "Environment": (
            "NEMOVOX_MANAGED_RESOURCE=1 NEMOVOX_RESOURCE_NAME=" + name
            + " NEMOVOX_RESOURCE_MODEL=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
            + " NEMOVOX_RESOURCE_ENDPOINT=http://127.0.0.1:8190/v1"
            if valid else "UNRELATED_SERVICE=1"
        ),
        "ActiveState": "active" if active else "inactive",
    }


def test_default_ownership_configuration_and_json_override(tmp_path):
    defaults = lifecycle._service_config()
    assert defaults["asr_local"].unit == "nemovox-asr.service"
    assert defaults["text_local"].unit == "nemovox-text-review.service"
    config = tmp_path / "lifecycle.json"
    config.write_text(json.dumps({
        "version": 1, "services": {
            "review_local": {"unit": "nemovox-review.service",
                             "startup_timeout_s": 300, "stop_after_stage": False}
        }
    }), encoding="utf-8")
    assert lifecycle._service_config(config)["review_local"].stop_after_stage is False
    assert "asr_local" not in lifecycle._service_config(config)


@pytest.mark.parametrize("document", [
    {"version": 3, "services": {}},
    {"version": 1, "services": []},
    {"version": 1, "services": {"asr_local": {"unit": "unrelated.service"}}},
    {"version": 1, "services": {"asr_local": {"unit": "nemovox-asr.service", "startup_timeout_s": 0}}},
    {"version": 1, "services": {"asr_local": {"unit": "nemovox-asr.service", "device": "cpu"}}},
])
def test_invalid_config_fails_closed(tmp_path, document):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        lifecycle._service_config(config)
    assert error.value.status == "invalid_lifecycle_config"


def test_healthy_unowned_endpoint_is_borrowed_without_mutation(manager, asr, monkeypatch):
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: True)
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(valid=False))
    attempted = []
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: attempted.append(args))
    with manager.endpoint(asr):
        assert attempted == []
    assert attempted == []


def test_missing_or_unowned_unit_never_starts(manager, asr, monkeypatch):
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: False)
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(valid=False))
    attempted = []
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: attempted.append(args))
    with pytest.raises(lifecycle.ResourceLifecycleError, match="no verified Nemovox-owned service") as error:
        with manager.endpoint(asr):
            pass
    assert error.value.status == "managed_unit_not_configured"
    assert attempted == []


def test_owned_asr_started_and_released_around_one_stage(manager, asr, monkeypatch):
    state = {"active": False}
    transitions = []
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(active=state["active"]))
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: state["active"])
    monkeypatch.setattr(lifecycle, "_external_clients", lambda _: False)
    def systemctl(*args):
        transitions.append(args)
        state["active"] = args[0] == "start"
        return ""
    monkeypatch.setattr(lifecycle, "_systemctl", systemctl)
    with manager.endpoint(asr):
        assert state["active"] is True
        assert transitions == [("start", "nemovox-asr.service")]
    assert state["active"] is False
    assert transitions[-1] == ("stop", "nemovox-asr.service")


def test_active_owned_unit_with_live_other_client_is_not_stopped(manager, asr, monkeypatch):
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(active=True))
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: True)
    monkeypatch.setattr(lifecycle, "_external_clients", lambda _: True)
    transitions = []
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: transitions.append(args))
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        with manager.endpoint(asr):
            pass
    assert error.value.status == "managed_resource_busy"
    assert transitions == []


def test_owned_service_start_failure_does_not_escape_cleanup(manager, asr, monkeypatch):
    state = {"active": False}
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(active=state["active"]))
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: state["active"])
    monkeypatch.setattr(lifecycle, "_external_clients", lambda _: False)
    def systemctl(*args):
        if args[0] == "start":
            state["active"] = True
            raise lifecycle.ResourceLifecycleError("start_transport_failed")
        state["active"] = False
    monkeypatch.setattr(lifecycle, "_systemctl", systemctl)
    with pytest.raises(lifecycle.ResourceLifecycleError):
        with manager.endpoint(asr):
            pass
    assert state["active"] is False


def test_worker_requires_gpu_capacity_and_never_implicitly_uses_cpu(manager, monkeypatch):
    monkeypatch.setattr(lifecycle, "_free_vram_mb", lambda: 3000)
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        with manager.worker(resource_name="riva_local"):
            pass
    assert error.value.status == "gpu_capacity_blocked"
    monkeypatch.setattr(lifecycle, "_free_vram_mb", lambda: 10000)
    with manager.worker(resource_name="riva_local"):
        pass


def test_text_client_is_lazily_started_only_when_inference_occurs(manager, monkeypatch):
    resource = load_model_resources(environ={}).resolve("translation_review")
    events = []

    @contextmanager
    def fake_endpoint(_resource):
        events.append("acquire")
        try:
            yield
        finally:
            events.append("release")

    class FakeClient:
        def __init__(self, **kwargs):
            events.append("create")
        def complete(self, **kwargs):
            events.append("inference")
            return {"choices": []}

    from nvidia_transcribe import refinement
    monkeypatch.setattr(refinement, "OpenAIChatTextClient", FakeClient)
    monkeypatch.setattr(manager, "endpoint", fake_endpoint)
    with lifecycle.managed_text_client(manager, resource) as client:
        assert events == []
    assert events == []
    with lifecycle.managed_text_client(manager, resource) as client:
        assert client.complete(messages=[]) == {"choices": []}
        assert client.complete(messages=[]) == {"choices": []}
    assert events == ["acquire", "create", "inference", "inference", "release"]


def test_systemd_transport_unavailable_does_not_block_healthy_external_endpoint(manager, asr, monkeypatch):
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: True)
    def inaccessible(_):
        raise lifecycle.ResourceLifecycleError("systemd_unavailable")
    monkeypatch.setattr(lifecycle, "_unit", inaccessible)
    with manager.endpoint(asr):
        pass


def test_owned_unit_wrong_model_marker_cannot_start(manager, asr, monkeypatch):
    marker = _unit()
    marker["Environment"] = marker["Environment"].replace(
        "NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4", "different-model"
    )
    monkeypatch.setattr(lifecycle, "_unit", lambda _: marker)
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: False)
    attempts = []
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: attempts.append(args))
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        with manager.endpoint(asr):
            pass
    assert error.value.status == "managed_unit_not_configured"
    assert attempts == []


def test_owned_asr_start_requires_reported_capacity(manager, asr, monkeypatch):
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit())
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: False)
    monkeypatch.setattr(lifecycle, "_free_vram_mb", lambda: 2000)
    attempts = []
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: attempts.append(args))
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        with manager.endpoint(asr):
            pass
    assert error.value.status == "gpu_capacity_blocked"
    assert attempts == []
