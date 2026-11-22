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



@pytest.mark.parametrize("token", ["secret-for-test", None])
def test_asr_readiness_auth_matches_legacy_inference_client(asr, monkeypatch, token):
    """Readiness and real ASR requests must use the same fallback credentials."""
    import os
    from nvidia_transcribe.client import NemotronVLLMClient

    if token is None:
        monkeypatch.delenv("VLLM_API_KEY", raising=False)
    else:
        monkeypatch.setenv("VLLM_API_KEY", token)
    seen = []

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, maximum):
            assert maximum == 250_000
            return json.dumps({"data": [{"id": asr.model}]}).encode()

    def fake_urlopen(request, *, timeout):
        assert timeout == 3.0
        seen.append((request.full_url, request.get_header("Authorization")))
        return Response()

    monkeypatch.setattr(lifecycle.urllib.request, "urlopen", fake_urlopen)
    assert lifecycle._models_healthy(asr)
    inference = NemotronVLLMClient(base_url=asr.endpoint, model=asr.model)
    assert seen == [(asr.endpoint + "/models", inference._headers()["Authorization"])]
    assert seen[0][1] == "Bearer " + (token or "EMPTY")


def test_asr_explicit_credential_overrides_legacy_fallback(asr, monkeypatch):
    from dataclasses import replace

    monkeypatch.setenv("VLLM_API_KEY", "legacy-token")
    monkeypatch.setenv("NEMOVOX_ASR_TOKEN_FOR_TEST", "explicit-token")
    selected = replace(asr, api_key_env="NEMOVOX_ASR_TOKEN_FOR_TEST")
    seen = []

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, limit):
            return json.dumps({"data": [{"id": selected.model}]}).encode()

    def fake_urlopen(request, *, timeout):
        seen.append(request.get_header("Authorization"))
        return Response()

    monkeypatch.setattr(lifecycle.urllib.request, "urlopen", fake_urlopen)
    assert lifecycle._models_healthy(selected)
    assert seen == ["Bearer explicit-token"]


def test_legacy_asr_token_does_not_leak_to_text_readiness(monkeypatch):
    resource = load_model_resources(environ={}).resolve("translation_review")
    monkeypatch.setenv("VLLM_API_KEY", "legacy-asr-only")
    seen = []

    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, limit):
            return json.dumps({"data": [{"id": resource.model}]}).encode()

    def fake_urlopen(request, *, timeout):
        seen.append(request.get_header("Authorization"))
        return Response()

    monkeypatch.setattr(lifecycle.urllib.request, "urlopen", fake_urlopen)
    assert lifecycle._models_healthy(resource)
    assert seen == [None]


def test_missing_explicit_asr_token_remains_fail_closed(asr, monkeypatch):
    from dataclasses import replace
    from nvidia_transcribe.model_resources import ResourceConfigError

    selected = replace(asr, api_key_env="NEMOVOX_ABSENT_ASR_TOKEN_FOR_TEST")
    monkeypatch.delenv("NEMOVOX_ABSENT_ASR_TOKEN_FOR_TEST", raising=False)
    monkeypatch.setenv("VLLM_API_KEY", "legacy-must-not-override-explicit")
    with pytest.raises(ResourceConfigError, match="NEMOVOX_ABSENT_ASR_TOKEN_FOR_TEST"):
        lifecycle._models_healthy(selected)


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
    monkeypatch.setattr(lifecycle, "_free_vram_mb", lambda: 31000)
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
    monkeypatch.setattr(lifecycle, "_free_vram_mb", lambda: 31000)
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


def test_inspection_is_read_only_and_reports_startable(manager, asr, monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(active=False))
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: False)
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: calls.append(args))
    state = manager.inspect(asr)
    assert state["state"] == "startable"
    assert state["startable"] is True
    assert calls == []


def test_offline_constructor_never_reads_invalid_lifecycle_config(tmp_path):
    # The rerender path instantiates a manager but never acquires a model.
    missing = tmp_path / "missing.json"
    manager = lifecycle.ModelLifecycle(config_path=missing, lock_path=tmp_path / "no-touch.lock")
    assert not (tmp_path / "no-touch.lock").exists()
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        _ = manager.services
    assert error.value.status == "invalid_lifecycle_config"


def test_lock_restores_sigterm_handler_and_rejects_symlink(tmp_path):
    import signal
    manager = lifecycle.ModelLifecycle(lock_path=tmp_path / "resources.lock")
    prior = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with manager._lock():
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
    assert signal.getsignal(signal.SIGTERM) is prior
    assert manager.lock_path.is_file()

    manager.lock_path.unlink()
    destination = tmp_path / "other-file"
    destination.write_text("preserve", encoding="utf-8")
    manager.lock_path.symlink_to(destination)
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        with manager._lock():
            pass
    assert error.value.status == "resource_lock_unavailable"
    assert destination.read_text(encoding="utf-8") == "preserve"


def test_resource_lease_wait_is_bounded(tmp_path, monkeypatch):
    import fcntl
    import os
    lock = tmp_path / "busy.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setenv("NEMOVOX_LIFECYCLE_LOCK_TIMEOUT_S", "0.1")
        contender = lifecycle.ModelLifecycle(lock_path=lock)
        with pytest.raises(lifecycle.ResourceLifecycleError) as error:
            with contender._lock():
                pass
        assert error.value.status == "resource_lease_timeout"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_cached_recording_does_not_acquire_asr_before_first_transcribe(manager, asr, monkeypatch):
    events = []

    @contextmanager
    def fake_endpoint(_):
        events.append("start")
        try:
            yield
        finally:
            events.append("stop")

    class FakeASR:
        model = "nemotron"
        base_url = "http://127.0.0.1:8190/v1"

        def transcribe(self, filename, config=None):
            events.append("transcribe")
            return filename

    monkeypatch.setattr(manager, "endpoint", fake_endpoint)
    with lifecycle.managed_asr_client(manager, asr, FakeASR()) as client:
        assert client.model == "nemotron"
        assert client.base_url.endswith("/v1")
        assert events == []
    assert events == []
    with lifecycle.managed_asr_client(manager, asr, FakeASR()) as client:
        assert client.transcribe("fixture.wav") == "fixture.wav"
        assert client.transcribe("next.wav") == "next.wav"
    assert events == ["start", "transcribe", "transcribe", "stop"]
