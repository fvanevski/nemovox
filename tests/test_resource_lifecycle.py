"""Deterministic CPU tests for automatic resource lifecycle and ownership."""
from __future__ import annotations

import json
from contextlib import contextmanager

import pytest

from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe import resource_lifecycle as lifecycle


@pytest.fixture
def text_resource():
    return load_model_resources(environ={}).resolve("translation_review")


@pytest.fixture
def manager(tmp_path):
    return lifecycle.ModelLifecycle(lock_path=tmp_path / "locks" / "models.lock")


def _unit(*, active=False, name="text_local", valid=True):
    return {
        "LoadState": "loaded",
        "FragmentPath": "/home/operator/.config/systemd/user/nemovox-text-review.service",
        "Environment": (
            "NEMOVOX_MANAGED_RESOURCE=1 NEMOVOX_RESOURCE_NAME=" + name
            + " NEMOVOX_RESOURCE_MODEL=chat"
            + " NEMOVOX_RESOURCE_ENDPOINT=http://127.0.0.1:8001/v1"
            if valid else "UNRELATED_SERVICE=1"
        ),
        "ActiveState": "active" if active else "inactive",
    }



def test_text_explicit_credential_overrides_legacy_fallback(text_resource, monkeypatch):
    from dataclasses import replace

    monkeypatch.setenv("VLLM_API_KEY", "legacy-token")
    monkeypatch.setenv("NEMOVOX_ASR_TOKEN_FOR_TEST", "explicit-token")
    selected = replace(text_resource, api_key_env="NEMOVOX_ASR_TOKEN_FOR_TEST")
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
    monkeypatch.setenv("VLLM_API_KEY", "legacy-text_resource-only")
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


def test_missing_explicit_text_token_remains_fail_closed(text_resource, monkeypatch):
    from dataclasses import replace
    from nvidia_transcribe.model_resources import ResourceConfigError

    selected = replace(text_resource, api_key_env="NEMOVOX_ABSENT_ASR_TOKEN_FOR_TEST")
    monkeypatch.delenv("NEMOVOX_ABSENT_ASR_TOKEN_FOR_TEST", raising=False)
    monkeypatch.setenv("VLLM_API_KEY", "legacy-must-not-override-explicit")
    with pytest.raises(ResourceConfigError, match="NEMOVOX_ABSENT_ASR_TOKEN_FOR_TEST"):
        lifecycle._models_healthy(selected)



@pytest.mark.parametrize("surface", ["inspect", "endpoint"])
def test_missing_explicit_token_blocks_before_host_service_access(
    manager, text_resource, monkeypatch, surface,
):
    from dataclasses import replace
    from nvidia_transcribe.model_resources import ResourceConfigError

    token_env = "NEMOVOX_MISSING_AUTH_TOKEN_FOR_TEST"
    selected = replace(text_resource, api_key_env=token_env)
    monkeypatch.delenv(token_env, raising=False)
    # The legacy fallback must never override a declared but absent key.
    monkeypatch.setenv("VLLM_API_KEY", "synthetic-legacy-token")

    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid explicit credentials must block before service or HTTP access")

    monkeypatch.setattr(lifecycle, "_unit", forbidden)
    monkeypatch.setattr(lifecycle, "_systemctl", forbidden)
    monkeypatch.setattr(lifecycle.urllib.request, "urlopen", forbidden)

    with pytest.raises(ResourceConfigError, match=token_env):
        if surface == "inspect":
            manager.inspect(selected)
        else:
            with manager.endpoint(selected):
                pytest.fail("Invalid configuration must not admit ASR inference")


def test_readiness_network_error_still_reports_unavailable(text_resource, monkeypatch):
    def unreachable(*_args, **_kwargs):
        raise lifecycle.urllib.error.URLError("synthetic unavailable endpoint")

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setattr(lifecycle.urllib.request, "urlopen", unreachable)
    assert lifecycle._models_healthy(text_resource) is False

def test_default_ownership_configuration_and_json_override(tmp_path):
    defaults = lifecycle._service_config()
    assert set(defaults) == {"text_local"}
    assert defaults["text_local"].unit == "nemovox-text-review.service"
    config = tmp_path / "lifecycle.json"
    config.write_text(json.dumps({
        "version": 1, "services": {
            "review_local": {"unit": "nemovox-review.service",
                             "startup_timeout_s": 300, "stop_after_stage": False}
        }
    }), encoding="utf-8")
    assert lifecycle._service_config(config)["review_local"].stop_after_stage is False
    assert "text_local" not in lifecycle._service_config(config)


@pytest.mark.parametrize("document", [
    {"version": 3, "services": {}},
    {"version": 1, "services": []},
    {"version": 1, "services": {"text_local": {"unit": "unrelated.service"}}},
    {"version": 1, "services": {"text_local": {"unit": "nemovox-text-review.service", "startup_timeout_s": 0}}},
    {"version": 1, "services": {"text_local": {"unit": "nemovox-text-review.service", "device": "cpu"}}},
])
def test_invalid_config_fails_closed(tmp_path, document):
    config = tmp_path / "invalid.json"
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        lifecycle._service_config(config)
    assert error.value.status == "invalid_lifecycle_config"


def test_healthy_unowned_endpoint_is_borrowed_without_mutation(manager, text_resource, monkeypatch):
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: True)
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(valid=False))
    attempted = []
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: attempted.append(args))
    with manager.endpoint(text_resource):
        assert attempted == []
    assert attempted == []


def test_missing_or_unowned_unit_never_starts(manager, text_resource, monkeypatch):
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: False)
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(valid=False))
    attempted = []
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: attempted.append(args))
    with pytest.raises(lifecycle.ResourceLifecycleError, match="no verified Nemovox-owned service") as error:
        with manager.endpoint(text_resource):
            pass
    assert error.value.status == "managed_unit_not_configured"
    assert attempted == []


def test_owned_text_started_and_released_around_one_stage(manager, text_resource, monkeypatch):
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
    with manager.endpoint(text_resource):
        assert state["active"] is True
        assert transitions == [("start", "nemovox-text-review.service")]
    assert state["active"] is False
    assert transitions[-1] == ("stop", "nemovox-text-review.service")


def test_active_owned_unit_with_live_other_client_is_not_stopped(manager, text_resource, monkeypatch):
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(active=True))
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: True)
    monkeypatch.setattr(lifecycle, "_external_clients", lambda _: True)
    transitions = []
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: transitions.append(args))
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        with manager.endpoint(text_resource):
            pass
    assert error.value.status == "managed_resource_busy"
    assert transitions == []


def test_owned_service_start_failure_does_not_escape_cleanup(manager, text_resource, monkeypatch):
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
        with manager.endpoint(text_resource):
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


def test_systemd_transport_unavailable_does_not_block_healthy_external_endpoint(manager, text_resource, monkeypatch):
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: True)
    def inaccessible(_):
        raise lifecycle.ResourceLifecycleError("systemd_unavailable")
    monkeypatch.setattr(lifecycle, "_unit", inaccessible)
    with manager.endpoint(text_resource):
        pass


def test_owned_unit_wrong_model_marker_cannot_start(manager, text_resource, monkeypatch):
    marker = _unit()
    marker["Environment"] = marker["Environment"].replace(
        "chat", "different-model"
    )
    monkeypatch.setattr(lifecycle, "_unit", lambda _: marker)
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: False)
    attempts = []
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: attempts.append(args))
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        with manager.endpoint(text_resource):
            pass
    assert error.value.status == "managed_unit_not_configured"
    assert attempts == []


def test_owned_text_start_requires_reported_capacity(manager, text_resource, monkeypatch):
    monkeypatch.setattr(lifecycle, "_service_config", lambda _: {"text_local": lifecycle.OwnedService("nemovox-text-review.service", min_free_vram_mb=16000)})
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit())
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: False)
    monkeypatch.setattr(lifecycle, "_free_vram_mb", lambda: 2000)
    attempts = []
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: attempts.append(args))
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        with manager.endpoint(text_resource):
            pass
    assert error.value.status == "gpu_capacity_blocked"
    assert attempts == []


def test_inspection_is_read_only_and_reports_startable(manager, text_resource, monkeypatch):
    calls = []
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(active=False))
    monkeypatch.setattr(lifecycle, "_models_healthy", lambda _: False)
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: calls.append(args))
    state = manager.inspect(text_resource)
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


def test_reconcile_idle_owned_reclaims_only_exact_abandoned_unit(manager, text_resource, monkeypatch):
    state = {"active": True}
    calls = []

    def unit(_unit_name):
        return {**_unit(active=state["active"]),
                "MainPID": "1009", "InvocationID": "same-invocation"}
    monkeypatch.setattr(lifecycle, "_unit", unit)
    monkeypatch.setattr(lifecycle, "_external_clients", lambda _: False)
    monkeypatch.setattr(lifecycle, "_service_config",
                        lambda _: {"text_local": lifecycle.OwnedService("nemovox-text-review.service")})

    def command(*args):
        calls.append(args)
        assert args == ("stop", "nemovox-text-review.service")
        state["active"] = False
        return ""
    monkeypatch.setattr(lifecycle, "_systemctl", command)
    assert manager.reconcile_idle_owned(resources={"text_local": text_resource}) == {"text_local": "recovered"}
    assert calls == [("stop", "nemovox-text-review.service")]
    assert manager.reconcile_idle_owned(resources={"text_local": text_resource}) == {"text_local": "inactive"}


def test_reconcile_refuses_unowned_or_mismatched_model_units(manager, text_resource, monkeypatch):
    attempts = []
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(valid=False, active=True))
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: attempts.append(args))
    monkeypatch.setattr(lifecycle, "_service_config",
                        lambda _: {"text_local": lifecycle.OwnedService("nemovox-text-review.service")})
    assert manager.reconcile_idle_owned(resources={"text_local": text_resource})["text_local"] == "unowned_or_identity_mismatch"
    assert attempts == []


def test_reconcile_respects_external_live_clients(manager, text_resource, monkeypatch):
    attempts = []
    # A properly identified owned unit is still untouchable while someone
    # else has an established client connection.
    monkeypatch.setattr(lifecycle, "_unit", lambda _: {
        **_unit(active=True), "MainPID": "1009",
        "InvocationID": "known-invocation",
    })
    monkeypatch.setattr(lifecycle, "_external_clients", lambda _: True)
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: attempts.append(args))
    monkeypatch.setattr(lifecycle, "_service_config",
                        lambda _: {"text_local": lifecycle.OwnedService("nemovox-text-review.service")})
    assert manager.reconcile_idle_owned(resources={"text_local": text_resource})["text_local"] == "active_clients"
    assert attempts == []


def test_reconcile_does_not_preempt_another_cli_lease(manager, text_resource, monkeypatch):
    import fcntl
    import os
    manager.lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(manager.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(lifecycle, "_systemctl",
                            lambda *args: pytest.fail("Recovery may not preempt a running stage"))
        assert manager.reconcile_idle_owned(resources={"text_local": text_resource}) == {"status": "active_model_stage"}
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)



def test_reconcile_requires_verified_unit_invocation_identity(manager, text_resource, monkeypatch):
    events = []
    monkeypatch.setattr(lifecycle, "_unit", lambda _: _unit(active=True))
    monkeypatch.setattr(lifecycle, "_external_clients", lambda _: False)
    monkeypatch.setattr(lifecycle, "_systemctl", lambda *args: events.append(args))
    monkeypatch.setattr(lifecycle, "_service_config",
                        lambda _: {"text_local": lifecycle.OwnedService("nemovox-text-review.service")})
    assert manager.reconcile_idle_owned(resources={"text_local": text_resource})["text_local"] == "unit_identity_unavailable"
    assert events == []



def test_cuda_worker_shortens_shared_lease_wait_without_preemption(manager, monkeypatch):
    monkeypatch.setattr(lifecycle, "_free_vram_mb", lambda: 20000)
    monkeypatch.setenv("NEMOVOX_LIFECYCLE_LOCK_TIMEOUT_S", "1")
    contender = lifecycle.ModelLifecycle(lock_path=manager.lock_path)
    with manager.worker(resource_name="whisper_local", lock_timeout_s=.1):
        with pytest.raises(lifecycle.ResourceLifecycleError) as error:
            with contender.worker(resource_name="riva_local", lock_timeout_s=.01):
                pytest.fail("Worker preempted active shared lease")
        assert error.value.status == "resource_lease_timeout"
    with contender.worker(resource_name="whisper_local", lock_timeout_s=.1):
        pass


@pytest.mark.parametrize("override", [-1., float("nan"), float("inf"), True])
def test_invalid_lease_wait_override_rejected(manager, override):
    with pytest.raises(lifecycle.ResourceLifecycleError) as error:
        with manager.worker(resource_name="whisper_local", lock_timeout_s=override):
            pytest.fail("Invalid override admitted")
    assert error.value.status == "invalid_lifecycle_lock_timeout"
