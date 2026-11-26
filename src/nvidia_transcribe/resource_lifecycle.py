"""Cooperative, job-scoped acquisition of *owned* model services.

The model registry selects inference identities; this module owns only lifecycle.
A healthy unowned endpoint may be used, never stopped or reconfigured.
Only explicitly marked Nemovox systemd user units can be started/stopped.
"""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
import signal
import threading
import shlex
import stat
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from pathlib import Path
from typing import Iterator, Mapping

from nvidia_transcribe.model_resources import ModelResource


class ResourceLifecycleError(RuntimeError):
    def __init__(self, status: str, detail: str = "") -> None:
        self.status = status
        super().__init__(detail or status)


@dataclass(frozen=True)
class OwnedService:
    unit: str
    startup_timeout_s: float = 240.0
    stop_after_stage: bool = True
    min_free_vram_mb: int = 0


_DEFAULT_UNITS = {
    "text_local": OwnedService("nemovox-text-review.service"),
}
_UNIT_RE = re.compile(r"^nemovox-[a-zA-Z0-9_.@-]+\.service$")
_RESOURCE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _is_loopback(endpoint: str | None) -> bool:
    parsed = urllib.parse.urlsplit(endpoint or "")
    return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}


def _service_config(path: str | Path | None = None) -> dict[str, OwnedService]:
    """Load declarative ownership; no startup, file creation, or inference."""
    selected = str(path) if path is not None else os.environ.get("NEMOVOX_LIFECYCLE_CONFIG")
    if not selected:
        return dict(_DEFAULT_UNITS)
    try:
        doc = json.loads(Path(selected).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ResourceLifecycleError("invalid_lifecycle_config", type(exc).__name__) from exc
    if not isinstance(doc, dict) or set(doc) != {"version", "services"} or doc["version"] != 1:
        raise ResourceLifecycleError("invalid_lifecycle_config", "Expected version 1 and services mapping")
    rows = doc["services"]
    if not isinstance(rows, dict):
        raise ResourceLifecycleError("invalid_lifecycle_config", "services must be a mapping")
    parsed: dict[str, OwnedService] = {}
    for name, value in rows.items():
        if not isinstance(name, str) or not _RESOURCE_RE.fullmatch(name):
            raise ResourceLifecycleError("invalid_resource_name")
        if not isinstance(value, dict) or not isinstance(value.get("unit"), str):
            raise ResourceLifecycleError("invalid_lifecycle_config", f"Invalid service definition for {name}")
        if set(value) - {"unit", "startup_timeout_s", "stop_after_stage", "min_free_vram_mb"}:
            raise ResourceLifecycleError("invalid_lifecycle_config", f"Unsupported service fields for {name}")
        unit = value["unit"]
        timeout = value.get("startup_timeout_s", 240)
        release = value.get("stop_after_stage", True)
        min_free = value.get("min_free_vram_mb", 0)
        if not _UNIT_RE.fullmatch(unit) or not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not (10 <= timeout <= 900) or not isinstance(release, bool) or not isinstance(min_free, int) or isinstance(min_free, bool) or not (0 <= min_free <= 120000):
            raise ResourceLifecycleError("invalid_lifecycle_config", f"Invalid unit, timeout or release policy for {name}")
        parsed[name] = OwnedService(unit=unit, startup_timeout_s=float(timeout), stop_after_stage=release,
                                    min_free_vram_mb=min_free)
    return parsed


def _systemctl(*args: str) -> str:
    try:
        p = subprocess.run(["systemctl", "--user", *args],
                           text=True, capture_output=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ResourceLifecycleError("systemd_unavailable", type(exc).__name__) from exc
    if p.returncode:
        raise ResourceLifecycleError("systemd_failed", p.stderr[-300:])
    return p.stdout


def _unit(unit: str) -> dict[str, str]:
    keys = "LoadState,ActiveState,MainPID,InvocationID,Environment,FragmentPath"
    raw = _systemctl("show", unit, "--no-page", "--property=" + keys)
    return dict(x.split("=", 1) for x in raw.splitlines() if "=" in x)


def _owned(properties: dict[str, str], resource: ModelResource) -> bool:
    if properties.get("LoadState") != "loaded" or not properties.get("FragmentPath"):
        return False
    try:
        assignments = shlex.split(properties.get("Environment") or "")
    except ValueError:
        return False
    markers: dict[str, list[str]] = {"NEMOVOX_MANAGED_RESOURCE": [], "NEMOVOX_RESOURCE_NAME": [],
        "NEMOVOX_RESOURCE_MODEL": [], "NEMOVOX_RESOURCE_ENDPOINT": []}
    for assignment in assignments:
        if "=" not in assignment:
            return False
        key, value = assignment.split("=", 1)
        if key in markers:
            markers[key].append(value)
    return (markers["NEMOVOX_MANAGED_RESOURCE"] == ["1"]
            and markers["NEMOVOX_RESOURCE_NAME"] == [resource.name]
            and markers["NEMOVOX_RESOURCE_MODEL"] == [resource.model]
            and markers["NEMOVOX_RESOURCE_ENDPOINT"] == [resource.endpoint])


def _models_healthy(resource: ModelResource, timeout: float = 3.0) -> bool:
    if not resource.endpoint or not resource.model:
        return False
    url = resource.endpoint.rstrip("/") + "/models"
    headers = {}
    # Resolve configuration before the fallible network probe: ResourceConfigError
    # subclasses ValueError, but a missing explicit token is never "unhealthy".
    # The selected credential must never be written to status or logs.
    credential = resource.credential()
    if credential:
        headers["Authorization"] = "Bearer " + credential
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read(250_000))
        return any(isinstance(row, dict) and row.get("id") == resource.model
                   for row in payload.get("data", []))
    except (OSError, ValueError, TypeError, KeyError, urllib.error.URLError):
        return False


def _free_vram_mb() -> int | None:
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           text=True, capture_output=True, timeout=5, check=False)
        if p.returncode == 0:
            return int(p.stdout.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def _external_clients(port: int | None) -> bool:
    """Unknown or failed inspection is not permission to evict clients."""
    if port is None:
        return True
    try:
        p = subprocess.run(["ss", "-Htn", "state", "established", f"( sport = :{port} )"],
                           text=True, capture_output=True, timeout=5, check=False)
        return p.returncode != 0 or bool(p.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return True


class ModelLifecycle:
    """Serialize local model work and start/release only cooperative resources.

    The flock spans each complete model stage (not GUI lifetime). Every CLI
    process, including those started by Studio, uses the same lock location.
    """

    def __init__(self, *, config_path: str | Path | None = None,
                 lock_path: str | Path | None = None) -> None:
        self.config_path = config_path
        self._services: dict[str, OwnedService] | None = None
        default = Path.home() / ".local/state/nemovox/model-lifecycle.lock"
        self.lock_path = Path(lock_path or os.environ.get("NEMOVOX_LIFECYCLE_LOCK", default)).expanduser()

    @property
    def services(self) -> dict[str, OwnedService]:
        # Cached on first real acquisition/status inspection. Offline rerender
        # must not fail merely because a dormant service config is unavailable.
        if self._services is None:
            self._services = _service_config(self.config_path)
        return self._services

    def inspect(self, resource: ModelResource) -> dict[str, str | bool]:
        """Read-only status for Studio; never starts/stops or loads any model."""
        if resource.adapter == "moss_events":
            installed = (Path(resource.checkpoint_path or "").is_dir()
                         and Path(resource.checkpoint_path or "", "config.json").is_file()
                         and bool(resource.worker_python) and Path(resource.worker_python).is_file())
            # This is availability, not proof of runtime or GPU admission.
            return {"state": "startable_local_worker" if installed else "worker_unavailable",
                    "ready": False, "managed": True, "startable": installed}
        healthy = _models_healthy(resource)
        settings = self.services.get(resource.name)
        if not settings:
            return {"state": "ready_external" if healthy else "unmanaged_unavailable",
                    "ready": healthy, "managed": False, "startable": False}
        try:
            state = _unit(settings.unit)
        except ResourceLifecycleError:
            return {"state": "ready_external" if healthy else "manager_unavailable",
                    "ready": healthy, "managed": False, "startable": False}
        owned = _owned(state, resource)
        if not owned:
            return {"state": "ready_external" if healthy else "managed_unit_not_configured",
                    "ready": healthy, "managed": False, "startable": False}
        if healthy:
            return {"state": "ready_managed", "ready": True,
                    "managed": True, "startable": True}
        unit_state = state.get("ActiveState")
        if unit_state == "activating":
            status = "starting"
        elif unit_state in {"inactive", "failed"}:
            status = "startable"
        else:
            status = "managed_model_mismatch"
        return {"state": status, "ready": False, "managed": True,
                "startable": status == "startable"}

    @contextmanager
    def _lock(self, *, timeout_override_s: float | None = None) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            fd = os.open(self.lock_path,
                         os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
                         0o600)
        except OSError as exc:
            raise ResourceLifecycleError("resource_lock_unavailable", type(exc).__name__) from exc
        with os.fdopen(fd, "a+b") as handle:
            st = os.fstat(handle.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
                raise ResourceLifecycleError("resource_lock_permissions")
            # Bounded cross-CLI admission, never an infinite wait on stale work.
            try:
                timeout = float(os.environ.get("NEMOVOX_LIFECYCLE_LOCK_TIMEOUT_S", "900"))
            except ValueError as exc:
                raise ResourceLifecycleError("invalid_lifecycle_lock_timeout") from exc
            if not 0.1 <= timeout <= 7200:
                raise ResourceLifecycleError("invalid_lifecycle_lock_timeout")
            if timeout_override_s is not None:
                # Zero preserves nonblocking orphan recovery. A worker can
                # shorten, but never extend, the configured shared lease wait.
                if (type(timeout_override_s) not in (int, float)
                        or not math.isfinite(timeout_override_s) or timeout_override_s < 0):
                    raise ResourceLifecycleError("invalid_lifecycle_lock_timeout")
                timeout = min(timeout, timeout_override_s)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() >= deadline:
                        raise ResourceLifecycleError("resource_lease_timeout") from exc
                    time.sleep(min(0.25, max(0., deadline - time.monotonic())))
            prior_sigterm = None
            restore_signal = threading.current_thread() is threading.main_thread()
            if restore_signal:
                prior_sigterm = signal.getsignal(signal.SIGTERM)

                def _on_sigterm(_signum, _frame):
                    # Studio cancellation sends TERM to the CLI's process
                    # group. Raising in Python unwinds owned-unit cleanup.
                    raise KeyboardInterrupt("Nemovox model-stage cancelled by SIGTERM")

                signal.signal(signal.SIGTERM, _on_sigterm)
            try:
                yield
            finally:
                try:
                    if restore_signal:
                        signal.signal(signal.SIGTERM, prior_sigterm)
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def reconcile_idle_owned(
        self, *, resources: Mapping[str, ModelResource] | None = None,
    ) -> dict[str, str]:
        """Reclaim orphaned owned units after a CLI subprocess dies.

        Caller must be a trusted Studio controller at startup or after a
        subprocess exits, never a passive status endpoint. No lock preemption.
        """
        try:
            with self._lock(timeout_override_s=0.0):
                from nvidia_transcribe.model_resources import load_model_resources

                resolved = resources if resources is not None else load_model_resources().resources
                results: dict[str, str] = {}
                for name, configured in self.services.items():
                    resource = resolved.get(name)
                    if not configured.stop_after_stage or resource is None:
                        results[name] = "not_recoverable"
                        continue
                    try:
                        before = _unit(configured.unit)
                    except ResourceLifecycleError:
                        results[name] = "unit_unavailable"
                        continue
                    if not _owned(before, resource):
                        results[name] = "unowned_or_identity_mismatch"
                        continue
                    if before.get("ActiveState") != "active":
                        results[name] = "inactive"
                        continue
                    # A loaded unit marker alone does not prove which
                    # invocation this controller would terminate.
                    if (not before.get("InvocationID")
                            or not before.get("MainPID")
                            or before.get("MainPID") == "0"):
                        results[name] = "unit_identity_unavailable"
                        continue
                    try:
                        port = urllib.parse.urlsplit(resource.endpoint or "").port
                    except ValueError:
                        results[name] = "invalid_endpoint"
                        continue
                    if _external_clients(port):
                        results[name] = "active_clients"
                        continue
                    # Re-read invocation identity before touching systemd.
                    current = _unit(configured.unit)
                    if (not _owned(current, resource)
                            or current.get("ActiveState") != "active"
                            or current.get("MainPID") != before.get("MainPID")
                            or current.get("InvocationID") != before.get("InvocationID")):
                        results[name] = "unit_changed"
                        continue
                    _systemctl("stop", configured.unit)
                    deadline = time.monotonic() + 30.0
                    while time.monotonic() < deadline:
                        if _unit(configured.unit).get("ActiveState") in {"inactive", "failed"}:
                            results[name] = "recovered"
                            break
                        time.sleep(0.3)
                    else:
                        raise ResourceLifecycleError("orphan_recovery_stop_unverified", name)
                return results
        except ResourceLifecycleError as exc:
            if exc.status == "resource_lease_timeout":
                return {"status": "active_model_stage"}
            raise

    @contextmanager
    def endpoint(self, resource: ModelResource) -> Iterator[None]:
        if resource.adapter != "openai_chat":
            raise ResourceLifecycleError("unsupported_lifecycle_adapter", resource.adapter)
        with self._lock():
            # A matching already-serving external endpoint is usable but never ours to stop.
            # A managed endpoint still obeys its explicit ownership/release policy.
            healthy = _models_healthy(resource)
            config = self.services.get(resource.name)
            try:
                properties = _unit(config.unit) if config else None
            except ResourceLifecycleError:
                if healthy:
                    # Healthy unmanaged endpoints never need systemd access.
                    yield
                    return
                raise
            owned = bool(config and properties and _owned(properties, resource))
            if healthy and not owned:
                yield
                return
            if not _is_loopback(resource.endpoint):
                if not healthy:
                    raise ResourceLifecycleError("unmanaged_endpoint_unavailable",
                                                 f"{resource.name}: configured endpoint is not ready")
                yield
                return
            if not config or not owned:
                if not healthy:
                    raise ResourceLifecycleError("managed_unit_not_configured",
                                                 f"{resource.name}: no verified Nemovox-owned service for unavailable endpoint")
                yield
                return
            active = properties.get("ActiveState") == "active"
            if active and not healthy:
                raise ResourceLifecycleError("owned_service_identity_mismatch",
                                             f"{resource.name}: active unit does not serve configured model")
            started = False
            try:
                if not healthy:
                    if properties.get("ActiveState") not in {"inactive", "failed"}:
                        raise ResourceLifecycleError("owned_service_busy", resource.name)
                    if config.min_free_vram_mb:
                        free = _free_vram_mb()
                        if free is None:
                            raise ResourceLifecycleError("gpu_telemetry_unavailable", resource.name)
                        if free < config.min_free_vram_mb:
                            raise ResourceLifecycleError("gpu_capacity_blocked",
                                                         f"{resource.name}: insufficient free VRAM for managed startup")
                    # Failed startup must enter finally too.
                    started = True
                    _systemctl("start", config.unit)
                    deadline = time.monotonic() + config.startup_timeout_s
                    while time.monotonic() < deadline:
                        if _models_healthy(resource):
                            break
                        current = _unit(config.unit)
                        if not _owned(current, resource) or current.get("ActiveState") == "failed":
                            raise ResourceLifecycleError("managed_start_failed", resource.name)
                        time.sleep(1.0)
                    else:
                        raise ResourceLifecycleError("managed_start_timeout", resource.name)
                yield
            finally:
                if config.stop_after_stage and (started or active):
                    # Never stop a replaced unit or one with other live clients.
                    current = _unit(config.unit)
                    if not _owned(current, resource):
                        raise ResourceLifecycleError("managed_owner_changed", resource.name)
                    if current.get("ActiveState") == "active":
                        port = urllib.parse.urlsplit(resource.endpoint or "").port
                        if _external_clients(port):
                            raise ResourceLifecycleError("managed_resource_busy",
                                                         f"{resource.name}: other live clients are connected")
                        _systemctl("stop", config.unit)
                        deadline = time.monotonic() + 30
                        while time.monotonic() < deadline:
                            if _unit(config.unit).get("ActiveState") in {"inactive", "failed"}:
                                break
                            time.sleep(0.3)
                        else:
                            raise ResourceLifecycleError("managed_stop_unverified", resource.name)

    @contextmanager
    def worker(self, *, resource_name: str, min_free_vram_mb: int = 8500,
               lock_timeout_s: float | None = None) -> Iterator[None]:
        """One on-demand CUDA worker, never a persistent service takeover."""
        with self._lock(timeout_override_s=lock_timeout_s):
            free = _free_vram_mb()
            if free is None:
                raise ResourceLifecycleError("gpu_telemetry_unavailable", resource_name)
            if free < min_free_vram_mb:
                raise ResourceLifecycleError("gpu_capacity_blocked",
                                             f"{resource_name}: insufficient free VRAM for explicitly configured CUDA worker")
            yield



class _DeferredTextClient:
    """Acquire a text service at the first uncached inference request."""

    def __init__(self, manager: ModelLifecycle, resource: ModelResource, *,
                 timeout_s: float = 180.0, reasoning_effort: str | None = None,
                 wire_protocol: str = "openai_chat") -> None:
        if wire_protocol not in ("openai_chat", "ninfer_messages"):
            raise ValueError("Unsupported managed text-review wire protocol")
        self.manager = manager
        self.resource = resource
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        self.wire_protocol = wire_protocol
        self._scope = None
        self._client = None

    def complete(self, **kwargs):
        if self._scope is None:
            scope = self.manager.endpoint(self.resource)
            scope.__enter__()
            self._scope = scope
            if self.wire_protocol == "ninfer_messages":
                from nvidia_transcribe.ninfer_messages_client import NInferMessagesTextClient
                client_type = NInferMessagesTextClient
            else:
                from nvidia_transcribe.refinement import OpenAIChatTextClient
                client_type = OpenAIChatTextClient
            self._client = client_type(
                base_url=self.resource.endpoint or "",
                model=self.resource.model or "",
                api_key=self.resource.credential(),
                timeout_s=self.timeout_s, reasoning_effort=self.reasoning_effort,
            )
        return self._client.complete(**kwargs)

    def close(self) -> None:
        if self._scope is not None:
            scope, self._scope = self._scope, None
            scope.__exit__(None, None, None)


@contextmanager
def managed_text_client(manager: ModelLifecycle, resource: ModelResource, *,
                        timeout_s: float = 180.0,
                        reasoning_effort: str | None = None,
                        wire_protocol: str = "openai_chat") -> Iterator[_DeferredTextClient]:
    """Use owned service lifecycle; opt in to NInfer Messages for bounded QC."""
    client = _DeferredTextClient(manager, resource, timeout_s=timeout_s,
                                 reasoning_effort=reasoning_effort,
                                 wire_protocol=wire_protocol)
    try:
        yield client
    finally:
        client.close()
