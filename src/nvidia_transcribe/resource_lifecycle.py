"""Cooperative, job-scoped acquisition of *owned* model services.

The model registry selects inference identities; this module owns only lifecycle.
A healthy unowned endpoint may be used, never stopped or reconfigured.
Only explicitly marked Nemovox systemd user units can be started/stopped.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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


_DEFAULT_UNITS = {
    "asr_local": OwnedService("nemovox-asr.service"),
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
        if set(value) - {"unit", "startup_timeout_s", "stop_after_stage"}:
            raise ResourceLifecycleError("invalid_lifecycle_config", f"Unsupported service fields for {name}")
        unit = value["unit"]
        timeout = value.get("startup_timeout_s", 240)
        release = value.get("stop_after_stage", True)
        if not _UNIT_RE.fullmatch(unit) or not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not (10 <= timeout <= 900) or not isinstance(release, bool):
            raise ResourceLifecycleError("invalid_lifecycle_config", f"Invalid unit, timeout or release policy for {name}")
        parsed[name] = OwnedService(unit=unit, startup_timeout_s=float(timeout), stop_after_stage=release)
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


def _owned(properties: dict[str, str], name: str) -> bool:
    if properties.get("LoadState") != "loaded" or not properties.get("FragmentPath"):
        return False
    try:
        assignments = shlex.split(properties.get("Environment") or "")
    except ValueError:
        return False
    markers: dict[str, list[str]] = {"NEMOVOX_MANAGED_RESOURCE": [], "NEMOVOX_RESOURCE_NAME": []}
    for assignment in assignments:
        if "=" not in assignment:
            return False
        key, value = assignment.split("=", 1)
        if key in markers:
            markers[key].append(value)
    return markers["NEMOVOX_MANAGED_RESOURCE"] == ["1"] and markers["NEMOVOX_RESOURCE_NAME"] == [name]


def _models_healthy(resource: ModelResource, timeout: float = 3.0) -> bool:
    if not resource.endpoint or not resource.model:
        return False
    url = resource.endpoint.rstrip("/") + "/models"
    headers = {}
    # The selected resource's credential is redacted from all status/log text.
    try:
        credential = resource.credential()
        if credential:
            headers["Authorization"] = "Bearer " + credential
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


class ModelLifecycle:
    """Serialize local model work and start/release only cooperative resources.

    The flock spans each complete model stage (not GUI lifetime). Every CLI
    process, including those started by Studio, uses the same lock location.
    """

    def __init__(self, *, config_path: str | Path | None = None,
                 lock_path: str | Path | None = None) -> None:
        self.services = _service_config(config_path)
        default = Path.home() / ".local/state/nemovox/model-lifecycle.lock"
        self.lock_path = Path(lock_path or os.environ.get("NEMOVOX_LIFECYCLE_LOCK", default)).expanduser()

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.lock_path.open("a+b") as handle:
            os.chmod(self.lock_path, 0o600)
            # Different CLI processes cannot overlap GPU stages.
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def endpoint(self, resource: ModelResource) -> Iterator[None]:
        if resource.adapter not in {"nemotron_ctc", "openai_chat"}:
            raise ResourceLifecycleError("unsupported_lifecycle_adapter", resource.adapter)
        with self._lock():
            # A matching already-serving external endpoint is usable but never ours to stop.
            # A managed endpoint still obeys its explicit ownership/release policy.
            config = self.services.get(resource.name)
            properties = _unit(config.unit) if config else None
            owned = bool(config and properties and _owned(properties, resource.name))
            healthy = _models_healthy(resource)
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
            if not healthy:
                if properties.get("ActiveState") not in {"inactive", "failed"}:
                    raise ResourceLifecycleError("owned_service_busy", resource.name)
                _systemctl("start", config.unit)
                started = True
                deadline = time.monotonic() + config.startup_timeout_s
                while time.monotonic() < deadline:
                    if _models_healthy(resource):
                        break
                    current = _unit(config.unit)
                    if not _owned(current, resource.name) or current.get("ActiveState") == "failed":
                        raise ResourceLifecycleError("managed_start_failed", resource.name)
                    time.sleep(1.0)
                else:
                    raise ResourceLifecycleError("managed_start_timeout", resource.name)
            try:
                yield
            finally:
                if config.stop_after_stage and (started or active):
                    # Never stop a renamed/replaced unit or one with unverified ownership.
                    current = _unit(config.unit)
                    if not _owned(current, resource.name):
                        raise ResourceLifecycleError("managed_owner_changed", resource.name)
                    _systemctl("stop", config.unit)
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if _unit(config.unit).get("ActiveState") == "inactive":
                            break
                        time.sleep(0.3)
                    else:
                        raise ResourceLifecycleError("managed_stop_unverified", resource.name)

    @contextmanager
    def worker(self, *, resource_name: str, min_free_vram_mb: int = 8500) -> Iterator[None]:
        """One on-demand Riva/Qwen worker, never a persistent service takeover."""
        with self._lock():
            free = _free_vram_mb()
            if free is None:
                raise ResourceLifecycleError("gpu_telemetry_unavailable", resource_name)
            if free < min_free_vram_mb:
                raise ResourceLifecycleError("gpu_capacity_blocked",
                                             f"{resource_name}: insufficient free VRAM for explicitly configured CUDA worker")
            yield
