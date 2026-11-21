"""Opt-in reversible control of a *cooperatively owned* Nemovox ASR user unit.

No generic GPU scheduler, guessed service names, or permission escalation.
Unmarked, busy, non-Nemovox or shared services are never stopped.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


class ManagedASRBlocked(RuntimeError):
    """Typed non-mutating admission failure, or a detected lifecycle fault."""

    def __init__(self, status: str, detail: str = "") -> None:
        super().__init__(detail or status)
        self.status = status


@dataclass(frozen=True)
class ManagedASRConfig:
    unit: str
    lease_path: Path
    base_url: str
    served_model: str
    journal_path: Path
    timeout_s: float = 45.0
    min_free_vram_mb: int = 2600

    def __post_init__(self) -> None:
        if not re.fullmatch(r"nemovox-[A-Za-z0-9_.@-]+\.service", self.unit):
            raise ValueError("Managed ASR unit must be explicitly named nemovox-*.service")
        if self.unit == "nemovox-studio.service":
            raise ValueError("Studio service is not an ASR-owned unit")
        parsed = urllib.parse.urlsplit(self.base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Managed ASR service must use loopback HTTP")
        if not 1 <= (parsed.port or 0) <= 65535:
            raise ValueError("Managed ASR base URL must declare a port")
        if not self.served_model.strip() or not 5 <= self.timeout_s <= 180:
            raise ValueError("Invalid managed ASR model/timeout")


def _run_systemctl(*args: str) -> str:
    try:
        proc = subprocess.run(
            ["systemctl", "--user", *args],
            check=False, text=True, capture_output=True, timeout=12,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ManagedASRBlocked("systemctl_unavailable", type(exc).__name__) from exc
    if proc.returncode:
        raise ManagedASRBlocked("systemctl_failed", proc.stderr[-250:])
    return proc.stdout


def _unit_properties(unit: str) -> dict[str, str]:
    keys = "LoadState,ActiveState,MainPID,InvocationID,ExecStart,Environment,FragmentPath"
    out = _run_systemctl("show", unit, "--no-page", "--property="+keys)
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def _free_vram_mb() -> int | None:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=5, check=False,
        )
        if p.returncode == 0:
            return int(p.stdout.splitlines()[0].strip())
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
        pass
    return None


def _models_healthy(base_url: str, model: str, timeout: float = 4.0) -> bool:
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    try:
        with urllib.request.urlopen(url + "/models", timeout=timeout) as response:
            if response.status != 200:
                return False
            result = json.loads(response.read(500_000))
            return any(str(m.get("id")) == model for m in result.get("data", []))
    except (OSError, ValueError, KeyError, urllib.error.URLError):
        return False


def _wait_unit(unit: str, active: bool, deadline: float) -> dict[str, str]:
    while time.monotonic() < deadline:
        props = _unit_properties(unit)
        state = props.get("ActiveState")
        if active and state == "active" and int(props.get("MainPID") or 0) > 0:
            return props
        if not active and state in {"inactive", "failed"}:
            return props
        time.sleep(0.3)
    raise ManagedASRBlocked("unit_transition_timeout", f"{unit} did not reach expected state")


def _has_exact_cooperative_markers(environment: str, lease_path: Path) -> bool:
    """Match full systemd Environment assignments, never path prefixes.

    systemctl show uses whitespace-separated assignments, with quoted values
    where needed. Malformed or duplicate ownership assignments fail closed.
    """
    try:
        assignments = shlex.split(environment)
    except ValueError:
        return False
    observed: dict[str, list[str]] = {
        "NEMOVOX_MANAGED_ASR": [], "NEMOVOX_ASR_LEASE_PATH": [],
    }
    for assignment in assignments:
        if "=" not in assignment:
            return False
        name, value = assignment.split("=", 1)
        if name in observed:
            observed[name].append(value)
    return (
        observed["NEMOVOX_MANAGED_ASR"] == ["1"]
        and observed["NEMOVOX_ASR_LEASE_PATH"] == [str(lease_path)]
    )


def _verified_owner(cfg: ManagedASRConfig) -> dict[str, str]:
    props = _unit_properties(cfg.unit)
    if props.get("LoadState") != "loaded" or props.get("ActiveState") != "active":
        raise ManagedASRBlocked("asr_unit_not_active")
    if not props.get("FragmentPath") or not props.get("InvocationID"):
        raise ManagedASRBlocked("asr_owner_unknown")
    pid = int(props.get("MainPID") or 0)
    if pid <= 0:
        raise ManagedASRBlocked("asr_owner_unknown")
    # The marker is a specific opt-in contract installed by the ASR unit's
    # operator, not a wildcard permission to stop any vLLM service.
    env = props.get("Environment") or ""
    if not _has_exact_cooperative_markers(env, cfg.lease_path):
        raise ManagedASRBlocked("asr_not_cooperatively_owned")
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ManagedASRBlocked("asr_owner_unknown", type(exc).__name__) from exc
    port = str(urllib.parse.urlsplit(cfg.base_url).port)
    if (port not in cmdline or
        "vllm" not in (cmdline + props.get("ExecStart", "")).lower() or
        "nemotron" not in (cmdline + props.get("ExecStart", "")).lower()):
        raise ManagedASRBlocked("asr_identity_mismatch")
    if not _models_healthy(cfg.base_url, cfg.served_model):
        raise ManagedASRBlocked("asr_model_health_mismatch")
    # Catch live non-cooperating clients at the handoff boundary. This is
    # supplemental to, not a replacement for, the cooperative lease contract.
    try:
        observed = subprocess.run(
            ["ss", "-Htn", "state", "established",
             f"( sport = :{urllib.parse.urlsplit(cfg.base_url).port} )"],
            text=True, capture_output=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ManagedASRBlocked("asr_clients_unverifiable") from exc
    if observed.returncode != 0:
        raise ManagedASRBlocked("asr_clients_unverifiable")
    if observed.stdout.strip():
        raise ManagedASRBlocked("asr_has_live_connections")
    return props


def _check_lease(path: Path) -> Any:
    path = path.resolve()
    try:
        st = path.stat()
    except OSError as exc:
        raise ManagedASRBlocked("asr_lease_missing", type(exc).__name__) from exc
    if st.st_uid != os.getuid() or st.st_mode & 0o077 or not path.is_file():
        raise ManagedASRBlocked("asr_lease_permissions")
    file = path.open("a+b")
    try:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        file.close()
        raise ManagedASRBlocked("asr_busy") from exc
    return file


def _pending_journal(path: Path) -> bool:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False
    except (OSError, ValueError) as exc:
        raise ManagedASRBlocked("handoff_journal_unreadable") from exc
    return state.get("phase") not in {"restored", "not_stopped"}


def _write_journal(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fp:
            json.dump(entry, fp, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def managed_asr_handoff(cfg: ManagedASRConfig) -> Iterator[dict[str, Any]]:
    """Release a cooperatively owned ASR instance only if CUDA lacks free VRAM.

    A journal is durably written before stop. On normal/error/interrupt,
    release proceeds first, then systemd restores the exact same pre-existing
    unit and confirms the exact served model. An interrupted journal blocks
    future automatic takeovers: operator must inspect/restore service first.
    """
    telemetry: dict[str, Any] = {"mode": "managed", "released": False, "restored": False}
    # A previous interrupted release must never be ignored simply because
    # VRAM happens to be free (the ASR unit might still be stopped).
    if _pending_journal(cfg.journal_path):
        raise ManagedASRBlocked("prior_handoff_requires_recovery")
    free = _free_vram_mb()
    telemetry["free_vram_before_mb"] = free
    if free is None:
        raise ManagedASRBlocked("gpu_telemetry_unavailable")
    if free >= cfg.min_free_vram_mb:
        telemetry["status"] = "cuda_already_available"
        yield telemetry
        return
    lease = _check_lease(cfg.lease_path)
    stopped = False
    original: dict[str, str] | None = None
    previous_sigterm: Any = None
    sigterm_installed = False
    try:
        # Unlike SIGKILL, SIGTERM can execute Python finally blocks. Treat
        # Studio/CLI cancellation as an exception so the owned ASR instance
        # is restored before the CLI exits. Non-main threads cannot install
        # signal handlers; their cancellation still requires normal context
        # unwinding or explicit interrupted-journal recovery.
        if threading.current_thread() is threading.main_thread():
            previous_sigterm = signal.getsignal(signal.SIGTERM)

            def _cancel_managed_handoff(_signum: int, _frame: Any) -> None:
                raise KeyboardInterrupt("Managed ASR handoff interrupted by SIGTERM")

            signal.signal(signal.SIGTERM, _cancel_managed_handoff)
            sigterm_installed = True
        original = _verified_owner(cfg)
        telemetry["owner_invocation_id"] = original.get("InvocationID")
        _write_journal(cfg.journal_path, {
            "phase": "pending_restore", "unit": cfg.unit,
            "invocation_id": original.get("InvocationID"),
            "served_model": cfg.served_model,
            "base_url": cfg.base_url, "originally_active": True,
            "unit_exec_sha256": hashlib.sha256(
                original.get("ExecStart", "").encode("utf-8")
            ).hexdigest(),
            "unit_fragment_path": original.get("FragmentPath"),
        })
        # From this point even a systemctl transport error is ambiguous:
        # always inspect/restore the prior unit in the finally path.
        stopped = True
        _run_systemctl("stop", cfg.unit)
        _wait_unit(cfg.unit, False, time.monotonic() + cfg.timeout_s)
        deadline = time.monotonic() + min(cfg.timeout_s, 30)
        while time.monotonic() < deadline:
            released_mb = _free_vram_mb()
            if released_mb is not None and released_mb >= cfg.min_free_vram_mb:
                telemetry["free_vram_after_release_mb"] = released_mb
                break
            time.sleep(0.5)
        else:
            raise ManagedASRBlocked("vram_not_released")
        telemetry["released"] = True
        yield telemetry
    finally:
        try:
            if stopped and original is not None:
                current = _unit_properties(cfg.unit)
                if current.get("ActiveState") != "active":
                    _run_systemctl("start", cfg.unit)
                    current = _wait_unit(cfg.unit, True, time.monotonic() + cfg.timeout_s)
                # An already-running unit can be the original process when
                # stop failed without side effects; do not restart it needlessly.
                if current.get("ActiveState") != "active":
                    raise ManagedASRBlocked("asr_restore_not_active")
                deadline = time.monotonic() + cfg.timeout_s
                while time.monotonic() < deadline:
                    if _models_healthy(cfg.base_url, cfg.served_model):
                        telemetry["restored"] = True
                        break
                    time.sleep(0.5)
                if not telemetry["restored"]:
                    raise ManagedASRBlocked("asr_restore_health_failed")
                _write_journal(cfg.journal_path, {
                    "phase": "restored", "unit": cfg.unit,
                    "original_invocation_id": original.get("InvocationID"),
                    "restored_invocation_id": current.get("InvocationID"),
                })
        finally:
            try:
                if sigterm_installed:
                    signal.signal(signal.SIGTERM, previous_sigterm)
            finally:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
                lease.close()


def recover_pending_managed_asr(cfg: ManagedASRConfig) -> dict[str, Any]:
    """Explicit crash-recovery operation: revalidate unit identity before start.

    This is NOT called automatically by an ordinary subtitle processing run.
    It never restarts a service unless its prior active state, unit file,
    original executable, cooperative marker and lease are all evidenced.
    """
    path = cfg.journal_path
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "no_pending_handoff"}
    except (OSError, ValueError) as exc:
        raise ManagedASRBlocked("handoff_journal_unreadable") from exc
    if data.get("phase") == "restored":
        return {"status": "already_restored"}
    if not (
        data.get("phase") == "pending_restore"
        and data.get("unit") == cfg.unit
        and data.get("served_model") == cfg.served_model
        and data.get("base_url") == cfg.base_url
        and data.get("originally_active") is True
        and data.get("unit_exec_sha256")
        and data.get("unit_fragment_path")
    ):
        raise ManagedASRBlocked("handoff_journal_identity_mismatch")
    lease = _check_lease(cfg.lease_path)
    try:
        props = _unit_properties(cfg.unit)
        env = props.get("Environment") or ""
        if props.get("LoadState") != "loaded":
            raise ManagedASRBlocked("asr_unit_not_loaded")
        if not _has_exact_cooperative_markers(env, cfg.lease_path):
            raise ManagedASRBlocked("asr_not_cooperatively_owned")
        if props.get("FragmentPath") != data["unit_fragment_path"]:
            raise ManagedASRBlocked("asr_unit_changed_since_handoff")
        if hashlib.sha256(props.get("ExecStart", "").encode()).hexdigest() != data["unit_exec_sha256"]:
            raise ManagedASRBlocked("asr_unit_changed_since_handoff")
        command = props.get("ExecStart", "").lower()
        port = str(urllib.parse.urlsplit(cfg.base_url).port)
        if "vllm" not in command or "nemotron" not in command or port not in command:
            raise ManagedASRBlocked("asr_identity_mismatch")
        if props.get("ActiveState") != "active":
            if props.get("ActiveState") not in {"inactive", "failed"}:
                raise ManagedASRBlocked("asr_state_unknown")
            _run_systemctl("start", cfg.unit)
            props = _wait_unit(cfg.unit, True, time.monotonic() + cfg.timeout_s)
        deadline = time.monotonic() + cfg.timeout_s
        while time.monotonic() < deadline:
            if _models_healthy(cfg.base_url, cfg.served_model):
                _write_journal(path, {
                    "phase": "restored", "unit": cfg.unit,
                    "restored_invocation_id": props.get("InvocationID"),
                    "recovered_from": data.get("invocation_id"),
                })
                return {"status": "restored", "unit": cfg.unit, "health_verified": True}
            time.sleep(0.5)
        raise ManagedASRBlocked("asr_restore_health_failed")
    finally:
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()
