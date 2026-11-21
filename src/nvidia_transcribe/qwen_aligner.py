"""Selective Qwen3-ForcedAligner-0.6B fallback adapter via isolated Python runtime."""

from __future__ import annotations

import json
import math
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nvidia_transcribe.contract import normalize_language_code
from nvidia_transcribe.media import extract_wav_segment

DEFAULT_QWEN_ALIGNER_PATH = Path("/workspace/models/Qwen3-ForcedAligner-0.6B")
DEFAULT_QWEN_PYTHON = Path("/workspace/nvidia-transcribe/outputs/modular-eval/.venv/bin/python")
WORKER_SCRIPT = Path(__file__).with_name("_qwen_align_worker.py")

MAX_QWEN_ALIGN_SECONDS = 180.0

SUPPORTED_QWEN_LANGUAGES: dict[str, str] = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "fr": "French",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "ar": "Arabic",
    "id": "Indonesian",
    "th": "Thai",
    "vi": "Vietnamese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}


def map_language_for_qwen_aligner(language: str | None) -> str | None:
    """Map canonical/arbitrary language codes strictly to supported Qwen3-ForcedAligner languages."""
    if not language:
        return None
    low = language.strip().lower().replace("_", "-")
    if low in {"cantonese", "yue", "zh-hk", "yue-hant"}:
        return "Cantonese"
    norm = normalize_language_code(low)
    if norm in SUPPORTED_QWEN_LANGUAGES:
        return SUPPORTED_QWEN_LANGUAGES[norm]
    if "-" in low:
        base_raw = low.split("-", 1)[0]
        base = "zh" if base_raw == "cmn" else normalize_language_code(base_raw)
        if base in SUPPORTED_QWEN_LANGUAGES:
            return SUPPORTED_QWEN_LANGUAGES[base]
    return None


def _wav_duration_s(wav_path: Path) -> float | None:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if frames > 0 and rate > 0:
                return frames / float(rate)
    except Exception:
        return None
    return None


def validate_qwen_items(
    raw_items: list[dict[str, Any]],
    *,
    audio_duration_s: float | None,
    source_offset_s: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate raw worker items, rejecting zero-duration/out-of-bounds spans without clamping."""
    validated: list[dict[str, Any]] = []
    zero_duration_count = 0
    invalid_bounds_count = 0
    non_monotonic_count = 0
    prev_start: float | None = None
    tol = 0.05

    for item in raw_items:
        tok = str(item.get("text") or "").strip()
        if not tok:
            continue
        st = item.get("start")
        en = item.get("end")
        valid = True
        err: str | None = None

        if st is None or en is None or isinstance(st, bool) or isinstance(en, bool):
            valid = False
            err = "missing_timestamp"
        else:
            st_f, en_f = float(st), float(en)
            if not (math.isfinite(st_f) and math.isfinite(en_f)) or st_f < -1e-4 or en_f < st_f:
                valid = False
                err = "negative_or_inverted"
                invalid_bounds_count += 1
            elif audio_duration_s is not None and en_f > audio_duration_s + tol:
                valid = False
                err = "past_audio_duration"
                invalid_bounds_count += 1
            elif abs(en_f - st_f) <= 1e-6:
                valid = False
                err = "zero_duration"
                zero_duration_count += 1
            elif prev_start is not None and st_f + 1e-4 < prev_start:
                valid = False
                err = "non_monotonic"
                non_monotonic_count += 1

        if valid:
            st_f, en_f = float(st), float(en)
            prev_start = st_f
            abs_s = round(max(0.0, st_f) + source_offset_s, 6)
            abs_e = round(max(0.0, en_f) + source_offset_s, 6)
            unit_type = "character" if len(tok) == 1 and ord(tok) >= 0x4E00 else "word"
            validated.append(
                {
                    "text": tok,
                    "start": abs_s,
                    "end": abs_e,
                    "raw_start": round(st_f, 6),
                    "raw_end": round(en_f, 6),
                    "unit_type": unit_type,
                    "valid_timing": True,
                    "timing_error": None,
                }
            )
        else:
            # Reject invalid timestamps explicitly (NEVER stretch/clamp/interpolate)
            unit_type = "character" if len(tok) == 1 and ord(tok) >= 0x4E00 else "word"
            validated.append(
                {
                    "text": tok,
                    "start": None,
                    "end": None,
                    "raw_start": float(st) if isinstance(st, (int, float)) else None,
                    "raw_end": float(en) if isinstance(en, (int, float)) else None,
                    "unit_type": unit_type,
                    "valid_timing": False,
                    "timing_error": err,
                }
            )

    summary = {
        "total_items": len(validated),
        "valid_items": sum(1 for u in validated if u["valid_timing"]),
        "zero_duration_count": zero_duration_count,
        "invalid_bounds_count": invalid_bounds_count,
        "non_monotonic_count": non_monotonic_count,
    }
    return validated, summary


@dataclass(frozen=True)
class QwenAlignerConfig:
    model_path: str = str(DEFAULT_QWEN_ALIGNER_PATH)
    worker_python: str = str(DEFAULT_QWEN_PYTHON)
    device: str = "cuda:0"  # GPU required; CPU only with explicit device="cpu".
    timeout_s: float = 120.0
    startup_timeout_s: float = 120.0
    min_free_vram_bytes: int = 2_600_000_000


class QwenWorkerError(RuntimeError):
    """A typed, non-silent failure of the isolated forced-aligner worker."""

    def __init__(self, status: str, detail: str = "") -> None:
        super().__init__(detail or status)
        self.status = status


class QwenAlignerSession:
    """One isolated model load for multiple requests; no implicit CPU downgrade.

    JSONL worker protocol v1: init -> ready -> align* -> shutdown. Parent owns
    subprocess session and kills the entire worker group on every exit path.
    """

    def __init__(self, config: QwenAlignerConfig) -> None:
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self._stderr: Any = None
        self._buffer = bytearray()
        self._pending = bytearray()
        self._next_id = 1
        self.started = False
        self.failed = False
        self.metrics: dict[str, Any] = {
            "model_load_count": 0, "align_count": 0,
            "startup_wall_s": 0.0, "inference_wall_s": 0.0,
            "peak_gpu_allocated_bytes": None,
        }

    def __enter__(self) -> "QwenAlignerSession":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _read(self, timeout_s: float) -> dict[str, Any]:
        import os
        import select
        import time

        assert self.process is not None and self.process.stdout is not None
        fd = self.process.stdout.fileno()
        deadline = time.monotonic() + timeout_s
        while True:
            nl = self._buffer.find(b"\n")
            if nl >= 0:
                raw = bytes(self._buffer[:nl])
                del self._buffer[:nl + 1]
                if len(raw) > 4_000_000:
                    raise QwenWorkerError("malformed_worker_output", "Worker JSON response too large")
                try:
                    response = json.loads(raw.decode("utf-8"))
                except (UnicodeError, ValueError) as exc:
                    raise QwenWorkerError("malformed_worker_output", str(exc)) from exc
                if not isinstance(response, dict) or response.get("protocol") != 1:
                    raise QwenWorkerError("malformed_worker_output", "Worker protocol mismatch")
                return response
            if len(self._buffer) > 4_000_000:
                raise QwenWorkerError("malformed_worker_output", "Worker response exceeds 4 MB")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QwenWorkerError("worker_timeout", "Qwen worker deadline exceeded")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise QwenWorkerError("worker_timeout", "Qwen worker deadline exceeded")
            chunk = os.read(fd, 65536)
            if not chunk:
                raise QwenWorkerError("worker_exited", f"Qwen worker exited ({self.process.poll()})")
            self._buffer.extend(chunk)

    def _request(self, action: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        assert self.process is not None and self.process.stdin is not None
        req_id = self._next_id
        self._next_id += 1
        message = json.dumps({"protocol": 1, "id": req_id, "action": action, **payload}, ensure_ascii=False)
        if len(message.encode("utf-8")) > 100_000:
            raise QwenWorkerError("invalid_request", "Worker request too large")
        try:
            self.process.stdin.write((message + "\n").encode("utf-8"))
            self.process.stdin.flush()
            response = self._read(timeout_s)
        except (BrokenPipeError, OSError) as exc:
            raise QwenWorkerError("worker_exited", str(exc)) from exc
        if response.get("id") != req_id:
            raise QwenWorkerError("malformed_worker_output", "Worker request/response ID mismatch")
        if response.get("status") != "ok":
            raise QwenWorkerError(str(response.get("status") or "worker_error"),
                                  str(response.get("error") or "")[:300])
        return response

    def start(self) -> None:
        import tempfile
        import time

        if self.started:
            return
        if self.failed:
            raise QwenWorkerError("worker_failed", "Prior session failure; cannot reuse")
        py_path = Path(self.config.worker_python)
        model_dir = Path(self.config.model_path)
        if not py_path.is_file() or not model_dir.is_dir() or not WORKER_SCRIPT.is_file():
            raise QwenWorkerError("unavailable", "Qwen worker Python, model, or worker script unavailable")
        if self.config.device not in {"cuda:0", "auto", "cpu"}:
            raise QwenWorkerError("invalid_device", "Use cuda:0, auto (CUDA-required), or explicit cpu")
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        t0 = time.monotonic()
        try:
            self.process = subprocess.Popen(
                [str(py_path), str(WORKER_SCRIPT), "--jsonl"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, bufsize=0, start_new_session=True,
            )
            ready = self._request("init", {
                "model_path": str(model_dir),
                "device": "cuda:0" if self.config.device == "auto" else self.config.device,
                "min_free_vram_bytes": self.config.min_free_vram_bytes,
            }, self.config.startup_timeout_s)
            used = str(ready.get("device_used") or "")
            expected = "cpu" if self.config.device == "cpu" else "cuda:0"
            if used != expected:
                raise QwenWorkerError("device_mismatch", f"Expected {expected}, worker used {used}")
            self.metrics.update(
                model_load_count=ready.get("model_load_count", 0),
                device_used=used, dtype=ready.get("dtype"),
                observed_dtype=ready.get("observed_dtype"),
                cuda_allocated_bytes=ready.get("cuda_allocated_bytes"),
                model_load_wall_s=ready.get("model_load_wall_s"),
                startup_wall_s=round(time.monotonic() - t0, 4),
                peak_gpu_allocated_bytes=ready.get("peak_gpu_allocated_bytes"),
            )
            if self.metrics["model_load_count"] != 1:
                raise QwenWorkerError("unexpected_model_load_count", "Worker did not load exactly once")
            self.started = True
        except Exception:
            self.failed = True
            self.close()
            raise

    def align(
        self, *, audio_path: Path, text: str, language: str, source_offset_s: float,
    ) -> dict[str, Any]:
        import time

        self.start()
        t0 = time.monotonic()
        try:
            result = self._request("align", {
                "audio_path": str(audio_path), "text": text, "language": language,
            }, self.config.timeout_s)
        except Exception:
            self.failed = True
            self.close()
            raise
        elapsed = time.monotonic() - t0
        self.metrics["align_count"] += 1
        self.metrics["inference_wall_s"] += elapsed
        current_peak = result.get("peak_gpu_allocated_bytes")
        if isinstance(current_peak, (int, float)):
            old_peak = self.metrics.get("peak_gpu_allocated_bytes") or 0
            self.metrics["peak_gpu_allocated_bytes"] = max(old_peak, int(current_peak))
        return result

    def close(self) -> None:
        import os
        import signal

        process = self.process
        self.process = None
        self.started = False
        if process is not None:
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write(
                        (json.dumps({"protocol": 1, "id": self._next_id, "action": "shutdown"}) + "\n").encode()
                    )
                    process.stdin.flush()
                    process.wait(timeout=3)
            except (OSError, BrokenPipeError, subprocess.TimeoutExpired):
                pass
            # Reap remaining live group members even when the model-worker
            # leader already exited. Zombies are not mistaken for active GPU
            # consumers, and no unrelated system service is signaled.
            import time
            def group_live() -> bool:
                proc_root = Path("/proc")
                if proc_root.is_dir():
                    for entry in proc_root.iterdir():
                        if not entry.name.isdecimal():
                            continue
                        try:
                            stat = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                            if stat[0] not in {"Z", "X"} and int(stat[2]) == process.pid and int(stat[3]) == process.pid:
                                return True
                        except (OSError, IndexError, ValueError):
                            continue
                    return False
                try:
                    os.killpg(process.pid, 0)
                    return True
                except ProcessLookupError:
                    return False
            for sig in (signal.SIGTERM, signal.SIGKILL):
                if not group_live():
                    break
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + 2
                while group_live() and time.monotonic() < deadline:
                    time.sleep(0.05)
            process.wait(timeout=2)
            if group_live():
                raise QwenWorkerError("worker_cleanup_failed", "Qwen worker descendants remain")
            if process.stdin is not None:
                process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None


class QwenForcedAlignerAdapter:
    """Lazy session-backed adapter; one worker/model across multiple span alignments."""

    def __init__(self, config: QwenAlignerConfig | None = None) -> None:
        self.config = config or QwenAlignerConfig()
        self.session = QwenAlignerSession(self.config)

    def __enter__(self) -> "QwenForcedAlignerAdapter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.session.close()

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self.session.metrics)

    def align(self, *, audio_path: Path, text: str, language: str | None,
              source_offset_s: float) -> dict[str, Any]:
        qwen_lang = map_language_for_qwen_aligner(language)
        if qwen_lang is None:
            return {"units": [], "diagnostic": {
                "backend": "qwen3_forced_aligner", "status": "unsupported_language",
                "requested_language": language,
            }}
        py_path = Path(self.config.worker_python)
        model_dir = Path(self.config.model_path)
        if not py_path.is_file() or not model_dir.is_dir() or not WORKER_SCRIPT.is_file():
            return {"units": [], "diagnostic": {
                "backend": "qwen3_forced_aligner", "status": "unavailable",
                "worker_python": str(py_path), "model_path": str(model_dir),
            }}
        dur_s = _wav_duration_s(Path(audio_path))
        if dur_s is not None and dur_s > MAX_QWEN_ALIGN_SECONDS:
            return {"units": [], "diagnostic": {
                "backend": "qwen3_forced_aligner", "status": "duration_exceeds_max_limit",
                "audio_duration_s": round(dur_s, 3), "max_seconds": MAX_QWEN_ALIGN_SECONDS,
            }}
        try:
            raw_out = self.session.align(
                audio_path=Path(audio_path), text=text,
                language=qwen_lang, source_offset_s=source_offset_s,
            )
        except (QwenWorkerError, OSError, RuntimeError) as exc:
            return {"units": [], "diagnostic": {
                "backend": "qwen3_forced_aligner",
                "status": exc.status if isinstance(exc, QwenWorkerError) else "worker_error",
                "error_type": type(exc).__name__, "error": str(exc)[:300],
            }}
        items = raw_out.get("items")
        if not isinstance(items, list):
            return {"units": [], "diagnostic": {
                "backend": "qwen3_forced_aligner", "status": "malformed_worker_output",
            }}
        validated, stats = validate_qwen_items(
            items, audio_duration_s=dur_s, source_offset_s=source_offset_s,
        )
        return {"units": validated, "diagnostic": {
            "backend": "qwen3_forced_aligner",
            "status": "ok" if stats["valid_items"] else "no_valid_units",
            "qwen_language": qwen_lang, "device_used": raw_out.get("device_used"),
            "dtype": raw_out.get("dtype"), "quantized_resolution_s": 0.08,
            "inference_wall_s": raw_out.get("inference_wall_s"),
            "model_load_count": self.session.metrics["model_load_count"],
            "observed_dtype": self.session.metrics.get("observed_dtype"),
            "cuda_allocated_bytes": self.session.metrics.get("cuda_allocated_bytes"),
            "peak_gpu_allocated_bytes": self.session.metrics["peak_gpu_allocated_bytes"],
            **stats,
        }}
