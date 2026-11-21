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
    device: str = "auto"
    timeout_s: float = 120.0


class QwenForcedAlignerAdapter:
    """``ExternalForcedAligner`` adapter invoking Qwen3-ForcedAligner-0.6B in its isolated venv."""

    def __init__(self, config: QwenAlignerConfig | None = None) -> None:
        self.config = config or QwenAlignerConfig()

    def align(
        self,
        *,
        audio_path: Path,
        text: str,
        language: str | None,
        source_offset_s: float,
    ) -> dict[str, Any]:
        qwen_lang = map_language_for_qwen_aligner(language)
        if qwen_lang is None:
            return {
                "units": [],
                "diagnostic": {
                    "backend": "qwen3_forced_aligner",
                    "status": "unsupported_language",
                    "requested_language": language,
                },
            }

        py_path = Path(self.config.worker_python)
        model_dir = Path(self.config.model_path)
        if not py_path.is_file() or not model_dir.is_dir() or not WORKER_SCRIPT.is_file():
            return {
                "units": [],
                "diagnostic": {
                    "backend": "qwen3_forced_aligner",
                    "status": "unavailable",
                    "worker_python": str(py_path),
                    "model_path": str(model_dir),
                },
            }

        dur_s = _wav_duration_s(Path(audio_path))
        if dur_s is not None and dur_s > MAX_QWEN_ALIGN_SECONDS:
            return {
                "units": [],
                "diagnostic": {
                    "backend": "qwen3_forced_aligner",
                    "status": "duration_exceeds_max_limit",
                    "audio_duration_s": round(dur_s, 3),
                    "max_seconds": MAX_QWEN_ALIGN_SECONDS,
                },
            }

        payload = {
            "model_path": str(model_dir),
            "audio_path": str(audio_path),
            "text": text,
            "language": qwen_lang,
            "device": self.config.device,
        }
        try:
            proc = subprocess.run(
                [str(py_path), str(WORKER_SCRIPT)],
                input=json.dumps(payload, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_s,
            )
        except Exception as exc:
            return {
                "units": [],
                "diagnostic": {
                    "backend": "qwen3_forced_aligner",
                    "status": "worker_invocation_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }

        if proc.returncode != 0:
            return {
                "units": [],
                "diagnostic": {
                    "backend": "qwen3_forced_aligner",
                    "status": "worker_error",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr[-500:],
                },
            }

        try:
            raw_out = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:
            return {
                "units": [],
                "diagnostic": {
                    "backend": "qwen3_forced_aligner",
                    "status": "malformed_worker_output",
                    "error": str(exc),
                },
            }

        if raw_out.get("status") != "ok":
            return {
                "units": [],
                "diagnostic": {
                    "backend": "qwen3_forced_aligner",
                    "status": "aligner_failed",
                    "error": raw_out.get("error"),
                },
            }

        validated, val_stats = validate_qwen_items(
            raw_out.get("items") or [],
            audio_duration_s=dur_s,
            source_offset_s=source_offset_s,
        )
        return {
            "units": validated,
            "diagnostic": {
                "backend": "qwen3_forced_aligner",
                "status": "ok" if val_stats["valid_items"] > 0 else "no_valid_units",
                "qwen_language": qwen_lang,
                "device_used": raw_out.get("device_used"),
                "dtype": raw_out.get("dtype"),
                "quantized_resolution_s": 0.08,
                **val_stats,
            },
        }
