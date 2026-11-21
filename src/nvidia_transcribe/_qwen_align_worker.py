#!/usr/bin/env python3
"""JSONL session worker for a single local Qwen3 ForcedAligner model load.

Only the parent process owns orchestration. The worker cannot stop any other
service and cannot silently fall back to CPU for a CUDA request.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _reply(request_id: object, status: str, **fields: Any) -> None:
    sys.stdout.write(json.dumps({
        "protocol": 1, "id": request_id, "status": status, **fields,
    }, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv != ["--jsonl"]:
        print("Qwen forced-aligner worker requires --jsonl protocol v1", file=sys.stderr)
        return 2
    os.environ.setdefault("HF_HOME", "/workspace/models/.hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/workspace/models/.hf-cache/hub")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    aligner: Any = None
    torch: Any = None
    device: str | None = None
    dtype: str | None = None
    model_load_count = 0
    for line in sys.stdin:
        if len(line) > 100_000:
            _reply(None, "invalid_request", error="Input exceeds 100 KB")
            continue
        try:
            req = json.loads(line)
            if not isinstance(req, dict) or req.get("protocol") != 1:
                raise ValueError("Protocol mismatch")
            action = req.get("action")
            req_id = req.get("id")
            if action == "shutdown":
                _reply(req_id, "ok")
                return 0
            if action == "init":
                if aligner is not None:
                    _reply(req_id, "already_initialized")
                    continue
                model_path = Path(str(req.get("model_path") or ""))
                requested = str(req.get("device") or "cuda:0")
                if requested not in {"cuda:0", "cpu"}:
                    _reply(req_id, "invalid_device")
                    continue
                if not model_path.is_dir():
                    _reply(req_id, "unavailable", error="Model directory missing")
                    continue
                import torch as torch_module
                torch = torch_module
                if requested == "cuda:0":
                    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
                        _reply(req_id, "blocked_gpu_unavailable")
                        continue
                    if not torch.cuda.is_bf16_supported():
                        _reply(req_id, "blocked_bf16_unsupported")
                        continue
                    free_b, _ = torch.cuda.mem_get_info(0)
                    if free_b < int(req.get("min_free_vram_bytes") or 2_600_000_000):
                        _reply(req_id, "blocked_insufficient_vram", free_bytes=free_b)
                        continue
                device = requested
                use_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
                t0 = time.monotonic()
                from qwen_asr import Qwen3ForcedAligner
                aligner = Qwen3ForcedAligner.from_pretrained(
                    str(model_path), dtype=use_dtype,
                    device_map=device, local_files_only=True,
                )
                dtype = str(use_dtype)
                observed_dtype = None
                for candidate in (aligner, *(getattr(aligner, name, None)
                                              for name in ("model", "_model", "aligner_model"))):
                    if candidate is None or not hasattr(candidate, "parameters"):
                        continue
                    try:
                        observed_dtype = str(next(candidate.parameters()).dtype)
                        break
                    except (AttributeError, StopIteration, TypeError):
                        continue
                cuda_allocated = (
                    int(torch.cuda.memory_allocated(0)) if device.startswith("cuda") else None
                )
                if device.startswith("cuda") and (cuda_allocated or 0) < 100_000_000:
                    _reply(req_id, "blocked_device_unverified",
                           error="CUDA was requested but substantial GPU model allocation was not observed")
                    return 3
                if observed_dtype is not None and observed_dtype != str(use_dtype):
                    _reply(req_id, "blocked_dtype_mismatch",
                           error=f"Requested {use_dtype}, observed {observed_dtype}")
                    return 3
                model_load_count += 1
                _reply(req_id, "ok", device_used=device, dtype=dtype,
                       observed_dtype=observed_dtype,
                       cuda_allocated_bytes=cuda_allocated,
                       model_load_count=model_load_count,
                       model_load_wall_s=round(time.monotonic() - t0, 4),
                       peak_gpu_allocated_bytes=(
                           int(torch.cuda.max_memory_allocated(0))
                           if device.startswith("cuda") else None
                       ))
                continue
            if action != "align" or aligner is None:
                _reply(req_id, "not_initialized")
                continue
            audio_path = Path(str(req["audio_path"]))
            text = str(req["text"])
            language = str(req["language"])
            if not audio_path.is_file() or not text.strip() or len(text) > 30_000:
                _reply(req_id, "invalid_request", error="Invalid audio or text")
                continue
            t0 = time.monotonic()
            output = aligner.align(audio=str(audio_path), text=text, language=language)[0]
            items = [
                {"text": str(item.text),
                 "start": float(item.start_time), "end": float(item.end_time)}
                for item in output.items
            ]
            _reply(req_id, "ok", items=items, device_used=device, dtype=dtype,
                   model_load_count=model_load_count,
                   inference_wall_s=round(time.monotonic() - t0, 4),
                   peak_gpu_allocated_bytes=(
                       int(torch.cuda.max_memory_allocated(0))
                       if device and device.startswith("cuda") else None
                   ))
        except Exception as exc:
            _reply(req.get("id") if "req" in locals() and isinstance(req, dict) else None,
                   "worker_error", error=f"{type(exc).__name__}: {str(exc)[:240]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
