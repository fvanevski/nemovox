#!/usr/bin/env python3
"""Isolated subprocess worker for Qwen3-ForcedAligner-0.6B.

Executed strictly inside ``outputs/modular-eval/.venv`` (qwen-asr==0.0.6,
transformers==4.57.6) via JSON-over-stdio/args so the primary Nemotron
environment never mixes incompatible Transformers dependencies.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("HF_HOME", "/workspace/models/.hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/workspace/models/.hf-cache/hub")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    req = json.loads(sys.stdin.read())
    model_path = str(req["model_path"])
    audio_path = str(req["audio_path"])
    text = str(req["text"])
    language = str(req["language"])
    device = str(req.get("device") or "auto")

    if not Path(model_path).is_dir():
        print(json.dumps({"status": "error", "error": f"model_dir_missing:{model_path}"}))
        return 0

    import torch
    from qwen_asr import Qwen3ForcedAligner

    if device == "auto":
        if torch.cuda.is_available():
            free_b, _ = torch.cuda.mem_get_info()
            # Aligner takes ~1.84 GiB BF16 weights + ~0.4 GiB activations
            use_device = "cuda:0" if free_b >= 2_600_000_000 else "cpu"
        else:
            use_device = "cpu"
    else:
        use_device = device

    dtype = torch.bfloat16 if use_device.startswith("cuda") else torch.float32
    aligner = Qwen3ForcedAligner.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=use_device,
        local_files_only=True,
    )
    res = aligner.align(audio=audio_path, text=text, language=language)[0]
    items: list[dict[str, Any]] = [
        {
            "text": str(it.text),
            "start": float(it.start_time),
            "end": float(it.end_time),
        }
        for it in res.items
    ]
    print(
        json.dumps(
            {
                "status": "ok",
                "device_used": use_device,
                "dtype": str(dtype),
                "items": items,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
