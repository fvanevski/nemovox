#!/usr/bin/env python3
"""Isolated subprocess worker for NVIDIA Riva-Translate-4B-Instruct-v2."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main() -> int:
    payload = json.loads(sys.stdin.read())
    model_path = payload["model_path"]
    items = payload.get("items") or []
    max_new_tokens = int(payload.get("max_new_tokens") or 384)
    req_device = str(payload.get("device") or "auto")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if req_device == "auto":
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            device = "cuda:0" if free_bytes >= 8_500_000_000 else "cpu"
        else:
            device = "cpu"
    else:
        device = req_device

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        local_files_only=True,
    )
    model.eval()
    load_s = round(time.perf_counter() - t_load, 4)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    t_inf = time.perf_counter()
    results: list[dict[str, Any]] = []
    for item in items:
        uid = item["utterance_id"]
        text = str(item.get("text") or "").strip()
        lang_pair = str(item.get("lang_pair") or "fr-en")
        if not text:
            results.append({"utterance_id": uid, "translated_text": "", "latency_s": 0.0})
            continue
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": lang_pair},
            {"role": "user", "content": text},
        ]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            out_ids = model.generate(
                tokenized,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out_ids[0][tokenized.shape[1] :]
        decoded = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append(
            {
                "utterance_id": uid,
                "translated_text": decoded,
                "latency_s": round(time.perf_counter() - t0, 4),
            }
        )
    infer_s = round(time.perf_counter() - t_inf, 4)
    peak_vram_mib = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if device.startswith("cuda") else 0.0
    )

    sys.stdout.write(
        json.dumps(
            {
                "status": "ok",
                "device_used": device,
                "dtype": str(dtype),
                "load_s": load_s,
                "infer_s": infer_s,
                "peak_vram_mib": peak_vram_mib,
                "results": results,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
