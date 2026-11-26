"""Isolated local MOSS worker. Invoked by the recording stage under its GPU lease."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def identity():
    spec = importlib.util.find_spec("moss_transcribe_diarize")
    if spec is None or spec.origin is None:
        raise RuntimeError("Pinned MOSS inference package is unavailable")
    root = Path(spec.origin).parent
    return {
        "python": sys.version,
        "executable": str(Path(sys.executable).resolve()),
        "packages": sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                           if d.metadata.get("Name")),
        "moss_code": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in sorted(root.rglob("*.py"))},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--identity", action="store_true")
    p.add_argument("--checkpoint")
    p.add_argument("--audio")
    p.add_argument("--output")
    p.add_argument("--max-new-tokens", type=int)
    a = p.parse_args()
    if a.identity:
        print(json.dumps(identity(), sort_keys=True))
        return
    if not all((a.checkpoint, a.audio, a.output, a.max_new_tokens)):
        p.error("inference requires checkpoint/audio/output/token limit")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize.inference_utils import (
        DEFAULT_PROMPT, build_transcription_messages, generate_transcription,
    )

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("MOSS requires admitted CUDA BF16; no CPU fallback")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(a.checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.checkpoint, trust_remote_code=True, local_files_only=True,
        torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()
    load_s = time.perf_counter() - start
    start = time.perf_counter()
    result = generate_transcription(
        model, processor, build_transcription_messages(a.audio, prompt=DEFAULT_PROMPT),
        max_length=131072, max_new_tokens=a.max_new_tokens, do_sample=False,
        device=device, dtype=dtype,
    )
    payload = {
        "text": result["text"], "prompt_len": result["prompt_len"],
        "generated_tokens": result["generated_tokens"], "max_new_tokens": a.max_new_tokens,
        "max_input_tokens": 131072, "prompt": DEFAULT_PROMPT,
        "device": str(device), "dtype": str(dtype), "load_s": load_s,
        "inference_s": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    Path(a.output).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
