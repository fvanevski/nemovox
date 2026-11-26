"""Explicit isolated CPU-only FireRed AED observation worker."""
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
    spec = importlib.util.find_spec("fireredvad")
    if spec is None or spec.origin is None:
        raise RuntimeError("FireRed AED package unavailable")
    root = Path(spec.origin).parent
    return {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
            "packages": sorted((d.metadata["Name"], d.version)
                               for d in importlib.metadata.distributions() if d.metadata.get("Name")),
            "code": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(root.rglob("*.py"))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--audio")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.identity:
        print(json.dumps(identity(), sort_keys=True))
        return
    if not all((args.checkpoint, args.audio, args.output)):
        parser.error("checkpoint, audio, output are required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from fireredvad import FireRedAed, FireRedAedConfig

    config = dict(use_gpu=False, smooth_window_size=5, speech_threshold=0.4,
                  singing_threshold=0.5, music_threshold=0.5, min_event_frame=20,
                  max_event_frame=2000, min_silence_frame=20, merge_silence_frame=0,
                  extend_speech_frame=0, chunk_max_frame=30000)
    start = time.perf_counter()
    model = FireRedAed.from_pretrained(args.checkpoint, FireRedAedConfig(**config))
    load_s = time.perf_counter() - start
    start = time.perf_counter()
    result, _ = model.detect(args.audio)
    payload = {"device": "cpu", "configuration": config, "duration_s": result["dur"],
               "event_intervals": result["event2timestamps"], "load_s": load_s,
               "inference_s": time.perf_counter() - start}
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
