#!/usr/bin/env python3
"""Work Package A Live GPU Verification: Verify Qwen fallback '钾' stays in its true utterance."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

WORKSPACE = Path("/workspace/nvidia-transcribe")
if str(WORKSPACE / "src") not in sys.path:
    sys.path.insert(0, str(WORKSPACE / "src"))

from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media

OUT_ROOT = WORKSPACE / "outputs/subtitle_runs/zh_qwen_fix_verify"


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    client = NemotronVLLMClient(base_url="http://127.0.0.1:8190/v1", model="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4", timeout_s=300.0)

    targets = [
        (
            "zh_fleurs_01_2006",
            WORKSPACE / "outputs/audio/zh/zh_fleurs_01_2006.wav",
            RecordingPipelineConfig(source_language="zh", alignment_fallback="qwen", resume=False),
        ),
        (
            "zh_fleurs_05_1805",
            WORKSPACE / "outputs/audio/zh/zh_fleurs_05_1805.wav",
            RecordingPipelineConfig(source_language="zh", alignment_fallback="qwen", resume=False),
        ),
        (
            "zh_2spk_overlap",
            WORKSPACE / "outputs/audio/overlap/synthetic_zh_2spk_overlap.wav",
            RecordingPipelineConfig(source_language="zh", alignment_fallback="qwen", resume=False),
        ),
        (
            "zh_multichunk_synthetic_off",
            WORKSPACE / "outputs/audio/zh/synthetic_zh_multichunk.wav",
            RecordingPipelineConfig(source_language="zh", chunk_seconds=30.0, context_seconds=5.0, min_tail_seconds=8.0, alignment_fallback="off", resume=False),
        ),
        (
            "zh_multichunk_synthetic_qwen",
            WORKSPACE / "outputs/audio/zh/synthetic_zh_multichunk.wav",
            RecordingPipelineConfig(source_language="zh", chunk_seconds=30.0, context_seconds=5.0, min_tail_seconds=8.0, alignment_fallback="qwen", resume=False),
        ),
    ]

    summary = {}
    for name, wav_path, cfg in targets:
        t0 = time.perf_counter()
        res = process_media(wav_path, output_dir=OUT_ROOT / name, client=client, config=cfg)
        c = res["contract"]
        utts = c.get("utterances") or []
        summary[name] = {
            "wall_time_s": round(time.perf_counter() - t0, 3),
            "contract_path": res["contract_path"],
            "utterances": [
                {
                    "id": u["id"],
                    "speaker": u["speaker_id"],
                    "start": u["source_start"],
                    "end": u["source_end"],
                    "timing_status": (u.get("timestamp_provenance") or {}).get("timing_status"),
                    "text": u["source_text"],
                    "qwen_chars": [
                        {"text": ch["text"], "start": ch["start"], "end": ch["end"]}
                        for ch in (u.get("source_characters") or [])
                        if ch.get("alignment_status") == "external_qwen_recovered"
                    ],
                }
                for u in utts
            ],
            "alignment_diagnostics": c.get("alignment_diagnostics"),
        }
        print(f"=== {name} ===")
        for u in summary[name]["utterances"]:
            print(f"  {u['id']} [{u['start']}..{u['end']}] {u['timing_status']} qwen={u['qwen_chars']} text={u['text']}")

    (OUT_ROOT / "qwen_fix_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
