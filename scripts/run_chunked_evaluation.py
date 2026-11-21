#!/usr/bin/env python3
"""Run Phase 2 chunked core GPU evaluation: 60/90/120s ablation, full French episode, and Mandarin validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.metrics import compute_wer
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media

ROOT = Path("/workspace/nvidia-transcribe")
MEDIA_FR = Path("/opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm")
PYANNOTE_JSON = Path(
    "/workspace/pyannote/outputs/runs/20261120-215532-Les-myst-rieuses-cit-s-d-or-Saison-4-Episode-1-YlVGUDNbyXk/result.json"
)
SINGLE_PASS_FR = ROOT / "outputs/responses/longform/fr_full_episode.contract.json"


def gpu_mem() -> dict[str, Any]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    u, f, t, util = [int(float(x.strip())) for x in out.split(",")]
    return {"memory_used_mib": u, "memory_free_mib": f, "memory_total_mib": t, "utilization_gpu_pct": util}


def summarize_recording(out: dict[str, Any]) -> dict[str, Any]:
    c = out["contract"]
    utts = c["utterances"]
    all_units = [u for utt in utts for u in utt.get("source_words", []) + utt.get("source_characters", [])]
    timed_units = [u for u in all_units if u.get("valid_timing")]
    untimed_units = [u for u in all_units if not u.get("valid_timing")]
    return {
        "duration_s": c["source"]["duration_s"],
        "source_language": c["source_language"],
        "language_provenance": c["language_provenance"],
        "chunk_config": c["processing"]["chunk_config"],
        "chunk_count": c["processing"]["chunk_count"],
        "total_wall_time_s": out["total_wall_time_s"],
        "chunk_latencies_s": c["model_provenance"].get("chunk_request_latencies_s"),
        "speaker_count_raw_local": c["speaker_reconciliation"]["raw_local_speaker_count"],
        "speaker_count_global": c["speaker_reconciliation"]["reconciled_global_speaker_count"],
        "unknown_speakers": c["speaker_reconciliation"]["unknown_speaker_count"],
        "unresolved_speakers": len(c["speaker_reconciliation"]["unresolved_speakers"]),
        "ambiguous_matches": len(c["speaker_reconciliation"]["ambiguous_matches"]),
        "collisions_prevented": len(c["speaker_reconciliation"]["collisions_prevented"]),
        "utterance_count": len(utts),
        "total_units": len(all_units),
        "timed_units": len(timed_units),
        "untimed_units": len(untimed_units),
        "boundary_dedup": c["processing"]["boundary_reconciliation"]["deduplicated_units"],
        "boundary_recovered": c["processing"]["boundary_reconciliation"]["recovered_boundary_units"],
        "boundary_ambiguities": len(c["processing"]["boundary_reconciliation"]["ambiguities"]),
        "alignment_failures": sum(1 for d in c["alignment_diagnostics"] if d["status"] not in ("ok", "subinterval_ctc_recovered")),
        "warnings": c["processing_warnings"],
    }


def main() -> int:
    base_url = "http://127.0.0.1:8190/v1"
    model = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
    client = NemotronVLLMClient(base_url=base_url, model=model, timeout_s=900.0)

    eval_dir = ROOT / "evaluations/chunked-core"
    eval_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = ROOT / "outputs/chunked_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract shared 300s French comparison slice (150s..450s: covers 03:16-03:55, 04:22-04:39, 05:39-05:52)
    slice_wav = ROOT / "outputs/audio/fr/fr_0230_0730_300s.wav"
    if not slice_wav.is_file():
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", "150", "-t", "300", "-i", str(MEDIA_FR), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(slice_wav)],
            check=True,
        )

    bench: dict[str, Any] = {
        "gpu_before": gpu_mem(),
        "ablation_300s": {},
        "french_full_episode": {},
        "mandarin": {},
    }

    # 2. Benchmark 60s / 90s / 120s cores with 10s context on the 300s slice
    for chunk_s in (60.0, 90.0, 120.0):
        key = f"chunk_{int(chunk_s)}s"
        out = process_media(
            slice_wav,
            output_dir=runs_dir / f"ablation_{key}",
            client=client,
            config=RecordingPipelineConfig(
                source_language=None,  # test auto LID resolution!
                mode="speaker",
                chunk_seconds=chunk_s,
                context_seconds=10.0,
                min_tail_seconds=15.0,
                speaker_embeddings="titanet",
                resume=False,
            ),
        )
        summary = summarize_recording(out)
        # Check child voices slice inside 150s-offset slice (03:16-03:55 = 46..85s relative to slice)
        child_utts = [
            {"speaker": u["speaker_id"], "start": u["source_start"], "end": u["source_end"], "text": u["source_text"]}
            for u in out["contract"]["utterances"]
            if u.get("source_start") is not None and u["source_end"] >= 46.0 and u["source_start"] <= 85.0
        ]
        summary["child_voices_window_speakers"] = sorted({u["speaker"] for u in child_utts})
        summary["child_voices_window_utterances"] = len(child_utts)
        summary["gpu_after"] = gpu_mem()
        bench["ablation_300s"][key] = summary

    # Select chunk_seconds=90.0 (balanced context & speaker stability; inspectable in ablation)
    selected_chunk_s = 90.0
    selected_context_s = 10.0

    # 3. Full French Episode chunked run
    fr_full_out = process_media(
        MEDIA_FR,
        output_dir=runs_dir / "fr_full_episode_90s",
        client=client,
        config=RecordingPipelineConfig(
            source_language="fr",
            mode="speaker",
            chunk_seconds=selected_chunk_s,
            context_seconds=selected_context_s,
            min_tail_seconds=15.0,
            speaker_embeddings="titanet",
            resume=False,
        ),
    )
    fr_summary = summarize_recording(fr_full_out)
    fr_summary["gpu_after"] = gpu_mem()

    # Compare full chunked vs Pyannote baseline and single-pass Nemotron
    py = json.loads(PYANNOTE_JSON.read_text(encoding="utf-8"))
    sp = json.loads(SINGLE_PASS_FR.read_text(encoding="utf-8"))
    chunked_text = " ".join(u["source_text"] for u in fr_full_out["contract"]["utterances"])
    py_text = " ".join(u.get("source_text", "") for u in py.get("subtitle_utterances", []))
    sp_text = sp.get("source_text", "")
    fr_summary["wer_vs_pyannote"] = compute_wer(py_text, chunked_text)
    fr_summary["wer_vs_single_pass_nemotron"] = compute_wer(sp_text, chunked_text)

    # Extract diagnostic window comparisons (03:16-03:55 and 04:22-04:39)
    for label, s_w, e_w in [("03:16-03:55", 196.0, 235.0), ("04:22-04:39", 262.0, 279.0)]:
        win_utts = [
            {
                "id": u["id"],
                "speaker": u["speaker_id"],
                "start": u["source_start"],
                "end": u["source_end"],
                "text": u["source_text"],
            }
            for u in fr_full_out["contract"]["utterances"]
            if u.get("source_start") is not None and u["source_end"] >= s_w and u["source_start"] <= e_w
        ]
        fr_summary[f"window_{label}"] = win_utts
    bench["french_full_episode"] = fr_summary

    # 4. Mandarin Validation
    # 4a. Single clip with '钾' OOV failure
    zh_oov_out = process_media(
        ROOT / "outputs/audio/zh/zh_fleurs_01_2006.wav",
        output_dir=runs_dir / "zh_oov_2006",
        client=client,
        config=RecordingPipelineConfig(
            source_language=None,  # test auto detection -> zh
            mode="speaker",
            chunk_seconds=90.0,
            context_seconds=10.0,
            speaker_embeddings="none",
            resume=False,
        ),
    )
    bench["mandarin"]["zh_fleurs_01_2006"] = {
        **summarize_recording(zh_oov_out),
        "utterances": [
            {"speaker": u["speaker_id"], "text": u["source_text"], "alignment_status": u["alignment_status"], "start": u["source_start"], "end": u["source_end"]}
            for u in zh_oov_out["contract"]["utterances"]
        ],
        "alignment_diagnostics": zh_oov_out["contract"]["alignment_diagnostics"],
    }

    # 4b. Synthetic 2-speaker Mandarin overlap (multispeaker_sot)
    zh_ov_out = process_media(
        ROOT / "outputs/audio/overlap/synthetic_zh_2spk_overlap.wav",
        output_dir=runs_dir / "zh_overlap",
        client=client,
        config=RecordingPipelineConfig(
            source_language="zh",
            mode="multispeaker_sot",
            chunk_seconds=90.0,
            context_seconds=10.0,
            speaker_embeddings="none",
            resume=False,
        ),
    )
    bench["mandarin"]["zh_2spk_overlap"] = {
        **summarize_recording(zh_ov_out),
        "utterances": [
            {"speaker": u["speaker_id"], "text": u["source_text"], "start": u["source_start"], "end": u["source_end"]}
            for u in zh_ov_out["contract"]["utterances"]
        ],
    }

    # 4c. Synthetic multi-chunk Mandarin recording (~67s across 6 FLEURS clips -> 2 chunks with chunk=35s, context=6s)
    import numpy as np
    import soundfile as sf

    zh_manifest = json.loads((ROOT / "outputs/audio/zh/zh_manifest.json").read_text(encoding="utf-8"))
    parts = []
    gap = np.zeros(int(0.5 * 16000), dtype=np.float32)
    for item in zh_manifest:
        arr, sr = sf.read(item["wav_path"], dtype="float32")
        parts.extend([arr, gap])
    synth_zh_long = ROOT / "outputs/audio/zh/synthetic_zh_multichunk.wav"
    sf.write(synth_zh_long, np.concatenate(parts), 16000, subtype="PCM_16")

    zh_multi_out = process_media(
        synth_zh_long,
        output_dir=runs_dir / "zh_multichunk_synthetic",
        client=client,
        config=RecordingPipelineConfig(
            source_language=None,
            mode="speaker",
            chunk_seconds=35.0,
            context_seconds=6.0,
            min_tail_seconds=10.0,
            speaker_embeddings="titanet",
            resume=False,
        ),
    )
    bench["mandarin"]["zh_multichunk_synthetic"] = {
        **summarize_recording(zh_multi_out),
        "synthetic_note": "Concatenation of 6 FLEURS cmn_hans_cn test clips (male + female) for multi-chunk boundary & OOV recovery validation.",
        "utterances": [
            {"id": u["id"], "speaker": u["speaker_id"], "start": u["source_start"], "end": u["source_end"], "alignment_status": u["alignment_status"], "text": u["source_text"]}
            for u in zh_multi_out["contract"]["utterances"]
        ],
        "alignment_diagnostics": zh_multi_out["contract"]["alignment_diagnostics"],
    }

    bench["gpu_final"] = gpu_mem()
    (eval_dir / "benchmark.json").write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Chunked Phase 2 evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
