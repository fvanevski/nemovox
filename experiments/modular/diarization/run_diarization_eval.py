#!/usr/bin/env python3
"""Experiment A: Continuous Nemotron-3-Diarization evaluation via HuggingFace Transformers."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/workspace/models/.hf-cache")
os.environ.setdefault("HF_HUB_CACHE", "/workspace/models/.hf-cache/hub")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import librosa
import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

from experiments.modular.common import (
    MODULAR_OUT,
    NEMOTRON_DIAR_DIR,
    PYANNOTE_ROOT,
    WORKSPACE_ROOT,
    compute_overlap_der,
    load_shared_corpus,
)


def _gpu_stats() -> dict[str, Any]:
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
        "reserved_mib": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
    }


def _load_mono_16k(path: str, offset_s: float = 0.0, duration_s: float | None = None) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=-1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000
    if offset_s > 0.0 or duration_s is not None:
        a = int(round(offset_s * sr))
        b = len(wav) if duration_s is None else min(len(wav), a + int(round(duration_s * sr)))
        wav = wav[a:b]
    return wav, sr


@torch.inference_mode()
def run_streaming_diarization(
    model: Any,
    processor: Any,
    wav: np.ndarray,
    *,
    sr: int = 16000,
    streaming_mode: str = "low_latency",
    threshold: float = 0.5,
    source_offset_s: float = 0.0,
) -> dict[str, Any]:
    """Run official streaming Nemotron-3-Diarization retaining `speaker_cache` continuously."""
    processor.set_streaming_mode(streaming_mode)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    n_first = processor.num_samples_first_audio_chunk
    n_chunk = processor.num_samples_per_audio_chunk
    step_mel = processor.num_mel_frames_per_step

    logits_parts: list[torch.Tensor] = []
    speaker_cache = None

    if len(wav) <= n_first:
        # Clip shorter than 1st full streaming buffer (~1.04s): treat as single first+last streaming chunk
        inputs = processor(
            wav,
            sampling_rate=sr,
            is_streaming=True,
            is_first_audio_chunk=True,
            is_last_audio_chunk=True,
        ).to(model.device, dtype=model.dtype)
        out = model(**inputs)
        logits_parts.append(out.logits.detach().cpu())
    else:
        # Chunk 0 (first chunk)
        c0 = wav[:n_first]
        in0 = processor(
            c0,
            sampling_rate=sr,
            is_streaming=True,
            is_first_audio_chunk=True,
            is_last_audio_chunk=False,
        ).to(model.device, dtype=model.dtype)
        out0 = model(**in0)
        logits_parts.append(out0.logits.detach().cpu())
        speaker_cache = out0.speaker_cache

        mel_cursor = step_mel
        total_samples = len(wav)
        while True:
            start_samp = processor.audio_chunk_start(mel_cursor)
            if start_samp >= total_samples:
                break
            end_samp = start_samp + n_chunk
            if end_samp < total_samples:
                chunk_wav = wav[start_samp:end_samp]
                inp = processor(
                    chunk_wav,
                    sampling_rate=sr,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    is_last_audio_chunk=False,
                ).to(model.device, dtype=model.dtype)
                out = model(**inp, speaker_cache=speaker_cache)
                logits_parts.append(out.logits.detach().cpu())
                speaker_cache = out.speaker_cache
                mel_cursor += step_mel
            else:
                chunk_wav = wav[start_samp:]
                inp = processor(
                    chunk_wav,
                    sampling_rate=sr,
                    is_streaming=True,
                    is_first_audio_chunk=False,
                    is_last_audio_chunk=True,
                ).to(model.device, dtype=model.dtype)
                out = model(**inp, speaker_cache=speaker_cache)
                logits_parts.append(out.logits.detach().cpu())
                speaker_cache = out.speaker_cache
                break

    elapsed_s = round(time.perf_counter() - t0, 4)
    full_logits = torch.cat(logits_parts, dim=1)
    probs = full_logits.sigmoid()[0].numpy()  # (T, 8)
    raw_segs = processor.extract_speaker_dict(full_logits, threshold=threshold)[0]
    segs = [
        {
            "speaker": int(s["Speaker"]),
            "start": round(float(s["Start"]) + source_offset_s, 3),
            "end": round(float(s["End"]) + source_offset_s, 3),
        }
        for s in raw_segs
    ]
    return {
        "mode": f"streaming:{streaming_mode}",
        "elapsed_s": elapsed_s,
        "chunk_calls": len(logits_parts),
        "num_frames": int(probs.shape[0]),
        "segments": segs,
        "probs": probs,
        "gpu": _gpu_stats(),
    }


@torch.inference_mode()
def run_offline_diarization(
    model: Any,
    processor: Any,
    wav: np.ndarray,
    *,
    sr: int = 16000,
    threshold: float = 0.5,
    source_offset_s: float = 0.0,
) -> dict[str, Any]:
    """Run official Transformers offline full-sequence forward (`is_streaming=False`)."""
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    inputs = processor(wav, sampling_rate=sr, is_streaming=False).to(model.device, dtype=model.dtype)
    out = model(**inputs)
    elapsed_s = round(time.perf_counter() - t0, 4)
    logits = out.logits.detach().cpu()
    attn = inputs.attention_mask.detach().cpu()
    probs = logits.sigmoid()[0].numpy()
    raw_segs = processor.extract_speaker_dict(logits, attn, threshold=threshold)[0]
    segs = [
        {
            "speaker": int(s["Speaker"]),
            "start": round(float(s["Start"]) + source_offset_s, 3),
            "end": round(float(s["End"]) + source_offset_s, 3),
        }
        for s in raw_segs
    ]
    return {
        "mode": "offline",
        "elapsed_s": elapsed_s,
        "chunk_calls": 1,
        "num_frames": int(probs.shape[0]),
        "segments": segs,
        "probs": probs,
        "gpu": _gpu_stats(),
    }


def _summarize_run(run_res: dict[str, Any], *, duration_s: float, source_offset_s: float = 0.0) -> dict[str, Any]:
    probs = run_res["probs"]  # (T, 8)
    segs = run_res["segments"]
    active = probs > 0.5
    active_counts = active.sum(axis=1)
    overlap_frames = int((active_counts >= 2).sum())
    speech_frames = int((active_counts >= 1).sum())
    silence_frames = int((active_counts == 0).sum())
    max_probs_per_spk = [round(float(probs[:, c].max()), 4) for c in range(probs.shape[1])]
    mean_active_prob = round(float(probs[active].mean()), 4) if speech_frames > 0 else 0.0
    low_conf_speech = int(((probs.max(axis=1) >= 0.25) & (probs.max(axis=1) < 0.5)).sum())
    spks_active = sorted({s["speaker"] for s in segs})

    first_arrival_s = {}
    for ch in spks_active:
        idx = next((i for i, v in enumerate(active[:, ch]) if v), None)
        if idx is not None:
            first_arrival_s[str(ch)] = round(idx * 0.01 + source_offset_s, 2)

    return {
        "mode": run_res["mode"],
        "elapsed_s": run_res["elapsed_s"],
        "rtf": round(run_res["elapsed_s"] / duration_s, 4) if duration_s > 0 else 0.0,
        "chunk_calls": run_res["chunk_calls"],
        "num_frames": run_res["num_frames"],
        "segment_count": len(segs),
        "speaker_count": len(spks_active),
        "speakers": spks_active,
        "channel_8_saturated": (7 in spks_active),
        "first_arrival_s": first_arrival_s,
        "speech_s": round(speech_frames * 0.01, 2),
        "overlap_s": round(overlap_frames * 0.01, 2),
        "silence_s": round(silence_frames * 0.01, 2),
        "uncertain_025_to_050_s": round(low_conf_speech * 0.01, 2),
        "mean_active_prob": mean_active_prob,
        "max_prob_per_channel": max_probs_per_spk,
        "gpu": run_res["gpu"],
    }


def _window_view(segs: list[dict[str, Any]], a: float, b: float) -> list[dict[str, Any]]:
    return [s for s in segs if s["end"] > a and s["start"] < b]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(MODULAR_OUT / "raw/nemotron_diar_eval.json"))
    args = parser.parse_args()

    corpus = load_shared_corpus()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(NEMOTRON_DIAR_DIR), local_files_only=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        str(NEMOTRON_DIAR_DIR),
        dtype=torch.float32,
        device_map="cuda:0",
        local_files_only=True,
    )
    model.eval()
    load_s = round(time.perf_counter() - t0, 4)
    load_gpu = _gpu_stats()

    results: dict[str, Any] = {
        "model_path": str(NEMOTRON_DIAR_DIR),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "config": {
            "num_speakers_max": int(model.config.head_config.num_speakers),
            "streaming_modes": dict(processor.streaming_modes),
            "default_streaming_mode": processor.streaming_mode,
            "streaming_config": {
                k: (str(v) if isinstance(v, torch.dtype) else v)
                for k, v in model.config.streaming_config.to_dict().items()
            },
            "offline_chunk_length_enc": int(model.config.chunk_length),
            "offline_fifo_length": int(model.config.fifo_length),
        },
        "load_s": load_s,
        "load_gpu": load_gpu,
        "cases": {},
    }

    # 1. Short verified & adversarial clips (streaming vs offline)
    short_targets: list[tuple[str, str, float, float, list[tuple[float, float, str]] | None]] = [
        ("zh_fleurs_01_2006", corpus["mandarin_fleurs"]["zh_fleurs_01_2006"]["wav_path"], 0.0, 8.28, None),
        ("zh_2spk_seq", corpus["zh_2spk_seq"]["path"], 0.0, 27.08, corpus["zh_2spk_seq"]["ref_diarization"]),
        ("zh_2spk_overlap", corpus["zh_2spk_overlap"]["wav_path"], 0.0, 10.38, corpus["zh_2spk_overlap"]["ref_diarization"]),
        ("fr_2spk_overlap", corpus["fr_2spk_overlap"]["wav_path"], 0.0, 12.2, corpus["fr_2spk_overlap"]["ref_diarization"]),
    ]
    for cid in ("S1", "S2", "S3", "O1", "O2", "O3_music_neg", "O4"):
        m = corpus["french_verified_overlap"][cid]
        short_targets.append((cid, m["path"], float(m.get("offset_s", 0.0)), float(m["duration_s"]), m.get("ref_diarization")))
    for cid in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"):
        m = corpus["french_regions"][cid]
        short_targets.append((cid, m["path"], float(m["offset_s"]), float(m["duration_s"]), None))

    for cid, wav_path, off_s, dur_s, ref_diar in short_targets:
        wav, sr = _load_mono_16k(wav_path)
        st_run = run_streaming_diarization(model, processor, wav, sr=sr, source_offset_s=off_s)
        of_run = run_offline_diarization(model, processor, wav, sr=sr, source_offset_s=off_s)
        st_sum = _summarize_run(st_run, duration_s=dur_s, source_offset_s=off_s)
        of_sum = _summarize_run(of_run, duration_s=dur_s, source_offset_s=off_s)
        entry: dict[str, Any] = {
            "duration_s": dur_s,
            "offset_s": off_s,
            "streaming": {**st_sum, "segments": st_run["segments"]},
            "offline": {**of_sum, "segments": of_run["segments"]},
        }
        if ref_diar is not None:
            entry["streaming"]["der_metrics"] = compute_overlap_der(
                ref_diar,
                [(s["start"] - off_s, s["end"] - off_s, str(s["speaker"])) for s in st_run["segments"]],
                total_duration_s=dur_s,
            )
            entry["offline"]["der_metrics"] = compute_overlap_der(
                ref_diar,
                [(s["start"] - off_s, s["end"] - off_s, str(s["speaker"])) for s in of_run["segments"]],
                total_duration_s=dur_s,
            )
        results["cases"][cid] = entry

    # 2. Continuous 300s French passage (02:30-07:30, spans 60/90/120s boundaries + F1..F4)
    f300 = corpus["french_regions"]["F_300s"]
    wav300, sr300 = _load_mono_16k(f300["path"])
    st300 = run_streaming_diarization(model, processor, wav300, sr=sr300, source_offset_s=f300["offset_s"])
    of300 = run_offline_diarization(model, processor, wav300, sr=sr300, source_offset_s=f300["offset_s"])
    results["cases"]["F_300s"] = {
        "duration_s": f300["duration_s"],
        "offset_s": f300["offset_s"],
        "streaming": {
            **_summarize_run(st300, duration_s=f300["duration_s"], source_offset_s=f300["offset_s"]),
            "segments": st300["segments"],
            "boundary_continuity": {
                f"b_{int(bt)}": _window_view(st300["segments"], bt - 5.0, bt + 5.0)
                for bt in (210.0, 240.0, 270.0, 330.0, 390.0)
            },
            "F1_0316_0355": _window_view(st300["segments"], 196.0, 235.0),
            "F2_0422_0439": _window_view(st300["segments"], 262.0, 279.0),
        },
        "offline": {
            **_summarize_run(of300, duration_s=f300["duration_s"], source_offset_s=f300["offset_s"]),
            "segments": of300["segments"],
            "F1_0316_0355": _window_view(of300["segments"], 196.0, 235.0),
            "F2_0422_0439": _window_view(of300["segments"], 262.0, 279.0),
        },
    }

    # 3. Full 23-min French episode (streaming low_latency with continuous speaker_cache + offline check)
    ffull = corpus["french_regions"]["F_full"]
    wav_full, sr_full = _load_mono_16k(ffull["path"])
    st_full = run_streaming_diarization(model, processor, wav_full, sr=sr_full, source_offset_s=0.0)
    of_full = run_offline_diarization(model, processor, wav_full, sr=sr_full, source_offset_s=0.0)

    # Save raw frame-probability sample (every 100ms + active window excerpts) for evidence archive
    prob_sample_path = MODULAR_OUT / "raw/nemotron_diar_full_probs_100ms.json"
    probs_st = st_full["probs"]
    sample_rows = [
        {"t_s": round(i * 0.1, 1), "probs": [round(float(x), 4) for x in probs_st[i * 10]]}
        for i in range(0, probs_st.shape[0] // 10, 5)  # every 0.5s across full 23m episode
    ]
    prob_sample_path.write_text(json.dumps(sample_rows, indent=2), encoding="utf-8")

    # Compare vs Pyannote & Nemotron 3.5 baselines
    py_old = json.loads((PYANNOTE_ROOT / "result.json").read_text(encoding="utf-8"))["segments"]
    py_new_path = (
        PYANNOTE_ROOT
        / "outputs/runs/20261120-215532-Les-myst-rieuses-cit-s-d-or-Saison-4-Episode-1-YlVGUDNbyXk/result.json"
    )
    py_new = json.loads(py_new_path.read_text(encoding="utf-8"))["segments"]
    nem35_raw = json.loads(
        (WORKSPACE_ROOT / "outputs/responses/longform/fr_full_episode.raw.json").read_text(encoding="utf-8")
    )
    nem35_diar = nem35_raw["ctc_timestamps"]["diarization"]
    nem35_chunked = json.loads(
        (WORKSPACE_ROOT / "outputs/chunked_runs/fr_full_episode_90s/recording.contract.json").read_text(encoding="utf-8")
    )

    st_full_sum = _summarize_run(st_full, duration_s=ffull["duration_s"])
    of_full_sum = _summarize_run(of_full, duration_s=ffull["duration_s"])

    # Compute agreement matrices (streaming Nemotron3 vs Pyannote, vs Nemotron 3.5 single-pass Sortformer)
    st_iv = [(s["start"], s["end"], f"s_{s['speaker']}") for s in st_full["segments"]]
    of_iv = [(s["start"], s["end"], f"o_{s['speaker']}") for s in of_full["segments"]]
    py_iv = [(float(s["start"]), float(s["end"]), str(s["speaker"])) for s in py_new if s.get("speaker")]
    nem35_iv = [(float(s["start"]), float(s["end"]), str(s["speaker"])) for s in nem35_diar]

    # Boundary continuity across 90s chunks in F_full
    boundary_continuity = {
        f"b_{int(bt)}": _window_view(st_full["segments"], bt - 5.0, bt + 5.0)
        for bt in (90.0, 180.0, 270.0, 540.0, 810.0, 1080.0, 1260.0)
    }

    # Late-episode channel activity check (after channel 7 first appears, if it does)
    ch_arrivals = st_full_sum["first_arrival_s"]
    late_window_segments = _window_view(st_full["segments"], 1000.0, ffull["duration_s"])

    results["cases"]["F_full"] = {
        "duration_s": ffull["duration_s"],
        "streaming": {
            **st_full_sum,
            "segments_file": str(MODULAR_OUT / "raw/nemotron_diar_full_streaming_segments.json"),
            "probs_sample_file": str(prob_sample_path),
            "F1_0316_0355": _window_view(st_full["segments"], 196.0, 235.0),
            "F2_0422_0439": _window_view(st_full["segments"], 262.0, 279.0),
            "F5_1031_1033": _window_view(st_full["segments"], 631.0, 633.0),
            "F6_1731_1735": _window_view(st_full["segments"], 1051.0, 1055.0),
            "F7_2208_2211": _window_view(st_full["segments"], 1328.0, 1331.0),
            "boundary_continuity": boundary_continuity,
            "late_episode_active_channels": sorted({s["speaker"] for s in late_window_segments}),
            "agreement_vs_nem35_sortformer": compute_overlap_der(nem35_iv, st_iv, total_duration_s=ffull["duration_s"]),
            "agreement_vs_pyannote_new": compute_overlap_der(py_iv, st_iv, total_duration_s=ffull["duration_s"]),
        },
        "offline": {
            **of_full_sum,
            "segments_file": str(MODULAR_OUT / "raw/nemotron_diar_full_offline_segments.json"),
            "F1_0316_0355": _window_view(of_full["segments"], 196.0, 235.0),
            "F2_0422_0439": _window_view(of_full["segments"], 262.0, 279.0),
            "F5_1031_1033": _window_view(of_full["segments"], 631.0, 633.0),
            "agreement_vs_nem35_sortformer": compute_overlap_der(nem35_iv, of_iv, total_duration_s=ffull["duration_s"]),
            "agreement_vs_streaming": compute_overlap_der(st_iv, of_iv, total_duration_s=ffull["duration_s"]),
        },
        "baselines": {
            "pyannote_old_result": {"segments": len(py_old), "speakers": sorted({s["speaker"] for s in py_old if s.get("speaker")})},
            "pyannote_new_run": {
                "segments": len(py_new),
                "speakers": sorted({s["speaker"] for s in py_new if s.get("speaker")}),
                "F1_0316_0355": [s for s in py_new if s["end"] > 196 and s["start"] < 235],
            },
            "nemotron35_single_pass_sortformer": {
                "segments": len(nem35_diar),
                "speakers": sorted({str(s["speaker"]) for s in nem35_diar}),
                "F1_0316_0355": [s for s in nem35_diar if s["end"] > 196 and s["start"] < 235],
            },
            "nemotron35_chunked_90s_titanet": {
                "raw_local_speakers": nem35_chunked["speaker_reconciliation"]["raw_local_speaker_count"],
                "reconciled_global_speakers": nem35_chunked["speaker_reconciliation"]["reconciled_global_speaker_count"],
                "unresolved_speakers": len(nem35_chunked["speaker_reconciliation"]["unresolved_speakers"]),
            },
        },
    }

    (MODULAR_OUT / "raw/nemotron_diar_full_streaming_segments.json").write_text(
        json.dumps(st_full["segments"], indent=2), encoding="utf-8"
    )
    (MODULAR_OUT / "raw/nemotron_diar_full_offline_segments.json").write_text(
        json.dumps(of_full["segments"], indent=2), encoding="utf-8"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "load_s": load_s,
                "full_streaming_speakers": st_full_sum["speaker_count"],
                "full_streaming_elapsed_s": st_full_sum["elapsed_s"],
                "full_offline_speakers": of_full_sum["speaker_count"],
                "full_offline_elapsed_s": of_full_sum["elapsed_s"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
