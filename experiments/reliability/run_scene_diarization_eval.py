#!/usr/bin/env python3
"""Phase D GPU evaluation: Scene-bounded Nemotron-3-Diarization + supplemental evidence adapter."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path("/workspace/nvidia-transcribe")
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

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
    FRENCH_FULL_EPISODE_WAV,
    NEMOTRON_DIAR_DIR,
    compute_overlap_der,
    load_shared_corpus,
    sha256_file,
)

_sd_spec = importlib.util.spec_from_file_location(
    "scene_diarization", WORKSPACE_ROOT / "src/nvidia_transcribe/scene_diarization.py"
)
assert _sd_spec and _sd_spec.loader
_sd_mod = importlib.util.module_from_spec(_sd_spec)
sys.modules["scene_diarization"] = _sd_mod
_sd_spec.loader.exec_module(_sd_mod)
build_scene_speaker_evidence = _sd_mod.build_scene_speaker_evidence


def _gpu_stats() -> dict[str, Any]:
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
        "reserved_mib": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
    }


def _load_mono_16k(path: str, offset_s: float = 0.0, duration_s: float | None = None) -> tuple[np.ndarray, int]:
    info = sf.info(path)
    sr_orig = info.samplerate
    start_frame = int(round(offset_s * sr_orig)) if offset_s > 0 else 0
    frames = int(round(duration_s * sr_orig)) if duration_s is not None else -1
    wav, sr = sf.read(path, start=start_frame, frames=frames, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=-1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000
    return wav, sr


@torch.inference_mode()
def run_streaming_scene(
    model: Any,
    processor: Any,
    wav: np.ndarray,
    *,
    sr: int = 16000,
    streaming_mode: str = "low_latency",
    threshold: float = 0.5,
    source_offset_s: float = 0.0,
) -> dict[str, Any]:
    """Run streaming Nemotron-3-Diarization with explicit scene-fresh speaker_cache."""
    processor.set_streaming_mode(streaming_mode)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    n_first = processor.num_samples_first_audio_chunk
    n_chunk = processor.num_samples_per_audio_chunk
    step_mel = processor.num_mel_frames_per_step

    logits_parts: list[torch.Tensor] = []
    speaker_cache = None

    if len(wav) <= n_first:
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
        in0 = processor(
            wav[:n_first],
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
                inp = processor(
                    wav[start_samp:end_samp],
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
                inp = processor(
                    wav[start_samp:],
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
        "segments": segs,
        "gpu": _gpu_stats(),
    }


@torch.inference_mode()
def run_offline_scene(
    model: Any,
    processor: Any,
    wav: np.ndarray,
    *,
    sr: int = 16000,
    threshold: float = 0.5,
    source_offset_s: float = 0.0,
) -> dict[str, Any]:
    """Run offline full-scene forward (`is_streaming=False`)."""
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    inputs = processor(wav, sampling_rate=sr, is_streaming=False).to(model.device, dtype=model.dtype)
    out = model(**inputs)
    elapsed_s = round(time.perf_counter() - t0, 4)
    logits = out.logits.detach().cpu()
    attn = inputs.attention_mask.detach().cpu()
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
        "segments": segs,
        "gpu": _gpu_stats(),
    }


def _summarize_segments(segs: list[dict[str, Any]], *, duration_s: float) -> dict[str, Any]:
    speakers = sorted({s["speaker"] for s in segs})
    speech_time = round(sum(s["end"] - s["start"] for s in segs), 3)
    intervals = sorted((s["start"], s["end"], s["speaker"]) for s in segs)
    ov_time = 0.0
    for i, (s1, e1, sp1) in enumerate(intervals):
        for s2, e2, sp2 in intervals[i + 1 :]:
            if s2 >= e1:
                break
            if sp1 != sp2:
                ov_time += max(0.0, min(e1, e2) - max(s1, s2))
    return {
        "segment_count": len(segs),
        "unique_speakers": speakers,
        "unique_speaker_count": len(speakers),
        "total_speech_s": speech_time,
        "overlap_pair_s": round(ov_time, 3),
        "rtf": round(speech_time / duration_s, 4) if duration_s > 0 else 0.0,
    }


def _window_view(segs: list[dict[str, Any]], t0: float, t1: float) -> list[dict[str, Any]]:
    return [
        {
            "speaker": s["speaker"],
            "start": round(max(t0, s["start"]), 3),
            "end": round(min(t1, s["end"]), 3),
        }
        for s in segs
        if s["end"] > t0 and s["start"] < t1
    ]


def _canonical_slice(utterances: list[dict[str, Any]], t0: float, t1: float) -> list[dict[str, Any]]:
    out = []
    for u in utterances:
        us = u.get("source_start")
        ue = u.get("source_end")
        if us is None or ue is None:
            continue
        if float(ue) > t0 and float(us) < t1:
            out.append(u)
    return out


def _pyannote_slice(pyannote_segs: list[dict[str, Any]], t0: float, t1: float) -> dict[str, Any]:
    hits = [
        {"speaker": s.get("speaker"), "start": s.get("start"), "end": s.get("end")}
        for s in pyannote_segs
        if s.get("end", 0.0) > t0 and s.get("start", 1e9) < t1
    ]
    return {
        "segment_count": len(hits),
        "unique_speakers": sorted({str(h["speaker"]) for h in hits}),
        "segments": hits,
    }


def main() -> None:
    out_path = WORKSPACE_ROOT / "outputs/reliability_runs/phase_d_scene_diarization.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = load_shared_corpus()
    fr_contract = json.loads(
        (WORKSPACE_ROOT / "outputs/chunked_runs/fr_full_episode_90s/recording.contract.json").read_text(encoding="utf-8")
    )
    fr_utterances = fr_contract.get("utterances") or []
    zh_mc_contract = json.loads(
        (WORKSPACE_ROOT / "outputs/chunked_runs/zh_multichunk_synthetic/recording.contract.json").read_text(encoding="utf-8")
    )
    zh_mc_utterances = zh_mc_contract.get("utterances") or []

    pyannote_raw = json.loads(Path("/workspace/pyannote/result.json").read_text(encoding="utf-8"))
    pyannote_segs = pyannote_raw.get("segments") or pyannote_raw.get("utterances") or []

    config_path = NEMOTRON_DIAR_DIR / "config.json"
    model_rev = f"local:{NEMOTRON_DIAR_DIR.name}:{sha256_file(config_path)[:12]}"

    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(NEMOTRON_DIAR_DIR), local_files_only=True)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        str(NEMOTRON_DIAR_DIR),
        dtype=torch.float32,
        device_map="cuda:0",
        local_files_only=True,
    )
    model.eval()
    load_s = round(time.perf_counter() - t_load, 4)
    load_gpu = _gpu_stats()

    streaming_modes = list(processor.streaming_modes.keys())

    # 1. Scene-bounded evaluation (explicit scene resets between intervals)
    scenes_spec: list[dict[str, Any]] = [
        # 60-120s bounded scenes
        {
            "scene_id": "scene_fr_100s_child_rapid",
            "scale": "60-120s",
            "path": corpus["french_regions"]["F_100s"]["path"],
            "file_offset_s": 0.0,
            "abs_offset_s": 185.0,
            "duration_s": 100.0,
            "canonical_ref": "fr_full_episode_90s",
            "subviews": {"F1_child_voices": (196.0, 235.0), "F2_rapid_turns": (262.0, 279.0)},
        },
        {
            "scene_id": "scene_fr_90s_overlaps_0530_0700",
            "scale": "60-120s",
            "path": str(FRENCH_FULL_EPISODE_WAV),
            "file_offset_s": 330.0,
            "abs_offset_s": 330.0,
            "duration_s": 90.0,
            "canonical_ref": "fr_full_episode_90s",
            "subviews": {"O1": (336.0, 345.0), "O2": (346.5, 355.5), "F3": (339.0, 342.0), "F4": (350.0, 352.0)},
        },
        {
            "scene_id": "scene_fr_90s_music_0930_1100",
            "scale": "60-120s",
            "path": str(FRENCH_FULL_EPISODE_WAV),
            "file_offset_s": 570.0,
            "abs_offset_s": 570.0,
            "duration_s": 90.0,
            "canonical_ref": "fr_full_episode_90s",
            "subviews": {"O3_music_neg": (627.5, 636.5), "F5_music_fp": (631.0, 633.0)},
        },
        {
            "scene_id": "scene_fr_90s_yelp_1700_1830",
            "scale": "60-120s",
            "path": str(FRENCH_FULL_EPISODE_WAV),
            "file_offset_s": 1020.0,
            "abs_offset_s": 1020.0,
            "duration_s": 90.0,
            "canonical_ref": "fr_full_episode_90s",
            "subviews": {"O4": (1048.0, 1057.0), "F6_overlap_vox": (1051.0, 1055.0)},
        },
        {
            "scene_id": "scene_zh_multichunk_70s",
            "scale": "60-120s",
            "path": corpus["zh_multichunk_synthetic"]["path"],
            "file_offset_s": 0.0,
            "abs_offset_s": 0.0,
            "duration_s": float(corpus["zh_multichunk_synthetic"]["duration_s"]),
            "canonical_ref": "zh_multichunk_synthetic",
            "subviews": {},
        },
        # 3-5m bounded scenes
        {
            "scene_id": "scene_fr_300s_early_0230_0730",
            "scale": "3-5m",
            "path": corpus["french_regions"]["F_300s"]["path"],
            "file_offset_s": 0.0,
            "abs_offset_s": 150.0,
            "duration_s": 300.0,
            "canonical_ref": "fr_full_episode_90s",
            "subviews": {"F1_child_voices": (196.0, 235.0), "F2_rapid_turns": (262.0, 279.0), "O1": (336.0, 345.0), "O2": (346.5, 355.5)},
        },
        {
            "scene_id": "scene_fr_240s_mid_0730_1130",
            "scale": "3-5m",
            "path": str(FRENCH_FULL_EPISODE_WAV),
            "file_offset_s": 450.0,
            "abs_offset_s": 450.0,
            "duration_s": 240.0,
            "canonical_ref": "fr_full_episode_90s",
            "subviews": {"O3_music_neg": (627.5, 636.5)},
        },
        {
            "scene_id": "scene_fr_300s_late_1200_1700",
            "scale": "3-5m",
            "path": str(FRENCH_FULL_EPISODE_WAV),
            "file_offset_s": 720.0,
            "abs_offset_s": 720.0,
            "duration_s": 300.0,
            "canonical_ref": "fr_full_episode_90s",
            "subviews": {},
        },
    ]

    scene_results: dict[str, Any] = {}
    evidence_examples: list[dict[str, Any]] = []

    for spec in scenes_spec:
        sid = spec["scene_id"]
        wav, sr = _load_mono_16k(spec["path"], offset_s=spec["file_offset_s"], duration_s=spec["duration_s"])
        dur_s = len(wav) / sr
        abs_off = spec["abs_offset_s"]

        mode_runs: dict[str, Any] = {}
        mode_runs["offline"] = run_offline_scene(model, processor, wav, sr=sr, source_offset_s=abs_off)
        for smode in streaming_modes:
            mode_runs[f"streaming:{smode}"] = run_streaming_scene(
                model, processor, wav, sr=sr, streaming_mode=smode, source_offset_s=abs_off
            )

        canon_src = fr_utterances if spec["canonical_ref"] == "fr_full_episode_90s" else zh_mc_utterances
        canon_slice = _canonical_slice(canon_src, abs_off, abs_off + dur_s)

        # Evaluate experimental supplemental evidence adapter on BOTH offline and low_latency streaming
        ev_offline = build_scene_speaker_evidence(
            scene_id=sid,
            scene_start_s=abs_off,
            scene_end_s=abs_off + dur_s,
            diar_segments=mode_runs["offline"]["segments"],
            canonical_utterances=canon_slice,
            model_revision=model_rev,
        )
        ev_stream = build_scene_speaker_evidence(
            scene_id=sid,
            scene_start_s=abs_off,
            scene_end_s=abs_off + dur_s,
            diar_segments=mode_runs[f"streaming:{streaming_modes[0]}"]["segments"],
            canonical_utterances=canon_slice,
            model_revision=model_rev,
        )

        subviews_out: dict[str, Any] = {}
        for sv_name, (a, b) in spec["subviews"].items():
            subviews_out[sv_name] = {
                m_key: _summarize_segments(_window_view(m_val["segments"], a, b), duration_s=(b - a))
                | {"segments": _window_view(m_val["segments"], a, b)}
                for m_key, m_val in mode_runs.items()
            }
            if spec["canonical_ref"] == "fr_full_episode_90s":
                subviews_out[sv_name]["pyannote_ref"] = _pyannote_slice(pyannote_segs, a, b)

        scene_results[sid] = {
            "scale": spec["scale"],
            "abs_interval": [abs_off, round(abs_off + dur_s, 3)],
            "duration_s": round(dur_s, 3),
            "runs": {
                k: {
                    "elapsed_s": v["elapsed_s"],
                    "chunk_calls": v["chunk_calls"],
                    "gpu": v["gpu"],
                    "summary": _summarize_segments(v["segments"], duration_s=dur_s),
                    "segments": v["segments"],
                }
                for k, v in mode_runs.items()
            },
            "canonical_chunked_summary": {
                "utterance_count": len(canon_slice),
                "unique_global_speakers": sorted(
                    {str(u.get("global_speaker_id") or u.get("speaker_id")) for u in canon_slice}
                ),
            },
            "evidence_adapter_offline": ev_offline,
            "evidence_adapter_streaming_default": ev_stream,
            "subviews": subviews_out,
        }
        evidence_examples.append(ev_offline)

    # 2. Ground-truth overlap & A-B-A sequence comparison across all modes
    gt_cases: list[tuple[str, str, float, float, list[tuple[float, float, str]]]] = [
        ("zh_2spk_seq", corpus["zh_2spk_seq"]["path"], 0.0, 27.08, corpus["zh_2spk_seq"]["ref_diarization"]),
        ("zh_2spk_overlap", corpus["zh_2spk_overlap"]["wav_path"], 0.0, 10.38, corpus["zh_2spk_overlap"]["ref_diarization"]),
        ("fr_2spk_overlap", corpus["fr_2spk_overlap"]["wav_path"], 0.0, 12.2, corpus["fr_2spk_overlap"]["ref_diarization"]),
    ]
    for cid in ("S1", "S2", "S3", "O1", "O2", "O3_music_neg", "O4"):
        m = corpus["french_verified_overlap"][cid]
        gt_cases.append((cid, m["path"], float(m.get("offset_s", 0.0)), float(m["duration_s"]), m.get("ref_diarization") or []))

    gt_results: dict[str, Any] = {}
    for cid, wav_path, off_s, dur_s, ref_diar in gt_cases:
        wav, sr = _load_mono_16k(wav_path)
        actual_dur = len(wav) / sr
        runs = {"offline": run_offline_scene(model, processor, wav, sr=sr, source_offset_s=off_s)}
        for smode in streaming_modes:
            runs[f"streaming:{smode}"] = run_streaming_scene(
                model, processor, wav, sr=sr, streaming_mode=smode, source_offset_s=off_s
            )
        comp: dict[str, Any] = {}
        for m_key, m_val in runs.items():
            hyp = [(s["start"] - off_s, s["end"] - off_s, str(s["speaker"])) for s in m_val["segments"]]
            comp[m_key] = {
                "elapsed_s": m_val["elapsed_s"],
                "summary": _summarize_segments(m_val["segments"], duration_s=actual_dur),
                "der_metrics": compute_overlap_der(ref_diar, hyp, total_duration_s=actual_dur),
                "segments": m_val["segments"],
            }
        gt_results[cid] = {
            "duration_s": round(actual_dur, 3),
            "offset_s": off_s,
            "ref_diarization": ref_diar,
            "modes": comp,
        }

    report = {
        "model_path": str(NEMOTRON_DIAR_DIR),
        "model_revision": model_rev,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "streaming_modes_available": {k: str(v) for k, v in processor.streaming_modes.items()},
        "load_s": load_s,
        "load_gpu": load_gpu,
        "scenes": scene_results,
        "ground_truth_cases": gt_results,
        "experimental_evidence_adapter_schema": {
            "authoritative_override_applied": False,
            "required_fields": [
                "evidence_source",
                "model_revision",
                "scene_id",
                "source_time_interval",
                "local_speaker_channel",
                "supported_identity_candidates",
                "contradictions",
                "unresolved_status",
            ],
        },
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote Phase D report to {out_path}")


if __name__ == "__main__":
    main()
