#!/usr/bin/env python3
"""Experiment B: Independent Qwen3-ForcedAligner-0.6B evaluation (B1 Nemotron, B2 Qwen ASR, B3 Ref)."""

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

import numpy as np
import soundfile as sf
import torch
from qwen_asr import Qwen3ForcedAligner

from experiments.modular.common import (
    MODULAR_OUT,
    QWEN_ALIGNER_DIR,
    WORKSPACE_ROOT,
    load_shared_corpus,
    validate_alignment_items,
)


def _gpu_stats() -> dict[str, Any]:
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
        "reserved_mib": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
    }


def _read_nemotron_text(rel_path: str) -> str:
    data = json.loads((WORKSPACE_ROOT / rel_path).read_text(encoding="utf-8"))
    return str(data["source_text"])


def _align_one(
    aligner: Qwen3ForcedAligner,
    *,
    case_id: str,
    condition: str,
    audio_input: Any,
    audio_duration_s: float,
    text: str,
    language: str,
    source_offset_s: float = 0.0,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        res = aligner.align(audio=audio_input, text=text, language=language)[0]
        elapsed_s = round(time.perf_counter() - t0, 4)
        items = [
            {
                "text": str(it.text),
                "start": round(float(it.start_time) + source_offset_s, 4),
                "end": round(float(it.end_time) + source_offset_s, 4),
                "raw_start": float(it.start_time),
                "raw_end": float(it.end_time),
            }
            for it in res.items
        ]
        audit = validate_alignment_items(
            items,
            expected_text=text,
            audio_duration_s=audio_duration_s,
            source_offset_s=source_offset_s,
        )
        return {
            "case_id": case_id,
            "condition": condition,
            "language": language,
            "audio_duration_s": round(audio_duration_s, 3),
            "source_offset_s": source_offset_s,
            "input_text": text,
            "elapsed_s": elapsed_s,
            "status": "ok",
            "error": None,
            "audit": audit,
            "items": items,
            "gpu": _gpu_stats(),
        }
    except Exception as exc:
        return {
            "case_id": case_id,
            "condition": condition,
            "language": language,
            "audio_duration_s": round(audio_duration_s, 3),
            "source_offset_s": source_offset_s,
            "input_text": text,
            "elapsed_s": round(time.perf_counter() - t0, 4),
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "audit": None,
            "items": [],
            "gpu": _gpu_stats(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qwen-asr-json",
        default=str(MODULAR_OUT / "raw/qwen_asr_eval.json"),
        help="Optional Qwen ASR JSON output to include B2 alignment conditions.",
    )
    parser.add_argument(
        "--out",
        default=str(MODULAR_OUT / "raw/qwen_aligner_eval.json"),
    )
    args = parser.parse_args()

    corpus = load_shared_corpus()
    qwen_asr_data: dict[str, Any] = {}
    qwen_asr_path = Path(args.qwen_asr_json)
    if qwen_asr_path.is_file():
        qwen_asr_data = json.loads(qwen_asr_path.read_text(encoding="utf-8")).get("cases", {})

    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    aligner = Qwen3ForcedAligner.from_pretrained(
        str(QWEN_ALIGNER_DIR),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
    )
    load_s = round(time.perf_counter() - t_load0, 4)
    load_gpu = _gpu_stats()

    results: list[dict[str, Any]] = []

    # 1. Mandatory Mandarin FLEURS cases (especially zh_fleurs_01_2006 with '钾')
    nem_zh_dir = "outputs/responses/zh"
    zh_nem_map = {
        "zh_fleurs_00_1906": f"{nem_zh_dir}/zh_fleurs_00_1906_auto.contract.json",
        "zh_fleurs_01_2006": f"{nem_zh_dir}/zh_fleurs_01_2006_auto.contract.json",
        "zh_fleurs_02_1883": f"{nem_zh_dir}/zh_fleurs_02_1883_auto.contract.json",
        "zh_fleurs_03_1734": f"{nem_zh_dir}/zh_fleurs_03_1734_auto.contract.json",
        "zh_fleurs_04_1890": f"{nem_zh_dir}/zh_fleurs_04_1890_auto.contract.json",
        "zh_fleurs_05_1805": f"{nem_zh_dir}/zh_fleurs_05_1805_explicit.contract.json",
    }
    for clip_id, meta in corpus["mandarin_fleurs"].items():
        wav = meta["wav_path"]
        dur = float(meta["duration_s"])
        ref_text = str(meta["raw_transcription"])
        results.append(
            _align_one(
                aligner,
                case_id=clip_id,
                condition="B3_reference_text",
                audio_input=wav,
                audio_duration_s=dur,
                text=ref_text,
                language="Chinese",
            )
        )
        nem_path = WORKSPACE_ROOT / zh_nem_map.get(clip_id, "")
        if nem_path.is_file():
            nem_txt = _read_nemotron_text(zh_nem_map[clip_id])
            results.append(
                _align_one(
                    aligner,
                    case_id=clip_id,
                    condition="B1_nemotron_text",
                    audio_input=wav,
                    audio_duration_s=dur,
                    text=nem_txt,
                    language="Chinese",
                )
            )
        if clip_id in qwen_asr_data and qwen_asr_data[clip_id].get("explicit", {}).get("text"):
            results.append(
                _align_one(
                    aligner,
                    case_id=clip_id,
                    condition="B2_qwen_asr_text",
                    audio_input=wav,
                    audio_duration_s=dur,
                    text=qwen_asr_data[clip_id]["explicit"]["text"],
                    language="Chinese",
                )
            )

    # 2. Mandarin sequential & simultaneous overlap + synthetic Chinese/English code-switch probe
    zh_seq = corpus["zh_2spk_seq"]
    results.append(
        _align_one(
            aligner,
            case_id="zh_2spk_seq",
            condition="B3_reference_text",
            audio_input=zh_seq["path"],
            audio_duration_s=float(zh_seq["duration_s"]),
            text=f"{zh_seq['speaker_a_ref']} {zh_seq['speaker_b_ref']} {zh_seq['speaker_a_ref_2']}",
            language="Chinese",
        )
    )
    if (WORKSPACE_ROOT / "outputs/responses/overlap/zh_2spk_seq_speaker.contract.json").is_file():
        results.append(
            _align_one(
                aligner,
                case_id="zh_2spk_seq",
                condition="B1_nemotron_text",
                audio_input=zh_seq["path"],
                audio_duration_s=float(zh_seq["duration_s"]),
                text=_read_nemotron_text("outputs/responses/overlap/zh_2spk_seq_speaker.contract.json"),
                language="Chinese",
            )
        )
    zh_ov = corpus["zh_2spk_overlap"]
    results.append(
        _align_one(
            aligner,
            case_id="zh_2spk_overlap",
            condition="B3_reference_combined",
            audio_input=zh_ov["wav_path"],
            audio_duration_s=float(zh_ov["total_duration_s"]),
            text=f"{zh_ov['speaker_a']['text']} {zh_ov['speaker_b']['text']}",
            language="Chinese",
        )
    )
    if (WORKSPACE_ROOT / "outputs/responses/overlap/zh_2spk_overlap_sot.contract.json").is_file():
        results.append(
            _align_one(
                aligner,
                case_id="zh_2spk_overlap",
                condition="B1_nemotron_text",
                audio_input=zh_ov["wav_path"],
                audio_duration_s=float(zh_ov["total_duration_s"]),
                text=_read_nemotron_text("outputs/responses/overlap/zh_2spk_overlap_sot.contract.json"),
                language="Chinese",
            )
        )

    # 3. French verified regions O1-O4 & S1-S3
    for cid, meta in corpus["french_verified_overlap"].items():
        if "reference_text" in meta:
            results.append(
                _align_one(
                    aligner,
                    case_id=cid,
                    condition="B3_reference_text",
                    audio_input=meta["path"],
                    audio_duration_s=float(meta["duration_s"]),
                    text=meta["reference_text"],
                    language="French",
                    source_offset_s=float(meta.get("offset_s", 0.0)),
                )
            )
        else:
            # Synthetic S1-S3: align speaker A alone AND combined A+B
            results.append(
                _align_one(
                    aligner,
                    case_id=cid,
                    condition="B3_reference_speaker_a_only",
                    audio_input=meta["path"],
                    audio_duration_s=float(meta["duration_s"]),
                    text=meta["speaker_a_ref"],
                    language="French",
                )
            )
            results.append(
                _align_one(
                    aligner,
                    case_id=cid,
                    condition="B3_reference_combined_ab",
                    audio_input=meta["path"],
                    audio_duration_s=float(meta["duration_s"]),
                    text=f"{meta['speaker_a_ref']} {meta['speaker_b_ref']}",
                    language="French",
                )
            )
        if cid in qwen_asr_data and qwen_asr_data[cid].get("explicit", {}).get("text"):
            results.append(
                _align_one(
                    aligner,
                    case_id=cid,
                    condition="B2_qwen_asr_text",
                    audio_input=meta["path"],
                    audio_duration_s=float(meta["duration_s"]),
                    text=qwen_asr_data[cid]["explicit"]["text"],
                    language="French",
                    source_offset_s=float(meta.get("offset_s", 0.0)),
                )
            )

    # 4. French animation Nemotron outputs (F1, F2, F_100s) + Qwen ASR B2 if present
    fr_nem_cases = [
        ("F1", corpus["french_regions"]["F1"], "outputs/responses/fr/fr_0316_0355_child_voices_speaker.contract.json"),
        ("F2", corpus["french_regions"]["F2"], "outputs/responses/fr/fr_0422_0439_rapid_turns_speaker.contract.json"),
        ("F_100s", corpus["french_regions"]["F_100s"], "outputs/responses/fr/fr_0305_0445_cont_100s_speaker.contract.json"),
    ]
    for cid, meta, nem_rel in fr_nem_cases:
        if (WORKSPACE_ROOT / nem_rel).is_file():
            results.append(
                _align_one(
                    aligner,
                    case_id=cid,
                    condition="B1_nemotron_text",
                    audio_input=meta["path"],
                    audio_duration_s=float(meta["duration_s"]),
                    text=_read_nemotron_text(nem_rel),
                    language="French",
                    source_offset_s=float(meta["offset_s"]),
                )
            )
        if cid in qwen_asr_data and qwen_asr_data[cid].get("explicit", {}).get("text"):
            results.append(
                _align_one(
                    aligner,
                    case_id=cid,
                    condition="B2_qwen_asr_text",
                    audio_input=meta["path"],
                    audio_duration_s=float(meta["duration_s"]),
                    text=qwen_asr_data[cid]["explicit"]["text"],
                    language="French",
                    source_offset_s=float(meta["offset_s"]),
                )
            )

    # 5. Long-form duration boundary check: 175s slice (within MAX_FORCE_ALIGN_INPUT_SECONDS=180)
    # and direct 300s test if B2_qwen_asr_text exists or Nemotron 300s text exists
    f300 = corpus["french_regions"]["F_300s"]
    wav300, sr300 = sf.read(f300["path"], dtype="float32")
    wav175 = wav300[: int(175.0 * sr300)]
    nem_300_path = WORKSPACE_ROOT / "outputs/responses/longform/fr_0000_0500_300s.contract.json"
    if "F_300s" in qwen_asr_data and qwen_asr_data["F_300s"].get("explicit", {}).get("text"):
        full300_txt = qwen_asr_data["F_300s"]["explicit"]["text"]
        # Use first 175s via ASR or chunked runs for F_175s
        results.append(
            _align_one(
                aligner,
                case_id="F_300s_over_limit_probe",
                condition="B2_qwen_asr_300s_direct",
                audio_input=(wav300, sr300),
                audio_duration_s=300.0,
                text=full300_txt,
                language="French",
                source_offset_s=float(f300["offset_s"]),
            )
        )
    nem_chunked_150 = WORKSPACE_ROOT / "outputs/chunked_runs/ablation_chunk_90s/recording.contract.json"
    if nem_chunked_150.is_file():
        rec = json.loads(nem_chunked_150.read_text(encoding="utf-8"))
        # ablation_chunk_90s was extracted from fr_0230_0730_300s.wav (local coords 0..300s -> global 150..450s)
        u_175 = [
            str(u.get("source_text") or u.get("text") or "")
            for u in rec["utterances"]
            if (u.get("source_end") or u.get("end") or 0.0) <= 175.0
        ]
        txt_175 = " ".join(u_175)
        results.append(
            _align_one(
                aligner,
                case_id="F_175s_near_180s_limit",
                condition="B1_nemotron_text",
                audio_input=(wav175, sr300),
                audio_duration_s=175.0,
                text=txt_175,
                language="French",
                source_offset_s=150.0,
            )
        )

    # 6. Windowed F_300s alignment for Experiment D modular & hybrid probes
    f300_probe_out = MODULAR_OUT / "raw/qwen_aligner_f300_chunked.json"
    variants: dict[str, Any] = {}
    if nem_chunked_150.is_file():
        rec = json.loads(nem_chunked_150.read_text(encoding="utf-8"))
        u_w1 = [
            str(u.get("source_text") or u.get("text") or "")
            for u in rec["utterances"]
            if ((u.get("source_start") or 0.0) + (u.get("source_end") or 0.0)) * 0.5 < 150.0
        ]
        u_w2 = [
            str(u.get("source_text") or u.get("text") or "")
            for u in rec["utterances"]
            if ((u.get("source_start") or 0.0) + (u.get("source_end") or 0.0)) * 0.5 >= 150.0
        ]
        mid_samp = int(150.0 * sr300)
        r1 = _align_one(
            aligner,
            case_id="F_300s_w1",
            condition="D2_hybrid_nemotron",
            audio_input=(wav300[:mid_samp], sr300),
            audio_duration_s=150.0,
            text=" ".join(u_w1),
            language="French",
            source_offset_s=150.0,
        )
        r2 = _align_one(
            aligner,
            case_id="F_300s_w2",
            condition="D2_hybrid_nemotron",
            audio_input=(wav300[mid_samp:], sr300),
            audio_duration_s=len(wav300[mid_samp:]) / float(sr300),
            text=" ".join(u_w2),
            language="French",
            source_offset_s=300.0,
        )
        variants["hybrid_nemotron35_plus_qwen_aligner"] = {
            "source_text": " ".join(str(u.get("source_text") or u.get("text") or "") for u in rec["utterances"]),
            "window_audits": [r1["audit"], r2["audit"]],
            "items": r1["items"] + r2["items"],
        }
    # Also run official Qwen3ASRModel(forced_aligner=...) chunked (<=180s) for F_300s modular D1 comparison
    if "F_300s" in qwen_asr_data and qwen_asr_data["F_300s"].get("explicit", {}).get("text"):
        q_txt = qwen_asr_data["F_300s"]["explicit"]["text"]
        r_direct = next((r for r in results if r["case_id"] == "F_300s_over_limit_probe"), None)
        if r_direct and r_direct["status"] == "ok":
            variants["modular_qwen3_asr_plus_aligner_direct300s"] = {
                "source_text": q_txt,
                "window_audits": [r_direct["audit"]],
                "items": r_direct["items"],
            }
    if variants:
        f300_probe_out.write_text(json.dumps({"variants": variants}, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "total_runs": len(results),
        "errors": sum(1 for r in results if r["status"] != "ok"),
        "total_items": sum((r["audit"] or {}).get("total_items", 0) for r in results),
        "total_zero_duration_items": sum((r["audit"] or {}).get("zero_duration_count", 0) for r in results),
        "runs_with_zero_duration": sum(
            1 for r in results if (r["audit"] or {}).get("zero_duration_count", 0) > 0
        ),
        "total_repeated_identical_spans": sum(
            (r["audit"] or {}).get("repeated_identical_span_count", 0) for r in results
        ),
        "total_negative_or_inverted": sum(
            (r["audit"] or {}).get("negative_or_inverted_count", 0) for r in results
        ),
        "total_past_duration": sum(
            (r["audit"] or {}).get("past_duration_count", 0) for r in results
        ),
        "total_non_monotonic": sum(
            (r["audit"] or {}).get("non_monotonic_count", 0) for r in results
        ),
    }

    payload = {
        "model_path": str(QWEN_ALIGNER_DIR),
        "backend": "qwen_asr.Qwen3ForcedAligner (transformers BF16)",
        "load_s": load_s,
        "load_gpu": load_gpu,
        "summary": summary,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "load_s": load_s, "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
