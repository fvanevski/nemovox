#!/usr/bin/env python3
"""Experiment C: Qwen3-ASR-1.7B evaluation (Transformers BF16, auto LID vs explicit)."""

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

import torch
import transformers
import qwen_asr
from qwen_asr import Qwen3ASRModel

from experiments.modular.common import (
    MODULAR_OUT,
    QWEN_ASR_DIR,
    compute_cer_detailed,
    compute_wer_detailed,
    load_shared_corpus,
    normalize_language,
)


def _gpu_stats() -> dict[str, Any]:
    return {
        "allocated_mib": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
        "reserved_mib": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
    }


def _run_asr(
    model: Qwen3ASRModel,
    audio_path: str,
    *,
    language: str | None,
    context: str = "",
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        out = model.transcribe(audio=audio_path, context=context, language=language, return_time_stamps=False)[0]
        elapsed_s = round(time.perf_counter() - t0, 4)
        return {
            "status": "ok",
            "error": None,
            "requested_language": language,
            "detected_language_raw": out.language,
            "detected_language_norm": normalize_language(out.language),
            "text": out.text,
            "elapsed_s": elapsed_s,
            "gpu": _gpu_stats(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "requested_language": language,
            "detected_language_raw": None,
            "detected_language_norm": "unknown",
            "text": "",
            "elapsed_s": round(time.perf_counter() - t0, 4),
            "gpu": _gpu_stats(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(MODULAR_OUT / "raw/qwen_asr_eval.json"))
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    args = parser.parse_args()

    corpus = load_shared_corpus()
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    model = Qwen3ASRModel.from_pretrained(
        str(QWEN_ASR_DIR),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=args.max_new_tokens,
        local_files_only=True,
    )
    load_s = round(time.perf_counter() - t_load0, 4)
    load_gpu = _gpu_stats()

    cases: dict[str, Any] = {}

    # Warmup
    _ = _run_asr(model, corpus["french_regions"]["F4"]["path"], language="French")

    # 1. French animation regions F1-F7 + F_100s + F_300s (exclude F_full per Section 9)
    for cid in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F_100s", "F_300s"):
        meta = corpus["french_regions"][cid]
        auto_res = _run_asr(model, meta["path"], language=None)
        exp_res = _run_asr(model, meta["path"], language="French")
        entry: dict[str, Any] = {
            "category": "french_animation",
            "duration_s": meta["duration_s"],
            "offset_s": meta["offset_s"],
            "auto": auto_res,
            "explicit": exp_res,
        }
        if cid == "F1":
            entry["explicit_with_context"] = _run_asr(
                model,
                meta["path"],
                language="French",
                context="Koumlar, Nat'cal, Ambrosius, Tao, Zia, Esteban, Mu",
            )
        cases[cid] = entry

    # 2. French FLEURS verified clips + O1-O4 verified clips + S1-S3 overlap mixtures + fr_2spk_overlap
    for cid, meta in corpus["french_fleurs"].items():
        ref = meta["raw_transcription"]
        auto_res = _run_asr(model, meta["wav_path"], language=None)
        exp_res = _run_asr(model, meta["wav_path"], language="French")
        auto_res["wer"] = compute_wer_detailed(ref, auto_res["text"], language="fr")
        exp_res["wer"] = compute_wer_detailed(ref, exp_res["text"], language="fr")
        cases[cid] = {
            "category": "french_fleurs",
            "duration_s": meta["duration_s"],
            "reference": ref,
            "auto": auto_res,
            "explicit": exp_res,
        }

    for cid, meta in corpus["french_verified_overlap"].items():
        auto_res = _run_asr(model, meta["path"], language=None)
        exp_res = _run_asr(model, meta["path"], language="French")
        if "reference_text" in meta:
            ref = meta["reference_text"]
            auto_res["wer"] = compute_wer_detailed(ref, auto_res["text"], language="fr")
            exp_res["wer"] = compute_wer_detailed(ref, exp_res["text"], language="fr")
        else:
            ref_a = meta["speaker_a_ref"]
            ref_b = meta["speaker_b_ref"]
            ref_ab = f"{ref_a} {ref_b}"
            for res in (auto_res, exp_res):
                res["wer_vs_speaker_a"] = compute_wer_detailed(ref_a, res["text"], language="fr")
                res["wer_vs_speaker_b"] = compute_wer_detailed(ref_b, res["text"], language="fr")
                res["wer_vs_combined_ab"] = compute_wer_detailed(ref_ab, res["text"], language="fr")
        cases[cid] = {
            "category": "french_overlap_or_neg",
            "duration_s": meta["duration_s"],
            "classification": meta["classification"],
            "reference": meta.get("reference_text") or {
                "speaker_a": meta["speaker_a_ref"],
                "speaker_b": meta["speaker_b_ref"],
            },
            "auto": auto_res,
            "explicit": exp_res,
        }

    fr_ov = corpus["fr_2spk_overlap"]
    fr_ov_auto = _run_asr(model, fr_ov["wav_path"], language=None)
    fr_ov_exp = _run_asr(model, fr_ov["wav_path"], language="French")
    for res in (fr_ov_auto, fr_ov_exp):
        res["wer_vs_speaker_a"] = compute_wer_detailed(fr_ov["speaker_a"]["text"], res["text"], language="fr")
        res["wer_vs_speaker_b"] = compute_wer_detailed(fr_ov["speaker_b"]["text"], res["text"], language="fr")
        res["wer_vs_combined_ab"] = compute_wer_detailed(
            f"{fr_ov['speaker_a']['text']} {fr_ov['speaker_b']['text']}", res["text"], language="fr"
        )
    cases["fr_2spk_overlap"] = {
        "category": "french_synthetic_overlap",
        "duration_s": fr_ov["total_duration_s"],
        "reference": {"speaker_a": fr_ov["speaker_a"]["text"], "speaker_b": fr_ov["speaker_b"]["text"]},
        "auto": fr_ov_auto,
        "explicit": fr_ov_exp,
    }

    # 3. Mandarin FLEURS (including '钾' & mixed Latin/numbers) + zh_2spk_seq + zh_2spk_overlap + zh_multichunk
    for cid, meta in corpus["mandarin_fleurs"].items():
        ref = meta["raw_transcription"]
        auto_res = _run_asr(model, meta["wav_path"], language=None)
        exp_res = _run_asr(model, meta["wav_path"], language="Chinese")
        auto_res["cer"] = compute_cer_detailed(ref, auto_res["text"])
        exp_res["cer"] = compute_cer_detailed(ref, exp_res["text"])
        cases[cid] = {
            "category": "mandarin_fleurs",
            "duration_s": meta["duration_s"],
            "reference": ref,
            "auto": auto_res,
            "explicit": exp_res,
        }

    zh_seq = corpus["zh_2spk_seq"]
    zh_seq_ref = f"{zh_seq['speaker_a_ref']} {zh_seq['speaker_b_ref']} {zh_seq['speaker_a_ref_2']}"
    zh_seq_auto = _run_asr(model, zh_seq["path"], language=None)
    zh_seq_exp = _run_asr(model, zh_seq["path"], language="Chinese")
    zh_seq_auto["cer"] = compute_cer_detailed(zh_seq_ref, zh_seq_auto["text"])
    zh_seq_exp["cer"] = compute_cer_detailed(zh_seq_ref, zh_seq_exp["text"])
    cases["zh_2spk_seq"] = {
        "category": "mandarin_sequential",
        "duration_s": zh_seq["duration_s"],
        "reference": zh_seq_ref,
        "auto": zh_seq_auto,
        "explicit": zh_seq_exp,
    }

    zh_ov = corpus["zh_2spk_overlap"]
    zh_ov_auto = _run_asr(model, zh_ov["wav_path"], language=None)
    zh_ov_exp = _run_asr(model, zh_ov["wav_path"], language="Chinese")
    for res in (zh_ov_auto, zh_ov_exp):
        res["cer_vs_speaker_a"] = compute_cer_detailed(zh_ov["speaker_a"]["text"], res["text"])
        res["cer_vs_speaker_b"] = compute_cer_detailed(zh_ov["speaker_b"]["text"], res["text"])
        res["cer_vs_combined_ab"] = compute_cer_detailed(
            f"{zh_ov['speaker_a']['text']} {zh_ov['speaker_b']['text']}", res["text"]
        )
    cases["zh_2spk_overlap"] = {
        "category": "mandarin_synthetic_overlap",
        "duration_s": zh_ov["total_duration_s"],
        "reference": {"speaker_a": zh_ov["speaker_a"]["text"], "speaker_b": zh_ov["speaker_b"]["text"]},
        "auto": zh_ov_auto,
        "explicit": zh_ov_exp,
    }

    zh_mc = corpus["zh_multichunk_synthetic"]
    zh_mc_auto = _run_asr(model, zh_mc["path"], language=None)
    zh_mc_exp = _run_asr(model, zh_mc["path"], language="Chinese")
    cases["zh_multichunk_synthetic"] = {
        "category": "mandarin_multichunk",
        "duration_s": zh_mc["duration_s"],
        "auto": zh_mc_auto,
        "explicit": zh_mc_exp,
    }

    payload = {
        "model_path": str(QWEN_ASR_DIR),
        "backend": "qwen_asr.Qwen3ASRModel (transformers BF16)",
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "qwen_asr": getattr(qwen_asr, "__version__", "0.0.6"),
        },
        "config": {
            "dtype": "bfloat16",
            "max_inference_batch_size": 1,
            "max_new_tokens": args.max_new_tokens,
        },
        "load_s": load_s,
        "load_gpu": load_gpu,
        "cases": cases,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out_path), "load_s": load_s, "case_count": len(cases)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
