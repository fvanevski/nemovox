#!/usr/bin/env python3
"""Aggregate machine-recomputed modular vs Nemotron/Pyannote comparison metrics."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from experiments.modular.common import (
    MODULAR_EVAL_DIR,
    MODULAR_OUT,
    PYANNOTE_ROOT,
    WORKSPACE_ROOT,
    compute_cer_detailed,
    compute_wer_detailed,
    load_shared_corpus,
)


def main() -> int:
    corpus = load_shared_corpus()
    qasr = json.loads((MODULAR_OUT / "raw/qwen_asr_eval.json").read_text(encoding="utf-8"))
    qalign = json.loads((MODULAR_OUT / "raw/qwen_aligner_eval.json").read_text(encoding="utf-8"))
    f300_align = json.loads((MODULAR_OUT / "raw/qwen_aligner_f300_chunked.json").read_text(encoding="utf-8"))
    diar = json.loads((MODULAR_OUT / "raw/nemotron_diar_eval.json").read_text(encoding="utf-8"))
    integ = json.loads((MODULAR_OUT / "raw/integration_probe.json").read_text(encoding="utf-8"))
    inv = json.loads((MODULAR_OUT / "logs/model_inventory_raw.json").read_text(encoding="utf-8"))
    nem35_b = json.loads((WORKSPACE_ROOT / "evaluations/nemotron35/benchmark.json").read_text(encoding="utf-8"))
    py_ov_b = json.loads(
        (PYANNOTE_ROOT / ".superpowers/audits/subtitle-quality/overlap-recovery/benchmark.json").read_text(encoding="utf-8")
    )["cases"]

    # 1. French FLEURS corpus WER
    fr_fleurs_rows: dict[str, Any] = {}
    q_fr_edits = q_fr_words = n_fr_edits = n_fr_words = 0
    for cid, meta in corpus["french_fleurs"].items():
        ref = meta["raw_transcription"]
        q_wer = qasr["cases"][cid]["explicit"]["wer"]
        n_clip = nem35_b["french_fleurs"]["clips"][cid]
        n_wer = compute_wer_detailed(ref, n_clip["result"]["source_text"], language="fr")
        q_fr_edits += q_wer["edits"]
        q_fr_words += q_wer["ref_words"]
        n_fr_edits += n_wer["edits"]
        n_fr_words += n_wer["ref_words"]
        fr_fleurs_rows[cid] = {
            "reference": ref,
            "qwen_asr_explicit": {"text": qasr["cases"][cid]["explicit"]["text"], "wer": q_wer},
            "qwen_asr_auto": {"text": qasr["cases"][cid]["auto"]["text"], "wer": qasr["cases"][cid]["auto"]["wer"]},
            "nemotron35_baseline": {"text": n_clip["result"]["source_text"], "wer": n_wer},
        }

    # 2. Mandarin FLEURS corpus CER
    zh_fleurs_rows: dict[str, Any] = {}
    q_zh_edits = q_zh_chars = n_zh_edits = n_zh_chars = 0
    for cid, meta in corpus["mandarin_fleurs"].items():
        ref = meta["raw_transcription"]
        q_cer = qasr["cases"][cid]["explicit"]["cer"]
        n_entry = nem35_b["mandarin"][cid]
        n_mode_key = "auto_ctc" if "auto_ctc" in n_entry else "explicit_ctc"
        n_text = n_entry[n_mode_key]["source_text"]
        n_cer = compute_cer_detailed(ref, n_text)
        q_zh_edits += q_cer["edits"]
        q_zh_chars += q_cer["ref_chars"]
        n_zh_edits += n_cer["edits"]
        n_zh_chars += n_cer["ref_chars"]
        zh_fleurs_rows[cid] = {
            "reference": ref,
            "qwen_asr_explicit": {"text": qasr["cases"][cid]["explicit"]["text"], "cer": q_cer},
            "nemotron35_baseline": {"mode": n_mode_key, "text": n_text, "cer": n_cer},
        }

    # 3. Synthetic overlap comparisons (Mandarin + French) & verified real O1-O4 + S1-S3
    zh_ov_ref_a = corpus["zh_2spk_overlap"]["speaker_a"]["text"]
    zh_ov_ref_b = corpus["zh_2spk_overlap"]["speaker_b"]["text"]
    zh_ov_ref_ab = f"{zh_ov_ref_a} {zh_ov_ref_b}"
    n_zh_ov_txt = nem35_b["overlap"]["zh_2spk_overlap"]["multispeaker_sot_mode"]["source_text"]
    n_zh_ov_cer = compute_cer_detailed(zh_ov_ref_ab, n_zh_ov_txt)

    fr_ov_ref_a = corpus["fr_2spk_overlap"]["speaker_a"]["text"]
    fr_ov_ref_b = corpus["fr_2spk_overlap"]["speaker_b"]["text"]
    fr_ov_ref_ab = f"{fr_ov_ref_a} {fr_ov_ref_b}"
    n_fr_ov_txt = nem35_b["overlap"]["fr_2spk_overlap"]["multispeaker_sot_mode"]["source_text"]
    n_fr_ov_wer = compute_wer_detailed(fr_ov_ref_ab, n_fr_ov_txt, language="fr")

    verified_fr_cases: dict[str, Any] = {}
    for cid in ("O1", "O2", "O3_music_neg", "O4"):
        py_key = "O3" if cid == "O3_music_neg" else cid
        verified_fr_cases[cid] = {
            "classification": py_ov_b[py_key]["classification"],
            "reference_text": py_ov_b[py_key]["reference_french_lexical"],
            "qwen_asr_explicit": {
                "text": qasr["cases"][cid]["explicit"]["text"],
                "wer": qasr["cases"][cid]["explicit"]["wer"],
            },
            "whisperx_pyannote_baseline": {
                "text": py_ov_b[py_key]["baseline"]["text"],
                "wer": compute_wer_detailed(
                    py_ov_b[py_key]["reference_french_lexical"], py_ov_b[py_key]["baseline"]["text"], language="fr"
                ),
            },
        }
    for cid in ("S1", "S2", "S3"):
        verified_fr_cases[cid] = {
            "classification": py_ov_b[cid]["classification"],
            "speaker_a_ref": py_ov_b[cid]["speaker_a_ref"],
            "speaker_b_ref": py_ov_b[cid]["speaker_b_ref"],
            "qwen_asr_explicit": {
                "text": qasr["cases"][cid]["explicit"]["text"],
                "wer_vs_a": qasr["cases"][cid]["explicit"]["wer_vs_speaker_a"],
                "wer_vs_b": qasr["cases"][cid]["explicit"]["wer_vs_speaker_b"],
                "wer_vs_combined_ab": qasr["cases"][cid]["explicit"]["wer_vs_combined_ab"],
            },
            "whisperx_pyannote_baseline": py_ov_b[cid]["baseline"],
            "nemotron3_diarization_streaming_der": diar["cases"][cid]["streaming"]["der_metrics"],
            "nemotron3_diarization_offline_der": diar["cases"][cid]["offline"]["der_metrics"],
        }

    # 4. Nemotron 3 Diarization compact summary (stripping full raw arrays for readable benchmark.json)
    diar_summary: dict[str, Any] = {
        "load_s": diar["load_s"],
        "load_gpu": diar["load_gpu"],
        "config": diar["config"],
        "cases": {},
    }
    for cid, c in diar["cases"].items():
        if cid == "F_full":
            st_sub = {k: v for k, v in c["streaming"].items() if k not in ("segments_file", "probs_sample_file")}
            of_sub = {k: v for k, v in c["offline"].items() if k != "segments_file"}
            diar_summary["cases"][cid] = {
                "duration_s": c["duration_s"],
                "streaming": st_sub,
                "offline": of_sub,
                "baselines": c["baselines"],
            }
        elif cid == "F_300s":
            diar_summary["cases"][cid] = {
                "duration_s": c["duration_s"],
                "offset_s": c["offset_s"],
                "streaming": {k: v for k, v in c["streaming"].items() if k != "segments"},
                "offline": {k: v for k, v in c["offline"].items() if k != "segments"},
            }
        else:
            diar_summary["cases"][cid] = {
                "duration_s": c["duration_s"],
                "offset_s": c["offset_s"],
                "streaming": {k: v for k, v in c["streaming"].items() if k != "segments"},
                "offline": {k: v for k, v in c["offline"].items() if k != "segments"},
                "streaming_segments": c["streaming"]["segments"],
            }

    # 5. Alignment summary (strip word-by-word arrays except zh_fleurs_01_2006 `'钾'`)
    align_rows = []
    potassium_case = None
    for r in qalign["results"]:
        row = {k: v for k, v in r.items() if k != "items"}
        align_rows.append(row)
        if r["case_id"] == "zh_fleurs_01_2006" and r["condition"] == "B3_reference_text":
            potassium_case = r["items"]

    # 6. Runtime & resource comparison across candidates
    benchmark = {
        "git": {
            "nvidia_transcribe_base": "721e73789ed2f575bc7c776665a9b04e9a6a097b",
            "nvidia_transcribe_current": subprocess.check_output(
                ["git", "-C", str(WORKSPACE_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "pyannote_reference_head": subprocess.check_output(
                ["git", "-C", str(PYANNOTE_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
        },
        "model_inventory": {
            name: {
                "local_dir": meta["local_dir"],
                "total_bytes": meta["total_bytes"],
                "architectures": meta["config"].get("architectures"),
                "transformers_version": meta["config"].get("transformers_version"),
                "file_count": len(meta["files"]),
            }
            for name, meta in inv.items()
        },
        "asr_comparison": {
            "qwen_load_s": qasr["load_s"],
            "qwen_load_gpu": qasr["load_gpu"],
            "french_fleurs": {
                "qwen_corpus_wer": round(q_fr_edits / q_fr_words, 4),
                "nemotron35_corpus_wer": round(n_fr_edits / n_fr_words, 4),
                "total_ref_words": q_fr_words,
                "clips": fr_fleurs_rows,
            },
            "mandarin_fleurs": {
                "qwen_corpus_cer": round(q_zh_edits / q_zh_chars, 4),
                "nemotron35_corpus_cer": round(n_zh_edits / n_zh_chars, 4),
                "total_ref_chars": q_zh_chars,
                "clips": zh_fleurs_rows,
            },
            "zh_2spk_seq": {
                "reference": qasr["cases"]["zh_2spk_seq"]["reference"],
                "qwen_asr_explicit": qasr["cases"]["zh_2spk_seq"]["explicit"],
                "nemotron35_speaker": {
                    "text": nem35_b["overlap"]["zh_2spk_seq"]["raw_text"],
                    "cer": compute_cer_detailed(
                        qasr["cases"]["zh_2spk_seq"]["reference"],
                        nem35_b["overlap"]["zh_2spk_seq"]["raw_text"].replace("<spk:0>", "").replace("<spk:1>", ""),
                    ),
                },
            },
            "zh_2spk_overlap": {
                "reference_ab": zh_ov_ref_ab,
                "qwen_asr_explicit": qasr["cases"]["zh_2spk_overlap"]["explicit"],
                "nemotron35_multispeaker_sot": {
                    "raw_text": nem35_b["overlap"]["zh_2spk_overlap"]["multispeaker_sot_mode"]["raw_text"],
                    "cer_vs_combined_ab": n_zh_ov_cer,
                },
            },
            "fr_2spk_overlap": {
                "reference_ab": fr_ov_ref_ab,
                "qwen_asr_explicit": qasr["cases"]["fr_2spk_overlap"]["explicit"],
                "nemotron35_multispeaker_sot": {
                    "raw_text": nem35_b["overlap"]["fr_2spk_overlap"]["multispeaker_sot_mode"]["raw_text"],
                    "wer_vs_combined_ab": n_fr_ov_wer,
                },
            },
            "french_verified_overlap_and_neg": verified_fr_cases,
            "french_animation_f1_to_f7_and_longform": {
                cid: {
                    "duration_s": qasr["cases"][cid]["duration_s"],
                    "auto_lid": qasr["cases"][cid]["auto"]["detected_language_raw"],
                    "explicit_text": qasr["cases"][cid]["explicit"]["text"],
                    "explicit_elapsed_s": qasr["cases"][cid]["explicit"]["elapsed_s"],
                    "explicit_with_context": qasr["cases"][cid].get("explicit_with_context"),
                }
                for cid in ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F_100s", "F_300s")
            },
        },
        "alignment_comparison": {
            "qwen_aligner_load_s": qalign["load_s"],
            "qwen_aligner_load_gpu": qalign["load_gpu"],
            "qwen_aligner_summary": qalign["summary"],
            "potassium_zh_fleurs_01_2006_items": potassium_case,
            "f300_combined_chunked_180s": f300_align.get("combined_runtime"),
            "f300_window_audits": {k: v["window_audits"] for k, v in f300_align["variants"].items()},
            "runs": align_rows,
        },
        "diarization_comparison": diar_summary,
        "integration_probe_f300s": {
            k: {sk: sv for sk, sv in v.items() if sk != "utterances"}
            for k, v in integ.items()
        },
    }

    MODULAR_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODULAR_EVAL_DIR / "benchmark.json"
    out_path.write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
    (MODULAR_OUT / "metrics/benchmark_summary.json").write_text(
        json.dumps(
            {
                "french_fleurs_wer": {
                    "qwen": benchmark["asr_comparison"]["french_fleurs"]["qwen_corpus_wer"],
                    "nemotron35": benchmark["asr_comparison"]["french_fleurs"]["nemotron35_corpus_wer"],
                },
                "mandarin_fleurs_cer": {
                    "qwen": benchmark["asr_comparison"]["mandarin_fleurs"]["qwen_corpus_cer"],
                    "nemotron35": benchmark["asr_comparison"]["mandarin_fleurs"]["nemotron35_corpus_cer"],
                },
                "qwen_aligner_summary": qalign["summary"],
                "diar_full_episode_streaming_s": diar["cases"]["F_full"]["streaming"]["elapsed_s"],
                "diar_full_episode_streaming_speakers": diar["cases"]["F_full"]["streaming"]["speaker_count"],
                "integration_f300s_stats": {k: v["stats"] for k, v in integ.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"benchmark_json": str(out_path), "size_bytes": out_path.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
