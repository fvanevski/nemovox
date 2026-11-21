# Handback: NVIDIA Nemotron 3.5 Transcribe NVFP4 Evaluation

## Executive Summary
NVIDIA Nemotron 3.5 Transcribe 30B-A4B NVFP4 runs natively on the local RTX 5090 (32 GB VRAM, Blackwell `sm_120`) using vLLM `0.32.0` + NeMo Speech (`99c252a77e7a1f57319618539882faa0d9c1abd2`) with active `modelopt_mixed` quantization (`19.28 GiB` model footprint, FP8 KV cache, 128 FP8 encoder Linears, CTC timestamp middleware). Throughput reached `53.8x–77.7x` real-time (full 23-minute French episode transcribed + word-aligned in `17.75s` at `28.8 GiB` peak VRAM).

However, **an immediate wholesale replacement of `/workspace/pyannote` is NOT recommended** due to three verified failure modes:
1. **Long-form speaker instability:** Unchunked 5-min and 23-min passes suffer tag collapse or rapid word-level `<spk:N>` ping-ponging (whereas 39–100s segments outperform Pyannote on similar child voices).
2. **CTC vocabulary OOV failure (`alignment_failed`):** Valid CJK characters outside the 23,862-token CTC vocabulary (observed on `'钾'`) abort word alignment for the entire request (`words=[]`).
3. **Short-clip (<4s) & loanword drift:** Cropped 2–3s clips induced cross-language hallucinations, and Mandarin loanwords exhibited Katakana leakage (`Scotturb` -> `スコート儿`).

A **chunked (60–120s) hybrid migration** with CTC-OOV fallback and separate text translation is the recommended path forward.

## Acceptance Summary
- **Checkpoint validation:** PASS (`/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`, rev `66c6adc4020179567074d0399cda2a4eeb2bb347`)
- **RTX 5090 NVFP4 execution:** PASS (`quantization=modelopt_mixed`, no BF16 fallback)
- **French ASR & diarization:** PASS (FLEURS WER `10.20%`; superior child-voice separation on 39s/100s)
- **Mandarin ASR & timing:** PARTIAL (FLEURS CER `12.36%` auto / `10.67%` explicit; `alignment_failed` on `'钾'`)
- **Simultaneous speakers:** PASS on `multispeaker_sot` controlled overlaps (FR & ZH)
- **CTC timestamps & subtitle compatibility:** PARTIAL (100% valid on French & 5/6 Mandarin; verified read-only with `/workspace/pyannote`'s `build_subtitles`)
- **CPU unit tests:** PASS (`15/15` via `.venv/bin/pytest -v -m "not gpu"`)

## Key Artifacts
- `evaluations/nemotron35/environment.md`
- `evaluations/nemotron35/benchmark.json`
- `evaluations/nemotron35/quality-review.md`
- `evaluations/nemotron35/runtime-contract.md`
- `evaluations/nemotron35/test-results.md`
- `evaluations/nemotron35/architecture-assessment.md`
- `evaluations/nemotron35/checkpoint_manifest.json`
- `evaluations/nemotron35/pip_list.txt`
- `evaluations/nemotron35/pytest_cpu.txt`
- `evaluations/nemotron35/artifact-manifest.json`
- `outputs/handbacks/nemotron35-nvfp4-evaluation.tar.gz`
