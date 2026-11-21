# Handback — Modular Multilingual Transcription Evaluation

**Workspace:** `/workspace/nvidia-transcribe`  
**Base HEAD:** `721e73789ed2f575bc7c776665a9b04e9a6a097b`  
**Reference Repo (unchanged):** `/workspace/pyannote` (`2b40b1c0c17ce324178af765d8d2c79234870acd`)  
**Models Root (`/workspace/models`):** `Nemotron-3-Diarization` (`52a43267...`), `Qwen3-ASR-1.7B` (`7278e1e7...`), `Qwen3-ForcedAligner-0.6B` (`c7cbfc20...`), plus untouched `NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`  
**Decision:** **Category B — Hybrid Architecture Preferred**

## Executive Summary

1. **Continuous Nemotron 3 Diarization (`transformers==5.23.0`):** Loads in `0.324s` (`~419–579 MiB` VRAM), achieves strong overlap F1 on controlled mixtures (`S1`: `0.8578`, `S2 -8 dB`: `0.9293`, `S3`: `0.7838`) and zero false-positive overlap on `O3_music_neg`, and keeps child voices (`F1`) separated across `F_300s` (`02:30–07:30`). Over the full `1379.9s` French episode (`27.42s` streaming / `1.46s` offline), however, no speaker channel activated beyond slot `6` (`t=265.52s`, channel `7` max prob `0.2192`), collapsing `F1` child voices and later characters onto channels `0..6` with almost zero frame overlap (`0.05s`).
2. **Independent `Qwen3-ForcedAligner-0.6B` (`qwen-asr==0.0.6`):** Solves Nemotron's Mandarin `'钾'` CTC-OOV abort (`21/21` valid timings on `zh_fleurs_01_2006`, `'钾'` at `1.36–1.60s`), **but** outputs `80ms`-quantized spans with `331/2644` (`12.5%`) zero-duration tokens across `31/47` runs (`14.1%` even when chunked `<=180s` on `F_300s`, `41–49%` on `zh_2spk_overlap`). Ideal **strictly as an OOV fallback**, not as a replacement for healthy Nemotron `10–20ms` CTC timing.
3. **`Qwen3-ASR-1.7B` (`bfloat16`):** Slightly edges Nemotron on clean French FLEURS (`0.0816` vs `0.1020` WER) and `"Scoturb"`, matches on Mandarin (`0.1461` vs `0.1236` CER due to Chinese vs Arabic numerals), **but degrades on noisy multi-speaker cartoon dialogue** (`"comme des chiens"`, `"Daisy Meadows"`, `"Abel"`) and fails on simultaneous overlap (`S1` blended hallucination `0.85` WER; `S2` & `zh_2spk_overlap` dropped `100%` of overlapping secondary speech).
4. **Recommendation (Category B — Hybrid):** Retain **Nemotron 3.5 NVFP4** as primary ASR/overlap engine (after fixing the 8 documented chunked-core bugs), attach **`Qwen3-ForcedAligner-0.6B`** behind `ExternalForcedAligner` solely for CTC-OOV recovery, and evaluate scene-bounded (`<=3–5 min`) **`Nemotron-3-Diarization`** as an acoustic diarization anchor rather than unchunked 23-minute rollout.

## Evidence Artifacts

- [`architecture-decision.md`](file:///workspace/nvidia-transcribe/evaluations/modular-bakeoff/architecture-decision.md)
- [`environment.md`](file:///workspace/nvidia-transcribe/evaluations/modular-bakeoff/environment.md)
- [`model-inventory.md`](file:///workspace/nvidia-transcribe/evaluations/modular-bakeoff/model-inventory.md)
- [`recognition-comparison.md`](file:///workspace/nvidia-transcribe/evaluations/modular-bakeoff/recognition-comparison.md)
- [`alignment-comparison.md`](file:///workspace/nvidia-transcribe/evaluations/modular-bakeoff/alignment-comparison.md)
- [`diarization-comparison.md`](file:///workspace/nvidia-transcribe/evaluations/modular-bakeoff/diarization-comparison.md)
- [`integration-assessment.md`](file:///workspace/nvidia-transcribe/evaluations/modular-bakeoff/integration-assessment.md)
- [`test-results.md`](file:///workspace/nvidia-transcribe/evaluations/modular-bakeoff/test-results.md)
- [`benchmark.json`](file:///workspace/nvidia-transcribe/evaluations/modular-bakeoff/benchmark.json)
- [`artifact-manifest.json`](file:///workspace/nvidia-transcribe/evaluations/modular-bakeoff/artifact-manifest.json)
- `/workspace/nvidia-transcribe/outputs/handbacks/modular-bakeoff-evidence.tar.gz`
