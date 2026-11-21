# Handback — Chunked Nemotron Transcription Core

**Repository:** `/workspace/nvidia-transcribe`  
**Base Commit:** `0fc70b1d5d15da616458f5f11915ecd4c31959af`  
**Checkpoint (unchanged):** `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`  
**Reference Repo (unchanged):** `/workspace/pyannote` (`2b40b1c0c17ce324178af765d8d2c79234870acd`)  
**Hardware:** NVIDIA RTX 5090 (32607 MiB VRAM; returned to idle `3445 MiB` after validation)

## 1. Delivered Capabilities

1. **CLI-driven chunked media pipeline (`nvidia-transcribe process-media`):** Accepts audio or video via `ffprobe`/`ffmpeg` ([`media.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/media.py)), plans gap-free core windows with configurable left/right context ([`chunking.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/chunking.py)), checkpoints per chunk into `manifest.json` for crash-safe `--resume` / `--fresh` execution ([`recording.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/recording.py)), and outputs canonical [`nemotron-recording-contract/v1`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/contract.md) (`recording.contract.json`) + `transcript.txt`.
2. **Per-speaker multilingual utterance reconstruction ([`utterances.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/utterances.py)):** Separates concurrent speakers before joining tokens, turning Mandarin `zh_2spk_overlap` from 32 interleaved 1–2 char fragments into 2 concurrent speaker streams while honoring CJK vs French spacing/punctuation rules.
3. **Conservative two-stage speaker reconciliation ([`speakers.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/speakers.py)):** Reconciles chunk-local IDs via adjacent temporal+lexical overlap plus optional CPU-only NeMo `titanet_small` embeddings (`0 MiB` GPU overhead) with strict `cannot_link` collision guards and explicit `provisional_single_chunk` / `unknown` uncertainty tracking.
4. **Non-destructive CTC alignment diagnostics & sub-interval recovery ([`alignment.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/alignment.py)):** Preserves exact verbatim text under CJK OOV failures (`'钾'`, `'诫'`, `'轱'`), recovers surrounding non-OOV sub-windows (`83/107` units recovered in `zh_multichunk_synthetic` chunk 0), isolates failing windows with `alignment_status="diarization_bounded_only"`, and leaves an `ExternalForcedAligner` hook.
5. **Reproducible CPU & GPU verification:** 30/30 CPU unit tests passing in `0.54s` (`-m "not gpu"`), plus 300s French window ablation (`40s`/`60s`/`90s`/`120s`), full 23-min French episode (`31.0437s`, `2016/2016` aligned units, zero ping-pong), temporal-only vs TitaNet comparison, Mandarin OOV + overlap + multi-chunk recovery runs, and prompt comparisons.

## 2. Key Open Architectural Observations

- Valid CTC alignment (`2016/2016` timed words in French episode) does not guarantee perfect speaker attribution: in `chunk_002`, Sortformer detected 5 speakers (`'0'..'4'`) while the LLM only emitted 3 (`'0'..'2'`), merging two voices into `SPK_06`, alongside two 1-word stray speaker tags (`Il`, `Diablement`). Preserving raw Sortformer `diarization` per chunk lets downstream subtitle/UI layers audit or refine diarization independently.
- Prompt comparison confirmed `mode="speaker"` remains the strongest multilingual baseline (outperforming `script_control` on Mandarin code-switched overlap).

## 3. Evidence Index

- [`architecture.md`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/architecture.md)
- [`contract.md`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/contract.md)
- [`benchmark.json`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/benchmark.json)
- [`supplemental.json`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/supplemental.json)
- [`quality-review.md`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/quality-review.md)
- [`test-results.md`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/test-results.md)
- [`speaker-reconciliation.md`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/speaker-reconciliation.md)
- [`alignment-diagnostics.md`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/alignment-diagnostics.md)
- [`artifact-manifest.json`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/artifact-manifest.json)
- `/workspace/nvidia-transcribe/outputs/handbacks/chunked-core-evidence.tar.gz`
