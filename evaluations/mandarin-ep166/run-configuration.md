# Run Configuration & Execution Summary — EP166

## Stage A: Nemotron 3.5 ASR + CTC Alignment (with Qwen Bottleneck Documentation)

- **Model:** `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` served via [start_vllm_server.sh](file:///workspace/nvidia-transcribe/scripts/start_vllm_server.sh) on `http://127.0.0.1:8190/v1`.
- **Verified server quantization:** `quantization=modelopt_mixed`, `kv_cache_dtype=fp8`, `gpu_memory_utilization=0.78` (`~27,548 MiB / 32,607 MiB` VRAM).
- **Command configuration:** `--source-language zh --mode speaker --chunk-seconds 90 --context-seconds 10 --alignment-fallback [qwen -> off] --speaker-embeddings titanet --audio-stream 0`.

### Qwen Alignment Bottleneck Analysis

1. **Initial `--alignment-fallback qwen` execution (`task-1428`, `18:30:11–18:40:31`):**
   - **All 15 / 15 Nemotron ASR chunks completed in ~1.5–2.0 minutes** (`chunk_000.raw.json` .. `chunk_014.raw.json`; `manifest.json` latency range `0.66s–2.11s` for chunks `0..13`, `71.19s` for Chunk `14` which hit `16,384` token limit on trailing silence/looping).
   - Post-ASR alignment then invoked `QwenForcedAlignerAdapter` sequentially on each OOV/degraded speaker span via [_qwen_align_worker.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/_qwen_align_worker.py).
   - **Root cause of latency:** `_qwen_align_worker.py` executes **`Qwen3ForcedAligner.from_pretrained(model_path, dtype=dtype, device_map=use_device, local_files_only=True)` in a cold subprocess for every single span**, loading and tearing down the `0.6B` model repeatedly (`3.0%` system RAM, `226%` CPU burst per invocation while `vLLM` remained resident at `27.5 GiB` VRAM with `0%` GPU utilization).
   - **Observed progress before graceful interrupt (~10m 20s elapsed):** Reached `chunk_004` (`0` final chunk contracts persisted; `78` `_sub_*.wav` clips already extracted across chunks `1–4`; `85` total vLLM HTTP `POST /v1/chat/completions` calls logged = `15` full chunks + `70` subinterval CTC calls). Given dense dialogue chunks `5, 6, 8, 9, 12, 14` ahead across 7 OOV-bearing chunks (`139` `_sub_*.wav` spans totaling `996.64s` of audio), Qwen-per-span projected `>25–30 minutes` end-to-end.
2. **Resumed `--alignment-fallback off` execution (`task-1447`, NO `--fresh`):**
   - `process-media` validated SHA-256 signatures and reused **all 15/15 completed full-chunk WAVs and `.raw.json` responses** (`0` full-chunk ASR requests repeated; `84` new `POST`s strictly for bounded subinterval CTC recovery + TitaNet CPU diarization reconciliation).
   - **Total resumed wall time:** **`36.21s`** (saving ~25+ minutes while preserving native CTC behavior for baseline diagnostics).
   - **Stage A output counts:** `15` chunks, `285` utterances (`175` `full`, `69` `partial`, `41` `failed`/untimed), `22` speakers, `66` boundary-deduplicated units, `4` boundary-recovered units.

## Stage B: Mandarin Glossary

- Created [a_will_eternal_zh.yaml](file:///workspace/nvidia-transcribe/configs/glossaries/a_will_eternal_zh.yaml) strictly from recurring verified entities/terminology in EP166 `transcript.txt` (including observed ASR variants for Bai Xiaochun, Bai Hao, Song Que, Yue'er, Hongchen, Tomb Keeper, Underworld Emperor/Palace/River, Arch-Emperor Dynasty/City, Starry Sky Dao Polarity Sect, Heavenly Dao Nascent Soul, Undying Bones, 21/24-colored Flame).

## Stage C: SwiftU Refinement & Translation

- **Server:** `/workspace/ninfer/swiftu` (`Qwen3.8-27B` NVFP4 + `dflash2`) served on `http://127.0.0.1:8001/v1` (reusing `reasoning_effort="low"`, `max_tokens=8192`).
- **Command:** `generate-subtitles` with `--refinement balanced --translator swiftu --style natural --formats srt,ass`.
- **Stage 2 (Refinement):** `48` API calls, **`94 / 285` utterances refined**, `0` screen rejections, `0` batch errors, **`348.447s`**.
- **Stage 3 (Translation):** `50` API calls, `284` translated (`1` `fallback_empty_translation`: single char `'的'`), **`15` adjacent same-speaker regroupings**, `0` batch errors, **`228.084s`**.
- **Stage 4 (Subtitles):** `307` SRT cues, `304` ASS cues (`50` `Top`-lane Layer `1`, `3` overflow), `41` untimed utterances marked `unrenderable_cues`; total CLI wall time **`576.624s`**.
