# Evaluated Model Inventory

All three candidate checkpoints were snapshotted at pinned Hugging Face revisions under `/workspace/models/` and verified offline via SHA-256 and safetensors metadata. Existing `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` was left unmodified.

| Model | Repository | Pinned Commit SHA | Local Path | License | Architecture (`config.json`) | Parameter Count | Weight Dtype | Total Disk Size (Files) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Nemotron 3 Diarization** | `nvidia/Nemotron-3-Diarization` | `52a432672732425e4c19765b9fec9b5810e6f4ba` | `/workspace/models/Nemotron-3-Diarization` | `openmdw-1.1` | `Nemotron3DiarizationForAudioFrameClassification` (`transformers>=5.18.0.dev0`) | `99,226,504` (`0.099B`) | `float32` | `676.97 MB` (`16` files; `model.safetensors`: `396,954,592` B) |
| **Qwen3-ASR-1.7B** | `Qwen/Qwen3-ASR-1.7B` | `7278e1e70fe206f11671096ffdd38061171dd6e5` | `/workspace/models/Qwen3-ASR-1.7B` | `apache-2.0` | `Qwen3ASRForConditionalGeneration` (`transformers==4.57.6`) | `2,349,217,408` (`2.349B`) | `bfloat16` | `4,485.24 MB` (`12` files; 2 shards) |
| **Qwen3-ForcedAligner-0.6B** | `Qwen/Qwen3-ForcedAligner-0.6B` | `c7cbfc2048c462b0d63a45797104fc9db3ad62b7` | `/workspace/models/Qwen3-ForcedAligner-0.6B` | `apache-2.0` | `Qwen3ASRForConditionalGeneration` (`transformers==4.57.6`) | `917,728,896` (`0.918B`) | `bfloat16` | `1,754.83 MB` (`10` files; 1 shard) |

## Key Checkpoint Configuration Observations

- **`Nemotron-3-Diarization`:** `head_config.num_speakers = 8`, `subsampling_factor = 8` (10ms mel frame -> 80ms encoder frame), streaming defaults `low_latency` `(chunk_length=9, right_context=4)` = `1.04s` buffer, `fifo_length=264`, `speaker_cache_length=264`, `speaker_cache_update_period=222`. Offline config uses `chunk_length=340` (~27.2s) and `fifo_length=40`.
- **`Qwen3-ASR-1.7B`:** AuT audio encoder + 1.7B LM (`qwen3_asr`), `MAX_ASR_INPUT_SECONDS = 1200` in `qwen_asr.inference.utils`. Default `max_new_tokens = 512` in `Qwen3ASRModel.from_pretrained` requires override (tested at `4096`) to avoid silently truncating 3–5 minute French passages.
- **`Qwen3-ForcedAligner-0.6B`:** Non-autoregressive timestamp-bin classifier quantized to `80ms` steps (`timestamp_segment_time`), hard-capped in `qwen_asr.inference.utils` at **`MAX_FORCE_ALIGN_INPUT_SECONDS = 180` seconds** (3 minutes, stricter than the ~5-minute HF card summary). Supports French and Mandarin Chinese (`get_supported_languages()`), strips punctuation except `'`, and splits CJK characters while keeping contiguous Latin words intact.

Complete per-file SHA-256 digests live in `outputs/modular-eval/logs/model_inventory_raw.json` (and staged into the evidence archive).
