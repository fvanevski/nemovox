# Phase C: GPU Memory Capacity & Stage Handoff Measurements (`20261122T054544Z`)

- **GPU:** Single NVIDIA RTX 5090 (`32607 MiB` total VRAM).
- **Pre-existing idle baseline (non-Nemovox desktop/embedding services untouched):** `3048 MiB used / 29079 MiB free` (`kwin_wayland` ~`326–328 MiB`, `/usr/local/bin/text-embeddings-router` `1726 MiB`, PID `1216176`).
- **Configured admission floor (`NEMOVOX_LIFECYCLE_CONFIG`):** `min_free_vram_mb: 26000` on both `asr_local` and `text_local` (measured against baseline `29079 MiB` free).

## Measured Peak VRAM per Serialized Stage (from live `nvidia-smi` polling + Riva telemetry)
| Stage | Model / Runtime | Measured Peak GPU Total Used | Min Free VRAM Observed | Post-Stage Baseline Restored |
|---|---|---:|---:|---|
| Idle baseline (before & after) | None (units `inactive`) | `3048–3049 MiB` | `29078–29079 MiB` | Baseline |
| Managed ASR (`process-media`) | Nemotron 3.5 Transcribe 30B-A4B (`modelopt_mixed` NVFP4 + `fp8` KV cache) | `31451 MiB` | `676 MiB` | Yes (`3048 MiB used / 29079 MiB free`) |
| Riva MT CUDA worker (`Studio` / `CLI`) | `Riva-Translate-4B-Instruct-v2` (`torch.bfloat16`, `cuda:0`) | `11533–11781 MiB` total (`8013.55–8014.75 MiB` torch peak) | `20346 MiB` | Yes (`3048 MiB used / 29079 MiB free`) |
| Managed English review (`text_local`) | `ninfer-serve` (`qwen3_8_27b_swift15abl_nvfp4full-dflash2`, 18.0 GiB weights, 7 GiB KV) | `30617 MiB` | `1510 MiB` | Yes (`3048–3049 MiB used`) |

## Handoff Observations
- Serialized single-GPU execution succeeded with `<= 1510 MiB` margin during `ninfer-serve` and `676 MiB` margin during `vllm` ASR without OOM, CPU fallback, or touching unrelated GPU services (`text-embeddings-router`, `local_embedding`, desktop compositor).
- Explicit release before stage boundaries kept peak VRAM below `32607 MiB` throughout all runs; concurrent execution would exceed `32607 MiB` and is prevented by `ModelLifecycle._lock()` plus `stop_after_stage: true`.
