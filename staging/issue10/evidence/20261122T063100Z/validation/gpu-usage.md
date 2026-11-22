# Phase C: GPU Memory Capacity & Stage Handoff Measurements (`20261122T063100Z`)

- **GPU:** Single NVIDIA RTX 5090 (`32607 MiB` total VRAM).
- **Pre-existing idle baseline:** `3048 MiB used / 29079 MiB free`.
- **Admission floor (`NEMOVOX_LIFECYCLE_CONFIG`):** `min_free_vram_mb: 26000` for `asr_local` and `text_local`.

## Observed VRAM Across Serialized Stages
| Stage | Model / Runtime | Observed Used VRAM | Observed Min Free VRAM | Restored Baseline |
|---|---|---:|---:|---|
| Idle baseline (pre/post) | None (`units inactive`) | `3048 MiB` | `29079 MiB` | Baseline |
| Studio ASR (`e1847762`) | Nemotron 30B-A4B (`modelopt_mixed` NVFP4 + `fp8` KV) | `26875–27339 MiB` | `4788 MiB` | Yes (`3048 MiB`) |
| Riva CUDA worker | `Riva-Translate-4B-Instruct-v2` (`torch.bfloat16`) | `11779 MiB` total (`8013.55 MiB` torch peak) | `20348 MiB` | Yes (`3048 MiB`) |
| Studio English review (`text_local`) | `ninfer-serve` Qwen3.8-27B NVFP4 (`weights 18.0 GiB`) | `30617 MiB` | `1510 MiB` | Yes (`3048 MiB`) |
| Post-`SIGKILL` recovery (`9387b8bb`) | ASR killed at `24689 MiB` used -> stopped -> restored | `24689 -> 3048 MiB` | `7438 -> 29079 MiB` | Yes (`3048 MiB`) |
| Post-startup reconcile (`text_local`) | Abandoned `ninfer-serve` at `30617 MiB` -> stopped -> restored | `30617 -> 3048 MiB` | `1510 -> 29079 MiB` | Yes (`3048 MiB`) |

- All pre-existing non-Nemovox GPU processes and desktop services remained running and untouched throughout.
