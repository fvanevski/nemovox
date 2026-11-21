# Bounded RTX 5090 CUDA BF16 Alignment Acceptance (`7b4112cc31784953abbb041fd0b5d7997851841d`)

**Provenance:** Freshly executed on merged `main` (`7b4112cc31784953abbb041fd0b5d7997851841d`) in the isolated worktree using the offline Qwen3-ForcedAligner-0.6B checkpoint (`/workspace/models/Qwen3-ForcedAligner-0.6B`) and isolated Qwen worker (`outputs/modular-eval/.venv/bin/python`), replaying recorded Mandarin Nemotron ASR responses (`--qwen-gpu-mode available`, `--qwen-device cuda:0`). Recorded ASR replay is not proof of live Nemotron ASR throughput.

Raw telemetry: [`logs/gpu_qualification_results.json`](logs/gpu_qualification_results.json).

## GPU & Cleanup State

- **GPU:** NVIDIA GeForce RTX 5090 (`32,607 MiB` total)
- **VRAM before / after:** `3052 MiB` used (`29,075 MiB` free) before -> `3052 MiB` used (`29,076 MiB` free) after -> **zero worker leaks**, zero residual CUDA allocations.
- **Checkpointing:** `native-alignment.checkpoint.json` written (`23,432` bytes in `two_span_subtitle_qwen`).

## Measured Replay Results

| Fixture (Recorded ASR Replay) | Mode / Policy | Windows (`[lo, hi] -> [start, end]`) | Wall (s) | Utterance Status | Timed / Total | Model Loads / Aligns | Device / Observed Dtype | Peak CUDA Alloc | Invalid / Zero-Dur | `'钾'` Timings (s) |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| `two_span_oov` (`18.56s`, 2 chunks) | `off` | — | 0.1558 | `failed: 2` | 0 / 42 | 0 / 0 | — | — | 0 | untimed |
| `two_span_oov` (`18.56s`, 2 chunks) | `qwen` (`subtitle`) | `[0, 21] -> [0.77, 6.98]`, `[0, 21] -> [10.77, 16.98]` | 8.2116 | `full: 2` | **42 / 42** | **1 / 2** | `cuda:0` / `torch.bfloat16` | 1,900,171,776 B | 0 | `[1.41, 1.65]` & `[11.65, 11.89]` (`1.3455s` -> `0.0365s`) |
| `zh_multichunk_synthetic` (`69.78s`, 3 chunks) | `off` | — | 0.4220 | `full: 6, partial: 1, failed: 1` | 176 / 198 | 0 / 0 | — | — | 0 | untimed |
| `zh_multichunk_synthetic` (`69.78s`, 3 chunks) | `qwen` (`subtitle`) | `[23, 44] -> [11.68, 17.84]` (`[44, 66]` skipped: `skipped_low_subtitle_value`) | 8.9182 | `full: 7, partial: 1` | **197 / 198** | **1 / 1** | `cuda:0` / `torch.bfloat16` | 1,900,167,680 B | 0 | `[12.24, 12.48]` (`1.3806s`) |
| `zh_multichunk_synthetic` (`69.78s`, 3 chunks) | `qwen` (`fine`) | `[23, 44] -> [11.68, 17.84]` (`[44, 66]` skipped: `skipped_partial_span_unanchored_edge`) | 8.6479 | `full: 7, partial: 1` | **197 / 198** | **1 / 1** | `cuda:0` / `torch.bfloat16` | 1,900,167,680 B | 0 | `[12.24, 12.48]` (`1.3675s`) |

- **No silent CPU fallback:** Actual `device_used="cuda:0"`, `observed_dtype="torch.bfloat16"`, `model_load_count=1`, `align_count=2` on `two_span_oov` with warm span #2 completing in `0.0365s` vs span #1 `1.3455s` (36.9x faster).
- **Text & neighbor preservation:** Native-timed utterances and source text preserved verbatim; `'钾'` recovered cleanly without fabricating skipped spans.
