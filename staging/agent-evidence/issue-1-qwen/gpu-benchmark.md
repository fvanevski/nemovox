# Bounded RTX 5090 CUDA BF16 Qualification (`28655e0d6fe57d9363e576cd0795a24aae77a4a3`)

## Execution Environment

- **GPU:** Single NVIDIA GeForce RTX 5090 (32,607 MiB total VRAM)
- **VRAM before / after qualification:** `3050 MiB` used / `29077 MiB` free before **and** `3050 MiB` used / `29077 MiB` free after (`0 MiB` leaked).
- **Pre-existing GPU processes (untouched):** `text-embeddings-router` (1728 MiB), `kwin_wayland` (328 MiB).
- **ASR Replay Provenance:** Because no live Nemotron ASR server was running on `127.0.0.1:8190` during qualification, `process_media` replayed the exact recorded Nemotron vLLM `raw.json` / subinterval responses from prior host runs, exercising the complete real Phase 1 -> `native-alignment.checkpoint.json` -> `AdmissionPolicy` -> persistent CUDA BF16 `QwenForcedAlignerAdapter` (`_qwen_align_worker.py` in `outputs/modular-eval/.venv`) -> Phase 2 speaker/utterance reconstruction pipeline.

## `off` vs. Optimized `qwen` Comparison

| Fixture | Mode / Policy | Wall (s) | Utterance Timing Status | Timed / Total Units | Model Loads | Align Calls | Device / Observed Dtype | Peak CUDA Alloc (MiB) | Invalid / Zero-Dur | Recovered `'钾'` Timings (s) |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `zh_fleurs_01_2006` (`8.28s`) | `off` | 0.1136 | `failed: 1` | 0 / 21 | 0 | 0 | — | — | 0 | untimed |
| `zh_fleurs_01_2006` (`8.28s`) | `qwen` (`subtitle`) | 11.8591 | `full: 1` | **21 / 21** | **1** | **1** | `cuda:0` / `torch.bfloat16` | 1812.1 | 0 | `[1.41, 1.65]` |
| `zh_multichunk_synthetic` (`69.78s`, 3 chunks) | `off` | 0.5574 | `full: 5, partial: 1, failed: 1` | 156 / 179 | 0 | 0 | — | — | 0 | untimed |
| `zh_multichunk_synthetic` (`69.78s`, 3 chunks) | `qwen` (`subtitle`) | 9.2508 | `full: 6, partial: 1` | **177 / 179** | **1** | **1** | `cuda:0` / `torch.bfloat16` | 1812.1 | 0 | `[12.24, 12.48]` (2-char gap skipped) |
| `zh_multichunk_synthetic` (`69.78s`, 3 chunks) | `qwen` (`fine`) | 9.2219 | `full: 7` | **179 / 179** | **1** | **2** | `cuda:0` / `torch.bfloat16` | 1812.2 | 1 zero-dur rejected | `[12.24, 12.48]` (span #1: `1.4028s`, span #2: **`0.0319s`**) |
| `two_span_oov` (`18.56s`, 2 chunks) | `off` | 0.1486 | `failed: 2` | 0 / 42 | 0 | 0 | — | — | 0 | untimed |
| `two_span_oov` (`18.56s`, 2 chunks) | `qwen` (`subtitle`) | 8.0702 | `full: 2` | **42 / 42** | **1** | **2** | `cuda:0` / `torch.bfloat16` | 1812.1 | 0 | `[1.41, 1.65]` & `[11.65, 11.89]` (span #1: `1.3054s`, span #2: **`0.0329s`**) |

## Key Observations

1. **Multi-span worker reuse speedup:** Inside `two_span_subtitle_qwen`, worker startup + model load took `5.4586s` (`model_load_wall_s=4.3662s`), span #1 inference took `1.3054s`, and warm span #2 inference in the **same worker process** (`model_load_count=1, align_count=2`) took **`0.0329s`** (39.7x faster than span #1; 166x faster than a cold worker reload). Similarly in `zh_multichunk_qwen_fine`, span #1 took `1.4028s` and span #2 took `0.0319s`.
2. **Selectivity preserved:** Under default `subtitle` policy on `zh_multichunk_synthetic`, the completely untimed 21-char `'钾'` utterance (`utt-0001`) was admitted and fully recovered (`[12.00, 17.44]`), while the 2-char gap in `utt-0002` (`20/22` already timed via subinterval CTC) was skipped (`skipped_low_subtitle_value`). Under opt-in `fine`, both spans were admitted in one worker session and `utt-0002` reached `22/22` (`full`).
3. **Strict Issue #197 validation:** In `zh_multichunk_qwen_fine` span #2, 1 item had zero duration (`zero_duration_count=1`) and was rejected without clamping; `'脚'` and `'劲'` received valid non-zero timings.
4. **Raw JSON telemetry:** Full benchmark metrics are stored in [`logs/gpu_qualification_results.json`](logs/gpu_qualification_results.json).
