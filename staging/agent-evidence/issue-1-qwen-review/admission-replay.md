# Bounded RTX 5090 CUDA BF16 Admission & Replay Comparison (`d5700b47a4e2f1abb60ebe99f53750c254de2696`)

## GPU & Host Service State

- **GPU:** NVIDIA GeForce RTX 5090 (32,607 MiB total)
- **VRAM before replay:** `3052 MiB` used / `29075 MiB` free (`kwin_wayland` 329 MiB, `text-embeddings-router` 1728 MiB)
- **VRAM after replay:** `3051 MiB` used / `29076 MiB` free (`kwin_wayland` 328 MiB, `text-embeddings-router` 1728 MiB) — zero worker leaks
- **Operating host services:** `nemovox-studio.service` and `local_embedding.service` remained `active` and undisturbed.

## Replay Results (`d5700b4` vs. Prior Qualified `28655e0`)

All `qwen` runs executed the real CUDA BF16 `Qwen3ForcedAligner` worker (`_qwen_align_worker.py` in `outputs/modular-eval/.venv`) under `--qwen-gpu-mode available` replaying recorded Nemotron ASR responses.

| Fixture | Mode / Policy | Candidate Windows (`[lo, hi] -> [start, end]`) | Wall (s) | Utterance Timing Status | Timed / Total Units | Model Loads / Aligns | Device / Observed Dtype | Peak CUDA Alloc | Invalid / Zero-Dur | `'钾'` Timings (s) |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| `zh_fleurs_01_2006` (`8.28s`) | `off` | — | 0.1064 | `failed: 1` | 0 / 21 | 0 / 0 | — | — | 0 | untimed |
| `zh_fleurs_01_2006` (`8.28s`) | `qwen` (`subtitle`) | `[0, 21] -> [0.77, 6.98]` | 9.9354 | `full: 1` | **21 / 21** | **1 / 1** | `cuda:0` / `torch.bfloat16` | 1,900,171,264 B | 0 | `[1.41, 1.65]` |
| `zh_multichunk_synthetic` (`69.78s`) | `off` | — | 0.4212 | `full: 5, partial: 1, failed: 1` | 156 / 179 | 0 / 0 | — | — | 0 | untimed |
| `zh_multichunk_synthetic` (`69.78s`) | `qwen` (`subtitle`) | `[23, 44] -> [11.68, 17.84]` (`[44, 66]` skipped) | 8.4218 | `full: 6, partial: 1` | **177 / 179** | **1 / 1** | `cuda:0` / `torch.bfloat16` | 1,900,167,680 B | 0 | `[12.24, 12.48]` |
| `zh_multichunk_synthetic` (`69.78s`) | `qwen` (`fine`) | `[23, 44] -> [11.68, 17.84]`, `[44, 66] -> [21.41, 28.035]` | 9.2949 | `full: 7` | **179 / 179** | **1 / 2** | `cuda:0` / `torch.bfloat16` | 1,900,198,912 B | 1 zero-dur rejected | `[12.24, 12.48]` |
| `two_span_oov` (`18.56s`) | `off` | — | 0.1478 | `failed: 2` | 0 / 42 | 0 / 0 | — | — | 0 | untimed |
| `two_span_oov` (`18.56s`) | `qwen` (`subtitle`) | `[0, 21] -> [0.77, 6.98]`, `[0, 21] -> [10.77, 16.98]` | 8.3157 | `full: 2` | **42 / 42** | **1 / 2** | `cuda:0` / `torch.bfloat16` | 1,900,171,776 B | 0 | `[1.41, 1.65]` & `[11.65, 11.89]` (`1.3313s` / `0.0366s`) |

## Delta Comparison Against `28655e0d6fe57d9363e576cd0795a24aae77a4a3`

- **Neighbor-clipping effect observed live:** In `zh_multichunk_qwen_fine`, span `[44, 66]` (`脚下垂直劲空十五米...`, ending `27.041s`, followed by `utt-0003` starting `27.985s`) previously had window `[21.41, 28.241]`; under `d5700b47a4e2f1abb60ebe99f53750c254de2696`, `following_timed[0] + 0.05` cleanly tightened it to **`[21.41, 28.035]`** while preserving `2/2` untimed character recovery and `full: 7` coverage.
- **Unchanged OOV recovery:** `zh_fleurs_01_2006`, `zh_multichunk_qwen_subtitle`, and `two_span_subtitle_qwen` retained identical candidate windows, 100% `'钾'` recovery, `model_load_count=1` worker reuse (`1.3313s` -> `0.0366s` warm span on `two_span_oov`), and `0` leaked VRAM.
- Raw telemetry: [`logs/gpu_qualification_results.json`](logs/gpu_qualification_results.json).
