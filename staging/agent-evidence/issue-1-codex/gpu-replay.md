# Bounded RTX 5090 CUDA BF16 Recorded ASR Replay (`1702608f82808fca30fb3eee1b62954230e6da32`)

**Provenance notice:** All runs below are **recorded ASR replays** replaying saved Nemotron vLLM responses (since no live `8190` ASR server or owned `nemovox-*.service` ASR unit is running on host) through the real Phase 1 -> `native-alignment.checkpoint.json` -> `AdmissionPolicy` -> persistent CUDA BF16 `QwenForcedAlignerAdapter` (`_qwen_align_worker.py` in `outputs/modular-eval/.venv`) -> Phase 2 reconstruction pipeline under `--qwen-gpu-mode available`.

Raw telemetry: [`logs/gpu_qualification_results.json`](logs/gpu_qualification_results.json) and [`logs/host_read_only_inspection.txt`](logs/host_read_only_inspection.txt).

## GPU & Host Service Invariants

- **GPU:** NVIDIA GeForce RTX 5090 (32,607 MiB total VRAM)
- **VRAM before replay:** `3051 MiB` used / `29076 MiB` free (`kwin_wayland` 328 MiB, `text-embeddings-router` 1728 MiB)
- **VRAM after replay:** `3052 MiB` used / `29076 MiB` free (`kwin_wayland` 328 MiB, `text-embeddings-router` 1728 MiB) — zero worker leaks
- **Unrelated host services:** `nemovox-studio.service` and `local_embedding.service` remained `active` and untouched throughout; port `8190` remained closed.

## Replay Results (`1702608f` vs. Prior Qualified `d5700b4`)

| Fixture | Mode / Policy | Candidate Windows (`[lo, hi] -> [start, end]`) | Wall (s) | Utterance Timing Status | Timed / Total Units | Model Loads / Aligns | Device / Observed Dtype | Peak CUDA Alloc | Invalid / Zero-Dur | `'钾'` Timings (s) |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| `zh_multichunk_synthetic` (`69.78s`, recorded 3-chunk replay) | `off` | — | 0.4316 | `full: 6, partial: 1, failed: 1` | 176 / 198 | 0 / 0 | — | — | 0 | untimed |
| `zh_multichunk_synthetic` (`69.78s`, recorded 3-chunk replay) | `qwen` (`subtitle`) | `[23, 44] -> [11.68, 17.84]` (`[44, 66]` skipped: `skipped_low_subtitle_value`) | 8.8599 | `full: 7, partial: 1` | **197 / 198** | **1 / 1** | `cuda:0` / `torch.bfloat16` | 1,900,167,680 B | 0 | `[12.24, 12.48]` (`1.3425s`) |
| `zh_multichunk_synthetic` (`69.78s`, recorded 3-chunk replay) | `qwen` (`fine`) | `[23, 44] -> [11.68, 17.84]` (`[44, 66]` skipped: `skipped_partial_span_unanchored_edge`) | 8.6083 | `full: 7, partial: 1` | **197 / 198** | **1 / 1** | `cuda:0` / `torch.bfloat16` | 1,900,167,680 B | 0 | `[12.24, 12.48]` (`1.3444s`) |
| `two_span_oov` (`18.56s`, recorded 2-chunk replay) | `off` | — | 0.1543 | `failed: 2` | 0 / 42 | 0 / 0 | — | — | 0 | untimed |
| `two_span_oov` (`18.56s`, recorded 2-chunk replay) | `qwen` (`subtitle`) | `[0, 21] -> [0.77, 6.98]`, `[0, 21] -> [10.77, 16.98]` | 8.2399 | `full: 2` | **42 / 42** | **1 / 2** | `cuda:0` / `torch.bfloat16` | 1,900,171,776 B | 0 | `[1.41, 1.65]` & `[11.65, 11.89]` (`1.3548s` / `0.0346s`) |

## Delta Notes vs. `d5700b47a4e2f1abb60ebe99f53750c254de2696`

1. **Multi-span single-worker CUDA BF16 reuse confirmed:** `two_span_oov` executed 2 spans inside one worker session (`model_load_count=1, align_count=2`, `cuda:0`, `torch.bfloat16`), with warm span #2 completing in **`0.0346s`** vs span #1 **`1.3548s`** (39.2x faster), recovering `42/42` units and both `'钾'` timings (`[1.41, 1.65]` and `[11.65, 11.89]`) with zero invalid/zero-duration items.
2. **`zh_multichunk_synthetic` `subtitle` unchanged, `fine` conservatively skips multi-diar partial edge:**
   - Under `subtitle`, `utt-0001` (`钙、钾等元素...`) is admitted at `[11.68, 17.84]` and fully recovered (`21/21`, `'钾'` `[12.24, 12.48]`), while `[44, 66]` (`1/22` untimed) is skipped as `skipped_low_subtitle_value`.
   - Under `fine`, at `d5700b4` `[44, 66]` (`utt-0002`, untimed prefix `'脚'`) was admitted via the old `native_start - 1.2` pad (`[21.41, 28.035]`). At `1702608f`, `[44, 66]` is skipped as **`skipped_partial_span_unanchored_edge`** because speaker `1` has two separated diarization intervals in chunk 0 (`[22.41, 27.06]` and `[28.06, 30.98]`, gap `1.0s > 0.6s`, `len(grouped_diar) == 2 != 1`). This reflects Central's conservative fail-closed rule (`len(grouped_diar) != 1`) in `selective_alignment.py` rather than cropping untimed edges.
