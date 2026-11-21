# Nemovox Issue #1 Final Admission-Neighbor Qualification Handback

- **Overall Disposition:** **`PASS`** for CPU compilation, both new adjacency regressions, full CPU suites (`109 passed` in both `.venv` and `.venv-studio`), synthetic adversarial adjacency checks, and bounded RTX 5090 CUDA BF16 replay qualification (`zh_fleurs_01_2006`, `zh_multichunk_synthetic`, `two_span_oov`); **`MANAGED_GPU_BLOCKED`** continuation for live `--qwen-gpu-mode managed` ASR takeover (no cooperatively owned `nemovox-*.service` ASR unit/lease or active `8190` ASR server on host; worker & managed controller sources remain blob-identical to `28655e0d6fe57d9363e576cd0795a24aae77a4a3`).
- **Repository:** `fvanevski/nemovox` (Issue #1)
- **Base `main` SHA:** `242c418f55cd2afb796ee2c4fa742a10ff04f7a3`
- **Previous Qualified `issue/1` SHA:** `28655e0d6fe57d9363e576cd0795a24aae77a4a3`
- **Central Bounds Fix Ancestor SHA:** `0bd775c56d720484e75c1e933628d3b58731a4ef`
- **Final `issue/1` Implementation SHA:** `d5700b47a4e2f1abb60ebe99f53750c254de2696`
- **Evidence Branch:** `evidence/issue-1-qwen-review-20261121` (branched directly from `d5700b47a4e2f1abb60ebe99f53750c254de2696`)
- **Evidence Directory:** `staging/agent-evidence/issue-1-qwen-review/`

## Delta Verification Summary (`28655e0..d5700b4`)

1. **Changed files since `28655e0`:** Only `docs/alignment/ISSUE1_ADMISSION_NEIGHBOR_REVERIFY.md`, `src/nvidia_transcribe/selective_alignment.py` (`+20/-1`), and `tests/test_qwen_session_and_policy.py` (`+44`). Verified blob-identical across `_qwen_align_worker.py`, `qwen_aligner.py`, `gpu_handoff.py`, `recording.py`, `cli.py`, and `subtitle_pipeline.py` (`git diff --exit-code` passed).
2. **Focused regressions (`32 passed` in 4.53s):** Both new tests pass verbatim:
   - `test_unique_diar_window_respects_adjacent_native_timed_speech`: `PASSED`
   - `test_neighbor_bounds_fail_closed_if_candidate_overlaps_native_timing`: `PASSED`
3. **Full CPU suites (`-m "not gpu"`):**
   - `.venv` (Python 3.13): **`109 passed` in 8.49s** (exit 0)
   - `.venv-studio`: **`109 passed` in 8.41s** (exit 0)
4. **Adversarial adjacency & provenance checks:** Verified in [`logs/adjacency_checks.json`](logs/adjacency_checks.json):
   - Same-speaker single diarization `[0.2, 8.0]` covering three sentences clips middle untimed `"世界啊。"` to `[1.65, 6.55]` (never including `0.5..1.7` or `6.5..7.2` aside from the documented `0.05s` boundary tolerance).
   - Cross-speaker untimed `"你好吗？"` (`spk_1`) between timed `spk_0` (`0.8..2.4`) and `spk_2` (`5.8..7.0`) clips cleanly to `[2.35, 5.85]`; original text, speaker IDs, and outer native timings remain untouched.
   - Conflicting neighbor timing (`1.0..3.8` and `3.9..5.0`) fails closed with `skipped_conflicting_or_long_window` and zero candidates.
5. **Bounded RTX 5090 CUDA BF16 replay (`available` mode):**
   - `zh_fleurs_01_2006`: `off` (`0/21`, `failed: 1`) vs `qwen subtitle` (`21/21`, `full: 1`, window `[0.77, 6.98]`, `'钾'` `[1.41, 1.65]`, `model_load_count=1, align_count=1`, `cuda:0` `torch.bfloat16`, `0` invalid/zero-duration).
   - `zh_multichunk_synthetic`: `off` (`156/179`) vs `qwen subtitle` (`177/179`, window `[11.68, 17.84]`, `'钾'` `[12.24, 12.48]`, 2-char gap skipped as `skipped_low_subtitle_value`) vs `qwen fine` (`179/179`, `full: 7`, `model_load_count=1, align_count=2`, span #2 clipped tightly to **`[21.41, 28.035]`** instead of `[21.41, 28.241]`, warm span #2 `0.0319s` vs span #1 `1.346s`).
   - `two_span_oov`: `off` (`0/42`, `failed: 2`) vs `qwen subtitle` (`42/42`, `full: 2`, windows `[0.77, 6.98]` & `[10.77, 16.98]`, `model_load_count=1, align_count=2`, span #1 `1.3313s` vs warm span #2 `0.0366s`).
   - Pre/post VRAM: `3052 MiB -> 3051 MiB` used (`29075 MiB -> 29076 MiB` free), zero worker leaks, `nemovox-studio.service` and `local_embedding.service` active and untouched.
6. **Managed-ASR continuation:** `MANAGED_GPU_BLOCKED` (unchanged host state: no `nemovox-*.service` ASR unit, no cooperative lease, port `8190` closed; prior mocked lifecycle & Linux `PR_SET_PDEATHSIG` evidence remains valid via blob-identical worker/controller sources).

## `COMMAND_SHAPE_FRICTION`

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events=[]
  omitted_sensitive_values=no
```
