# Nemovox Issue #1 Local Integration & Qualification Handback

- **Overall Disposition:** `PASS` for CPU regressions, `qwen-asr` worker compatibility, and real RTX 5090 CUDA BF16 selective alignment + multi-span worker reuse (`--qwen-gpu-mode available`); **`MANAGED_GPU_BLOCKED`** for live `--qwen-gpu-mode managed` ASR takeover (no cooperatively owned `nemovox-*.service` ASR unit or active ASR instance on host; lifecycle/restore/SIGTERM/journal-recovery gates verified via CPU state-machine regressions).
- **Repository:** `fvanevski/nemovox` (Issue #1)
- **Base `main` SHA:** `242c418f55cd2afb796ee2c4fa742a10ff04f7a3`
- **Initial Central `issue/1` SHA:** `570f2de312cee1444defa671073e83e8dab4dd6b`
- **Final `issue/1` Implementation SHA:** `28655e0d6fe57d9363e576cd0795a24aae77a4a3`
- **Evidence Branch:** `evidence/issue-1-qwen-20261121` (branched directly from `28655e0d6fe57d9363e576cd0795a24aae77a4a3`)
- **Evidence Directory:** `staging/agent-evidence/issue-1-qwen/`

## Narrow Local Integration Fix (`570f2de..28655e0`)

Testing Central's `570f2de312cee1444defa671073e83e8dab4dd6b` against the host's canonical Mandarin OOV fixtures (`zh_fleurs_01_2006` and `zh_multichunk_synthetic`) revealed two narrow admission gaps in `src/nvidia_transcribe/selective_alignment.py` (`plan_selective_recovery`):

1. **Intra-utterance Sortformer diarization pauses (`<= 0.6s`):** Sortformer emitted 3 diarization segments (`[1.12, 3.32]`, `[3.75, 5.08]`, `[5.47, 6.63]`) with `0.43s` and `0.39s` pauses inside the single `zh_fleurs_01_2006` sentence. Because `plan_selective_recovery` required `len(matching_diar) == 1` on raw unmerged intervals rather than reusing `alignment.group_diarization_windows(..., max_gap_s=0.6, pad_s=0.0)`, `zh_fleurs_01_2006` was skipped as `skipped_unanchored_or_ambiguous`. Fixed by grouping `matching_diar` with `max_gap_s=0.6, pad_s=0.0` (only when `untimed_spans_by_speaker[speaker] == 1`), preserving rejection of `> 0.6s` multi-window ambiguity (`[0.2, 1.2]` vs `[2.0, 3.0]`).
2. **Cross-speaker neighbor boundary with unique grouped diarization:** In `zh_multichunk_synthetic` chunk 0, the untimed `'钾'` utterance (`spk_0`) sits between a timed `spk_0` utterance (`..7.281s`) and a timed `spk_1` utterance (`22.61s..`). Because `right` only accepted same-speaker timed units, `right` was `None` and the `'钾'` utterance was skipped. Fixed by allowing `left`/`right` fallback to the nearest timed unit across speakers *only* when grouped `matching_diar` strictly inside `[left[1]-0.25, right[0]+0.25]` forms a single speech window (`[11.68, 17.84]`).

Added 5 focused CPU regressions in `tests/test_qwen_session_and_policy.py` (default-off zero worker/service action, intra-utterance pause merging & cross-speaker neighbor localization, protocol ID mismatch worker reap, real Linux `PR_SET_PDEATHSIG` parent-death SIGKILL cleanup, and `_verified_owner`/Studio exclusion checks).

## Criterion-to-Evidence Matrix

| Acceptance Criterion | Status | Evidence |
| --- | --- | --- |
| Default `alignment_fallback=off` never loads worker or touches GPU services | **PASS** | `test_default_off_never_loads_worker_or_touches_managed_services` + `gpu-benchmark.md` (`qwen_recovery.status == "disabled"`) |
| Selective admission preserves native CTC/source/speakers, skips overlap/low-value gaps | **PASS** | `test_qwen_fallback.py` + `test_qwen_session_and_policy.py` + `zh_multichunk` (`subtitle` skips 2-char gap vs `fine` admits both) |
| One isolated GPU-loaded worker/session across multiple spans; CUDA BF16 verified, no silent CPU downgrade | **PASS** | `two_span_subtitle_qwen` & `zh_multichunk_qwen_fine`: `model_load_count=1`, `align_count=2`, `cuda:0`, `observed_dtype=torch.bfloat16`, warm call `0.0329s` vs cold `1.3054s` |
| Explicit managed-GPU mode safely frees/restores owned ASR unit; blocks non-owned/busy/pending-journal | **BLOCKED (live) / PASS (mocked lifecycle)** | Live host is `MANAGED_GPU_BLOCKED` (no ASR unit on 8190, no `NEMOVOX_MANAGED_ASR=1` or lease); CPU state-machine tests verify busy/unmarked/changed-unit blocks, failure & SIGTERM unwind restore |
| End-to-end overhead & subtitle coverage vs `off` baseline | **PASS** | `gpu-benchmark.md`: `zh_fleurs_01_2006` `0/21 -> 21/21` (`failed -> full`), `zh_multichunk` `156/179 -> 177/179` (`subtitle`) / `179/179` (`fine`), `two_span` `0/42 -> 42/42` |
| Focused + full CPU suites and host GPU checks pass; evidence SHA-bound | **PASS** | `107 passed` in `.venv` (`8.46s`) & `107 passed` in `.venv-studio` (`8.35s`) on `28655e0d6fe57d9363e576cd0795a24aae77a4a3`; GPU VRAM `3050 MiB` before & after |

## Frontend Equivalence Note

No frontend files under `web/` were modified between `main@242c418f55cd2afb796ee2c4fa742a10ff04f7a3` and `issue/1@28655e0d6fe57d9363e576cd0795a24aae77a4a3` (verified via `changed_files.txt`). `nemovox-studio.service` remained untouched and `active` throughout.

## Artifacts in This Directory

- [`test-results.md`](test-results.md) — Python compilation and CPU regression suites across `.venv` and `.venv-studio`
- [`gpu-benchmark.md`](gpu-benchmark.md) — Bounded RTX 5090 CUDA BF16 `off` vs `qwen` benchmarks and VRAM telemetry
- [`ownership-and-restore.md`](ownership-and-restore.md) — Read-only host service audit (`MANAGED_GPU_BLOCKED`) and mocked lifecycle/parent-death verification
- [`changed_files.txt`](changed_files.txt), [`diff_vs_base_main.patch`](diff_vs_base_main.patch), [`diff_local_fix.patch`](diff_local_fix.patch)
- [`SHA256SUMS.txt`](SHA256SUMS.txt), [`manifest.json`](manifest.json), and [`nemovox-issue-1-qwen-evidence.tar.gz`](nemovox-issue-1-qwen-evidence.tar.gz)

## `COMMAND_SHAPE_FRICTION`

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events=[]
  omitted_sensitive_values=no
```
