# Local Qualification Handback — PR #20 Final CPU Regression Qualification (#17 / #18)

`RESULT=PASS` (executed CPU-only regression qualification and frozen sealed-archive replays; Central owns final exact-head adjudication, Codex sequencing, and merge readiness).

## 1. Identities

- **Repository / Branch / PR**: `fvanevski/nemovox` / `issue/17` / PR `#20`
- **Execution Class**: `CPU_ONLY` (`MODEL_INFERENCE=forbidden`, verified zero GPU/service activity)
- **Tested Source SHA**: `1d0dc2a3c95d7657967088087effb5267dd05fcf`
- **Tested Source Tree**: `50bb17a93d36b3b01019856f257c891824d036f5`
- **Pre-Remediation Reviewed Head**: `ee0caf027d9df7394407b625ee58608e352db708`
- **Base SHA**: `9dd863841f72ab5f9aebf5bb4deb322bee99edf1`
- **Source Edits**: None (isolated detached worktree clean)
- **Private Review Archive**: `/workspace/nvidia-transcribe/outputs/handbacks/pr20-regression-20261124-1d0dc2a/pr20-regression-review-1d0dc2a.tar.gz` (`114836` bytes, SHA-256 `16da4b7d97636610c6dc7b2681134982df32f6de963abe697d481a4621e6b4a6`)

## 2. Native Checks & Exits

| Check | Exit | Result |
|---|---|---|
| Core & Studio import binding (`PYTHONPATH=src`) | `0` | Both `.venv` and `.venv-studio` resolved `nvidia_transcribe` to candidate worktree |
| `compileall` | `0` | Clean |
| `git diff --check` | `0` | Clean |
| Focused affected-path suite (all 11 named test files, `-m "not gpu"`) | `0` | `157 passed` (`0` missing files) |
| Full native CPU suite (`-m "not gpu"`) | `0` | `489 passed` |
| Full native Studio suite (`tests/test_studio*.py`, `-m "not gpu"`) | `0` | `50 passed` |
| Literal scheduling replay (`scripts.replay_recovery_schedule`, no `--boundary-wav`) | `0` | `status=completed`, `new_worker=not_requested`, all 5 cases preserved |
| Literal frozen acoustic-resolution replay (`scripts.replay_acoustic_resolution`) | `0` | `status=completed`, `new_model_inference=false`, all 6 cases verified |

## 3. Sealed Replay Findings

- **Scheduling replay** (`occurrence-recovery-review-029f969.tar.gz`, SHA-256 `4d46e32e...6755`):
  - `all_evidence_ids_preserved=True` across all 5 regions; zero duplicate task IDs.
  - `boundary`: previously deferred untimed target admitted globally at `[91.535, 93.57]`; counterfactual primary plan admits focused `3.535s` bracket; `candidate_resolution_statuses={"unresolved": 7, "event_supported_word_timing_unresolved": 1}`.
- **Frozen acoustic-resolution replay** (both sealed archives, `new_model_inference=false`):
  - `missing_original_row_ids=[]` and `new_row_ids=[]` across all 6 cases (`compression`, `boundary`, `targeted_boundary`, `late_gap`, `low_flag_control`, `multilingual_control`).
  - `legacy_subinterval_attempted=false` across all 5 primary regions.
  - `boundary`: `16 -> 16` rows (`untimed 1 -> 1`), `decisions={"unresolved_competing_source": 1, "already_represented": 1}`, `receipts={"unresolved": 7, "event_supported_word_timing_unresolved": 1}`.
  - `targeted_boundary`: `16 -> 16` rows (`untimed 1 -> 1`), `decisions={}`, `receipts={"unresolved": 8}`, `frozen_independent_requests=2`.
  - `compression` (`30 -> 30`, `4` unresolved), `late_gap` (`14 -> 14`, `1` unresolved), `low_flag_control` (`14 -> 14`, `1` unresolved), `multilingual_control` (`2 -> 2`, `0` unresolved).

## 4. Resource Proof, Friction, and Stop Conditions

- **CPU-only proof**: GPU memory identical before & after (`1662 MiB` used / `30466 MiB` free); `nemovox-asr.service` and `nemovox-text-review.service` remained `inactive/dead`; `nemovox-studio.service` unchanged (`PID 4071576`).
- **`COMMAND_SHAPE_FRICTION`**: None (`events: []`).
- **Stop conditions**: No Codex review requested, no PR state mutation, no merge, no issue closure, no deployment, no `#19` work, no source edits.
