# Local Qualification Handback — Recovery Scheduling & Evidence Completion (#17 / #18)

`RESULT=FAIL` (literal `scripts/replay_recovery_schedule.py:93` failed with `ValueError` due to a `tuple != list` comparison bug; all CPU/Studio tests, unmodified-core 5-case replay, targeted CPU Whisper crop pair, and 9/9 primary raw-response authentication passed).

## 1. Identities

- **Repository / Branch**: `fvanevski/nemovox` / `issue/17`
- **Execution Class**: `CPU_ONLY` (zero GPU inference, zero Nemotron/Qwen service startup)
- **Tested Source SHA**: `ebdc06f0bd2997c06d73e5e32158e07beb82e3a6` (tree `2865d1773ca385d6ea5a7695cc50a2c9d28fc51a`)
- **Prior Qualified Source SHA**: `029f969b43c58d53fd27975ff122c097ead04167`
- **Prior Evidence Carrier SHA**: `25fd9d15c24bd5623a3ba8cf93d132b713fa4bd4`
- **Source Edits**: None (candidate worktree clean)
- **Private Supplementary Archive**: `/workspace/nvidia-transcribe/outputs/handbacks/recovery-scheduling-20261124-ebdc06f/recovery-scheduling-supplement-ebdc06f.tar.gz` (`51812` bytes, SHA-256 `a512f262035860a9f3aaa9ac78b523b517c1cc673573af05eacc9eee2b4ffc4d`)

## 2. Native Checks & Exits

| Check | Exit | Result |
|---|---|---|
| Core & Studio import binding (`PYTHONPATH=src`) | `0` | Both `.venv` and `.venv-studio` resolved `nvidia_transcribe` to candidate worktree |
| `compileall` (`src/nvidia_transcribe`, `scripts/replay_recovery_schedule.py`, `scripts/evaluate_occurrence_recovery.py`) | `0` | Clean |
| Focused regressions (5 modules, `-m "not gpu"`) | `0` | `69 passed` |
| Full CPU test suite (`-m "not gpu"`) | `0` | `476 passed` |
| Studio test suite (`tests/test_studio*.py`, `-m "not gpu"`) | `0` | `50 passed` |
| `git diff --check` | `0` | Clean |
| `.venv-challenger` vs `requirements-challenger.txt` | `0` | Reused without reinstall: `faster-whisper==1.2.1`, `ctranslate2==4.9.1`, `av==18.0.0`, `uv pip check` clean |
| Literal `python -m scripts.replay_recovery_schedule` | `1` | **FAIL** at line 94 (`ValueError: Explicit previously deferred target not admitted`) due to `tuple != list` comparison on line 93 |
| Unmodified-core 5-case replay + targeted CPU Whisper crop | `0` | Completed via external scratch driver calling unmodified `nvidia_transcribe` functions with `list(bounds)` comparison |
| 9 omitted primary raw responses | `0` | `9/9` matched manifest `raw_response_sha256` exactly (zero rejected generations) |

## 3. Literal Runner Defect Reproducer (`scripts/replay_recovery_schedule.py:93`)

- **Failing expression**: `now[0]["bounds"] != prior[0]["bounds"]` at `scripts/replay_recovery_schedule.py:93`.
- **Observed types & values**:
  - `prior[0]["bounds"]` (loaded via `json.loads` from archived `source-quality.json`): `list` `[91.535, 93.57]`
  - `now[0]["bounds"]` (returned in-memory by `schedule_independent`): `tuple` `(91.535, 93.57)`
  - `now[0]["bounds"] == prior[0]["bounds"]` -> `False`; `list(now[0]["bounds"]) == list(prior[0]["bounds"])` -> `True`.
- Candidate source was left untouched per `SEMANTIC_REMEDIATION=forbidden` and `SOURCE_REMOTE_UPDATE=forbidden`.

## 4. Unmodified-Core 5-Case Replay & Targeted CPU Whisper Results

- **All 5 cases**: `all_evidence_ids_preserved=True`, zero duplicate task IDs, original recording unmutated.
- **`boundary` independent re-admission (on frozen prior observed tasks)**:
  - Previously deferred untimed target `utt_789bf4bc4d62fa748554d081` (`[91.535, 93.57]`) IS admitted globally (`prior_independent_kinds=["occurrence", "occurrence"]` -> `candidate_independent_kinds=["untimed", "occurrence"]`, `2` explicit deferrals retained).
  - `candidate_resolution_statuses`: `unresolved: 7`, `event_supported_word_timing_unresolved: 1` (previously `unresolved: 8`).
- **`boundary` counterfactual primary plan (NOT executed acoustically)**:
  - Admits focused `3.535s` bracket `[90.785, 94.32]` (`recovery_959043e4997f5a1ca58125de`, reasons `["untimed_occurrence", "untimed_source_dialogue"]`); admitted target seconds reduced `36.0155s -> 18.173s` with `8` explicit deferrals retained.
- **Other 4 cases**: `compression` counterfactual primary plan admits `5` evidence IDs (`23.952s`, `9` deferrals) vs prior `3` (`13.53s`, `8` deferrals); `late_gap`, `low_flag_control`, and `multilingual_control` unchanged.
- **Targeted CPU Whisper crop pair** (exact authenticated prior `boundary` WAV SHA-256 `d53c26af1f8dc1e0cfeb3b19735b69973cf9a5512bce0664e8445fe27139f066`):
  - `status=observed`, `2` requests, `7.512s` audio, `22.1143s` wall, `receipt_sha256=61d35650760112a8afdcf09630a917102587743084e55eea76a1dbe19199b6df`.
  - Produced `4` worker segment rows, `0` cross-model events, `decisions={}`, `target_still_untimed=True`, `canonical_rows_before=16 -> canonical_rows_after=16`.
  - Unresolved acoustic outcome preserved honestly without threshold tuning.

## 5. Omitted Primary Raw-Response Completion (`9/9` Authenticated)

All 9 primary chunk raw responses were located under the retained live run `recording/chunks/chunk_00*.raw.json` paths recorded in the archived manifests and verified byte-for-byte against `raw_response_sha256` (`9/9` matched, `0` rejected generations). Full path-independent mapping is in `summaries/omitted_primary_raw_responses.json`; exact raw JSON bytes are included in the private supplementary archive only.

## 6. CPU-Only Resource Proof & Stop Conditions

- GPU remained untouched (`1662 MiB` used before & after); `nemovox-asr.service` and `nemovox-text-review.service` remained `inactive/dead`; `nemovox-studio.service` unchanged (`PID 4071576`).
- No PR created, no merge, no issue closure, no deployment, no `#19` work, no source edits.
