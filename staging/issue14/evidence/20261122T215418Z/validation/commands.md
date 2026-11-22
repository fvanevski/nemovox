# Phase B & Phase C Executed Commands (`866aa734` / `05ac52db`)

## Source Edits
- `SOURCE_EDITS=none` (zero source/test modifications; test failure reported unmodified per handoff boundary).

## Phase B: Deterministic CPU & Studio Tests (`PYTHONDONTWRITEBYTECODE=1`)
1. **Compileall:**
   - `PYTHONPATH=<ISOLATED_WORKTREE>/src <PROD_VENV>/bin/python -m compileall -q src tests scripts`
   - `exit=0`
2. **Focused Regressions (`.venv`):**
   - `PYTHONPATH=<ISOLATED_WORKTREE>/src <PROD_VENV>/bin/python -m pytest -q tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_utterances_alignment_speakers.py tests/test_recording_and_normalization.py`
   - `exit=1`: `1 failed, 31 passed, 1 warning in 0.68s`
   - Failure: `tests/test_issue14_timing_boundaries.py::test_long_displaced_duplicate_from_adjacent_excerpts_is_not_rendered_twice` (line 64 asserts `timing_evidence_status == "displaced_sequence_ctc_unverified"`, whereas `src/nvidia_transcribe/boundary_evidence.py:245` was updated to emit `"cross_chunk_ctc_disagreement"`).
3. **Full Non-GPU Suite (`.venv`):**
   - `PYTHONPATH=<ISOLATED_WORKTREE>/src <PROD_VENV>/bin/python -m pytest -q -m 'not gpu'`
   - `exit=1`: `1 failed, 273 passed, 1 warning in 9.69s` (`0` skipped; same single failure at `test_issue14_timing_boundaries.py:64`; `1` `anyio` `PytestRemovedIn10Warning`).
4. **Studio Suites (`.venv-studio`):**
   - `PYTHONPATH=<ISOLATED_WORKTREE>/src <PROD_VENV_STUDIO>/bin/python -m pytest -q tests/test_studio_exports.py tests/test_studio_export_api.py tests/test_studio_api.py`
   - `exit=0`: `49 passed in 3.67s` (`0` skipped).

## Phase C: 16-Chunk Offline Episode Replay
- `PYTHONPATH=<ISOLATED_WORKTREE>/src PYTHONDONTWRITEBYTECODE=1 <PROD_VENV>/bin/python <PRIVATE_SCRATCH>/replay_ab.py`
- Replay flow: 16 raw chunks -> `recover_chunk_alignment(model_path=None, subinterval_runner=None)` -> `audit_ctc_timing_units` -> `normalize_chunk_speakers` + archived `recording.contract.json` `local_to_global` -> `reconcile_chunk_boundaries` -> `reconstruct_speaker_utterances`.
