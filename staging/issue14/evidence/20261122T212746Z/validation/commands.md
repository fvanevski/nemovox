# Executable & Replay Verification Commands (`50b1408f` / `6986cf5a`)

## Source Edits
- `SOURCE_EDITS=none` (zero source/test modifications; `src/` and `tests/` at `6986cf5a` are byte-identical to `50b1408f`).

## Phase B: CPU & Studio Test Suites (Isolated Worktree, `PYTHONDONTWRITEBYTECODE=1`)
1. **Compileall:**
   - Command: `PYTHONPATH=<ISOLATED_WORKTREE>/src <PROD_VENV>/bin/python -m compileall -q src tests scripts`
   - Result: `exit=0`
2. **Focused Issue #14 & Boundary Suites (`.venv`):**
   - Command: `PYTHONPATH=<ISOLATED_WORKTREE>/src <PROD_VENV>/bin/python -m pytest -q tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_utterances_alignment_speakers.py tests/test_recording_and_normalization.py`
   - Result: `exit=0`, `27 passed, 1 warning in 0.68s` (`0` skipped; `1` `anyio` `PytestRemovedIn10Warning`)
3. **Full Non-GPU Suite (`.venv`):**
   - Command: `PYTHONPATH=<ISOLATED_WORKTREE>/src <PROD_VENV>/bin/python -m pytest -q -m 'not gpu'`
   - Result: `exit=0`, `269 passed, 1 warning in 10.21s` (`0` skipped; `1` `anyio` `PytestRemovedIn10Warning`)
4. **Studio API & Export Suites (`.venv-studio`):**
   - Command: `PYTHONPATH=<ISOLATED_WORKTREE>/src <PROD_VENV_STUDIO>/bin/python -m pytest -q tests/test_studio_exports.py tests/test_studio_export_api.py tests/test_studio_api.py`
   - Result: `exit=0`, `49 passed in 3.76s` (`0` skipped)

## Phase C: Offline Cached-Episode Replay
- Command: `PYTHONPATH=<ISOLATED_WORKTREE>/src PYTHONDONTWRITEBYTECODE=1 <PROD_VENV>/bin/python <PRIVATE_SCRATCH>/replay_issue14.py`
- Replay flow: 16 archived chunks -> `recover_chunk_alignment(model_path=None, subinterval_runner=None)` -> `audit_ctc_timing_units` -> `normalize_chunk_speakers` -> archived `recording.contract.json` `local_to_global` map -> `reconcile_chunk_boundaries` -> `reconstruct_speaker_utterances(source_language="fr", chunk_windows=...)`.
- Outputs & per-token raw dumps retained privately in `<PRIVATE_SCRATCH>` (`chmod 700`).
