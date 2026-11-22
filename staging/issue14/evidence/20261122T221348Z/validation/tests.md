# Exact-Head Deterministic Verification (`66275bdb`)

## Import Isolation & Source Invariance
- **Isolated detached worktree:** `<ISOLATED_WORKTREE>` at `b3aa3846682b3f1636fea500fb5a001c63d0ccd7`.
- **Import origin verified:** `<ISOLATED_WORKTREE>/src/nvidia_transcribe/boundary_evidence.py`.
- **`git diff 866aa734 66275bdb -- src/`:** Empty (`0` files changed under `src/`). Prior 16-chunk offline replay (`99dce3f8bfa1a93670b28473209dbf8bd924a54e:staging/issue14/evidence/20261122T215418Z/`) is reused verbatim without redundant replay.
- **`git diff 866aa734 66275bdb -- tests/`:** Single Central-authored update at `tests/test_issue14_timing_boundaries.py:64` (`timing_evidence_status == "cross_chunk_ctc_disagreement"` and `timing_evidence_reason == "relative_sequence_shift_unverified"`).

## Test Commands & Outcomes (Production `.venv` & `.venv-studio`, Read-Only Interpreters)

| Gate | Command (`PYTHONPATH=<ISOLATED_WORKTREE>/src`) | Outcome | Exit | Warnings / Skips |
|---|---|---|---|---|
| `compileall` | `.venv/bin/python -m compileall -q src tests scripts` | **PASS** | `0` | `0` warnings / `0` skips |
| Focused timing/chunking/speakers/recording | `.venv/bin/pytest -q tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_utterances_alignment_speakers.py tests/test_recording_and_normalization.py` | **PASS (`32 passed`)** | `0` | `1` warning (`anyio` `PytestRemovedIn10Warning`); `0` skipped |
| Full non-GPU Python suite | `.venv/bin/pytest -q -m 'not gpu'` | **PASS (`274 passed`)** | `0` | `1` warning (`anyio` `PytestRemovedIn10Warning`); `0` skipped |
| Studio API & export suite | `.venv-studio/bin/pytest -q tests/test_studio_exports.py tests/test_studio_export_api.py tests/test_studio_api.py` | **PASS (`49 passed`)** | `0` | `0` warnings / `0` skips |
