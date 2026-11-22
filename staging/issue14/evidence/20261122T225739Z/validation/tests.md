# Phase A Deterministic Suite Verification (`81704133`)

## Isolation & Import Verification
- **Isolated detached worktree:** `<ISOLATED_WORKTREE>` at `1a60b53703937546594e97aac59ea3a703eb0794` (0 unapproved edits vs `81704133`).
- **Production `.venv` & `.venv-studio` (read-only):** `PYTHONPATH=<ISOLATED_WORKTREE>/src`.

## Suite Results

| Gate | Command | Result | Exit | Skips / Warnings |
|---|---|---|---|---|
| `compileall` | `.venv/bin/python -m compileall -q src tests scripts` | **PASS** | `0` | `0` skips / `0` warnings |
| Focused Issue #14 & Core | `.venv/bin/pytest -q tests/test_issue14_pause_chunks.py tests/test_issue14_speaker_review_guards.py tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_utterances_alignment_speakers.py tests/test_recording_and_normalization.py tests/test_translation_review.py` | **PASS (`59 passed`)** | `0` | `0` skipped; `1` `anyio` `PytestRemovedIn10Warning` |
| Full non-GPU Python | `.venv/bin/pytest -q -m 'not gpu'` | **PASS (`289 passed`)** | `0` | `0` skipped; `1` `anyio` `PytestRemovedIn10Warning` |
| Studio API & Export | `.venv-studio/bin/pytest -q tests/test_studio_exports.py tests/test_studio_export_api.py tests/test_studio_api.py` | **PASS (`50 passed`)** | `0` | `0` skipped / `0` warnings |
| Web Frontend | `npm test && npm run build` | **PASS (`6` files, `14` tests + `tsc && vite build`)** | `0` | `0` skipped / `0` warnings |

## Verification of Both Repaired CPU Tests & Resume Isolation
- `_near_threshold_single_embedding_candidate`: `0.6999` / `0.6637` / `0.6536` return `True`; `0.70` / `0.7001` / `0.78` / `0.649` return `False`.
- `plan_chunks(200, ChunkConfig(chunk_seconds=90, context_seconds=10))`: returns `3` cores `[0,90], [90,180], [180,200]`.
- Signature separation: `sig(fixed) == 175113b5fd6523ff != sig(pause_aware) == 569258217266b091`; full-run CLI produced distinct `config_signature=276658302aed6344683f05048f99a7d0e413c9ecb1a4d7fd6f2db27837744e4e` with `resumed_chunks=0`.
