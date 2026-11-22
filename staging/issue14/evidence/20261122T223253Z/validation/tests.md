# Phase A Deterministic Test & Admission Report (`4c6c7ac5`)

## Isolation & Import Verification
- **Detached isolated worktree:** `<ISOLATED_WORKTREE>` at `b110657d02788b13d32ffa7b031a2c57122e17a8` (`diff vs 4c6c7ac5 -- src tests web` empty).
- **Verified import origin:** `<ISOLATED_WORKTREE>/src/nvidia_transcribe/{speech_chunks,speakers,translation_review}.py`.
- **CLI `--chunk-policy` parse:** Verified `{pause_aware,fixed}` option help and `studio_api.py:653` forwarding.

## Deterministic Suite Results (Production `.venv` & `.venv-studio`, Read-Only Interpreters)

| Gate | Command | Result | Exit | Skips / Warnings |
|---|---|---|---|---|
| `compileall` | `.venv/bin/python -m compileall -q src tests scripts` | **PASS** | `0` | `0` skips / `0` warnings |
| Focused Issue #14 & Core | `.venv/bin/pytest -q tests/test_issue14_pause_chunks.py tests/test_issue14_speaker_review_guards.py tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_utterances_alignment_speakers.py tests/test_recording_and_normalization.py tests/test_translation_review.py` | **FAIL (`2 failed, 55 passed`)** | `1` | `0` skipped; `1` `anyio` `PytestRemovedIn10Warning` |
| Full non-GPU Python | `.venv/bin/pytest -q -m 'not gpu'` | **FAIL (`2 failed, 285 passed`)** | `1` | `0` skipped; `1` `anyio` `PytestRemovedIn10Warning` |
| Studio API & Export | `.venv-studio/bin/pytest -q tests/test_studio_exports.py tests/test_studio_export_api.py tests/test_studio_api.py` | **PASS (`50 passed`)** | `0` | `0` skipped / `0` warnings |
| Web Frontend | `npm test && npm run build` | **PASS (`6` files, `14` tests + `tsc && vite build`)** | `0` | `0` skipped / `0` warnings |

## Exact Failing Test Reproducers (Stopped Phase C GPU Execution)
1. **`tests/test_issue14_pause_chunks.py::test_short_media_and_fixed_compatibility:59`**
   - `plan_chunks(200, ChunkConfig(chunk_seconds=90, context_seconds=10))` returns **`3`** `ChunkSpec`s (`[0..90]`, `[90..180]`, `[180..200]`, since `20s > min_tail_seconds=15.0`), failing `assert len(fixed) == 2`.
2. **`tests/test_issue14_speaker_review_guards.py::test_single_near_threshold_embedding_cannot_force_merge:45`**
   - `assert not _near_threshold_single_embedding_candidate(0.70, 0.65)` returns **`True`** instead of `False` because `src/nvidia_transcribe/speakers.py:311` evaluates `threshold <= similarity < threshold + ambiguity_margin` and in IEEE-754 float `0.65 + 0.05 == 0.7000000000000001 > 0.70` (`0.70 - 0.65 == 0.04999999999999993 < 0.05`).

## Fixed Mode Compatibility & Resume Isolation Check
- Default `RecordingPipelineConfig()` uses `chunk_policy="fixed"` (`sig=175113b5fd6523ff`, identical to explicit `chunk_policy="fixed"`).
- `chunk_policy="pause_aware"` produces a distinct deterministic signature (`b79c681ea12ca1a4`), preventing accidental reuse of historical `90s` raw chunk caches while allowing identical adaptive plans to resume cleanly.
- Studio `Options` schema accepts `"fixed"` (default) and `"pause_aware"`, rejects invalid strings (`ValidationError`), and passes `--chunk-policy` to `process-media`.
