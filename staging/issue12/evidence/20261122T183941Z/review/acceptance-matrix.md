
# Issue #12 / PR #13 Acceptance Matrix

| Criterion | Status | Evidence |
|---|---|---|
| Exact source SHA/tree & candidate import isolation | PASS | `source/identity.json`, `validation/cpu-tests.txt` |
| Compileall + focused CPU pytest (83 .venv, 42 .venv-studio) | PASS | `validation/cpu-tests.txt` |
| Full non-GPU pytest (244 passed, 0 skipped) | PASS | `validation/cpu-tests.txt` |
| Frontend Vitest (6 files, 14 tests) + tsc/Vite build | PASS | `validation/web-tests.txt` |
| Host output root authorization, symlinks & traversal checks | PASS | `validation/export-paths-and-collisions.md` |
| Task-aware export selection & internal archive retention | PASS | `validation/export-paths-and-collisions.md` |
| Collision-safe child dirs, idempotent retry, fail-closed overwrite & partial/failed status | PASS | `validation/export-paths-and-collisions.md` |
| Remote media retention, early export (`save_remote_media`) & URL guardrails | PASS | `validation/remote-source-retention.md` |
| Separate typed `NEMOVOX_EVENT` stage transitions from single CLI invocation | PASS | `validation/stage-event-and-log-contract.md` |
| Persistent logs, chunked UTF-8 Copy full, Download full & reload retention | PASS | `validation/stage-event-and-log-contract.md`, `validation/isolated-studio-browser.md` |
| Paginated job history (>50 runs), legacy job defaults & missing job alert | PASS | `validation/isolated-studio-browser.md` |
| Chromium Playwright end-to-end UI qualification | PASS | `validation/isolated-studio-browser.md` |
| Real French ASR -> Riva CUDA -> English review -> SRT/ASS -> export | PASS | `validation/french-gpu-studio-e2e.md` |
| Production Studio & model-unit/GPU isolation | PASS | `validation/service-and-production-invariance.md` |
