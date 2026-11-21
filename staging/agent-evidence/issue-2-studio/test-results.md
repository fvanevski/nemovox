# Issue #2 Test & Build Verification Results

Implementation commit under test: `2f0b16e890b2877ec0779818d09d837c79b83086` (`integration/gui-foundation-20261121`).

## 1. Focused Studio API Lifecycle & Contract Regressions (`.venv-studio`)

```bash
.venv-studio/bin/pytest -v tests/test_studio_api.py
```

- **Exit code:** `0`
- **Result:** `10 passed in 3.11s`
  - `test_health_and_media_roots` — `PASSED`
  - `test_path_traversal_is_denied` — `PASSED`
  - `test_job_submission_and_preview` — `PASSED`
  - `test_invalid_options_and_bad_url` — `PASSED`
  - `test_parse_srt` — `PASSED`
  - `test_upload_and_extension_rejection` — `PASSED`
  - `test_interrupted_job_persistence_and_contract_media` — `PASSED`
  - `test_cancel_process_group_and_env_forwarding` — `PASSED`
  - `test_contract_transcription_import_and_empty_contract` — `PASSED`
  - `test_cancel_cleans_descendant_after_group_leader_exits` — `PASSED`

## 2. Full CPU Suite in Studio Environment (`.venv-studio`)

```bash
.venv-studio/bin/pytest -q -m "not gpu"
```

- **Exit code:** `0`
- **Result:** `81 passed in 4.07s`

## 3. Full CPU Suite in CLI Environment (`.venv`)

```bash
.venv/bin/pytest -q -m "not gpu"
```

- **Exit code:** `0`
- **Result:** `81 passed in 4.04s` (1 `PytestRemovedIn10Warning` from `.venv`'s `anyio` plugin)

## 4. Frontend Unit Tests & Production Build (`web/`)

```bash
cd web && npm test && npm run build
```

- **Exit code:** `0`
- **Vitest (`vitest run`):** `2 passed` test files (`src/api.test.ts` 2 tests, `src/source-selection.test.ts` 2 tests; `4` total tests passed in 132ms)
- **TypeScript + Vite build (`tsc --noEmit && vite build`):** `2036 modules transformed`, built in `565ms`

## 5. Live HTTP / CLI Subprocess / B1 Cleanup Smoke

- **B1 Live HTTP Cancel (`127.0.0.1:8769`):** Leader exited after spawning `SIGTERM`-resistant descendant (`PID 3513991`); `POST /api/jobs/{id}/cancel` terminated session descendants (`job_state=cancelled`, `descendant_alive=False`, exit `0`).
- **B1 Live FastAPI Lifespan Shutdown (`SIGTERM` to Uvicorn):** Active job spawned `SIGTERM`-resistant descendant (`PID 3514534`); `SIGTERM` to Uvicorn triggered `runner.shutdown()`, yielding `uvicorn_rc=-15`, `job_state=cancelled`, `descendant_alive=False` (exit `0`).
- **B2 Empty & Invalid Contracts via HTTP (`127.0.0.1:8765`):** Empty contract transcription reached `complete`; invalid contract rejected with HTTP `422`.
- **Live GPU Inference:** `UNVERIFIED` (no live Nemotron vLLM / SwiftU service running on ports `8190`/`8001`).
