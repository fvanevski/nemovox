# Issue #2 Final-Head Verification Results (`49fac0f4ae9c607078cbc71e34bc448c5aab318f`)

## 1. Focused Studio API Tests (`.venv-studio`)

```bash
.venv-studio/bin/pytest -v tests/test_studio_api.py
```

- **Exit code:** `0`
- **Result:** `12 passed in 3.12s`
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
  - `test_app_creation_does_not_reconcile_another_running_server` — `PASSED`
  - `test_cancel_and_completion_have_serialized_terminal_state` — `PASSED`

## 2. Full CPU Suites

- **`.venv-studio/bin/pytest -q -m "not gpu"`:** **exit `0`**, `83 passed in 3.95s`
- **`.venv/bin/pytest -q -m "not gpu"`:** **exit `0`**, `83 passed in 4.01s` (1 `PytestRemovedIn10Warning` from `.venv`'s `anyio` plugin)

## 3. Frontend Unit Tests & Production Build (`web/`)

```bash
cd web && npm test && npm run build
```

- **Exit code:** `0`
- **Vitest (`vitest run`):** `2 passed` test files (`src/api.test.ts` 2 tests, `src/source-selection.test.ts` 2 tests; `4` tests passed)
- **TypeScript + Vite build (`tsc --noEmit && vite build`):** `2036 modules transformed`, built in `586ms`

## 4. Bounded Host-Process HTTP Smoke Output (`127.0.0.1:8765` & `127.0.0.1:8769`)

```text
1_HEALTH: {"status":"ok","backend":"local","ffmpeg":true,"ytdlp":true,"media_roots":["/workspace/nvidia-transcribe/outputs","<USER_HOME>/Videos","/opt/mcog","/workspace/pyannote/outputs/sources","/workspace/nvidia-transcribe/outputs/final_smoke/media"]}
1_ROOT_HTML: 200 OK
2_AUTHORIZED_STARTUP_RECONCILE: interrupted | Previous Studio process stopped
3_IMPORT_IN_SECOND_PROCESS_PRESERVES_ACTIVE: running stdout= IMPORT_OK
4_SECONDARY_SERVER_EXIT: 3 LOCK_ERROR_PRESENT: True
5_CANCEL_PRECEDES_COMPLETION: state=cancelled descendant_alive=False
6_COMPLETION_PRECEDES_CANCEL: complete | Complete
7_GRACEFUL_SHUTDOWN: uvicorn_rc=-15 job_state=cancelled descendant_alive=False
EXIT_CODE=0
```

## 5. Unchanged Frontend & Browser Qualification Reference

- `web/` Git tree SHA is unchanged (`b36450e7a5b0970a1af609bde38a6330846d3e1f` on both `2f0b16e8` and `49fac0f4`).
- B2 contract transcription UI, I1 mode-switch/late-upload regression + `POST /api/jobs` payload interception, JASSUB Chromium (`--disable-gpu`) playback, and FireDragon `13.13.2` verification remain backed by `evidence/issue-2-studio-20261121` (`e97e5dc0cf644bb9b614a933154ab559aabea466`).
