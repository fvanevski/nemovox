# Test & Build Verification Results

Integration commit under test: `d6962a58783d982343f99f697339fc3fdd22ec87` (`integration/gui-foundation-20261121`).

## 1. Studio Python CPU Test Suite (`.venv-studio`)

Command:

```bash
.venv-studio/bin/pytest -v -m "not gpu"
```

- **Exit code:** `0`
- **Result:** `79 passed in 1.83s`
- Includes 8 tests in `tests/test_studio_api.py`:
  - `test_health_and_media_roots`
  - `test_path_traversal_is_denied`
  - `test_job_submission_and_preview`
  - `test_invalid_options_and_bad_url`
  - `test_parse_srt`
  - `test_upload_and_extension_rejection`
  - `test_interrupted_job_persistence_and_contract_media`
  - `test_cancel_process_group_and_env_forwarding`

## 2. Main CLI Environment CPU Test Suite (`.venv`)

Command:

```bash
.venv/bin/pytest -v -m "not gpu"
```

- **Exit code:** `0`
- **Result:** `79 passed in 1.75s` (1 `PytestRemovedIn10Warning` from `.venv`'s `anyio` plugin)

## 3. Frontend Unit Tests & Production Build (`web/`)

Command:

```bash
cd web && npm test && npm run build
```

- **Vitest (`vitest run`):** `Test Files 1 passed (1)`, `Tests 2 passed (2)` (`web/src/api.test.ts`), exit `0`
- **TypeScript + Vite build (`tsc --noEmit && vite build`):** `2035 modules transformed`, built in `960ms` including JASSUB worker/WASM/font assets (`dist/assets/jassub-worker*.wasm`, `dist/assets/jassub-worker*.js`, `dist/assets/default-*.woff2`), exit `0`

## 4. Live HTTP API + CLI Subprocess + Playwright E2E Smoke

Executed against `.venv-studio/bin/python -m uvicorn nvidia_transcribe.studio_api:app --host 127.0.0.1 --port 8765 --workers 1` with `NEMOVOX_CLI_PYTHON=/workspace/nvidia-transcribe/.venv/bin/python`, synthetic FFmpeg WebM/MKV fixtures, synthetic Mandarin `recording.contract.json`, and a local ephemeral OpenAI-compatible translator endpoint:

- `GET /api/health` -> `200`, `{"status": "ok", "backend": "local", "ffmpeg": true, "ytdlp": true, ...}`
- `GET /api/glossaries` -> `200`, `["a_will_eternal_zh", "mcog_fr"]`
- `GET /api/media?path=/etc` -> `403` (path traversal denied)
- `POST /api/media/upload` (via browser `<input type="file">`) -> `200`, uploaded synthetic WebM into isolated upload root
- `POST /api/jobs` -> `202`, ran real `.venv/bin/python -m nvidia_transcribe.cli generate-subtitles` subprocess, reached `complete`
- `GET /api/jobs/{id}/artifacts` -> returned all 8 generated files (`refined.contract.json`, `translated.contract.json`, `raw_source_transcript.txt`, `refined_source_transcript.txt`, `final_english_subtitles.txt`, `subtitles.metadata.json`, `synthetic_demo.en.srt`, `synthetic_demo.en.ass`)
- `GET /api/jobs/{id}/artifact/output/synthetic_demo.en.srt` -> `200`, `137` bytes
- Browser reload (`page.reload()`) preserved `?job=<id>&step=preview` and completed preview state
