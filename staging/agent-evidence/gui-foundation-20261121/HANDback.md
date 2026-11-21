# Nemovox Studio Integration Handback

- **Repository:** `https://github.com/fvanevski/nemovox`
- **Baseline `origin/main`:** `9b4c8b539c67f2f6266fef135a259e92f5477b19`
- **Central staged branch (`origin/central/gui-foundation-20261121`):** `cf74bc8c5a793286bd424b3c3667451a92d8079e`
- **Integration branch (`integration/gui-foundation-20261121`):** `d6962a58783d982343f99f697339fc3fdd22ec87`
- **Evidence branch (`evidence/gui-foundation-20261121`):** committed on top of `d6962a58783d982343f99f697339fc3fdd22ec87`
- **Evidence archive (`staging/agent-evidence/gui-foundation-20261121/gui-foundation-evidence.tar.gz`):** `1,678,558` bytes, SHA-256 `faa6083ecdea4ae7ce9ceaf35aac6f5b7e15b0f1a760c05f2a65eefa6f6ac38b`

## Host Environment

- **Host:** Garuda Linux, NVIDIA GeForce RTX 5090 (32 GB VRAM)
- **Python (Studio API):** `.venv-studio` (`Python 3.13.15`, FastAPI `0.165.0`, Uvicorn `0.54.0`, `python-multipart` `0.0.34`, installed via `uv`)
- **Python (CLI pipeline):** `.venv` (`Python 3.13.15`, invoked via `NEMOVOX_CLI_PYTHON`)
- **Node / Frontend:** Node `v26.13.1`, npm `12.2.3`, React `19`, Vite `8.3.8`, TypeScript `6`, Tailwind CSS `4`, JASSUB `2.0.0`, Vitest `4.1.12`
- **Media / Browsers:** `ffmpeg`, `ffprobe`, `yt-dlp`, Chromium (`/usr/bin/chromium`), Garuda FireDragon `13.13.2`

## Summary of Integration Repairs

All central architecture and design choices were preserved (`web/` React/Vite frontend -> `studio_api.py` FastAPI single-worker queue -> existing `nvidia_transcribe.cli` subprocesses):

1. **Process-group cancellation & queued-job cancellation hardening (`src/nvidia_transcribe/studio_api.py`):**
   - Added `_terminate_process_group()` with `ProcessLookupError` guards and `SIGTERM` -> `wait(8s)` -> `SIGKILL` escalation used consistently across both `_command()` and `_download()`.
   - Checked `self._cancelled` at the top of `_run()` and `_download()` so queued jobs cancelled before launch transition cleanly to `cancelled` without spawning subprocesses, and discarded completed IDs from `self._cancelled`.
2. **Environment-configured ASR/SwiftU endpoint forwarding (`src/nvidia_transcribe/studio_api.py`):**
   - Forwarded `ASR_BASE_URL`, `ASR_MODEL`, `SWIFTU_BASE_URL`, and `SWIFTU_MODEL` when present in environment to `--base-url`, `--model`, `--swiftu-url`, and `--swiftu-model` CLI flags.
3. **Artifacts & contract media preview resolution (`src/nvidia_transcribe/studio_api.py`):**
   - Added `asr/transcript.txt`, `output/raw_source_transcript.txt`, `output/refined_source_transcript.txt`, and `output/subtitles.metadata.json` to `GET /api/jobs/{id}/artifacts`.
   - Resolved embedded `source.path` on `contract` jobs inside `GET /api/jobs/{id}/media` while enforcing root allowlisting and `MEDIA_EXTENSIONS`.
4. **Browser refresh persistence, accessibility, fallback UI & artifact labels (`web/src/App.tsx`, `web/index.html`):**
   - Synced active `job` and `step` with URL query params (`?job=...&step=...`) so refreshing the page preserves the active run and preview.
   - Wrapped the main layout inside `<Tabs.Root>` with `<Tabs.Content>` panels so Radix Tabs `aria-controls` references and vertical keyboard arrow navigation work properly.
   - Added `<video onError>` graceful fallback UI for unsupported containers/codecs or contract-only jobs without media, labeled HTML vs `ASS · libass` preview modes accurately, enabled log disclosure for `cancelled`/`interrupted` states, and displayed full artifact filenames (`item.name`) instead of ambiguous `.slice(-2)` suffixes.
   - Added an inline SVG favicon in `web/index.html` eliminating the `/favicon.ico` 404 console error.
5. **Lockfile & `.gitignore` fix (`web/package-lock.json`, `web/src/api.test.ts`, `.gitignore`):**
   - Added `!web/**/*.ts` in `.gitignore` so TypeScript test/source files under `web/` are not matched by the media MPEG-TS `*.ts` rule.
   - Committed `web/package-lock.json` and unit test `web/src/api.test.ts`.
   - Expanded `tests/test_studio_api.py` with CPU-only tests for upload validation, interrupted-job persistence across `JobRunner` restarts, contract media streaming, process-group child termination on cancel, and endpoint env-var forwarding.

## Scope & Honest Limitations

- Live GPU Nemotron vLLM (`127.0.0.1:8190`) and SwiftU (`127.0.0.1:8001`) subtitle inference services were not running during this session (port `8788` returned HTTP 500 to unconfigured chat completion probes). Full-episode ASR was intentionally avoided per handoff instructions.
- Real CLI execution (`NEMOVOX_CLI_PYTHON=.venv/bin/python -m nvidia_transcribe.cli generate-subtitles`) was verified end-to-end against a synthetic Mandarin `recording.contract.json`, synthetic FFmpeg WebM media, and a local OpenAI-compatible mock SwiftU translator, producing real refined/translated contracts, `.srt`, and `.ass` rendered in-browser via JASSUB `libass` WASM.
- Neither `/workspace/pyannote` nor `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` was modified, and no existing baselines or `main` branch were altered.
