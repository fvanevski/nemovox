# Issue #2 Browser & UX Verification Review

Verified against the built frontend served by FastAPI (`http://127.0.0.1:8765/`, commit `2f0b16e890b2877ec0779818d09d837c79b83086`) using synthetic WebM video and synthetic Mandarin/untimed contracts.

## Browsers Tested

1. **Chromium (`/usr/bin/chromium`) via Playwright (`--disable-gpu`, `1440x960`):**
   - `console` / `pageerror` errors (excluding intentional `422` submission rejections): `[]`
   - Unexpected HTTP `>=400` responses: `[]`
2. **Garuda Linux FireDragon `13.13.2` (`/usr/bin/firedragon --headless`):**
   - Loaded `http://127.0.0.1:8765/?job=<id>&step=preview`, exit code `0`, rendered complete preview UI with synthetic WebM video, cue timeline, and honest `HTML text fallback` indicator.

## Verified Issue #2 Flows

- **B2 Contract Transcription (UI):**
  - Submitting an invalid contract (`{"utterances": [{"id": "broken"}]}`) displays `[role="alert"]`: `"Every contract utterance must contain source_text."` without queueing a job.
  - Submitting a populated synthetic contract (`utt-0000` timed, `utt-0001` untimed) for `Transcribe` transitions cleanly to `complete`, renders both `recording.contract.json` and `transcript.txt` download buttons (`HTTP 200`), and shows `Transcript cues` with speaker labels (`SPK_00`, `SPK_01`) and `— → —` for the untimed line.
- **I1 Active Media-Mode Invariant (UI + Network Interception):**
  - Selecting a host file then clicking `Upload` clears the active selection (`Selected source` hidden).
  - Intercepting `POST /api/media/upload` with a 1.2s delay, starting upload of `synthetic_demo.webm`, and switching immediately to `Remote URL` ensures the late upload completion is ignored (`Selected source` remains empty).
  - Entering `https://example.org/synthetic-clip.webm` and clicking `Start processing` produces an intercepted `POST /api/jobs` JSON payload whose `source` is strictly `{"kind": "url", "value": "https://example.org/synthetic-clip.webm"}`.
- **Subtitle Preview, JASSUB & Refresh Persistence:**
  - Running `Subtitles` on the synthetic contract via `.venv/bin/python -m nvidia_transcribe.cli generate-subtitles` (using an ephemeral local OpenAI-compatible mock translator) renders `ASS · libass` subtitles (`Today's audio check is ready.`) onto the synthetic video canvas in Chromium, preserves `?job=<id>&step=preview` across `page.reload()`, and supports cue click-to-seek.

## Archived Synthetic Screenshots

Stored under `screenshots/` inside `issue-2-studio-evidence.tar.gz`:
- `b2_contract_transcription.png`
- `b2_subtitles_jassub.png`
- `i1_remote_url_active.png`
- `firedragon_preview.png`
