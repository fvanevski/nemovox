# Browser & UX Verification Review

Verified against the compiled production frontend served by FastAPI (`http://127.0.0.1:8765/`) as well as Vite dev mode (`http://127.0.0.1:5173/`).

## Browsers Exercised

1. **Chromium (`/usr/bin/chromium`) via Playwright headless (`--disable-gpu`):**
   - Viewport: `1440x960`
   - Console/Page errors on normal workflow: `0` (`CONSOLE ERRORS: []`, `UNEXPECTED HTTP >=400: []`; the single expected `404` on `/api/jobs/ccc.../media` occurred only during the intentional contract-without-media fallback check).
2. **Garuda Linux FireDragon `13.13.2` (`/usr/bin/firedragon --headless`):**
   - Loaded `http://127.0.0.1:8765/?job=<id>&step=preview` directly via URL query persistence, rendered the full dark Studio UI, loaded the synthetic WebM stream, and displayed `HTML text fallback` honestly in the preview header pill.

## Verified Interactive Flows

- **Tab 1 (`Media & task`):**
  - Submitting without selecting media shows accessible `[role="alert"]` validation banner and allows dismissal.
  - Host directory browser navigates approved roots and selects `recording.contract.json` / `.webm` media.
  - Browser file upload (`Upload` mode) uploads synthetic WebM and selects the staged server path.
  - `Remote URL` mode renders bounded URL input field with non-playlist hint.
  - Task selection (`Transcribe`, `Translate`, `Subtitles`) updates downstream configuration sections cleanly.
- **Tab 2 (`Configure`):**
  - Keyboard accessibility: focusing `Media & task` tab trigger and pressing `ArrowDown` moves focus and active selection to `Configure` (`aria-selected="true"`).
  - Subtitle presentation options (`Standard` vs `Raised main`, `Condense hard-to-read English cues`) appear conditionally for `subtitles` tasks; `Advanced acoustic settings` expands/collapses cleanly.
- **Tab 3 (`Process & preview`):**
  - Live polling tracks job progress (`queued` -> `running` -> `complete`).
  - Refreshing the browser (`page.reload()`) retains `?job=<id>&step=preview`, job status, recent run list, preview cues, and downloads.
  - In Chromium (`--disable-gpu`), JASSUB initialized (`ASS · libass`), attached a canvas overlay over the synthetic video, and rendered crisp English ASS subtitles (`Today's audio check is ready.`).
  - Clicking a cue seeks the video timeline; typing in `Search dialogue…` filters cues (`2` -> `1` for `"verify"`).
  - For a contract-only job without playable source media, `<video onError>` swaps cleanly to `"Browser video preview unavailable"` while keeping cues and `.srt` download functional.

## Synthetic Screenshots Archived

Included inside `gui-foundation-evidence.tar.gz` (`screenshots/`):
- `tab1_media_task.png`
- `tab2_configure.png`
- `tab3_preview_ass.png`
- `tab3_preview_fallback.png`
- `firedragon_preview.png`
