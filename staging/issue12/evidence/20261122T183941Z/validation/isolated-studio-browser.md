
# Isolated Candidate Studio & Chromium Playwright Acceptance

- **Candidate Server #1:** `127.0.0.1:8769` (`cwd=/workspace/nemovox-issue12-verify`, `PYTHONPATH=/workspace/nemovox-issue12-verify/src`, `NEMOVOX_CLI_PYTHON=/workspace/nvidia-transcribe/.venv/bin/python`)
- **Browser:** `/usr/bin/chromium` via Playwright Python (`sync_playwright`, clipboard permissions enabled)
- **Uncaught `pageerror` count:** `0`

## Verified Browser Flows
1. **Host-folder notice:** Verified visible text `"not the browser's Downloads folder"` (`True`).
2. **Source tab switching & URL checkbox:** Verified `"Also save downloaded original media, even if later processing fails"` is visible only on `Remote URL` and hidden on `Host files` / `Upload` (`True`).
3. **Task-aware export checkboxes & defaults:**
   - `Transcribe`: `2` choices
   - `Translate`: `9` choices (`.ass` omitted)
   - `Subtitles`: `10` choices; default 4 checked: `['Source transcript (.txt)', 'Final English dialogue (.txt)', 'Timed subtitle track (.srt)', 'Styled subtitle track (.ass)']`
4. **Export root browser & keyboard/label interaction:** Navigated `exports_b` -> `exports_a` -> `subdir` -> `Up` (`True`); toggled checkboxes by `<label>` click and `Space` key; entered manual nested path `exports_a/manual/nested`.
5. **UI error alerts:**
   - Outside-root destination -> `role="alert"`: `Output directory is outside configured writable export roots.`
   - Missing job (`/?job=dddd...`) -> `role="alert"`: `Could not load this job. It may have been removed or the service is unavailable.`
6. **End-to-end UI job submission & post-completion retention:**
   - Submitted `recording.contract.json` (`en -> en`), intercepted `POST /api/jobs` `export` plan: `{"destination": "/tmp/nemovox-issue12-phaseb/s1/exports_a/manual/nested", "artifacts": ["final_english", "srt"], "save_remote_media": false}`.
   - `Saved files` reported `Export status: complete · 2 saved`; no fixed wall of individual export buttons in prominent UI; all 8 internal files accessible under `Advanced: all retained job artifacts`.
   - `Processing log (retained after completion)` visible after completion; `Copy visible` and `Copy full log` succeeded; `Download full log` linked to `/api/jobs/23dff28c22ea4cbdbba4f267b9e4a6ed/log/download`.
   - State persisted across `page.reload()` (`reload_retained=True`).
7. **Job history pagination & legacy job migration:**
   - `Browse all runs` paginated `verified 1-20, 21-40, 41-58 of 58` across 58 total jobs.
   - Reopening legacy job `000000000000000000000000000000aa` (no `export` / `export_report`) displayed `Export status: disabled` and `Internal job archive only` (`legacy_job_ui=True`).
