
# Phase B1–B2 Fresh Chromium Browser & Production Contract Acceptance

- **Target:** `http://127.0.0.1:8765` (`MainPID=1121444`, `InvocationID=6cd5cf36a6774b0693d37a196c088464`, `index-BmJZkZRY.js`)
- **Browser:** Fresh `/usr/bin/chromium` headless context (`1280x900`, clipboard permissions)
- **UI Checks (B1):**
  - `"not the browser's Downloads folder"` host notice present (`True`)
  - Task-aware export checkboxes: `transcription=2`, `translation=9`, `subtitles=10`
  - Keyboard `Space` toggle on `ASR contract (.json)`: `True`
  - Export folder picker selected unique subfolder and confirmed folder pill BEFORE Start (`True`)
  - `Browse all runs` loaded (`True`), older preexisting job `980af31f014245c4b200e0af8b33ebca` reopened cleanly with `Export status: disabled` / `Internal job archive only` (`True`), missing job `dddd...` displayed `Could not load this job` alert (`True`).
- **Synthetic 2-Chunk Contract Acceptance (B2, `336` compact bytes, `sha256=3ea2975074d4a1e9868a29a5c007bffb902f5985abf9e67e59df6e189fb52484`):**
  - `subtitles` UI (`6bb42e5a62d141f7afa9db248e526140`): `Export status: complete · 5 saved`, `exported_files=['final_english_subtitles.txt', 'recording.contract.json', 'subtitles.en.ass', 'subtitles.en.srt', 'transcript.txt']`, `archived==exported==original=3ea2975074d4a1e9868a29a5c007bffb902f5985abf9e67e59df6e189fb52484`, `cli_used_original_contract=True`, `refined_order=[['chunk0-timed', 150.0, 151.0], ['chunk1-untimed', None, None]]`, `Copy full log` -> `"Full log copied."`, retained across `page.reload()`.
  - `transcription` API (`0e2b5e5f3ea845ec9f2e77b1486e342b`): `archived==exported==original=3ea2975074d4a1e9868a29a5c007bffb902f5985abf9e67e59df6e189fb52484`, `exported_files=['recording.contract.json', 'transcript.txt']`.
  - `translation` API (`6d13b1b8f4a6461bbb7ac0c6da3d6100`): `archived==exported==original=3ea2975074d4a1e9868a29a5c007bffb902f5985abf9e67e59df6e189fb52484`, `cli_used_original_contract=True`, `refined_order=[['chunk0-timed', 150.0, 151.0], ['chunk1-untimed', None, None]]`, `exported_files=['final_english_subtitles.txt', 'recording.contract.json', 'subtitles.en.srt', 'transcript.txt']`.
  - `original_contract_untouched=True`.
- **Browser errors:** `page_errors=0`, `console_errors=12` (all 12 were expected HTTP `404` responses from missing-job check and synthetic contract `/media` preview).
- **Sanitized Screenshots:** `validation/sanitized-tab1-exports.png`, `validation/sanitized-tab3-synthetic.png` (sidebar prior job titles redacted before capture).
