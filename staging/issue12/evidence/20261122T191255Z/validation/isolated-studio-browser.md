
# Isolated Studio & Browser Verification (`1eaf682262dea4b286db111fcb1b728c980d2e4c`)

- **Candidate Server:** `127.0.0.1:8769` (`cwd=/workspace/nemovox-issue12-verify`, `PYTHONPATH=/workspace/nemovox-issue12-verify/src`, `NEMOVOX_CLI_PYTHON=/workspace/nvidia-transcribe/.venv/bin/python`, serving `/workspace/nemovox-issue12-verify/web/dist`)
- **Browser:** `/usr/bin/chromium` via Playwright `sync_playwright`
- **Errors:** `page_errors=0`, `console_errors=0`

## Synthetic Contract Job (`e3431ae564e844c5aaf130d1364d7a81`)
- Submitted `recording.contract.json` (`subtitles`, default 4 artifacts) via Chromium UI.
- Recorded `stage_history`: `[('asr', 'skipped'), ('translation', 'skipped'), ('review', 'skipped'), ('condensation', 'skipped'), ('render', 'complete'), ('export', 'complete')]`
- UI stage panel before reload: `['skipped\nasr\nUsing an existing recording contract', 'skipped\nPrimary translation\nbypassed_same_language', 'skipped\nEnglish review\nSame-language translation bypass', 'skipped\ncondensation\nOptional condensation disabled', 'complete\nRender subtitles', 'complete\nexport\ncomplete']`
- UI stage panel after `page.reload()`: `['skipped\nasr\nUsing an existing recording contract', 'skipped\nPrimary translation\nbypassed_same_language', 'skipped\nEnglish review\nSame-language translation bypass', 'skipped\ncondensation\nOptional condensation disabled', 'complete\nRender subtitles', 'complete\nexport\ncomplete']`
- Exported files: `['final_english_subtitles.txt', 'synthetic.en.ass', 'synthetic.en.srt', 'transcript.txt']` (`Export status: complete · 4 saved`)

## Empty Export Plan (`1406c56774b749e0a816b247f8582771`)
- Submitted `artifacts=[]`, `save_remote_media=false` with valid destination -> terminal `export_report.status=disabled`, no external directory created.
