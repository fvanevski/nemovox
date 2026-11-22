
# Corrective Contract Export, Retained Contract & Zero-Artifact Disabled Status

Tested on isolated candidate Studio `127.0.0.1:8769` serving `web/dist` built from `a15b360d396d7b6e028d4551efdea0d356d7f8bb` with `/usr/bin/chromium`.

## 1. Imported `recording.contract.json` + `subtitles` + All 4 Default Exports
- **Job ID:** `f485dc442f7d4c50820e2f48954b5ef3`
- **Submitted artifacts:** `['source_transcript', 'final_english', 'srt', 'ass']`
- **ASR stage:** `skipped` (`Using an existing recording contract`, no ASR inference)
- **Internal archive retained:**
  - `asr/recording.contract.json` retained with unchanged `utterances` (`source_text` and `utterance_id` identical to input)
  - `asr/transcript.txt` derived and retained
- **Terminal export status:** `complete` (UI: `Export status: complete · 4 saved`; **no false `partial`**)
- **External `nemovox-f485dc442f7d4c50820e2f48954b5ef3/` files:** `['final_english_subtitles.txt', 'synthetic.en.ass', 'synthetic.en.srt', 'transcript.txt']`, each SHA-256 verified against internal originals.
- **Reload retention:** Verified `Export status: complete · 4 saved` and retained log panel persisted across `page.reload()`.

## 2. Optional `recording_contract` Export on Contract `subtitles` Job
- **Job ID:** `d00c7a9d1cba4f1abd2820db3332b3a8`
- **Terminal export status:** `complete`
- **Exported files:** `['final_english_subtitles.txt', 'recording.contract.json', 'synthetic.en.srt', 'transcript.txt']` (exported `recording.contract.json` utterances verified identical to input).

## 3. Zero Selected Artifacts (`artifacts=[]`, `save_remote_media=false`)
- **Job ID:** `ca9ce2bb6ef0462e8943fc864020b4b0`
- **Submitted export plan:** `{"destination": "/tmp/nemovox-issue12-requal/exports", "artifacts": [], "save_remote_media": false}`
- **Terminal export status:** `disabled` (UI: `Export status: disabled · 0 saved`; never `pending`)
- **External directory created:** `False`
