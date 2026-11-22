
# Bounded Sidecar Fixture & Isolated Studio/Browser Check (`be977b64c6a73adda895fb7c517bbcb2d1334794`)

- **Candidate Server:** `127.0.0.1:8769` (`cwd=/workspace/nemovox-issue12-verify`, `PYTHONPATH=/workspace/nemovox-issue12-verify/src`, `NEMOVOX_CLI_PYTHON=/workspace/nvidia-transcribe/.venv/bin/python`)
- **Browser:** `/usr/bin/chromium` via Playwright `sync_playwright`
- **Two-Chunk Fixture:** `chunk0-timed` (`150.0 -> 151.0`, `chunk_index=0`), `chunk1-untimed` (`source_start=null, source_end=null`, `chunk_index=1`), `manifest.json` (`chunk 0: 0..300`, `chunk 1: 290..610`), `chunks/chunk_001.raw.json`.

## Direct Candidate `audit_and_reconstruct_contract` Check
- `chunk1-untimed` audit entry: `chunk_window_s=[290.0, 610.0]`, `estimated_order_s=450.0`, `classification=plausible_independent_missing_timing`
- Reconstructed utterance order: `[("chunk0-timed", 150.0, 151.0), ("chunk1-untimed", None, None)]` (no invented timestamps).

## Live Studio Runs (`translation` & `subtitles`)
- **`translation` job (`44cde6822d3f4068b492cbed2df2bff3`):**
  - `state=complete`, `export_status=complete`
  - `cli_received_original_contract=True` (`Loading existing recording contract .../two_chunk_clip/recording.contract.json`)
  - `refined_utterance_order`: `[{'id': 'chunk0-timed', 'orig_id': 'chunk0-timed', 'start': 150.0, 'end': 151.0}, {'id': 'chunk1-untimed', 'orig_id': 'chunk1-untimed', 'start': None, 'end': None}]`
  - `exported_files`: `['recording.contract.json', 'transcript.txt']`
  - `archived_contract_exists=True`, `archived_transcript_exists=True`
  - `exported_contract_json_equal=True`, `exported_contract_bytes_equal_to_compact_original=False` (`406` vs `316` bytes)
- **`subtitles` job (`7355c6b6f0c14bdeb4bbf4d1781d7bb2`, Chromium UI + reload):**
  - `state=complete`, `export_status=complete`
  - `cli_received_original_contract=True`
  - `refined_utterance_order`: `[{'id': 'chunk0-timed', 'orig_id': 'chunk0-timed', 'start': 150.0, 'end': 151.0}, {'id': 'chunk1-untimed', 'orig_id': 'chunk1-untimed', 'start': None, 'end': None}]`
  - `exported_files`: `['final_english_subtitles.txt', 'recording.contract.json', 'subtitles.en.ass', 'subtitles.en.srt', 'transcript.txt']` (`Export status: complete · 5 saved`, retained across `page.reload()`)
  - `ui_stages`: `['skipped\nasr\nUsing an existing recording contract', 'skipped\nPrimary translation\nbypassed_same_language', 'skipped\nEnglish review\nSame-language translation bypass', 'skipped\ncondensation\nOptional condensation disabled', 'complete\nRender subtitles', 'complete\nexport\ncomplete']`
  - `exported_contract_json_equal=True`, `exported_contract_bytes_equal_to_compact_original=False` (`406` vs `316` bytes)
- `original_unchanged=True`
- `page_errors=0`, `console_errors=2` (`404` on `/api/jobs/<id>/media` because synthetic contract fixture had no `synthetic.wav` on disk).
