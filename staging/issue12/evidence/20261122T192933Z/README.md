
# Issue #12 / PR #13 — Codex Remediation Qualification (`20261122T192933Z`)

- **Repository:** `fvanevski/nemovox`
- **Issue / PR:** `#12` / `#13` (draft)
- **Branch:** `issue/12-studio-output-workflow`
- **Tested Source SHA:** `be977b64c6a73adda895fb7c517bbcb2d1334794`
- **Tested Tree SHA:** `1faacabc4b43d4c18ec4fd28bc0c3c788370d819`
- **Prior Evidence Carrier (historical):** `07a6e7b25cbafaf110035e6a665225ee28bbfe2e`
- **Source Edits by Local Agent:** `none` (`SOURCE_EDITS=not_authorized`)
- **Result:** `FAIL`

## Summary
The Codex P2 sidecar-path fix in `src/nvidia_transcribe/studio_api.py` (`82787a0`) passes the original contract directory to `generate-subtitles`, preserves untimed chunk-1 ordering after chunk-0 (`estimated_order_s=450.0 > 150.0`, `chunk_window_s=[290.0, 610.0]`, no invented timestamps), and completes both `translation` and `subtitles` Studio runs cleanly.

However, the newly added test `test_imported_contract_processing_keeps_adjacent_reconstruction_sidecars` (`[translation]` and `[subtitles]`) **fails** at `tests/test_studio_export_api.py:207`:
- Input `original` is written as compact JSON (`316` bytes, `json.dumps(source_contract)`).
- `JobRunner._run` (`studio_api.py:618`) archives `asr/recording.contract.json` via `_atomic_json(contract, value)` (`indent=2`, `406` bytes), and `export_selected` exports that archived copy (`406` bytes).
- Line 207 asserts `(saved / "recording.contract.json").read_bytes() == original_contents`, causing `AssertionError` (`406` indented bytes != `316` compact bytes), even though parsed JSON equality holds.
