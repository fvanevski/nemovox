
# Isolated Studio Byte-Preservation & Sidecar Chronology (`51f59f65249ff4fb9705bc1f67638abb521d9062`)

- **Candidate Studio:** `127.0.0.1:8769` (`cwd=/workspace/nemovox-issue12-verify`, `PYTHONPATH=/workspace/nemovox-issue12-verify/src`, `NEMOVOX_CLI_PYTHON=/workspace/nvidia-transcribe/.venv/bin/python`)
- **Controlled Compact Input:** `316` bytes, `sha256=9bc27859f8198d00f3c684846eac0dc3e0d327b39e598e651856434534f72289` (`chunk1-untimed` at `chunk_index=1` with `source_start=null`, `chunk0-timed` at `150.0->151.0`, adjacent `manifest.json` `[0-300, 290-610]`, `chunks/chunk_001.raw.json`)
- **Direct Candidate Audit:** `chunk_window_s=[290.0, 610.0]`, `estimated_order_s=450.0`, reconstructed order `[("chunk0-timed", 150.0, 151.0), ("chunk1-untimed", None, None)]`

## All Three Tasks (`transcription`, `translation`, `subtitles`)
- **`transcription` (`b0fe268ef4d24e88bcb5303661d1a15b`):** `state=complete`, `export_status=complete`, `bytes=316`, `original==archived==exported=9bc27859f8198d00f3c684846eac0dc3e0d327b39e598e651856434534f72289`, `exported_files=['recording.contract.json', 'transcript.txt']`
- **`translation` (`72f7d3b096f74e2580361db71e5b3759`):** `state=complete`, `export_status=complete`, `bytes=316`, `original==archived==exported=9bc27859f8198d00f3c684846eac0dc3e0d327b39e598e651856434534f72289`, `cli_used_original_contract=True`, `refined_utterances=[{'id': 'chunk0-timed', 'orig_id': 'chunk0-timed', 'start': 150.0, 'end': 151.0}, {'id': 'chunk1-untimed', 'orig_id': 'chunk1-untimed', 'start': None, 'end': None}]`, `exported_files=['recording.contract.json', 'transcript.txt']`
- **`subtitles` (`cb37c5ed6e5d48ee83018513bb33bfd2`):** `state=complete`, `export_status=complete`, `bytes=316`, `original==archived==exported=9bc27859f8198d00f3c684846eac0dc3e0d327b39e598e651856434534f72289`, `cli_used_original_contract=True`, `refined_utterances=[{'id': 'chunk0-timed', 'orig_id': 'chunk0-timed', 'start': 150.0, 'end': 151.0}, {'id': 'chunk1-untimed', 'orig_id': 'chunk1-untimed', 'start': None, 'end': None}]`, `exported_files=['recording.contract.json', 'transcript.txt']`
- **`inputs_unmodified`:** `True` (`recording.contract.json`, `manifest.json`, `chunks/chunk_001.raw.json` unchanged).
