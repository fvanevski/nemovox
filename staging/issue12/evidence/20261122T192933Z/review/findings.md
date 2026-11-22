
# Blocking Findings (`be977b64c6a73adda895fb7c517bbcb2d1334794`)

1. **BLOCKING (`tests/test_studio_export_api.py:207` vs `src/nvidia_transcribe/studio_api.py:618`):**
   - `test_imported_contract_processing_keeps_adjacent_reconstruction_sidecars[translation]` and `[subtitles]` both fail at line 207:
     `assert (saved / "recording.contract.json").read_bytes() == original_contents`
   - **Cause:** The test writes `original` compactly (`json.dumps(source_contract)` -> `147` chars, no indentation), while `studio_api.py:618` writes the archived job copy via `_atomic_json(contract, value)` (`indent=2`), and `export_selected` exports `job_root / "asr/recording.contract.json"`. The exported file is semantically identical (`json.loads(...) == source_contract`), but its raw bytes differ (`indent=2` vs single-line JSON).
   - **Disposition:** `SOURCE_EDITS=not_authorized`; returned exact reproducer to Central to either preserve exact raw source bytes when archiving `recording.contract.json` in `studio_api.py` (after JSON validation) or compare parsed JSON / `indent=2` in `test_studio_export_api.py`.
