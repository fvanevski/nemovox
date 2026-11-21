# Test Results & Acceptance Matrix

## CPU-Only Unit Tests
Command: `.venv/bin/pytest -v -m "not gpu"` (no 30B model weights loaded)
- **Result:** **15 / 15 PASSED** in `0.14s` (full log in `pytest_cpu.txt`).
- Coverage:
  - Checkpoint layout, `modelopt` MIXED_PRECISION validation, missing shard rejection, BF16 rejection (`tests/test_checkpoint.py`)
  - Language-agnostic prompt construction (no French default), generation-loop guardrail, vLLM payload formatting, clear `CTCMetadataError` failure when CTC metadata is unavailable (`tests/test_client.py`)
  - `<spk:N>` / `<ev:*>` extraction, French Unicode & `source_offset_s` conversion, Chinese Unicode & character-level timing without whitespace splitting, non-fabrication of NaN/negative/inverted timestamps, source-transcript preservation alongside translation, pyannote adapter (`tests/test_parser_and_contract.py`)
  - French WER & Mandarin CER Unicode normalization (`tests/test_metrics.py`)

## Evaluation Acceptance Criteria

| Criterion | Status | Measured Evidence |
|---|---|---|
| 1. Native NVFP4 inference on RTX 5090 (32 GB) | **PASS** | Loaded in `19.28 GiB` VRAM (`quantization=modelopt_mixed`, `fp8` KV cache, `128` FP8 encoder Linears, `FLASHINFER_CUTLASS` NVFP4). Peak `28,794 / 32,607 MiB` during 23-min run. |
| 2. Source-language speech recognition (FR & ZH) | **PASS (with caveats)** | French FLEURS WER `10.20%`; Mandarin FLEURS CER `12.36%` (auto) / `10.67%` (explicit). Short <4s clips & Katakana loanwords (`Scotturb`) showed errors. |
| 3. Speaker-attributed transcription & identity consistency | **PARTIAL** | Beats Pyannote on 39s similar child voices (`2` speakers vs collapsed `1`) and 100s continuous (`4` clean speakers) and A-B-A Mandarin (`<spk:0>-<spk:1>-<spk:0>`). Degrades on 300s (2-tag collapse) and 23-min full episode (word-level `<spk:N>` ping-ponging). |
| 4. Overlapping voice recognition | **PASS (for speech-on-speech)** | `multispeaker_sot` recovered both simultaneous lexical streams in `fr_2spk_overlap` and `zh_2spk_overlap` with `<spk:0>`/`<spk:1>` tags. Real 3s nonlexical overlap (`05:39`) hallucinated when cropped. |
| 5. Word/character timestamps & diarization metadata | **PARTIAL** | `100%` valid timings on all French clips (`114/114`, `246/246`, `517/517`, `1992/1992`) and 5/6 Mandarin clips (`23/23`, `37/37`, `31/31`, `31/31`, `28/28`). **Failed** (`alignment_failed`) on Mandarin `'钾'` due to 23,862-token CTC vocab OOV. |
| 6. Direct English speech translation | **PARTIAL** | Fast & mostly fluent, but exhibits entity distortion (`mu` -> `dead language`, `Nadia` -> `giraffe`) and factual omission (dropped `"钙"` / calcium). |
| 7. Language-independent transcript contract & subtitle compat | **PASS** | `nemotron-transcript-contract/v1` preserves source text, Unicode, offsets, diarization, and `null` uncertainty; tested read-only with `/workspace/pyannote`'s `build_subtitles` for FR and ZH. |
| 8. CPU-only unit test suite | **PASS** | `15/15` passed without loading GPU weights. |
| 9. Reproducible benchmarks & verified evidence archive | **PASS** | `benchmark.json`, manifests, and verified `nemotron35-nvfp4-evaluation.tar.gz` generated. |
