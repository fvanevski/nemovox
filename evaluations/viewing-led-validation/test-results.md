# Verification & Test Results

## 1. Bytecode Compilation & CPU Unit Tests
Executed on `/workspace/nvidia-transcribe/.venv` after all source, overlay, and test edits:

```bash
.venv/bin/python -m compileall -q src tests experiments && .venv/bin/pytest -q -m "not gpu"
```

- **Exit code:** `0`
- **`pytest -q -m "not gpu"`:** **`71 passed, 1 warning in 0.98s`** (`0` failed, `0` skipped — 69 existing tests + 2 new tests in [test_viewing_validation_regressions.py](file:///workspace/nvidia-transcribe/tests/test_viewing_validation_regressions.py) verifying coarse-timing recovery, raw-ASR/pre-overlay provenance preservation, mismatch guards, duplicate suppression, CLI `--validation-overlay`, and `--re-render` SRT/ASS export).

## 2. CPU `--re-render --validation-overlay` Regeneration Checks
Executed cleanly with `0` GPU / `0` LLM inference calls:

- **French (`fr_full_episode`):** Exit `0` (`--ass-layout standard --ass-max-lanes 3 --re-render --validation-overlay configs/overlays/fr_s04e01_validation.json`)
  - Applied: `relocated_timing_count=1, edited_text_count=3, suppressed_count=4, skipped_ids=[]`
  - Parity: `renderable_cues=280, ass_displayed_cues=280, ass_overflow_cues=0, invalid=0, leaks=0`
- **Mandarin (`zh_full_episode_ep166`):** Exit `0` (`--ass-layout raised_main --ass-margin-v 150 --ass-max-lanes 3 --re-render --validation-overlay configs/overlays/zh_ep166_validation.json`)
  - Applied: `recovered_timing_count=19, relocated_timing_count=2, edited_text_count=24, suppressed_count=9, skipped_ids=[]`
  - Parity: `renderable_cues=301, ass_displayed_cues=301, ass_overflow_cues=0, invalid=0, leaks=0`
- **Baseline Integrity:** `outputs/subtitle_runs/viewing_quality/` untouched; Nemotron ASR & Qwen fallback never restarted.

## 3. Preview Rendering & Visual Spot-Check Verification
- Rendered **22 bounded 720p MP4 previews** (original audio + burned ASS) under `outputs/subtitle_runs/viewing_validation/previews/`, inspected directly via multimodal playback, re-rendered **15 updated clips** after overlay fixes, and verified **8 1080p FFmpeg screenshots** under `outputs/subtitle_runs/viewing_validation/screenshots/`.
