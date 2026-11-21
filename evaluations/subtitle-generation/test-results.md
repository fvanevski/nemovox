# Test & Verification Results

All automated CPU checks passed after the final source edit (`src/nvidia_transcribe/translation.py` + `tests/test_translation.py`), with zero failures or skipped assertions.

## 1. Bytecode Compilation & CPU Unit Suites

1. **Syntax & bytecode check:**
   - Command: `.venv/bin/python -m compileall -q src tests`
   - Result: Exit `0`
2. **Primary project CPU test suite:**
   - Command: `.venv/bin/pytest -v -m "not gpu"`
   - Result: **`63 passed, 1 warning in 1.01s`** (`PytestRemovedIn10Warning` from `anyio.pytest_plugin`)
   - Key new/updated test modules:
     - `tests/test_qwen_fallback.py`: 8 passed (includes `test_qwen_fallback_localizes_oov_span_and_never_migrates_to_earlier_utterance`)
     - `tests/test_refinement.py`: 4 passed (includes `test_batched_swiftu_refinement_profiles_screening_and_resume`)
     - `tests/test_translation.py`: 5 passed (includes `test_translate_vetoed_regrouping_never_leaves_untranslated_source_orphan` & `test_end_to_end_generate_subtitles_from_contract_and_rerender_without_asr`)
     - `tests/test_subtitles.py`: 4 passed (phrase wrapping, min/max duration clamp, acoustic-only concurrency & 2-lane ASS, safe sidecar overwrite)
3. **Modular evaluation verification suite:**
   - Command: `PYTHONPATH=. outputs/modular-eval/.venv/bin/pytest -v experiments/modular/tests/test_modular_eval.py`
   - Result: **`8 passed in 0.33s`**

## 2. Live GPU & End-to-End Pipeline Runs

| Check | Command / Script | Observed Outcome |
| --- | --- | --- |
| Live Qwen OOV Fix Verification | `.venv/bin/python experiments/subtitle/verify_qwen_fix_gpu.py` | **PASSED**: `zh_multichunk_synthetic_qwen` (`24.182s`, `7/7` full timing, `'钾'` at `[12.24..12.48]`), `zh_fleurs_01_2006` (`11.86s`, `21/21` chars), `zh_fleurs_05_1805`, `zh_2spk_overlap` |
| Translation Bakeoff (Riva-4B vs SwiftU) | `.venv/bin/python experiments/subtitle/compare_translations.py` | **PASSED**: `20/20` FR+ZH cases (`riva: 28.95s, 8017.8 MiB`, `swiftu: 23.677s`) |
| Profile Comparison (FR `00:00–06:15`) | `.venv/bin/python experiments/subtitle/run_live_subtitles.py --profile-comparison` | **PASSED**: `off (0 edits, 0s)`, `conservative (13, 44.618s)`, `balanced (10, 46.195s)`, `expressive (29, 62.165s)`, `0` batch errors |
| Full FR Episode + Orphan Repair | `.venv/bin/python experiments/subtitle/run_live_subtitles.py --full-fr-episode` | **PASSED**: `221` utts (`48` refinements, `4` merges, `0` orphans) -> `286` cues (`281` ASS displayed, `21` Top-lane, `5` overflow, `0` unrenderable) |
| Mandarin Live Subtitle Samples | `.venv/bin/python experiments/subtitle/run_live_subtitles.py --mandarin-samples` | **PASSED**: all 4 contracts -> `.en.srt` & `.en.ass`, `0` errors, `0` unrenderable cues |
| CLI End-to-End & `--re-render` | `.venv/bin/nvidia-transcribe generate-subtitles ... --re-render` | **PASSED**: `status="OK"`, reused `translated.contract.json` cleanly in `0.035s` |
| FFmpeg Visual ASS Render | `.venv/bin/python experiments/subtitle/run_live_subtitles.py --render-screenshots` | **PASSED**: 6 PNG frames rendered & visually verified (`Main` + `Top` simultaneous separation) |

## 3. Workspace Integrity Checks

- `/workspace/pyannote`: Read-only reference unchanged (`HEAD == 2b40b1c0c17ce324178af765d8d2c79234870acd`).
- GPU VRAM clean after daemon teardown (`3337 MiB / 32607 MiB`, `1%` utilization).
