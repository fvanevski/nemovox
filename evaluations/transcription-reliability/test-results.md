# Verification & Test Results

All checks recorded below ran in the active workspace environments after final production edits to `src/nvidia_transcribe/`. CPU unit tests (`-m "not gpu"`) remained strictly separated from live GPU inference runs.

---

## 1. Static Syntax & CPU Unit Tests

| Check | Environment | Command | Result | Scope Covered |
| :--- | :--- | :--- | :--- | :--- |
| **Bytecode Compilation** | `.venv` (`Python 3.13.15`) | `.venv/bin/python -m compileall -q src tests` | **PASS (exit `0`)** | All production and test modules |
| **Main Production CPU Suite** | `.venv` (`Python 3.13.15`) | `.venv/bin/pytest -v -m "not gpu"` | **`52 passed, 1 warning in 0.95s`** | Checkpoint validation, chunking & resume, client payload/errors, WER/CER metrics, parser/Unicode/offsets, Phase A regressions (`A1`–`A8`), Qwen fallback adapter & Issue #197 zero-duration rejection (`test_qwen_fallback.py`), experimental refinement contract (`test_refinement.py`), experimental scene diarization adapter (`test_scene_diarization.py`) |
| **Isolated Modular-Eval CPU Suite** | `outputs/modular-eval/.venv` (`Python 3.12.14`) | `PYTHONPATH=. outputs/modular-eval/.venv/bin/pytest -v experiments/modular/tests/test_modular_eval.py` | **`8 passed in 0.36s`** | Modular language normalization, alignment bounds validator, overlap DER metrics, attribution invariants |
| **Total Verified CPU Tests** | Both venvs | — | **`60 passed, 0 failed`** | Zero skipped or relaxed assertions |

### CPU Suite Breakdown by Test File (`52` Main + `8` Modular)

- `tests/test_checkpoint.py`: `4 passed`
- `tests/test_chunking_and_resume.py`: `4 passed`
- `tests/test_client.py`: `4 passed`
- `tests/test_metrics.py`: `2 passed`
- `tests/test_parser_and_contract.py`: `5 passed`
- `tests/test_phase_a_regressions.py`: `8 passed` (`A1`–`A8` regression tests)
- `tests/test_qwen_fallback.py`: `7 passed` (Phase B adapter, zero-duration rejection, selective preservation of CTC, mixed-overlap skip, CLI/config)
- `tests/test_recording_and_normalization.py`: `4 passed`
- `tests/test_refinement.py`: `3 passed` (Phase C localization, `raw_decode` repetition/contradiction handling, false-correction metrics)
- `tests/test_scene_diarization.py`: `4 passed` (Phase D scene partitioning, music FP detection, collapsed child voices advisory without merge)
- `tests/test_utterances_alignment_speakers.py`: `7 passed`
- `experiments/modular/tests/test_modular_eval.py`: `8 passed`

---

## 2. Live GPU Integration & Acceptance Runs (RTX 5090 32 GB)

| Run Group | Environment / Engine | Key Checks & Measured Outcomes |
| :--- | :--- | :--- |
| **vLLM Server Launch Check** | `.venv` + `modelopt_mixed` NVFP4 (`VLLM_CACHE_ROOT` + FlashInfer cache) | Verified `quantization=modelopt_mixed`, `kv_cache_dtype=fp8`, `ctc_timestamp_middleware`, `--reasoning-parser nemotron_v3`, FlashInfer autotune cache hit (`109 configs loaded`), `/v1/models` health `200` |
| **Phase B Subprocess Qwen Smoke** | Standalone `.venv` -> `outputs/modular-eval/.venv` (`cuda:0`) | `zh_fleurs_01_2006.wav`: `21/21` chars valid (`'钾' [1.36, 1.60]`), VRAM `3340 -> 3341 MiB` on exit |
| **Combined Acceptance Suite (`9` live runs)** | Live `vLLM` + co-resident `Qwen3-ForcedAligner-0.6B` subprocess | • `zh_fleurs_01_2006_off`: `0.3829s`, `CER=0.0`, `'钾'` preserved, `timing_status=failed`<br>• `zh_fleurs_01_2006_qwen`: `11.5695s`, `CER=0.0`, `21/21` `external_qwen_recovered` (`cuda:0`)<br>• `zh_fleurs_05_1805_qwen`: `0.9144s`, `"Scourthorpe四零三路..."` Latin spacing intact<br>• `zh_2spk_overlap_qwen`: `0.5694s`, `2/2` speakers & `49/49` units `ctc_aligned`<br>• `zh_multichunk_synthetic_off` vs `qwen`: `'钾'` recovered `[1.92, 2.08]` via Qwen, zero-duration `'桥'` rejected cleanly (`zero_duration_count=1`), `74` subinterval CTC units untouched<br>• `fr_child_voices_f1_qwen`: `4.6106s`, `6` utts, `2` speakers, `114/114` `ctc_aligned`<br>• `fr_300s_repaired_qwen`: `9.5031s`, `46` utts, `7` speakers, `532/532` `ctc_aligned`<br>• `fr_full_episode_repaired_90s`: `31.4241s`, `221` utts / `23` speakers / `2064` units (up from `213` / `22` / `2016`), `3` pre-solver ambiguous matches preserved unresolved |
| **Phase D Scene-Bounded Diarization** | `outputs/modular-eval/.venv-diar` (`Nemotron-3-Diarization`) | `8` bounded scenes (`60–120s` and `240–300s`, explicit cache reset per scene) + `10` GT clips x `4` modes (`offline`, `low_latency`, `very_low_latency`, `ultra_low_latency`); peak VRAM `427–448 MiB`; exact `3/3` match on `scene_zh_multichunk_70s` (`consistent_support: 3`) |
| **Phase C Refinement (`C0/C1/C2`)** | Live `vLLM` (`C0`/`C1`/`C2b`) + `/workspace/ninfer/swiftu` (`C2a`) | `39` French & Mandarin cases across `verified_natural` (`19`), `controlled_probe` (`8`), `episode_unverified_audit` (`12`). Documented Nemotron text-only `C2` degeneration stop + evaluated `swiftu` two-stage reasoning + Nemotron audio verification |

---

## 3. Workspace Integrity & Boundary Checks

- `/workspace/pyannote`: untouched at `2b40b1c0c17ce324178af765d8d2c79234870acd`.
- `/workspace/models/*`: all checkpoints untouched and loaded offline (`local_files_only=True`).
- GPU left completely idle (`~3.34 GiB used / 28.1 GiB free`) between sequential engine phases and upon completion.
