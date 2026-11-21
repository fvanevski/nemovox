# CPU Compilation, Adjacency & Regression Results (`d5700b47a4e2f1abb60ebe99f53750c254de2696`)

## Test Runs

| Check | Environment | Result | Log |
| --- | --- | --- | --- |
| `python -m compileall -q src/nvidia_transcribe` | `.venv` (Python 3.13) | **PASS** (exit 0) | [`logs/compileall.txt`](logs/compileall.txt) |
| `pytest -v tests/test_qwen_session_and_policy.py tests/test_qwen_fallback.py` | `.venv` (Python 3.13) | **32 passed** in 4.53s (exit 0) | [`logs/focused_pytest.txt`](logs/focused_pytest.txt) |
| `pytest -q -m "not gpu"` (full suite) | `.venv` (Python 3.13) | **109 passed** in 8.49s (exit 0) | [`logs/pytest_venv_cpu.txt`](logs/pytest_venv_cpu.txt) |
| `pytest -q -m "not gpu"` (full suite) | `.venv-studio` | **109 passed** in 8.41s (exit 0) | [`logs/pytest_venv_studio_cpu.txt`](logs/pytest_venv_studio_cpu.txt) |

## New Adjacency Regressions Verified (`PASSED`)

- `tests/test_qwen_session_and_policy.py::test_unique_diar_window_respects_adjacent_native_timed_speech` — **PASSED**
- `tests/test_qwen_session_and_policy.py::test_neighbor_bounds_fail_closed_if_candidate_overlaps_native_timing` — **PASSED**

## Synthetic Adversarial Adjacency & Provenance Verification

Recorded in [`logs/adjacency_checks.json`](logs/adjacency_checks.json):

1. **Same-speaker single diarization `[0.2, 8.0]` (`"你好。世界啊。再见。"`):** Middle untimed span `[2, 5]` (`"世界啊。"`) clipped to `[1.65, 6.55]` (`>= 1.7 - 0.05` and `<= 6.5 + 0.05`), excluding both outer timed sentences.
2. **Cross-speaker neighbors (`"早。"` `spk_0` `0.8..2.4`, `"你好吗？"` `spk_1` untimed, `"行。"` `spk_2` `5.8..7.0`):** Middle span `[1, 4]` clipped to `[2.35, 5.85]`; after applying mock aligner timings, outer native units `[0]` and `[-1]` remained identical (`native_first_untouched=true`, `native_last_untouched=true`), and all unit texts and speakers remained unchanged.
3. **Conflicting neighbor bounds (`1.0..3.8` vs `3.9..5.0`):** Returned `candidates=[]` and `decisions[0].status == "skipped_conflicting_or_long_window"` (fail-closed, no worker call).

## Source Identity Reuse for Worker & Managed Controller

`git diff --exit-code 28655e0d6fe57d9363e576cd0795a24aae77a4a3..d5700b47a4e2f1abb60ebe99f53750c254de2696` confirmed zero changes in `_qwen_align_worker.py`, `qwen_aligner.py`, `gpu_handoff.py`, `recording.py`, `cli.py`, and `subtitle_pipeline.py`. Prior worker lifecycle, Linux `PR_SET_PDEATHSIG`, and mocked managed-ASR lifecycle/SIGTERM/journal-recovery evidence remains bound and valid; live host managed handoff remains **`MANAGED_GPU_BLOCKED`** (port `8190` closed, no `nemovox-*.service` ASR unit, no cooperative lease).
