# CPU Compilation & Regression Test Results (`1702608f82808fca30fb3eee1b62954230e6da32`)

## Summary Table

| Check | Python Environment | Command (`PYTHONPATH=/workspace/nemovox-issue1/src`) | Observed Result | Exit | Log |
| --- | --- | --- | --- | ---: | --- |
| Package bytecode compile | `.venv` (Python 3.13.15) | `python -m compileall -q src/nvidia_transcribe` | `compileall_exit=0` | `0` | [`logs/compileall.txt`](logs/compileall.txt) |
| Focused Qwen / ownership / admission suite | `.venv` (Python 3.13.15) | `python -m pytest -v tests/test_qwen_session_and_policy.py tests/test_qwen_fallback.py` | **`43 passed, 1 warning in 4.66s`** | `0` | [`logs/focused_pytest.txt`](logs/focused_pytest.txt) |
| Full CPU test suite | `.venv` (Python 3.13.15) | `python -m pytest -q -m "not gpu"` | **`120 passed, 1 warning in 8.65s`** | `0` | [`logs/pytest_venv_cpu.txt`](logs/pytest_venv_cpu.txt) |
| Full CPU test suite | `.venv-studio` (Python 3.13) | `python -m pytest -q -m "not gpu"` | **`120 passed in 8.64s`** | `0` | [`logs/pytest_venv_studio_cpu.txt`](logs/pytest_venv_studio_cpu.txt) |

## Newly Added Codex Regression Tests Verified (`PASSED`)

All 11 newly added test cases (up from 32 focused / 109 total at `d5700b47a4e2f1abb60ebe99f53750c254de2696`) executed and passed verbatim in `tests/test_qwen_session_and_policy.py`:

1. `test_managed_owner_rejects_nonexact_or_duplicate_lease_markers[NEMOVOX_MANAGED_ASR=1 NEMOVOX_ASR_LEASE_PATH={lease}.other]` — **PASSED**
2. `test_managed_owner_rejects_nonexact_or_duplicate_lease_markers[NEMOVOX_MANAGED_ASR=10 NEMOVOX_ASR_LEASE_PATH={lease}]` — **PASSED**
3. `test_managed_owner_rejects_nonexact_or_duplicate_lease_markers[NEMOVOX_MANAGED_ASR=1 NEMOVOX_ASR_LEASE_PATH={lease} NEMOVOX_ASR_LEASE_PATH={lease}.other]` — **PASSED**
4. `test_managed_owner_rejects_nonexact_or_duplicate_lease_markers[NEMOVOX_MANAGED_ASR=1 NEMOVOX_ASR_LEASE_PATH={lease}.other NEMOVOX_ASR_LEASE_PATH={lease}]` — **PASSED**
5. `test_cooperative_marker_parser_accepts_exact_quoted_assignment` — **PASSED**
6. `test_managed_crash_recovery_rejects_prefix_lease_before_start` — **PASSED**
7. `test_partly_timed_span_covers_diarized_untimed_edge[True]` — **PASSED**
8. `test_partly_timed_span_covers_diarized_untimed_edge[False]` — **PASSED**
9. `test_partly_timed_untimed_edge_without_unique_diarization_is_skipped` — **PASSED**
10. `test_restoration_systemctl_transport_error_is_fatal_and_journal_stays_pending` — **PASSED**
11. `test_recording_propagates_typed_post_stop_failure` — **PASSED**

## Runtime Import Provenance

Verified before execution that `nvidia_transcribe` resolves strictly to `/workspace/nemovox-issue1/src/nvidia_transcribe/__init__.py` in both `.venv` and `.venv-studio`.
