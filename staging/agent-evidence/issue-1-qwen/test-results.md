# CPU Compilation & Regression Test Results (`28655e0d6fe57d9363e576cd0795a24aae77a4a3`)

## Summary

| Check | Interpreter | Command | Result | Log |
| --- | --- | --- | --- | --- |
| Bytecode compilation | `.venv` (Python 3.13) | `PYTHONPATH=/workspace/nemovox-issue1/src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q /workspace/nemovox-issue1/src/nvidia_transcribe` | **PASS** (exit 0) | [`logs/compileall.txt`](logs/compileall.txt) |
| Full CPU test suite | `.venv` (Python 3.13) | `PYTHONPATH=/workspace/nemovox-issue1/src /workspace/nvidia-transcribe/.venv/bin/pytest -v -m "not gpu" /workspace/nemovox-issue1/tests` | **107 passed** in 8.46s (exit 0) | [`logs/pytest_venv_cpu.txt`](logs/pytest_venv_cpu.txt) |
| Full CPU test suite | `.venv-studio` | `PYTHONPATH=/workspace/nemovox-issue1/src /workspace/nvidia-transcribe/.venv-studio/bin/pytest -v -m "not gpu" /workspace/nemovox-issue1/tests` | **107 passed** in 8.35s (exit 0) | [`logs/pytest_venv_studio_cpu.txt`](logs/pytest_venv_studio_cpu.txt) |

## `qwen-asr` Compatibility Check

- Worker interpreter: `/workspace/nvidia-transcribe/outputs/modular-eval/.venv/bin/python` (Python 3.12, `qwen_asr==0.0.6`)
- Verified `Qwen3ForcedAligner.from_pretrained(pretrained_model_name_or_path, **kwargs)` accepts `dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True` and `align(audio=..., text=..., language=...)[0].items` yields `ForcedAlignItem(text, start_time, end_time)`.
- Observed `observed_dtype="torch.bfloat16"` via `aligner.model.parameters()` and `cuda_allocated_bytes=1841616896` (~1.72 GiB).
- Native-HF comparison: `native-HF comparison skipped (optional)`.

## Covered Regression Areas

1. **Default `off` path:** `test_default_off_never_loads_worker_or_touches_managed_services` confirms zero worker spawn, zero `native-alignment.checkpoint.json` write, and zero `managed_asr_handoff` call even when untimed OOV units exist.
2. **Two-phase ordering:** `test_qwen_pipeline_finishes_all_native_chunks_before_worker` confirms all native ASR and subinterval calls finish and `native-alignment.checkpoint.json` is written before any Qwen call.
3. **Selective admission & integrity:** `test_qwen_fallback.py` and `test_qwen_session_and_policy.py` verify `subtitle` vs `fine` thresholds, intra-utterance Sortformer pause merging (`<= 0.6s`) vs ambiguous multi-window rejection (`> 0.6s`), cross-speaker neighbor bounds with unique grouped diarization, mixed-speaker overlap rejection, zero-duration/non-monotonic/out-of-bounds rejection without clamping, and intact native CTC timings/verbatim text/speakers.
4. **Persistent worker & Linux parent-death cleanup:** Verified `model_load_count=1` across multiple requests, protocol ID mismatch rejection + group reap, timeout non-retry + group reap, CUDA-required rejection of misreported CPU, and real Linux `prctl(PR_SET_PDEATHSIG, SIGKILL)` termination when parent process is killed with `SIGKILL` (`test_linux_parent_death_sigkill_reaps_harmless_worker`).
5. **Managed ASR lifecycle state machine:** Verified `Studio` exclusion, `nemovox-*.service` naming gate, `NEMOVOX_MANAGED_ASR=1` + lease marker checks, live `ss` connection block, busy-lease block, pending-journal re-entry block even with free VRAM, changed-unit recovery refusal, normal/failure restoration, and Python main-thread `SIGTERM` unwind restoration.
