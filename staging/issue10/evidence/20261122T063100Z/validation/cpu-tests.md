# Phase A: CPU Compilation & Test Verification (`20261122T063100Z`)

- **Source SHA:** `73760ee8f969a5927ac10cd8985de93f33f71e6a` (`origin/issue/10-auto-model-resources`)
- **Base Main SHA:** `8fee1e177db9742dbdb38c4a22bdd663c20ac80f`
- **Worktree:** `/workspace/nemovox-issue10-verify4` (clean detached HEAD)
- **Provenance class:** Direct command output recorded from `/workspace/nemovox-issue10-verify4` using `PYTHONPATH=/workspace/nemovox-issue10-verify4/src`.

## 1. Imported Package Identity
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -c "import sys, nvidia_transcribe, yaml; print(sys.executable, nvidia_transcribe.__file__, yaml.__version__)"
/workspace/nvidia-transcribe/.venv/bin/python /workspace/nemovox-issue10-verify4/src/nvidia_transcribe/__init__.py 6.0.3
EXIT_CODE: 0
```

## 2. Bytecode Compilation
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests experiments
EXIT_CODE: 0
```

## 3. Focused CPU Tests
Includes `tests/test_resource_lifecycle.py`, `tests/test_resource_integration.py`, `tests/test_translation_review.py`, `tests/test_studio_api.py`, and `tests/test_model_resources.py`:
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_resource_lifecycle.py tests/test_resource_integration.py tests/test_translation_review.py tests/test_studio_api.py tests/test_model_resources.py
116 passed, 1 warning in 3.63s
EXIT_CODE: 0
```
- Verified newly added & previously failing credential-parity regressions (`test_missing_explicit_asr_token_remains_fail_closed`, `test_missing_explicit_token_blocks_before_host_service_access`, `test_readiness_network_error_still_reports_unavailable`, explicit precedence, and no text-model token leakage) and Studio orphan-recovery regressions all pass with zero systemd mutation.

## 4. Full Non-GPU CPU Suite
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
222 passed, 1 warning in 9.01s
EXIT_CODE: 0
```
- Single warning is `PytestRemovedIn10Warning` from `anyio/pytest_plugin.py:14`. Zero failures or skipped assertions.
