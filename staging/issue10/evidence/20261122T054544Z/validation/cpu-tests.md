# Phase A: CPU Compilation & Test Verification (`20261122T054544Z`)

- **Source SHA:** `8d6f70f161c9100d4e64fca49da5139c4d109615` (`origin/issue/10-auto-model-resources`)
- **Base Main SHA:** `8fee1e177db9742dbdb38c4a22bdd663c20ac80f`
- **Worktree:** `/workspace/nemovox-issue10-verify2` (clean detached HEAD)
- **Provenance class:** Direct command output recorded from `/workspace/nemovox-issue10-verify2` using `PYTHONPATH=/workspace/nemovox-issue10-verify2/src`.

## 1. Imported Package Identity
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -c "import sys, nvidia_transcribe, yaml; print(sys.executable, nvidia_transcribe.__file__, yaml.__version__)"
/workspace/nvidia-transcribe/.venv/bin/python /workspace/nemovox-issue10-verify2/src/nvidia_transcribe/__init__.py 6.0.3
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
101 passed, 1 warning in 3.72s
EXIT_CODE: 0
```
- The single warning is `PytestRemovedIn10Warning` from `anyio/pytest_plugin.py:14`.

## 4. Full Non-GPU CPU Suite
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
207 passed, 1 warning in 9.06s
EXIT_CODE: 0
```
- Zero failures, zero skipped assertions. Confirms remediation of the prior `test_single_clip_asr_role_default_and_legacy_explicit_endpoint` failure recorded under `20261122T035837Z`.
