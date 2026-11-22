# Phase A: CPU Compilation & Test Verification (`20261122T062100Z`)

- **Source SHA:** `a0ab446556ab983bdc04be24820a4f12f17df98c` (`origin/issue/10-auto-model-resources`)
- **Base Main SHA:** `8fee1e177db9742dbdb38c4a22bdd663c20ac80f`
- **Worktree:** `/workspace/nemovox-issue10-verify3` (clean detached HEAD)
- **Provenance class:** Direct command output recorded from `/workspace/nemovox-issue10-verify3` with `PYTHONPATH=/workspace/nemovox-issue10-verify3/src`.

## 1. Imported Package Identity
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -c "import sys, nvidia_transcribe, yaml; print(sys.executable, nvidia_transcribe.__file__, yaml.__version__)"
/workspace/nvidia-transcribe/.venv/bin/python /workspace/nemovox-issue10-verify3/src/nvidia_transcribe/__init__.py 6.0.3
EXIT_CODE: 0
```

## 2. Bytecode Compilation
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests experiments
EXIT_CODE: 0
```

## 3. Focused CPU Tests (FAIL)
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_resource_lifecycle.py tests/test_resource_integration.py tests/test_translation_review.py tests/test_studio_api.py tests/test_model_resources.py
....F................................................................... [ 63%]
.........................................                                [100%]
=================================== FAILURES ===================================
_____________ test_missing_explicit_asr_token_remains_fail_closed ______________

asr = ModelResource(name='asr_local', adapter='nemotron_ctc', model='nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4', e...th='/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4', worker_python=None, device=None, api_key_env=None)
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7f8292f0cf30>

    def test_missing_explicit_asr_token_remains_fail_closed(asr, monkeypatch):
        from dataclasses import replace
        from nvidia_transcribe.model_resources import ResourceConfigError
    
        selected = replace(asr, api_key_env="NEMOVOX_ABSENT_ASR_TOKEN_FOR_TEST")
        monkeypatch.delenv("NEMOVOX_ABSENT_ASR_TOKEN_FOR_TEST", raising=False)
        monkeypatch.setenv("VLLM_API_KEY", "legacy-must-not-override-explicit")
>       with pytest.raises(ResourceConfigError, match="NEMOVOX_ABSENT_ASR_TOKEN_FOR_TEST"):
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E       Failed: DID NOT RAISE ResourceConfigError

tests/test_resource_lifecycle.py:128: Failed
1 failed, 112 passed, 1 warning in 3.80s
EXIT_CODE: 1
```

## 4. Full Non-GPU CPU Suite (FAIL)
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
........................................................................ [ 32%]
......................................................F................. [ 65%]
........................................................................ [ 98%]
...                                                                      [100%]
=================================== FAILURES ===================================
_____________ test_missing_explicit_asr_token_remains_fail_closed ______________
...
E       Failed: DID NOT RAISE ResourceConfigError

tests/test_resource_lifecycle.py:128: Failed
1 failed, 218 passed, 1 warning in 9.31s
EXIT_CODE: 1
```

- All other 218 non-GPU tests (including the new `reconcile_idle_owned` / Studio orphan-recovery tests in `tests/test_resource_lifecycle.py` and `tests/test_studio_api.py`, plus `VLLM_API_KEY` fallback/precedence tests aside from fail-closed missing explicit token) passed; see `review/findings.md` for root cause.
