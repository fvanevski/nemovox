# CPU Verification Results (`20261122T035837Z`, SHA `ea64f4e33f25e2a71413fcba1655e1b513290f92`)

## 1. Python Import Identity (`exit=0`)
```text
COMMAND: PYTHONPATH=/workspace/nemovox-issue10-verify/src /workspace/nvidia-transcribe/.venv/bin/python -c "import sys, nvidia_transcribe, yaml; print(sys.executable, nvidia_transcribe.__file__, yaml.__version__)"
/workspace/nvidia-transcribe/.venv/bin/python /workspace/nemovox-issue10-verify/src/nvidia_transcribe/__init__.py 6.0.3
EXIT_CODE: 0
```

## 2. Python Compileall (`exit=0`)
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests experiments
EXIT_CODE: 0
```

## 3. Focused CPU Regressions (`exit=0`, `87 passed`)
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_resource_lifecycle.py tests/test_translation_review.py tests/test_studio_api.py tests/test_model_resources.py
87 passed, 1 warning in 3.68s
EXIT_CODE: 0
```

## 4. Full CPU Suite (`exit=1`, `1 failed, 205 passed`)
```text
COMMAND: /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
........................................................................ [ 34%]
..............................................F......................... [ 69%]
..............................................................           [100%]
=================================== FAILURES ===================================
________ test_single_clip_asr_role_default_and_legacy_explicit_endpoint ________

monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7fb4db46e6d0>
capsys = <_pytest.capture.CaptureFixture object at 0x7fb4da40b360>

    def test_single_clip_asr_role_default_and_legacy_explicit_endpoint(monkeypatch, capsys):
        for key in ("ASR_BASE_URL", "ASR_MODEL", "NEMOVOX_MODEL_RESOURCES"):
            monkeypatch.delenv(key, raising=False)
        constructed = []
    
        class FakeASR:
            def __init__(self, *, base_url, model, api_key=None):
                constructed.append((base_url, model))
    
            def transcribe(self, audio, config):
                return {"contract": {"raw_text": "synthetic"}, "raw_response": {}}
    
        monkeypatch.setattr(cli, "NemotronVLLMClient", FakeASR)
>       assert cli.main(["transcribe", "synthetic.wav"]) == 0, capsys.readouterr()
E       AssertionError: CaptureResult(out='', err='{
E           "status": "RESOURCE_BLOCKED",
E           "reason": "managed_unit_not_configured",
E           "error": "asr_local: no verified Nemovox-owned service for unavailable endpoint"
E         }
E         ')
E       assert 1 == 0
E        +  where 1 = <function main at 0x7fb4db355440>(['transcribe', 'synthetic.wav'])
E        +    where <function main at 0x7fb4db355440> = cli.main

tests/test_resource_integration.py:249: AssertionError
=========================== short test summary info ============================
FAILED tests/test_resource_integration.py::test_single_clip_asr_role_default_and_legacy_explicit_endpoint
1 failed, 205 passed, 1 warning in 9.29s
EXIT_CODE: 1
```
