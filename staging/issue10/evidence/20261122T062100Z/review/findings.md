# PR #11 / Issue #10 Findings (`20261122T062100Z`)

- **Result:** `FAIL` (Phase A CPU gate failed; Phase B/C host GPU verification halted per `staging/issue10/HOST_VERIFY.md`).
- **Blocking Finding (`BLOCKING`):**
  - **Files/lines:** `src/nvidia_transcribe/resource_lifecycle.py:134-152` vs `src/nvidia_transcribe/model_resources.py:19,101-109` vs `tests/test_resource_lifecycle.py:121-130`.
  - **Observed failure:** Both the focused CPU command (`1 failed, 112 passed in 3.80s`, exit `1`) and `pytest -q -m 'not gpu'` (`1 failed, 218 passed in 9.31s`, exit `1`) fail in `tests/test_resource_lifecycle.py::test_missing_explicit_asr_token_remains_fail_closed`:
    ```text
    E Failed: DID NOT RAISE ResourceConfigError
    tests/test_resource_lifecycle.py:128: Failed
    ```
  - **Root cause:** `ResourceConfigError` is a subclass of `ValueError` (`src/nvidia_transcribe/model_resources.py:19`: `class ResourceConfigError(ValueError):`). Inside `_models_healthy()` (`src/nvidia_transcribe/resource_lifecycle.py:134-152`), `credential = resource.credential()` is evaluated *inside* the broad `try: ... except (OSError, ValueError, TypeError, KeyError, urllib.error.URLError): return False` handler. When an explicit `api_key_env` is configured and absent from `os.environ`, `resource.credential()` raises `ResourceConfigError`, which is immediately caught by `except ValueError` and converted to `return False` instead of propagating `ResourceConfigError` to fail closed.
  - **Consequence:** Besides failing `test_missing_explicit_asr_token_remains_fail_closed`, a missing explicit `api_key_env` silently masquerades as an unhealthy-but-startable loopback unit (`startable` in `inspect()` / `systemctl start` attempt in `endpoint()`), rather than failing closed on configuration error.
- **Unverified / Halted Scope:** Live GPU re-qualification of F-01 (`VLLM_API_KEY` without explicit `api_key_env`), fresh French Studio/CLI E2E at `a0ab446`, and controlled Studio-supervised `SIGKILL` crash-recovery testing were not executed because Phase A failed. Prior `8d6f70f161c9100d4e64fca49da5139c4d109615` evidence (`20261122T054544Z`) remains historical baseline only.
- **GitHub Evidence Sufficiency:** Complete Phase A failure output, exact root-cause trace, and state metadata are published to `staging/issue10/evidence/20261122T062100Z/`. No private archive upload is needed.
