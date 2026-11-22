# PR #11 / Issue #10 Findings (`20261122T035837Z`)

- **Result:** `FAIL` (Phase A CPU full-suite gate failed; Phase B/C host provisioning and live GPU verification halted per `staging/issue10/HOST_VERIFY.md`).
- **Blocking Finding (`BLOCKING`):**
  - **File/line:** `src/nvidia_transcribe/cli.py:247` vs `tests/test_resource_integration.py:236-260`.
  - **Observed failure:** `pytest -q -m "not gpu"` fails in `tests/test_resource_integration.py::test_single_clip_asr_role_default_and_legacy_explicit_endpoint` (`1 failed, 205 passed`, exit 1):
    `cli.main(["transcribe", "synthetic.wav"])` returns exit code `1` with stderr `{"status": "RESOURCE_BLOCKED", "reason": "managed_unit_not_configured", "error": "asr_local: no verified Nemovox-owned service for unavailable endpoint"}`.
  - **Root cause:** PR #11 wrapped `client.transcribe(args.audio, cfg)` inside `with ModelLifecycle().endpoint(asr_resource):` in `cli.py:247`, which executes unconditionally on the `transcribe` subcommand before `FakeASR.transcribe(...)` is called. In `test_single_clip_asr_role_default_and_legacy_explicit_endpoint`, only `cli.NemotronVLLMClient` is monkeypatched with `FakeASR`; `ModelLifecycle` is neither mocked in the test nor injectable/bypassed for fake clients, so it inspects live `127.0.0.1:8190/v1/models` and `systemctl --user show nemovox-asr.service` and fails on any CPU test environment without a live ASR server or provisioned unit.
  - **Note on why focused tests missed it:** `staging/issue10/HOST_VERIFY.md` omitted `tests/test_resource_integration.py` from the focused pytest command (`tests/test_resource_lifecycle.py tests/test_translation_review.py tests/test_studio_api.py tests/test_model_resources.py` passed 87/87), whereas `pytest -q -m "not gpu"` runs `tests/test_resource_integration.py` and immediately catches the regression.
- **Host Discovery Note for Next Round (Non-Blocking):**
  - Read-only inspection confirmed `/workspace/ninfer/build/rtx5090-release/apps/ninfer-serve` and the 27B NVFP4 `.ninfer` model exist on disk for `nemovox-text-review.service` once Central resolves Phase A.
- **GitHub Evidence Sufficiency:** Complete Phase A logs, exact failure trace, identity metadata, acceptance matrix, and friction report are published to `staging/issue10/evidence/20261122T035837Z/`. No private archive upload is needed.
