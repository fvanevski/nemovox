# PR #11 / Issue #10 Findings (`20261122T063100Z`)

- **Result:** `PASS` against exact source SHA `73760ee8f969a5927ac10cd8985de93f33f71e6a` (`origin/issue/10-auto-model-resources`).
- **Blocking Findings:** `[]` (none). Both prior issues — F-01 (`VLLM_API_KEY` readiness fallback without `api_key_env`) and the Phase A fail-closed regression (`ResourceConfigError` swallowed by `except ValueError` in `_models_healthy()`) — are verified fixed in CPU tests (`116/116` focused, `222/222` full) and live French Studio/CLI E2E.

## Non-Blocking Finding for Central
- **F-02 (`NON-BLOCKING`, timing boundary): `reconcile_idle_owned()` 30.0s stop-wait deadline vs systemd `TimeoutStopSec=30` during heavy vLLM initialization.**
  - **Observed behavior (`job_id=9387b8bb16c247e5bb3c21cbd2634010`):** When candidate CLI child `100970` was `SIGKILL`ed mid-safetensors load (`GPU=24689 MiB`, `22:43:38`), Studio correctly detected exit `-9`, marked the job `failed`, and `reconcile_idle_owned()` immediately issued `systemctl --user stop nemovox-asr.service`. Because `vllm` EngineCore was blocked inside 20 GiB checkpoint loading and did not exit on `SIGTERM`, systemd reached `TimeoutStopSec=30` at `22:44:08` and forcibly `SIGKILL`ed the unit cgroup (`MainPID=0`, `ActiveState=failed`, GPU freed to `3048 MiB`). Simultaneously, `reconcile_idle_owned()`'s `30.0s` polling deadline (`src/nvidia_transcribe/resource_lifecycle.py:336-343`) expired at `22:44:08`, raising `ResourceLifecycleError("orphan_recovery_stop_unverified", "asr_local")`, so Studio stderr logged `[nemovox] owned model recovery unavailable: ResourceLifecycleError` even though cleanup actually succeeded right at the 30s mark and `/api/resource-status` immediately returned `startable`.
  - **Comparison check:** When `nemovox-text-review.service` (`ninfer-serve`) had finished loading, fresh candidate Studio startup reconciliation stopped it cleanly in `<1s` with zero stderr errors.
  - **Suggested polish:** Extend the `_unit` stop-verification polling deadline slightly past systemd `TimeoutStopSec=30` (e.g., `35.0s`) so a `TimeoutStopSec=30` forced cgroup kill settles cleanly before `orphan_recovery_stop_unverified` is raised.

## GitHub Evidence Sufficiency
- Complete sanitized Phase A, B, and C results, acceptance matrix, findings, and GHDEV-10 friction record are published under `staging/issue10/evidence/20261122T063100Z/`. No private archive upload is needed.
