# PR #11 / Issue #10 Findings (20261122T170806Z)

- **Result:** `PASS` against exact source SHA `2bce0d8b3730c7e8784eef9b10cc616c273e86d2` (`origin/issue/10-auto-model-resources`).
- **Codex P1 Finding (`#discussion_r4658943795`):** **RESOLVED & VERIFIED.**
  - Uncached Riva `device="auto"` now enters `lifecycle.worker(resource_name=riva_resource_name)` before subprocess launch and before automatic device selection.
  - Confirmed via CPU admission/unwind mocks (`126/126` focused, `227/227` full), real canonical-lock contention probes (timeout denial with `0` worker spawns; wait-then-admit spawning `1` worker `0.165s` after lock release returning `device_used="cuda:0"`), and live French Studio + CLI E2E runs (`device_used="cuda:0"`, `8013.55–8014.75 MiB` torch peak, zero concurrent ASR/text-review residency).
- **Blocking Findings:** `none`.

## Non-Blocking Recorded Limitation (F-02)
- **F-02 (`NON-BLOCKING`, unchanged from `20261122T063100Z`):** `reconcile_idle_owned()` 30.0s stop-verification deadline vs systemd `TimeoutStopSec=30` during heavy vLLM weight-load SIGKILL (`src/nvidia_transcribe/resource_lifecycle.py:336-343`). No new F-02 trigger occurred during this run, and no F-02 change was requested or applied under the Issue #10 Riva auto-lease campaign boundary.

## Scope & Residuals
- **Not re-executed (by explicit instruction):** Prior accepted SIGKILL stress scenario and standalone CLI SIGKILL (unchanged since `73760ee8` / `20261122T063100Z`).
- **Evidence sufficiency:** All required public-safe artifacts are included in GitHub under `staging/issue10/evidence/20261122T170806Z/`; no private archive upload is required.
