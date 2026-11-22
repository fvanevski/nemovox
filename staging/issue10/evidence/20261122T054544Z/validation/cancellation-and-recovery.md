# Phase C: Cancellation & Recovery Verification (`20261122T054544Z`)

## 1. Live SIGTERM Cancellation During Managed ASR (`job_id=d415c65ae0bc4979aa74c004486bbd5d`)
- **Target:** `fr_fleurs_02_1913.wav` submitted via `POST http://127.0.0.1:8767/api/jobs`.
- **Observed acquisition:** `nemovox-asr.service` transitioned to `ActiveState=active`, `SubState=running` (`MainPID=3987360`, `22:02:59 PST`).
- **Cancellation action:** Issued `POST http://127.0.0.1:8767/api/jobs/d415c65ae0bc4979aa74c004486bbd5d/cancel` after ~2s (`dt=0.51s` + `2.0s` hold). Immediate HTTP response: `state="running"`, `stage="Cancelling"`.
- **Observed unwinding & cleanup:**
  - `ModelLifecycle._lock()` installed `_on_sigterm` which raised `KeyboardInterrupt`, allowing `endpoint()` `finally:` to issue `systemctl --user stop nemovox-asr.service`.
  - Journal (`22:03:01` -> `22:03:02`): `Stopping Nemovox explicitly managed ASR model...` -> `Stopped Nemovox explicitly managed ASR model.` (`2.371s` wall time).
  - Final job state: `state="cancelled"`, `stage="Cancelled"`, `progress=8`, `error=null`.
  - Immediate post-cancel check (`i=0`): `nemovox-asr.service` `ActiveState=inactive`, `MainPID=0`; `nemovox-text-review.service` `ActiveState=inactive`, `MainPID=0`; `orphan_procs=[]`; GPU VRAM restored to `3048 MiB used / 29079 MiB free`; `/api/resource-status` restored to `{"asr":{"state":"startable",...},"translation_review":{"state":"startable",...}}`.

## 2. Timeout Failure Unwind (`job_id=4d0c3f21f3fc495bab3ea3babcf693af`)
- Confirmed `managed_start_timeout` after 300s also entered `finally:` and stopped `nemovox-asr.service` cleanly (`ActiveState=inactive`, `MainPID=0`, `Result=success`, `ExecMainStatus=0`), freeing VRAM back to `3046 MiB used / 29081 MiB free`.

## 3. Scope Limitation (`UNVERIFIED`)
- **SIGKILL (`kill -9`) recovery of an active child CLI/systemd process:** `UNVERIFIED` in live host testing (bounded non-destructive test scope did not forcibly `SIGKILL` mid-transaction; `studio_api.py` startup reconciliation for interrupted jobs is covered by CPU tests in `tests/test_studio_api.py`).
