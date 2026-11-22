# Phase C: Controlled SIGKILL Crash Recovery, Startup Reconciliation & Guard Protections (`20261122T063100Z`)

## 1. Controlled `SIGKILL` of Candidate CLI Child Process (`job_id=9387b8bb16c247e5bb3c21cbd2634010`)
- **Target:** Only candidate Studio's CLI subprocess (`cli_pid=100970`, parent `studio_pid=20248` on `127.0.0.1:8767`) running `nvidia_transcribe.cli process-media` on `fr_fleurs_01_1857.wav`. Neither `nemovox-asr.service`, GPU drivers, nor production Studio (`2922624`) were signaled.
- **Pre-kill state (`dt=21.54s`):** `nemovox-asr.service` `ActiveState=active/running` (`MainPID=101033`, `InvocationID=889fa72f32df49c5832ce88fdc872c4f`), GPU VRAM `24689 MiB used / 7438 MiB free` (vLLM EngineCore `102334` mid-safetensors load).
- **Supervised Studio detection & cleanup:**
  - `JobRunner._command()` detected child death (`returncode=-9`), immediately invoked `_reconcile_idle_models()`, and issued `systemctl --user stop nemovox-asr.service` (`22:43:38 PST`).
  - Job transitioned accurately to `state="failed"`, `stage="Failed"`, `progress=8`, `error="Transcribing audio and speakers failed (exit -9); see execution.log"` (never `complete`).
  - `vllm` EngineCore (`102334`) was busy inside weight loading and did not exit within systemd `TimeoutStopSec=30`, so at `22:44:08 PST` systemd forcibly `SIGKILL`ed the unit's cgroup (`101033`, `102333`, `102334`), leaving `ActiveState=failed`, `MainPID=0`. (Note minor non-blocking 30.0s timing observation F-02 in `review/findings.md`.)
  - Post-recovery state (`dt=44.55s`): `MainPID=0`, `orphan_procs=[]`, GPU VRAM restored to `3048 MiB used / 29079 MiB free`, and `GET /api/resource-status` reported `{"asr": {"state": "startable", "ready": false, "managed": true, "startable": true}, ...}`.

## 2. Guard Protections & Fresh Candidate Studio Startup Reconciliation
Tested directly against a live running `nemovox-text-review.service` (`ninfer-serve`, `MainPID=112714`, `InvocationID=168d9a1511d5489daad7fbb3b4c40b79`, `GPU=30617 MiB used / 1510 MiB free`):
1. **Active canonical lease protection:** Held `fcntl.flock` on `~/.local/state/nemovox/model-lifecycle.lock`. `ModelLifecycle.reconcile_idle_owned()` returned `{"status": "active_model_stage"}` non-blockingly and left `nemovox-text-review.service` `active` with identical `MainPID=112714` and `InvocationID`.
2. **External client protection:** Kept an active TCP connection open on `127.0.0.1:8001`. `reconcile_idle_owned()` returned `{"asr_local": "inactive", "text_local": "active_clients"}` and preserved `MainPID=112714` / `InvocationID`.
3. **Mismatched unit marker protection:** Configured mismatched bindings (`asr_local -> nemovox-text-review.service`, `text_local -> nemovox-asr.service`). `reconcile_idle_owned()` returned `{"asr_local": "unowned_or_identity_mismatch", "text_local": "unowned_or_identity_mismatch"}` and preserved `MainPID=112714` / `InvocationID`.
4. **Fresh candidate Studio startup reconciliation:** With lock released and external client closed while `nemovox-text-review.service` remained genuinely abandoned and `active`, started fresh candidate Studio (`127.0.0.1:8767`, PID `113310`). Startup `_reconcile_idle_models()` automatically stopped `nemovox-text-review.service` (`ActiveState=inactive`, `SubState=dead`, `MainPID=0`), restored GPU memory to `3048 MiB used / 29079 MiB free`, served `/api/resource-status` as `startable`, and exited with zero errors in controller logs.

## 3. Standalone CLI Limitation
- Direct standalone `SIGKILL` of a CLI invocation without a supervising Studio process has no independent always-running watchdog; reclaim occurs when the next Studio controller starts/reconciles (matches `docs/model-resources/LIFECYCLE.md`). Live `SIGTERM` cancellation cleanup was additionally verified on prior baseline run `8d6f70f`.
