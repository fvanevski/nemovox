# Phase B: Service Invariance, Cleanup & Historical Scope Reuse (20261122T170806Z)

## 1. Before / After Host & Production Studio Invariance
- **Production Studio (`nemovox-studio.service`, `127.0.0.1:8765`):**
  - Before: `ActiveState=active`, `SubState=running`, `UnitFileState=enabled`, `MainPID=2922624`, `GET /api/health` -> HTTP `200`.
  - After: `ActiveState=active`, `SubState=running`, `UnitFileState=enabled`, `MainPID=2922624`, `GET /api/health` -> HTTP `200`. Untouched throughout.
- **Managed Units (`nemovox-asr.service`, `nemovox-text-review.service`):**
  - Before: both `ActiveState=inactive`, `SubState=dead`, `UnitFileState=static`, `MainPID=0`.
  - After: both `ActiveState=inactive`, `SubState=dead`, `UnitFileState=static`, `MainPID=0`.
- **Candidate Studio (`127.0.0.1:8767`):** Stopped cleanly after verification; TCP port `8767` verified closed.
- **Unrelated Host GPU/Local Services:**
  - `/usr/bin/kwin_wayland` (`PID=11480`), `/usr/local/bin/text-embeddings-router` (`PID=1216176`, `1726 MiB` GPU), `mcp-mux-2.0-dev` (`127.0.0.1:8012`) remained running and undisturbed before, during, and after all runs.
- **GPU Baseline Restoration:** `3049 MiB used / 29078 MiB free` before -> `3049 MiB used / 29078 MiB free` after.
- **Repository Checkouts:** `/workspace/nvidia-transcribe` (`main` at `8fee1e1`) and `/workspace/nemovox-issue10-verify5` (`2bce0d8`) remained clean of source/config edits.

## 2. Historical-Scoped Evidence Reuse (`20261122T063100Z` at `73760ee8`)
- Central remediation `2bce0d8` modified only `src/nvidia_transcribe/translation.py` (+4/-1 lines) and `tests/test_translation.py` (+102 lines).
- Owned unit definitions (`nemovox-asr.service`, `nemovox-text-review.service`), `ModelLifecycle.reconcile_idle_owned()`, Studio SIGKILL supervision, active-client/marker-change guards, and negative ownership checks were unchanged since `73760ee8`, so their prior passing verification in `staging/issue10/evidence/20261122T063100Z/` was not duplicated per `RIVA_AUTO_LEASE_VERIFY.md`.
