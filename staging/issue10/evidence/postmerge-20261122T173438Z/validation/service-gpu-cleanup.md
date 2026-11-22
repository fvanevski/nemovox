# Phase 3: Final Host State, Cleanup & External Resource Invariance (postmerge-20261122T173438Z)

- **Production Studio (`nemovox-studio.service`, `127.0.0.1:8765`):**
  - Remains `ActiveState=active`, `SubState=running`, `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea` (unchanged across Phase 3).
  - Final `GET /api/health` -> HTTP `200`; `GET /api/resource-status` -> HTTP `200` (`asr: startable`, `translation_review: startable`).
- **Managed Units (`nemovox-asr.service`, `nemovox-text-review.service`):**
  - Final `ActiveState=inactive`, `SubState=dead`, `MainPID=0`.
- **GPU Baseline Restoration (`nvidia-smi`):**
  - Before Phase 3: `3049 MiB used / 29078 MiB free` (`32607 MiB` total).
  - After Phase 3: `3046 MiB used / 29081 MiB free` (`32607 MiB` total).
- **External Services Untouched:**
  - `kwin_wayland` (`PID=11480`), `text-embeddings-router` (`PID=1216176`, `1726 MiB` GPU), and `mcp-mux-2.0-dev` (`PID=1739`, `127.0.0.1:8012`) retained identical PIDs and remained untouched throughout.
- **Historical SIGKILL / F-02 Note:**
  - Accepted premerge crash-recovery evidence (`staging/issue10/evidence/20261122T063100Z/`) remains historical provenance; destructive production SIGKILL stress was not repeated per deployment handoff. F-02 remains nonblocking and unchanged.
