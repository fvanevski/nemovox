
# Production Studio & GPU Isolation Invariance

- **Production `nemovox-studio.service` (`127.0.0.1:8765`):**
  - Before: `ActiveState=active`, `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea`, `/api/health` HTTP `200`
  - After: `ActiveState=active`, `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea`, `/api/health` HTTP `200`
- **Managed model units (`nemovox-asr.service`, `nemovox-text-review.service`):**
  - Both `inactive (dead)` before requalification and cleanly returned to `inactive (dead)` after job `f98406d658e44b28ac4e32757d3d1949` completed.
- **GPU VRAM:**
  - Before: `NVIDIA GeForce RTX 5090, 3102 MiB, 29025 MiB, 32607 MiB`
  - After: `NVIDIA GeForce RTX 5090, 3105 MiB, 29022 MiB, 32607 MiB`
