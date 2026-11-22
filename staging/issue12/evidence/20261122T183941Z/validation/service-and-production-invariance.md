
# Model Resource & Production Isolation Invariance

## Production Studio (`nemovox-studio.service`, `127.0.0.1:8765`)
- **Before Phase B:** `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea`, `/api/health` HTTP `200`
- **After Phase B:** `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea`, `/api/health` HTTP `200`
- **Before Phase C:** `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea`, `/api/health` HTTP `200`
- **After Phase C:** `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea`, `/api/health` HTTP `200`

## Managed GPU Units & VRAM Baseline
- **GPU Before Phase C:** `NVIDIA GeForce RTX 5090, 3103 MiB, 29024 MiB, 32607 MiB`
- **GPU After Phase C:** `NVIDIA GeForce RTX 5090, 3103 MiB, 29024 MiB, 32607 MiB`
- **`nemovox-asr.service`:** `inactive (dead)` before Phase B, after Phase B, before Phase C, and after Phase C.
- **`nemovox-text-review.service`:** `inactive (dead)` before Phase B, after Phase B, before Phase C, and after Phase C.
- Candidate Studio on `127.0.0.1:8769` (port `8766` was already occupied by host PID `12704`) used isolated `NEMOVOX_STUDIO_JOBS`, `NEMOVOX_STUDIO_UPLOADS`, and `NEMOVOX_EXPORT_ROOTS` and never modified `/workspace/nvidia-transcribe` or production Studio state.
