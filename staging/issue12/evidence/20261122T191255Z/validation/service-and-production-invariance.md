
# Production Studio, Main & GPU Isolation

- **Production `nemovox-studio.service` (`127.0.0.1:8765`):**
  - Before: `ActiveState=active`, `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea`, `/api/health` HTTP `200`
  - After: `ActiveState=active`, `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea`, `/api/health` HTTP `200`
- **`nemovox-asr.service` & `nemovox-text-review.service`:** Both `inactive (dead)` before and after verification.
- **GPU VRAM:** Before `NVIDIA GeForce RTX 5090, 3106 MiB, 29021 MiB, 32607 MiB` -> After `NVIDIA GeForce RTX 5090, 3105 MiB, 29022 MiB, 32607 MiB`.
- **Primary `/workspace/nvidia-transcribe` (`main`):** Unmodified at `332a99697edcc0b72fcc842168d5b1f57782e203`.
