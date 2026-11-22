
# Production Studio, Main & Model-Unit Invariance

- **Production `nemovox-studio.service` (`127.0.0.1:8765`):**
  - Before & After: `ActiveState=active`, `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea`, `/api/health` HTTP `200`
- **`nemovox-asr.service` & `nemovox-text-review.service`:** Both `inactive (dead)` before and after verification; no GPU lease acquired or models started.
- **Primary `/workspace/nvidia-transcribe` (`main`):** Unmodified (`## main...origin/main`) at `332a99697edcc0b72fcc842168d5b1f57782e203`.
