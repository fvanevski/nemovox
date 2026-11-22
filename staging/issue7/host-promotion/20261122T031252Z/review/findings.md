# Host Promotion & Live Studio Findings (`20261122T031252Z`)

- **Promotion Result:** `PASS` — `/workspace/nvidia-transcribe` promoted via clean fast-forward from `8205d7783230272c39a669c3f51d89bca50669e6` (`index-Bwa8rYL-.js`, pre-PR #8) to `8fee1e177db9742dbdb38c4a22bdd663c20ac80f` (`tree=2c733a3036443121cc4c1c7c2b7a09bb0d5680a5`, `index-B12FMBjD.js`). `nemovox-studio.service` restarted cleanly (`PID 1006972 -> 2922624`).
- **Blocking Findings:** `[]` (no application or deployment defects found).
- **Unverified / Blocked-by-Env Scenarios:**
  - `LIVE_RIVA_TEXT_E2E=BLOCKED_BY_ENV`: Host OpenAI-compatible text server (`127.0.0.1:8001`) and ASR server (`127.0.0.1:8190`) are not running; per `EXTERNAL_GPU_SERVICE_MUTATIONS=forbidden`, external inference services were not started or modified.
  - `SCREEN_READER_AUDIO=UNVERIFIED`: Headless Chromium verified all DOM ARIA attributes (`role="combobox"`, `aria-expanded`, `aria-controls`, `aria-activedescendant`, `aria-live="polite"`, focus restoration), while actual screen-reader TTS audio output was not captured.
- **GitHub Evidence Sufficiency:** All required receipts, HTTP checks, media track probes, interactive browser/DOM observations, E2E job verification, and final live readback are published directly to `staging/issue7/host-promotion/20261122T031252Z/`. No private archive upload is required.
