# Final Live Readback (`20261122T031252Z`)

- **`nemovox-studio.service`:** `ActiveState=active`, `SubState=running`, `MainPID=2922624`, `ExecMainStartTimestamp=Sat 2026-11-21 19:13:11 PST` (unchanged after all smoke tests).
- **HTTP `/`:** `200 OK` (serves `assets/index-B12FMBjD.js`).
- **HTTP `/api/health`:** `200 OK` (`{"status":"ok","backend":"local","ffmpeg":true,"ytdlp":true,...}`).
- **Repo checkout (`<REPO_ROOT>`):** `HEAD=8fee1e177db9742dbdb38c4a22bdd663c20ac80f`, `tree=2c733a3036443121cc4c1c7c2b7a09bb0d5680a5`, `git status --porcelain=v1` clean.
- **GPU status (`nvidia-smi`):** `NVIDIA GeForce RTX 5090, 3051 MiB, 32607 MiB` (pre-promotion `3050 MiB / 32607 MiB`; zero GPU model load/evict/restart performed).
