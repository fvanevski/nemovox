# Studio Fresh-Process & Runtime Health Verification

Raw sanitized trace: [`logs/studio_fresh_process_sanitized.txt`](logs/studio_fresh_process_sanitized.txt).

- **Service Transition:**
  - Prior PID: `4157910` (stopped cleanly while idle).
  - New PID: `1006972` (`InvocationID=8870cac30b7e449c9fb674d91ca1303e`).
  - Unit state: `enabled`, `active` (`Linger=yes`), unit file SHA-256 unchanged (`118c256bc94768b52c2f286c8b3cbdc9cf28423718413bdd6e9e3ba2107342d3`).
  - Command line: `/workspace/nvidia-transcribe/.venv-studio/bin/python -m uvicorn nvidia_transcribe.studio_api:app --host 127.0.0.1 --port 8765 --workers 1`.
  - Redacted env check (`/proc/1006972/environ`): `PYTHONPATH=/workspace/nvidia-transcribe/src`, `NEMOVOX_CLI_PYTHON=/workspace/nvidia-transcribe/.venv/bin/python`, `ASR_BASE_URL=<UNSET>`, `SWIFTU_BASE_URL=<UNSET>`.
- **Network & HTTP Probes:**
  - `ss -ltnp`: exclusive loopback listener `127.0.0.1:8765` (`pid=1006972`).
  - `GET /api/health` -> HTTP `200` (`{"status":"ok","backend":"local","ffmpeg":true,"ytdlp":true,...}`).
  - `GET /` -> HTTP `200`; `/assets/index-Bwa8rYL-.js` -> HTTP `200`; `/assets/index-DJ2GBhMf.css` -> HTTP `200`.
  - `GET /api/jobs` -> HTTP `200` (`1` preserved job `9be92af3ba144e6eb984ee43af0807de`, `state=complete`).
  - `GET /api/jobs/9be92af3ba144e6eb984ee43af0807de`, `/artifacts`, `/preview`, `/logs` -> all HTTP `200` with preserved smoke artifacts.
- **Unrelated Services & Cleanup:**
  - `local_embedding.service` remained `active` (`MainPID=25721`).
  - Zero stale CLI or Qwen worker child processes; RTX 5090 `3052 MiB` used / `29075 MiB` free.
