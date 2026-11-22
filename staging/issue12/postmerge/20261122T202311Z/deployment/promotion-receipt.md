
# Phase A4–A5 Promotion & Fresh-Process Activation Receipt

1. **Idle recheck & clean stop:** Verified 0 active/queued jobs (`JOBS_IDLE_CONFIRMED`); `systemctl --user stop nemovox-studio.service` exited PID `4180532` cleanly (`OLD_LISTENER_EXITED`).
2. **Fast-forward promotion:** `git -C /workspace/nvidia-transcribe merge --ff-only 8f6c03cbc70e84bf1970ba364207a02c9fe22b44` -> `HEAD=8f6c03cbc70e84bf1970ba364207a02c9fe22b44`, `tree=21ae1c6d1b682f1061ae36f6896cd0ac87db1580`.
3. **Atomic same-filesystem `web/dist` replacement:**
   - Old `web/dist/index.html` sha256: `d745cf10189c6c67f7bde1aef66e72360f5ecc4df8d9c71796aeae2eafeca38b` (`index-62Ntthch.js`)
   - New `web/dist/index.html` sha256: `05088b0a3e9e6735685f8d72c2d0c24164eb3c0dbb81a8ff3fc3ab365f932f54` (`index-BmJZkZRY.js`, `index-BIuIduzU.css`)
4. **Fresh service activation:** `systemctl --user start nemovox-studio.service`
   - New `MainPID=1121444`, new `InvocationID=6cd5cf36a6774b0693d37a196c088464`, `ExecMainStartTimestamp=Sun 2026-11-22 12:18:03 PST`, 1 Uvicorn worker listening on `127.0.0.1:8765`.
   - `/proc/1121444/cwd=/workspace/nvidia-transcribe`, `PYTHONPATH=/workspace/nvidia-transcribe/src`, `NEMOVOX_CLI_PYTHON=/workspace/nvidia-transcribe/.venv/bin/python`.
   - Live HTTP checks: `GET /api/health` (`200`), `GET /` serves `index-BmJZkZRY.js` & `index-BIuIduzU.css`, `GET /api/export-directories` (`200`), `GET /api/job-history` (`200`).
   - All 6 pre-promotion `job.json` hashes verified unchanged (`sha256sum -c pre_jobs.sha256`: `6/6 OK`).
