# Deploy Receipt (`20261122T031252Z`)

## Authority & Pre-Promotion Discovery (HOST_ATTESTED)
- **Previous host state:** Clean `main` checkout at `8205d7783230272c39a669c3f51d89bca50669e6` (`tree=885db0684e5fa2a5f7a6031601ebe05b9df9f243`); `nemovox-studio.service` (`systemd --user`, `MainPID=1006972`, started `Sat 2026-11-21 14:23:55 PST`) serving `web/dist/assets/index-Bwa8rYL-.js` on `127.0.0.1:8765` without `/api/translation-backends` (`GET /api/translation-backends` returned HTTP 404).
- **Pre-promotion job check:** `outputs/studio/jobs` contained 1 completed historical job (`9be92af3ba144e6eb984ee43af0807de`); zero `queued` or `running` jobs.
- **External GPU state:** RTX 5090 `3050 MiB / 32607 MiB`; ports `8001` (text endpoint) and `8190` (ASR endpoint) were not listening. No external GPU service was started, stopped, or modified.

## Promotion Steps Executed (`exit=0`)
1. `git fetch origin --prune` -> verified `origin/main` = `8fee1e177db9742dbdb38c4a22bdd663c20ac80f` (`tree=2c733a3036443121cc4c1c7c2b7a09bb0d5680a5`).
2. `git merge --ff-only 8fee1e177db9742dbdb38c4a22bdd663c20ac80f` -> clean fast-forward (`8205d77..8fee1e1`), `git status --porcelain=v1` empty.
3. Verified module import identity in both `.venv` (`nvidia_transcribe.__file__` in `<REPO_ROOT>/src/nvidia_transcribe/__init__.py`, `PyYAML 6.0.3`) and `.venv-studio` (`<REPO_ROOT>/src/nvidia_transcribe/__init__.py`, `PyYAML 6.0.4`, `FastAPI 0.165.0`).
4. `cd web && npm run build` (`tsc --noEmit && vite build`) -> produced `web/dist/index.html` (`1543908b439b...`), `assets/index-B12FMBjD.js` (`de1eb65f929e...`), `assets/index-mYr5Cx9X.css` (`7bc75f3747c5...`).
5. `systemctl --user restart nemovox-studio.service` -> new `MainPID=2922624`, `ActiveState=active`, `SubState=running`, `ExecMainStartTimestamp=Sat 2026-11-21 19:13:11 PST`.
6. Post-promotion HTTP readback: `GET /` returned `index-B12FMBjD.js`, `GET /api/health` returned `{"status":"ok","backend":"local","ffmpeg":true,"ytdlp":true,...}`, `GET /api/translation-backends` returned `{"riva":{"configured":true,"device":"cuda:0","detail":"Dedicated Riva translation worker is configured."}}`.
