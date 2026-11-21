# Service Deployment Verification & Test Results (`242c418f55cd2afb796ee2c4fa742a10ff04f7a3`)

## 1. Frontend Build & Tests from `main` Checkout (`/workspace/nvidia-transcribe/web`)

```bash
cd /workspace/nvidia-transcribe/web && npm test && npm run build
```

- **Exit code:** `0`
- **Vitest (`vitest run`):** `Test Files 3 passed (3)`, `Tests 8 passed (8)` in `398ms` (`source-selection.test.ts` 2, `api.test.ts` 2, `job-display.test.ts` 4)
- **TypeScript + Vite build (`tsc --noEmit && vite build`):** `2037 modules transformed`, built in `492ms`

## 2. Systemd User Unit Installation, Enablement & Activation

- `systemd-analyze verify --user` on the staged unit: parsed cleanly without unit errors (2 pre-existing warnings in unrelated user `environment.d` files ignored by systemd).
- `systemctl --user daemon-reload`: exit `0`.
- `systemctl --user enable --now nemovox-studio.service`: exit `0`; created `default.target.wants/nemovox-studio.service` symlink.
- `systemctl --user is-enabled` = `enabled`; `systemctl --user is-active` = `active`.
- `loginctl show-user "$USER" -p Linger` = `Linger=yes`.

## 3. Live Health, GUI Assets & Restart Persistence Proof

- **Before restart (`MainPID=4154778`):**
  - `GET http://127.0.0.1:8765/api/health` -> `200`, `{"status": "ok", "backend": "local", "ffmpeg": true, "ytdlp": true, ...}`
  - `GET http://127.0.0.1:8765/` -> `200` (`<title>Nemovox Studio</title>`)
  - `/assets/index-Bwa8rYL-.js` -> `200` (`442,992` bytes); `/assets/index-DJ2GBhMf.css` -> `200` (`24,855` bytes)
  - Submitted CPU-only synthetic contract transcription job `9be92af3ba144e6eb984ee43af0807de` -> reached `complete` (`2` cues, `recording.contract.json` + `transcript.txt`).
- **Idle restart (`systemctl --user restart nemovox-studio.service` -> exit `0`):**
  - New `MainPID=4157910` active on `127.0.0.1:8765` with zero leftover processes from `4154778`.
  - `GET http://127.0.0.1:8765/api/health` -> `200` (`status=ok`, `ffmpeg=true`, `ytdlp=true`).
  - `GET http://127.0.0.1:8765/` -> `200`.
  - Pre-restart job `9be92af3ba144e6eb984ee43af0807de` remained `complete` with `2` preview cues and `2` downloadable artifacts (`recording.contract.json`, `transcript.txt` verified via HTTP `200`).

## 4. Python Regression Equivalence & GPU Exclusion

- Exact source commit `242c418f55cd2afb796ee2c4fa742a10ff04f7a3` (tree `122f8466db22098ba9427373af7e87ceb8421076`) passed `tests/test_studio_api.py` (`14 passed`, exit `0`) and full CPU suites (`85 passed` in both `.venv-studio` and `.venv`) on the merged-main acceptance campaign (`evidence/issue-2-postmerge-20261121@7ee376c07fcac0d3125231dc16426564094a3332`).
- Live GPU ASR/SwiftU endpoints remain **UNVERIFIED** (no external model servers running or altered).
