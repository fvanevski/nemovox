# Post-Merge Test & Build Results (`242c418f55cd2afb796ee2c4fa742a10ff04f7a3`)

All tests and builds below were executed freshly inside the detached `main` worktree (`/tmp/nemovox-postmerge-wt-20261121`, tree `122f8466db22098ba9427373af7e87ceb8421076`).

## 1. Focused Studio API Regressions (`.venv-studio`, `PYTHONPATH=$WT/src`)

```bash
PYTHONPATH=/tmp/nemovox-postmerge-wt-20261121/src /workspace/nvidia-transcribe/.venv-studio/bin/pytest -v tests/test_studio_api.py
```

- **Exit code:** `0`
- **Result:** `14 passed in 3.18s` (logs/pytest_studio_focused.log)

## 2. Full CPU Suites in Both Python Environments

- **`.venv-studio`:** `PYTHONPATH=.../.venv-studio/bin/pytest -q -m "not gpu"` -> **exit `0`**, **85 passed in 4.18s** (`logs/pytest_studio_cpu.log`)
- **`.venv`:** `PYTHONPATH=.../.venv/bin/pytest -q -m "not gpu"` -> **exit `0`**, **85 passed, 1 warning in 4.12s** (`logs/pytest_cli_cpu.log`; `PytestRemovedIn10Warning` from `anyio`)

## 3. Frontend Unit Tests & Fresh Production Build (`$WT/web/`)

```bash
cd /tmp/nemovox-postmerge-wt-20261121/web && npm test && npm run build
```

- **Exit code:** `0` (`logs/web_test_build.log`)
- **Vitest (`vitest run`):** `Test Files 3 passed (3)`, `Tests 8 passed (8)` in `153ms`
- **TypeScript + Vite build (`tsc --noEmit && vite build`):** `2037 modules transformed`, built cleanly in `511ms`

## 4. GPU & External Model Services Scope Note

- GPU tests (`-m gpu`) and live Nemotron ASR (`127.0.0.1:8190`) / SwiftU (`127.0.0.1:8001`) model inference remain **UNVERIFIED** (not started or modified per post-merge acceptance contract boundaries).
