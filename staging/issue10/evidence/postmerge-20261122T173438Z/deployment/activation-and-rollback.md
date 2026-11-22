# Production Promotion, Activation & Rollback Disposition (postmerge-20261122T173438Z)

## 1. Promotion Execution
1. Stopped `nemovox-studio.service` (`MainPID=2922624` terminated cleanly).
2. Executed `git -C /workspace/nvidia-transcribe merge --ff-only 332a99697edcc0b72fcc842168d5b1f57782e203` (`Updating 8fee1e1..332a996`).
3. Verified exact deployed `HEAD=332a99697edcc0b72fcc842168d5b1f57782e203` and tree `6f1b41c26f8439d9eb1c5a4aff11b64e4c40afaa`.
4. Verified both `/workspace/nvidia-transcribe/.venv/bin/python` and `/workspace/nvidia-transcribe/.venv-studio/bin/python` import `nvidia_transcribe` from `/workspace/nvidia-transcribe/src/nvidia_transcribe/__init__.py`.
5. Rebuilt pinned frontend in `/workspace/nvidia-transcribe/web` (`npm run build`, exit `0`, `576ms`) producing `dist/assets/index-62Ntthch.js` and `dist/assets/index-CTKQH0KC.css` (replacing prior `index-B12FMBjD.js`).
6. Started `nemovox-studio.service` via `systemctl --user start nemovox-studio.service`. No edits to `nemovox-studio.service`, `nemovox-asr.service`, or `nemovox-text-review.service` were needed or made.

## 2. Rollback Disposition
- **Rollback executed:** `no` (deployment and all Phase 3 acceptance checks passed).
- **Private rollback readiness:** `yes` — retained at `~/.local/state/nemovox/rollback-issue10-20261122T1728Z/` until Central closes Issue #10.
