
# Issue #12 / PR #13 — Final Stage-Telemetry Verification (`20261122T191255Z`)

- **Repository:** `fvanevski/nemovox`
- **Issue / PR:** `#12` / `#13` (draft)
- **Branch:** `issue/12-studio-output-workflow`
- **Tested Source SHA:** `1eaf682262dea4b286db111fcb1b728c980d2e4c`
- **Tested Tree SHA:** `6ffee8186223354eabbb26445d5750e135da5694`
- **Prior Evidence Carrier (historical):** `e4adb206a5a815003b7285cf6dc14dbb85fb07dc`
- **Source Modifications:** `none`
- **Result:** `PASS`

## Summary
Verified `1eaf682262dea4b286db111fcb1b728c980d2e4c` (`_drain_stage_events` >256 KiB post-exit drain + `nemovox-progress/v1` schema enforcement) in isolated worktree `/workspace/nemovox-issue12-verify`:
- `compileall -q src tests experiments`: `exit=0`
- `.venv-studio` `test_completed_cli_drains_all_stage_events_after_large_final_log`: `1 passed in 0.29s`
- `.venv-studio` Studio tests (`test_studio_exports.py`, `test_studio_export_api.py`, `test_studio_api.py`): `46 passed in 3.50s`
- `.venv` full non-GPU suite (`pytest -q -m 'not gpu'`): `248 passed, 0 skipped, 1 warning in 9.35s`
- `web/` `npm test`: `6 files, 14 passed in 318ms`; `npm run build`: `exit=0` (`2040` modules)
- Isolated Studio (`127.0.0.1:8769`) + Chromium (`/usr/bin/chromium`): `0` page/console errors, contract-subtitles default 4-file export & reload persistence (`e3431ae564e844c5aaf130d1364d7a81`), empty-plan `disabled` (`1406c56774b749e0a816b247f8582771`), and short French FLEURS CUDA E2E (`231c643546df46f48c182001fe258e5d`).
- Production Studio (`nemovox-studio.service`, `127.0.0.1:8765`, PID `4180532`) and `/workspace/nvidia-transcribe` (`main`) unchanged.
