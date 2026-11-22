
# Issue #12 / PR #13 — Requalification Evidence (`20261122T185434Z`)

- **Repository:** `fvanevski/nemovox`
- **Issue / PR:** `#12` / `#13` (draft)
- **Branch:** `issue/12-studio-output-workflow`
- **Tested Source SHA:** `a15b360d396d7b6e028d4551efdea0d356d7f8bb`
- **Tested Tree SHA:** `76aa38e97beb9ceae9ae5d2f829a263a0f8232a8`
- **Prior Evidence Carrier (historical):** `515cfeb1a4c530f6a9ed9cf82094f1213d02777f` (source `99a3abce9cf8e8e5d894892f91a899c46e71a508`)
- **Source Changes by Local Agent:** `none` (`a15b360d396d7b6e028d4551efdea0d356d7f8bb` tested unmodified)
- **Result:** `PASS`

## Summary
Single consolidated requalification pass at `a15b360d396d7b6e028d4551efdea0d356d7f8bb`: compileall (`exit=0`), focused CPU tests (`86` passed in `.venv`, `45` passed in `.venv-studio`), full non-GPU suite (`247` passed, `0` skipped), frontend Vitest (`14` passed) + TypeScript/Vite build (`exit=0`), fresh corrective browser/API regressions (contract-subtitles default 4-file export without false `partial`, retained `asr/recording.contract.json` & `asr/transcript.txt`, optional `recording_contract` export, zero-artifact `disabled` status with no directory write, log filter + filtered `Copy visible` + unfiltered `Copy full log` + reload persistence, `0` console/page errors), and one bounded real French FLEURS CUDA E2E (`f98406d658e44b28ac4e32757d3d1949`). Production Studio (`127.0.0.1:8765`, PID `4180532`) and model units remained unaffected.
