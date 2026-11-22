
# Requalification Acceptance Matrix (`a15b360d396d7b6e028d4551efdea0d356d7f8bb`)

| Criterion | Status | Evidence |
|---|---|---|
| Exact source SHA/tree & candidate import isolation | PASS | `source/identity.json`, `validation/cpu-tests.txt` |
| Compileall + focused CPU pytest (86 .venv, 45 .venv-studio) | PASS | `validation/cpu-tests.txt` |
| Full non-GPU pytest (247 passed, 0 skipped) | PASS | `validation/cpu-tests.txt` |
| Frontend Vitest (6 files, 14 tests) + tsc/Vite build | PASS | `validation/web-tests.txt` |
| Imported contract + subtitles default 4-file export (no false partial) | PASS | `validation/corrective-contract-and-empty-export.md` |
| Retained `asr/recording.contract.json` & `asr/transcript.txt` + optional contract export | PASS | `validation/corrective-contract-and-empty-export.md` |
| Zero selected artifacts -> terminal `disabled` & no external dir | PASS | `validation/corrective-contract-and-empty-export.md` |
| Log filter, filtered Copy visible, unfiltered Copy full & reload persistence | PASS | `validation/log-search-and-browser.md` |
| Isolated browser (0 page/console errors) | PASS | `validation/log-search-and-browser.md` |
| Short French CUDA E2E (ASR -> Riva CUDA -> Review -> SRT/ASS -> export) | PASS | `validation/french-gpu-studio-e2e.md` |
| Production Studio & GPU isolation | PASS | `validation/service-and-production-invariance.md` |
