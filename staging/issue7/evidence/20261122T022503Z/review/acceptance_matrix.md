# PR #8 / Issue #7 Requalification Acceptance Matrix (`4d560467fdcf97c0b01b54a9b7cf711e44317eaf`)

| Check / Criterion | Outcome | Evidence |
|---|---|---|
| Exact source admission (`4d560467fdcf97c0b01b54a9b7cf711e44317eaf`) | PASS | `source/identity.json`, detached clean worktree `/workspace/nemovox-issue7-requal` |
| Python import identity (`PYTHONPATH=/workspace/nemovox-issue7-requal/src`) | PASS | `validation/import_and_pyyaml.txt` (`/workspace/nemovox-issue7-requal/src/nvidia_transcribe/__init__.py`) |
| `PyYAML>=6.0` declared & satisfied | PASS | `validation/import_and_pyyaml.txt` & `validation/acceptance_probes.txt` (`6.0.3` in `.venv`, `6.0.4` in `.venv-studio`, YAML glossaries loaded via CLI & Studio) |
| `Options.condense_high_cps` defaults `false` & persisted `job.json` | PASS | `validation/acceptance_probes.txt` & `validation/cpu_focused.txt` (omitted -> `false`, no `--condense-high-cps`; explicit `true` -> `true` + `--condense-high-cps`) |
| Python bytecode compile (`compileall`) | PASS | `validation/compileall.txt` (exit 0) |
| Focused CPU regressions (5 test files) | PASS | `validation/cpu_focused.txt` (78 passed, exit 0) |
| Full CPU pytest suite (`-m "not gpu"`) | PASS | `validation/cpu_full.txt` (179 passed, exit 0) |
| Frontend Vitest (`npm test`) | PASS | `validation/web_test.txt` (4 files, 10 passed, exit 0) |
| Frontend TypeScript & Vite build (`npm run build`) | PASS | `validation/web_build.txt` (exit 0) |
| Updated GUI glossary hint & Studio stage label | PASS | `review/four_file_diff_audit.md` (`web/src/App.tsx:411`, `src/nvidia_transcribe/studio_api.py:492`) |
| Single-track FFprobe & synthetic multi-track test reconciliation | PREVIOUS_SOURCE_EVIDENCE | Unchanged media inspection code verified at prior `e824321d5680e61044cdbccc9564ca25d2b2f9e9` (`198f92cf5f3a3920bf54c9a81ea477cb2bb3bc36`); synthetic `test_audio_stream_parser_uses_audio_only_indices` re-ran and passed at `4d56046` |
| Live Riva / ASR / text-model GPU inference, real multi-track FFprobe, interactive browser JASSUB | UNVERIFIED | Not executed per requalification scope |
