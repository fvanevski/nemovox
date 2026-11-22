# PR #8 / Issue #7 Codex Remediation Acceptance Matrix (`15101b37ae970e5c5ec6711343abe728701ecd30`)

| Criterion | Outcome | Evidence |
|---|---|---|
| Exact source admission (`15101b37ae970e5c5ec6711343abe728701ecd30`) | PASS | `source/identity.json`, detached clean worktree `/workspace/nemovox-issue7-codex-requal` |
| Python import identity (`PYTHONPATH=/workspace/nemovox-issue7-codex-requal/src`) | PASS | `validation/import_and_pyyaml.txt` (`/workspace/nemovox-issue7-codex-requal/src/nvidia_transcribe/__init__.py`, PyYAML `6.0.3`) |
| Python compile (`compileall -q src tests experiments`) | PASS | `validation/compileall.txt` (exit 0) |
| Focused CPU regressions (5 test modules) | PASS | `validation/cpu_focused.txt` (82 passed, exit 0) |
| Full CPU pytest suite (`-m "not gpu"`) | PASS | `validation/cpu_full.txt` (183 passed, exit 0) |
| Frontend Vitest (`npm test`) | PASS | `validation/web_test.txt` (5 test files, 11 passed, exit 0) |
| Frontend TypeScript + Vite build (`npm run build`) | PASS | `validation/web_build.txt` (exit 0) |
| English-source bypass (`bypassed_same_language`, `skipped_same_language`, SRT & offline rerender) | PASS | `tests/test_translation_review.py::test_english_source_skips_riva_worker_and_text_review`, `validation/codex_remediation_probes.txt` |
| CLI English contract with deliberately nonexistent Riva worker/checkpoint | PASS | `tests/test_translation_review.py::test_cli_riva_english_contract_does_not_require_local_worker`, `validation/codex_remediation_probes.txt` (auto-detected `"en"` & explicit `"eng"` bypass; `"fr"` rejects) |
| Studio English-to-English bypass & UI `englishOnly` handling | PASS | `tests/test_studio_api.py::test_english_studio_subtitles_do_not_require_riva_checkpoint`, `validation/codex_remediation_probes.txt`, `web/src/config-options.test.ts`, `web/src/App.tsx:390-402,472` |
| Malformed `.yaml`/`.json` glossary isolation (`null`, scalar, broken syntax, invalid list) | PASS | `tests/test_studio_api.py::test_invalid_glossaries_do_not_disable_valid_entries`, `validation/codex_remediation_probes.txt` (2 shipped glossaries intact) |
| `ExplainedSelect` combobox SSR semantics & static code audit | PASS (SSR + static); UNVERIFIED (interactive DOM) | `web/src/ExplainedSelect.test.tsx`, `web/src/ExplainedSelect.tsx:16-76`; no `jsdom`/`happy-dom`/`@testing-library` installed |
| Core preserved invariants (untouched ASR, audio-only tracks, Riva->review, cache/rerender, PyYAML, `condense_high_cps=false`) | PASS | `validation/cpu_full.txt`, `validation/web_test.txt` |
| Prior single-track FFprobe & live GPU/Riva/ASR/text-model/JASSUB | PREVIOUS_SOURCE / UNVERIFIED | Earlier single-track FFprobe observed on `e824321d5680e61044cdbccc9564ca25d2b2f9e9`; live inference and interactive browser DOM/JASSUB unverified |
