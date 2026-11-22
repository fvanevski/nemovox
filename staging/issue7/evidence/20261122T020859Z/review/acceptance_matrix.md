# PR #8 / Issue #7 Acceptance Matrix

| Check / Invariant | Outcome | Evidence |
|---|---|---|
| Exact source admission (`e824321d5680e61044cdbccc9564ca25d2b2f9e9`) | PASS | `source/identity.json`, clean detached worktree |
| Python bytecode compile (`compileall`) | PASS | `validation/compileall.txt` (exit 0) |
| Focused Python regressions (5 modules) | PASS | `validation/cpu_focused.txt` (76 passed, exit 0) |
| Full CPU pytest suite (`-m "not gpu"`) | PASS | `validation/cpu_full.txt` (177 passed, exit 0) |
| Frontend Vitest suite (`npm test`) | PASS | `validation/web_test.txt` (4 files, 10 passed, exit 0) |
| Frontend TypeScript + Vite build (`npm run build`) | PASS | `validation/web_build.txt` (exit 0) |
| Studio `Options` compatibility & frontend defaults | PASS | `validation/behavioral_probes.txt`, `web/src/config-options.test.ts`, `tests/test_studio_api.py` |
| Audio-only FFprobe indexing, path/index validation, URL/contract handling | PASS | `validation/audio_probe.txt` (single-track real media), `tests/test_studio_api.py`, `web/src/source-selection.test.ts` |
| Untouched ASR -> Riva -> single English review pass -> offline rerender | PASS | `validation/behavioral_probes.txt`, `tests/test_translation_review.py` |
| Multi-batch reviewer receives original `machine_translated_text` & `source_text`, never `refined_text` or edited English | PASS | `validation/behavioral_probes.txt`, `tests/test_translation_review.py` |
| Documentation & CLI/API/TS schema audit | PASS (minor non-blocking copy/packaging notes) | `review/doc_audit.json`, `validation/cli_help_and_host_readonly.txt` |
| Live Riva / ASR / text-model GPU inference | UNVERIFIED | Not executed; read-only config verified (`validation/cli_help_and_host_readonly.txt`) |
| Multi-track real media FFprobe | UNVERIFIED (single-track verified) | No multi-track media in inspected approved roots; synthetic multi-track unit test passed |
