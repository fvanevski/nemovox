
# Codex-Remediation Acceptance Matrix (`be977b64c6a73adda895fb7c517bbcb2d1334794`)

| Criterion | Status | Evidence |
|---|---|---|
| Exact source/tree SHA & candidate import isolation | PASS | `source/identity.json`, `validation/cpu-tests.txt` |
| Compileall | PASS | `validation/cpu-tests.txt` |
| Focused sidecar tests (`test_imported_contract_processing_keeps_adjacent_reconstruction_sidecars[translation/subtitles]`) | FAIL | `validation/cpu-tests.txt` (failed at `tests/test_studio_export_api.py:207`) |
| Focused `.venv-studio` (46 passed, 2 failed) & full `.venv` non-GPU (248 passed, 2 failed) | FAIL | `validation/cpu-tests.txt` |
| Web Vitest (14) + TypeScript/Vite build | PASS | `validation/web-tests.txt` |
| Chunk window [290-610] > 150s ordering & original contract input path | PASS | `validation/sidecar-and-browser.md` |
| Job archive & selected exports (semantic JSON equal; raw byte formatting differs) | FAIL (byte-equality test) | `validation/sidecar-and-browser.md`, `review/findings.md` |
| Isolated Studio + Chromium stage/log persistence | PASS | `validation/sidecar-and-browser.md` |
| Production Studio, main & GPU invariance | PASS | `validation/service-and-production-invariance.md` |
