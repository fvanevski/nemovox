
# Codex P2 Stage-Truthfulness Acceptance Matrix (`134c4f064b320777640726509ba0d03582f7c395`)

| Criterion | Status | Evidence |
|---|---|---|
| Exact source/tree/main identity & candidate imports | PASS | `source/identity.json`, `validation/cpu-tests.txt` |
| Compileall | PASS | `validation/cpu-tests.txt` |
| Six translation success/fallback cases (`openai_chat`, `swiftu` × `ok`, `request_error`, `missing_id`) | PASS | `validation/cpu-tests.txt`, `validation/stage-truth-table.md` |
| Studio failed-stage persistence on completed job | PASS | `validation/cpu-tests.txt`, `validation/stage-truth-table.md` |
| Studio tests (`.venv-studio`, 49 passed, 0 skipped) | PASS | `validation/cpu-tests.txt` |
| Full non-GPU suite (`.venv`, 257 passed, 0 skipped) | PASS | `validation/cpu-tests.txt` |
| Source text & error preservation on degraded translation | PASS | `validation/stage-truth-table.md` |
| Production Studio, main & model-unit invariance | PASS | `validation/service-and-production-invariance.md` |
