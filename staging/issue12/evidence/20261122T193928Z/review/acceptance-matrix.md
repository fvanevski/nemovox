
# Final Verification Acceptance Matrix (`51f59f65249ff4fb9705bc1f67638abb521d9062`)

| Criterion | Status | Evidence |
|---|---|---|
| Exact source/tree/main identity & candidate imports | PASS | `source/identity.json`, `validation/cpu-tests.txt` |
| Compileall | PASS | `validation/cpu-tests.txt` |
| Studio tests (.venv-studio, 48 passed, 0 skipped) | PASS | `validation/cpu-tests.txt` |
| Full non-GPU suite (.venv, 250 passed, 0 skipped) | PASS | `validation/cpu-tests.txt` |
| All 3 import tasks byte-identical archive & export | PASS | `validation/contract-byte-and-sidecar-check.md` |
| Original contract input path & 290-610s / >150s chronology | PASS | `validation/contract-byte-and-sidecar-check.md` |
| Original input files unmodified | PASS | `validation/contract-byte-and-sidecar-check.md` |
| Production Studio, main & model-unit invariance | PASS | `validation/service-and-production-invariance.md` |
