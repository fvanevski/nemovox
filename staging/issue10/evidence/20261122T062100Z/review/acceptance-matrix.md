# PR #11 / Issue #10 Acceptance Matrix (`20261122T062100Z`)

- **Tested source SHA:** `a0ab446556ab983bdc04be24820a4f12f17df98c`
- **Overall Result:** `FAIL` (Phase A CPU gate failed)

| Criterion | Status | Evidence Reference |
|---|---|---|
| Clean detached worktree & `PYTHONPATH` import identity | PASS | `source/identity.json`, `validation/cpu-tests.md` |
| Python compile (`compileall`) | PASS | `validation/cpu-tests.md` (`exit 0`) |
| Focused CPU tests (`5` files) | FAIL (`1 failed, 112 passed`) | `validation/cpu-tests.md` (`test_missing_explicit_asr_token_remains_fail_closed`) |
| Full non-GPU CPU suite | FAIL (`1 failed, 218 passed`) | `validation/cpu-tests.md` (`test_missing_explicit_asr_token_remains_fail_closed`) |
| Frontend tests (`5 files / 11 passed`) & Vite build | PASS | `validation/web-tests.md` |
| Live host GPU / F-01 `VLLM_API_KEY` readiness / French E2E / CLI parity | NOT EXECUTED (halted at Phase A) | `validation/live-stage-transitions.md` |
| Supervised `SIGKILL` orphan recovery on host | NOT EXECUTED (halted at Phase A) | `validation/cancellation-and-recovery.md` |
| Unowned/production services (`8765`, GPU) untouched | PASS | `promotion/unit-configuration.md`, `promotion/rollback.md` |
