# PR #11 / Issue #10 Acceptance Matrix (`20261122T035837Z`)

| Phase / Check | Outcome | Evidence |
|---|---|---|
| Exact source admission (`ea64f4e33f25e2a71413fcba1655e1b513290f92`) | PASS | `source/identity.json`, detached clean worktree |
| Python import identity & `compileall` | PASS | `validation/cpu-tests.md` (exit 0) |
| Focused CPU tests (4 modules) | PASS | `validation/cpu-tests.md` (`87 passed`, exit 0) |
| Full CPU pytest suite (`-m "not gpu"`) | **FAIL** | `validation/cpu-tests.md` (`1 failed, 205 passed`, exit 1: `tests/test_resource_integration.py::test_single_clip_asr_role_default_and_legacy_explicit_endpoint`) |
| Frontend Vitest + TypeScript/Vite build | PASS | `validation/web-tests.md` (`5 files, 11 passed`, `npm run build` exit 0) |
| Phase B host unit provisioning | NOT_RUN | Gated behind Phase A pass (`HOST_VERIFY.md`) |
| Phase C live GPU / Studio / CLI E2E | NOT_RUN | Gated behind Phase A pass (`HOST_VERIFY.md`) |
