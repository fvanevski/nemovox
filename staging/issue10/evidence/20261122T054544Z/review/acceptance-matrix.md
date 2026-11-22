# PR #11 / Issue #10 Acceptance Matrix (`20261122T054544Z`)

- **Tested source SHA:** `8d6f70f161c9100d4e64fca49da5139c4d109615`
- **Overall Result:** `PASS`

| Criterion | Status | Evidence Reference |
|---|---|---|
| Clean detached worktree & `PYTHONPATH` import identity | PASS | `source/identity.json`, `validation/cpu-tests.md` |
| Python compile (`compileall`) | PASS | `validation/cpu-tests.md` (`exit 0`) |
| Focused CPU tests (`5` files, `101 passed`) | PASS | `validation/cpu-tests.md` |
| Full non-GPU CPU suite (`207 passed`) | PASS | `validation/cpu-tests.md` |
| Frontend tests (`5 files / 11 passed`) & Vite build (`web/dist`) | PASS | `validation/web-tests.md` |
| Provisioned `nemovox-asr.service` & `nemovox-text-review.service` (un-enabled, 4 markers verified) | PASS | `promotion/unit-configuration.md` |
| Read-only `/api/resource-status` (`startable`, no model start) | PASS | `validation/live-stage-transitions.md` |
| Negative ownership & marker-mismatch probes | PASS | `validation/negative-ownership.md` |
| Live Studio French E2E (`asr_local` -> `riva` CUDA -> `text_local` -> SRT/ASS, UI & downloads) | PASS | `validation/live-stage-transitions.md`, `validation/studio-and-cli-e2e.md` |
| Direct CLI parity (`process-media` + `generate-subtitles`) | PASS | `validation/live-stage-transitions.md`, `validation/studio-and-cli-e2e.md` |
| GPU capacity & serialized handoff margins | PASS | `validation/gpu-usage.md` |
| No-model-start (`cached resume`, `--re-render`, English contract) | PASS | `validation/live-stage-transitions.md` |
| Bounded SIGTERM cancellation & cleanup + timeout cleanup + actionable Studio block error | PASS | `validation/cancellation-and-recovery.md`, `validation/live-stage-transitions.md` |
| Unowned/production services (`8765`, desktop, external GPU) untouched | PASS | `promotion/rollback.md` |
| SIGKILL mid-transaction recovery | UNVERIFIED | `validation/cancellation-and-recovery.md` |
| Pre-merge promotion to production `8765` | NOT EXECUTED (FORBIDDEN) | `HOST_VERIFY.md` constraint |
