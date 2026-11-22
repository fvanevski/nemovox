# PR #11 / Issue #10 Acceptance Matrix (`20261122T063100Z`)

- **Tested source SHA:** `73760ee8f969a5927ac10cd8985de93f33f71e6a`
- **Overall Result:** `PASS`

| Criterion | Status | Evidence Reference |
|---|---|---|
| Clean detached worktree & `PYTHONPATH` import identity | PASS | `source/identity.json`, `validation/cpu-tests.md` |
| Python compile (`compileall`) | PASS | `validation/cpu-tests.md` (`exit 0`) |
| Focused CPU tests (`5` files, `116 passed` incl. credential fail-closed & recovery regressions) | PASS | `validation/cpu-tests.md` |
| Full non-GPU CPU suite (`222 passed`) | PASS | `validation/cpu-tests.md` |
| Frontend tests (`5 files / 11 passed`) & Vite build (`web/dist`) | PASS | `validation/web-tests.md` |
| Reused verified `nemovox-asr.service` & `nemovox-text-review.service` (static, 4 markers verified) | PASS | `promotion/unit-configuration.md` |
| Read-only `/api/resource-status` & negative ownership probes | PASS | `validation/live-stage-transitions.md`, `validation/negative-ownership.md` |
| Live F-01 `VLLM_API_KEY` ASR readiness without `api_key_env` (Studio E2E `e1847762`) | PASS | `validation/live-stage-transitions.md`, `validation/studio-and-cli-e2e.md` |
| Direct CLI parity without `api_key_env` + no-start checks (cached / `--re-render` / English) | PASS | `validation/live-stage-transitions.md`, `validation/studio-and-cli-e2e.md` |
| GPU capacity & serialized stage handoffs | PASS | `validation/gpu-usage.md` |
| Controlled `SIGKILL` of candidate CLI child (`9387b8bb`), failed job reporting, GPU restoration | PASS (with non-blocking F-02 timing note) | `validation/cancellation-and-recovery.md`, `review/findings.md` |
| Guard protections (active canonical lease, external TCP client, mismatched markers) | PASS | `validation/cancellation-and-recovery.md` |
| Fresh candidate Studio startup reconciliation of abandoned `nemovox-text-review.service` | PASS | `validation/cancellation-and-recovery.md` |
| Unowned/production services (`8765`, external GPU/desktop) untouched | PASS | `promotion/rollback.md` |
| Standalone CLI `SIGKILL` without supervising controller | UNVERIFIED / NOT GUARANTEED | `validation/cancellation-and-recovery.md`, `docs/model-resources/LIFECYCLE.md` |
| Pre-merge promotion to production `8765` | NOT EXECUTED (FORBIDDEN) | `HOST_VERIFY.md` |
