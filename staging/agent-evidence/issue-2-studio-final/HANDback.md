# Nemovox Issue #2 Final Lifecycle Review-Correction Qualification

**Result:** `PASS` *(final lifecycle review-corrections, focused/full CPU suites, frontend tests/build, and live multi-process HTTP qualification passed; live GPU inference remains `UNVERIFIED` as no Nemotron/SwiftU endpoints were running)*

## Authoritative SHAs & Branches

- **Repository:** `https://github.com/fvanevski/nemovox`
- **Issue:** `https://github.com/fvanevski/nemovox/issues/2`
- **Baseline `origin/main`:** `9b4c8b539c67f2f6266fef135a259e92f5477b19`
- **Final implementation branch (`integration/gui-foundation-20261121`):** `49fac0f4ae9c607078cbc71e34bc448c5aab318f`
  - Review-correction commits over prior qualified commit `2f0b16e890b2877ec0779818d09d837c79b83086`:
    - `fb0c02b` (`fix(studio): guard job reconciliation with exclusive lifespan ownership`)
    - `49fac0f4ae9c607078cbc71e34bc448c5aab318f` (`fix(studio): serialize cancellation against job completion`)
- **Final evidence branch (`evidence/issue-2-studio-final-20261121`):** branched directly from `49fac0f4ae9c607078cbc71e34bc448c5aab318f`
- **Prior evidence branch reused for unchanged frontend/B2/I1 screenshots (`evidence/issue-2-studio-20261121`):** `e97e5dc0cf644bb9b614a933154ab559aabea466` *(preserved untouched)*
- **Compact archive (`staging/agent-evidence/issue-2-studio-final/issue-2-studio-final-evidence.tar.gz`):** `18,647` bytes, SHA-256 `a7853414a55a2b110f86fe1bdc94cd5361ff48d4d446397b399fecc38f206b43`

## Scope of Review Corrections & Unchanged Source Identity

Only two tracked files changed between `2f0b16e890b2877ec0779818d09d837c79b83086` and `49fac0f4ae9c607078cbc71e34bc448c5aab318f`:

- `src/nvidia_transcribe/studio_api.py` (`+54 / -11`)
- `tests/test_studio_api.py` (`+68`)

Frontend tree equivalence:
- `git rev-parse 2f0b16e890b2877ec0779818d09d837c79b83086:web` = `b36450e7a5b0970a1af609bde38a6330846d3e1f`
- `git rev-parse 49fac0f4ae9c607078cbc71e34bc448c5aab318f:web` = `b36450e7a5b0970a1af609bde38a6330846d3e1f`
- `git diff 2f0b16e8..49fac0f4 -- web/` is empty; B2/I1/Chromium/FireDragon UI evidence in `evidence/issue-2-studio-20261121` (`e97e5dc0cf644bb9b614a933154ab559aabea466`) remains valid without duplicating its screenshot set.

## Host-Process Qualification Summary

1. **Authorized Startup Reconciliation:** Pre-seeded orphaned `running` job transitioned cleanly to `interrupted` (`Previous Studio process stopped`) once the primary Uvicorn server acquired `owned_job_root()` lock.
2. **Import & Secondary App Construction Safety:** Importing `nvidia_transcribe.studio_api` and calling `create_app()` in a separate Python host process while server #1 had an active `running` job left that job's `job.json` untouched (`running`).
3. **Secondary Server Lock Rejection:** Launching a second `uvicorn` process (`127.0.0.1:8769`) against the same `job_root` failed startup (`exit 3`, `"Studio job directory is already owned by another running server."`) without disturbing server #1 (`127.0.0.1:8765`).
4. **Serialized Cancellation vs Completion:**
   - Cancellation preceding completion transitioned the job to `cancelled` (never `complete`) and terminated the `SIGTERM`-resistant descendant (`/proc/<pid>` absent).
   - Completion preceding cancellation preserved `state=complete`, `stage=Complete` across subsequent `POST /api/jobs/{id}/cancel`.
5. **Graceful Shutdown & Descendant Cleanup:** Sending `SIGTERM` to Uvicorn triggered FastAPI `lifespan` (`runner.shutdown()`), cleanly terminating active `SIGTERM`-resistant descendants (`/proc/<pid>` absent) and recording `cancelled`.

## Remaining Limitations

- Live GPU ASR (`127.0.0.1:8190`) and SwiftU (`127.0.0.1:8001`) subtitle inference remain `UNVERIFIED` in this session (no live GPU servers running; RTX 5090 idle at `3230 MiB / 32607 MiB`).

## Command-Shape Friction

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events: []
  omitted_sensitive_values=no
```
