# PR #11 / Issue #10 Acceptance Matrix (20261122T170806Z)

- **Tested source SHA:** `2bce0d8b3730c7e8784eef9b10cc616c273e86d2`
- **Base main SHA:** `8fee1e177db9742dbdb38c4a22bdd663c20ac80f`
- **Overall Result:** `PASS`

| Gate / Criterion | Status | Evidence Reference |
|---|---|---|
| Remote head authenticated & clean isolated worktree (`PYTHONPATH` import provenance) | PASS | `source/identity.json`, `source/remediation.diff` |
| Python compile (`compileall -q src tests experiments`) | PASS | `validation/cpu-tests.txt` (`exit 0`) |
| Focused CPU tests (`6` files, `126 passed`) | PASS | `validation/cpu-tests.txt` (`exit 0`) |
| Full non-GPU CPU suite (`227 passed`) | PASS | `validation/cpu-tests.txt` (`exit 0`) |
| Frontend tests (`5 files / 11 passed`) & Vite build (`web/dist`) | PASS | `validation/frontend.txt` (`exit 0`) |
| P1 admission invariants (`auto`/`cuda:0`/`cpu`, denied timeout, failure unwind, EN bypass, cache hit) | PASS | `validation/admission-trace.md` |
| Real canonical-lock contention (Case A timeout `0` spawns; Case B wait-then-admit `cuda:0`) | PASS | `validation/host-riva-auto.md` |
| Real uncached `device=auto` Riva CUDA (`device_used="cuda:0"`, MT preserved, bounded VRAM) | PASS | `validation/host-riva-auto.md` |
| Foreign-language Candidate Studio (`127.0.0.1:8767`) & Direct CLI parity E2E | PASS | `validation/foreign-studio-cli.md` |
| Service/GPU invariance & production Studio (`127.0.0.1:8765`) untouched | PASS | `validation/service-invariance.md` |
| F-02 stop-poll boundary classification | NON-BLOCKING (recorded) | `review/findings.md` |
| Pre-merge promotion / main write / new PR | NOT EXECUTED (forbidden) | `review/findings.md` |
