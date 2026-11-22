
# Post-Merge Acceptance Matrix (`8f6c03cbc70e84bf1970ba364207a02c9fe22b44`)

| Gate | Status | Evidence |
|---|---|---|
| Phase A1–A2 admission, rollback backup & preimage hashes | PASS | `deployment/preflight.md`, `deployment/rollback-readiness.md` |
| Phase A3 staged compileall, 257 CPU, 49 Studio, 14 Vitest & Vite build | PASS | `validation/tests.txt` |
| Phase A4–A5 FF-only promotion, atomic dist swap & fresh Studio PID `1121444` | PASS | `deployment/promotion-receipt.md`, `source/source-and-tree-identity.json` |
| Preexisting 6 `job.json` & host config invariance | PASS | `deployment/promotion-receipt.md` |
| Phase B1 fresh Chromium browser against `http://127.0.0.1:8765` | PASS | `validation/fresh-studio-browser.md`, screenshots |
| Phase B2 live 2-chunk synthetic contract across 3 tasks (SHA-256 + sidecars) | PASS | `validation/fresh-studio-browser.md` |
| Phase B3 live French FLEURS ASR -> Riva CUDA -> review -> render -> 2 exports | PASS | `validation/real-french-job.md` |
| Phase B4 isolated negative, remote stub, legacy, cancel, chunk-log & >50 history | PASS | `validation/synthetic-and-failure-tests.md` |
| Model units & GPU VRAM invariance | PASS | `validation/real-french-job.md` |
