
# Final Bounded Verification Acceptance Matrix (`1eaf682262dea4b286db111fcb1b728c980d2e4c`)

| Criterion | Status | Evidence |
|---|---|---|
| Exact source/tree SHA & candidate import isolation | PASS | `source/identity.json`, `validation/cpu-tests.txt` |
| Large-final-log (>256 KiB) event drain & schema spoof rejection | PASS | `validation/cpu-tests.txt` |
| Compileall + focused `.venv-studio` (46) + full `.venv` non-GPU (248) | PASS | `validation/cpu-tests.txt` |
| Frontend Vitest (14) + TypeScript/Vite build | PASS | `validation/web-tests.txt` |
| Isolated candidate Studio + Chromium stage display & reload | PASS | `validation/isolated-studio-browser.md` |
| Bounded French CUDA smoke E2E (ASR -> Riva CUDA -> Review -> Render -> Export) | PASS | `validation/french-gpu-studio-e2e.md` |
| Production Studio, main checkout & GPU isolation | PASS | `validation/service-and-production-invariance.md` |
