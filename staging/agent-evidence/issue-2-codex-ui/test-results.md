# Test & Build Results (`9ee4757ea24bbd68cb190f606b75a7f1af360ea1`)

## 1. Frontend Unit Tests & Production Build (`web/`)

Command:

```bash
cd web && npm test && npm run build
```

- **Exit code:** `0`
- **Vitest (`vitest run`):** `Test Files 3 passed (3)`, `Tests 8 passed (8)` in `154ms`
  - `src/source-selection.test.ts` (2 tests) — `PASSED`
  - `src/api.test.ts` (2 tests) — `PASSED`
  - `src/job-display.test.ts` (4 tests) — `PASSED`
- **TypeScript + Vite build (`tsc --noEmit && vite build`):** `2037 modules transformed`, built in `737ms`

## 2. Unchanged Backend Suite Equivalence

- `src/` tree (`d5447a9f845579826c694473fde17cfde930049f`) and `tests/` tree (`3d119fc42e2bab9c0159bcc491255a254013b667`) are bit-for-bit identical to qualified head `49fac0f4ae9c607078cbc71e34bc448c5aab318f` (`evidence/issue-2-studio-final-20261121`, `83 passed` in both `.venv-studio` and `.venv`, `12 passed` in `tests/test_studio_api.py`).
- Live GPU ASR/SwiftU endpoints remain `UNVERIFIED` (external GPU services not running).
