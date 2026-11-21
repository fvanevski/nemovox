# Nemovox PR #3 Codex UI Remediation — Qualification Handback

**Result:** `PASS` *(Vitest suite, TypeScript/Vite production build, and deterministic Chromium polling-order + remote/local media-gating browser tests passed on exact head `9ee4757ea24bbd68cb190f606b75a7f1af360ea1`)*

## Authoritative Branches, SHAs & Equivalence

- **Repository:** `https://github.com/fvanevski/nemovox`
- **PR / Issue:** `https://github.com/fvanevski/nemovox/pull/3`, `https://github.com/fvanevski/nemovox/issues/2`
- **Final implementation branch (`integration/gui-foundation-20261121`):** `9ee4757ea24bbd68cb190f606b75a7f1af360ea1`
  - Commits over prior qualified SHA `49fac0f4ae9c607078cbc71e34bc448c5aab318f`:
    - `91979bb` (`fix(studio): refresh completed job results and defer remote media preview`)
    - `9ee4757ea24bbd68cb190f606b75a7f1af360ea1` (`test(studio): prove final-cache transition with fresh working results retained`)
- **Changed files (`49fac0f4..9ee4757e`):**
  - `docs/gui/ISSUE2_CODEX_UI_REMEDIATION.md`
  - `web/src/App.tsx`
  - `web/src/job-display.ts`
  - `web/src/job-display.test.ts`
- **Unchanged backend identity (reused prior evidence):**
  - `src/` tree SHA: `d5447a9f845579826c694473fde17cfde930049f` (identical to `49fac0f4ae9c607078cbc71e34bc448c5aab318f`)
  - `tests/` tree SHA: `3d119fc42e2bab9c0159bcc491255a254013b667` (identical to `49fac0f4ae9c607078cbc71e34bc448c5aab318f`)
  - Python API/lifecycle/CPU evidence remains valid on `evidence/issue-2-studio-final-20261121` (`0f0a87097b781d8cddfc8b5469490cb7eab5cf53`), and initial FireDragon/B2/I1 screenshots remain on `evidence/issue-2-studio-20261121` (`e97e5dc0cf644bb9b614a933154ab559aabea466`).
- **Evidence branch (`evidence/issue-2-codex-ui-20261121`):** branched directly from `9ee4757ea24bbd68cb190f606b75a7f1af360ea1`.
- **Evidence archive (`staging/agent-evidence/issue-2-codex-ui/issue-2-codex-ui-evidence.tar.gz`):** `977,967` bytes (~0.93 MiB), SHA-256 `246f8e7c2dcc66a121a2bd2464af296ada24a8cecd41ec288d5f055694f7a530`.

## Summary of Controlled Qualification

1. **Final-cache key transition (`resultQueryKey`):** Controlled Playwright Chromium polling switched `GET /api/jobs/{id}` to `complete` between fast (1.5s) status polls and slow (4s/5s) result polls; both `/preview` and `/artifacts` refetched immediately on completion (`final` cache key) and rendered `"Codex regression cue visible."` plus download links without manual refresh.
2. **Deferred remote media & JASSUB gating (`mediaPreviewState`, `canShowVideo`):**
   - While `source.kind === "url"` and `source_path === null` (`pending`), the UI displayed `"Downloading remote media…"`, emitted zero `/api/jobs/{id}/media` requests, mounted zero `<video>` elements, and kept JASSUB uninitialized.
   - Once the status poll supplied `source_path`, `<video>` mounted and requested `/api/jobs/{id}/media` without page reload; upon `complete` + `.ass` artifact arrival, JASSUB attached (`ASS · libass`, 1 canvas).
   - Terminal `failed` remote job with `source_path === null` showed `"Browser video preview unavailable"` and `"The remote source was not downloaded..."` with zero `/media` requests.
   - Ordinary local `path` job rendered video, synchronized cue, and JASSUB `ASS · libass` overlay cleanly (`CONSOLE_ERRORS: []`).

## Command-Shape Friction

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events: []
  omitted_sensitive_values=no
```
