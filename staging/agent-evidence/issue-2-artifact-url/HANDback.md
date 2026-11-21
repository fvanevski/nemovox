# Nemovox PR #3 Artifact URL Remediation — Qualification Handback

**Result:** `PASS` *(focused SRT/ASS reserved/Unicode URL tests, complete CPU suites in `.venv-studio` and `.venv`, and localhost HTTP + Playwright Chromium verification passed on exact head `bc5f067340d8bc84b7129bbdea7589ea401222c8`)*

## Branches & Exact SHAs

- **Repository:** `https://github.com/fvanevski/nemovox`
- **PR / Issue / Finding:** `https://github.com/fvanevski/nemovox/pull/3`, `https://github.com/fvanevski/nemovox/issues/2`, `https://github.com/fvanevski/nemovox/pull/3#discussion_r4654453576`
- **Final implementation branch (`integration/gui-foundation-20261121`):** `bc5f067340d8bc84b7129bbdea7589ea401222c8`
- **New evidence branch (`evidence/issue-2-artifact-url-20261121`):** branched directly from `bc5f067340d8bc84b7129bbdea7589ea401222c8`
- **Evidence archive (`staging/agent-evidence/issue-2-artifact-url/issue-2-artifact-url-evidence.tar.gz`):** `19,484` bytes, SHA-256 `4262f575c48914462cd9038469b2424a2b84da111a59768b62bda282681a3868`

## Diff & Equivalence Summary

- Changed files (`9ee4757ea24bbd68cb190f606b75a7f1af360ea1..bc5f067340d8bc84b7129bbdea7589ea401222c8`):
  - `docs/gui/ISSUE2_CODEX_ARTIFACT_URL_REMEDIATION.md`
  - `src/nvidia_transcribe/studio_api.py` (`urllib.parse.quote(p.relative_to(root).as_posix(), safe="/")`)
  - `tests/test_studio_api.py` (`test_artifact_links_url_encode_reserved_and_unicode_filenames[.srt/.ass]`)
- **Unchanged frontend & prior lifecycle evidence:**
  - `web/` Git tree SHA is identical (`d571b1080aa9ce8f45468d6f5ee40deeb16c46ab`) between `9ee4757ea24bbd68cb190f606b75a7f1af360ea1` and `bc5f067340d8bc84b7129bbdea7589ea401222c8`. Frontend Vitest/Vite build and polling-order UI verification from `evidence/issue-2-codex-ui-20261121` (`936ded2dd2ff280292795c74cc052c94c992a3d7`) and lifecycle verification from `evidence/issue-2-studio-final-20261121` (`0f0a87097b781d8cddfc8b5469490cb7eab5cf53`) remain valid.

## Verification Summary

- `.venv-studio/bin/pytest -v tests/test_studio_api.py -k artifact_links_url_encode_reserved_and_unicode_filenames`: **2 passed in 0.21s** (exit `0`).
- `.venv-studio/bin/pytest -q -m "not gpu"`: **85 passed in 3.99s** (exit `0`).
- `.venv/bin/pytest -q -m "not gpu"`: **85 passed in 4.02s** (exit `0`, 1 `PytestRemovedIn10Warning` from `anyio`).
- Localhost HTTP + Chromium check (`Episode #1? 50% 日本語.en.srt` & `.ass`):
  - Encoded URLs preserved `/artifact/output/...` and encoded spaces, `#`, `?`, `%`, and UTF-8 bytes without query/fragment truncation.
  - Both direct HTTP GET and Playwright browser download clicks returned `HTTP 200` and byte-identical payloads (`56` bytes SRT SHA-256 `4220b69dd79b4f8e9b7410890da65ca9ce662b202c1bd32d308ee2873a7614ca`; `596` bytes ASS SHA-256 `6fc7c0d9d8c06994475a84f907cdd462433c30c492c780b5075692ba6d00ff5b`).
  - Browser JASSUB fetched the encoded `.ass` artifact (`HTTP 200`) and rendered `ASS · libass` with `0` console errors.

## Limitations

- Live GPU ASR/SwiftU endpoints remain `UNVERIFIED` (not running and not required for this API URL-encoding remediation).

## Command-Shape Friction

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events: []
  omitted_sensitive_values=no
```
