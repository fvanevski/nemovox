# Test Results (`bc5f067340d8bc84b7129bbdea7589ea401222c8`)

## 1. Focused Artifact URL Regression Tests

```bash
.venv-studio/bin/pytest -v tests/test_studio_api.py -k artifact_links_url_encode_reserved_and_unicode_filenames
```

- **Exit code:** `0`
- **Result:** `2 passed, 12 deselected in 0.21s`
  - `test_artifact_links_url_encode_reserved_and_unicode_filenames[.srt]` — `PASSED`
  - `test_artifact_links_url_encode_reserved_and_unicode_filenames[.ass]` — `PASSED`

(Full `tests/test_studio_api.py` also ran cleanly: `14 passed in 3.14s`, exit `0`.)

## 2. Complete CPU Suites

- **`.venv-studio/bin/pytest -q -m "not gpu"`:** **exit `0`**, `85 passed in 3.99s`
- **`.venv/bin/pytest -q -m "not gpu"`:** **exit `0`**, `85 passed in 4.02s` (1 `PytestRemovedIn10Warning` from `.venv`'s `anyio` plugin)

## 3. Unchanged Frontend Equivalence

- `web/` tree hash is unchanged (`d571b1080aa9ce8f45468d6f5ee40deeb16c46ab` at both `9ee4757ea24bbd68cb190f606b75a7f1af360ea1` and `bc5f067340d8bc84b7129bbdea7589ea401222c8`), reusing `evidence/issue-2-codex-ui-20261121` (`936ded2dd2ff280292795c74cc052c94c992a3d7`).
