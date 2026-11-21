# PR #3: Second Codex artifact URL remediation, focused validation

## Exact source and scope

PR: https://github.com/fvanevski/nemovox/pull/3
Issue: https://github.com/fvanevski/nemovox/issues/2
Codex finding: https://github.com/fvanevski/nemovox/pull/3#discussion_r4654453576
Pre-correction head: `9ee4757ea24bbd68cb190f606b75a7f1af360ea1`

Central has authored the complete correction on the same PR implementation branch:
- `src/nvidia_transcribe/studio_api.py`: URL-encode artifact relative paths using `urllib.parse.quote(..., safe="/")`, keeping filesystem-facing artifact metadata unchanged.
- `tests/test_studio_api.py`: parameterized API test for SRT/ASS filenames with `#`, `?`, spaces, `%`, and Unicode; verify URL parsing, decoded path identity, HTTP status, and actual download bytes.

Do not redesign the route, frontend, CLI, job scheduler, ASR or model runtime. The current exact implementation SHA must be fetched/read from GitHub before host verification; the pre-correction SHA is not a validation target.

## One bounded host execution and handback

1. Read current `AGENTS.md`, this contract and current PR #3/issue #2. Fetch and authenticate the exact implementation branch head.
2. Run the focused artifact regression in the Studio environment (e.g. `.venv-studio/bin/pytest -v tests/test_studio_api.py -k artifact_links_url_encode_reserved_and_unicode_filenames`); **do not** alter or skip a failing test to make it green.
3. Run `.venv-studio/bin/pytest -q -m "not gpu"` and `.venv/bin/pytest -q -m "not gpu"` where both preserved environments remain available. The backend API change merits retesting Python; React source and previous Chromium polling/UI code are unchanged, so repeat `npm test && npm run build` only if relevant.
4. Exercise localhost Studio against synthetic job output SRT and ASS with reserved characters (not copyrighted media). Verify `GET /api/jobs/{id}/artifacts` returns percent-encoded URLs with preserved path separators, each link downloads exactly the original payload, and an ASS URL can be fetched by the browser/JASSUB route without fragment/query truncation. A minimal Playwright intercepted download/network check is sufficient; do not repeat the full GUI campaign.
5. Resolve only demonstrated in-scope integration defects, with root-cause fix and regression proof. Preserve existing GPU servers and host/media source state.
6. Publish one compact sanitized evidence archive and browsable handback/test/browser/manifest under an **evidence-only branch based directly on the final implementation SHA**. Include exact source/evidence SHA, focused/full Python test commands with exit codes, actual encoded URL examples, HTTP request response/byte match, artifact checksums, changed file diff, any limitations, and canonical `COMMAND_SHAPE_FRICTION`.
7. Stop with PASS/FAIL/BLOCKED after evidence publication. Do **not** create PRs, merge, close #2, request Codex again (two requests already used), or resolve Codex threads. Do not modify unrelated code or evidence branches.

Unchanged frontend/UI and earlier complete browser, lifecycle and GPU-excluded evidence can be reused only with exact Git tree/source identity proof. Live GPU inference remains UNVERIFIED and is not required for this narrow repair.
