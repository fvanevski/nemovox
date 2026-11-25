# Required-Review Failure-Boundary Qualification Evidence (`20261125T221400Z-67c055d2`)

Sanitized public evidence packet for issue #19 / PR #21 (`issue/19`) qualifying candidate commit `67c055d285779ad8f36206040c163d96f54122dc` (tree `8670b9aa167cb82e7e39d4d2fe437415e2bf8653`), two commits ahead of previous reviewed head `e9e9aeab6d6e554f258c46ead8ca1b0c76e7c459` via intermediate commit `a614e5f0d8c3c1ad815dac3df12615ace77c2273`.

**Result:** `PASS`

No private episode dialogue, raw model responses, private source excerpts, or credentials are included in this public packet. Full private replay contracts/deltas, rendered SRT/ASS outputs, raw render CLI logs, and complete diffs are retained in the host-local private archive:
`review-failure-boundaries-67c055d2.tar.gz` (`1319942` bytes, SHA-256 `380088d1b2a1c28c52bd21449a92ec9d3f69b3ff2553470dbbe53b0b58f6084f`).

## Summary

- **Static, CPU, Studio & Frontend Gates**:
  - `compileall -q src scripts tests`: `PASS` (exit 0)
  - `git diff --check e9e9aeab..HEAD`: `PASS` (exit 0)
  - Focused B1/B2 + replay regressions: `64 passed` (exit 0, `0.28s`)
  - Full CPU (`-m 'not gpu' tests`, including Studio): `575 passed` (exit 0, `12.11s`)
  - Studio (`.venv-studio`): `51 passed` (exit 0, `3.73s`)
  - `web/` (`1011e5c23e77821de13e8a1043735f792d9d92c7`) and `configs/` (`c10b6aaa015ef8f442bff87dd850189631a83aa3`) unchanged from `f45c91d5`; prior 14 frontend tests + build reused from `3e6f8d82`.
- **Failure Boundaries (B1 & B2)**:
  - **B1**: Terminal native comparative output exhaustion enters compact recovery; successful compact recovery retains normal comparison authority; repeated exhaustion explicitly defers complete primary MT without partial cache entries; transport, lifecycle, and non-output-stop failures remain errors.
  - **B2**: Failed required review persists contract, invalidates stale metadata, and fails; public CLI `--re-render` explicitly salvages with zero model calls, retained errors, and byte-identical contract/SRT/ASS repeatability; concurrent contract replacement and stale/mismatched/incomplete salvage receipts are rejected.
- **Retained Positive-Path Replay (`67c055d2`)**: `PASS` (exit 0), `249/249` cache hits, `0` API calls, `137` accepted review edits (`4` text changes + `1` status-only change vs original completion), `100%` exact per-row, delta, and summary correspondence with authenticated `review-anchors-f45c91d5.tar.gz` (53 verified manifest members).
- **Normal Offline Public Rerenders (`--re-render`)**: `PASS` across all 3 cases (`bounded`, `mandarin`, `french`), zero model calls, **100% byte-identical** `translated.contract.json`, SRT, and ASS hashes to `f45c91d5`.
- **Host & Resource State**: CPU-only, `0` GPU inference invocations, GPU memory `3048 MiB` pre / `3046 MiB` post, `nemovox-studio.service` undisturbed, owner checkout unchanged.
