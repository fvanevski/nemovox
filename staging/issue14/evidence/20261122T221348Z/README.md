# Issue #14 Evidence Packet — Exact-Head Gate & Bounded CTC Timing Probes (`20261122T221348Z`)

- **Repository:** `fvanevski/nemovox` | **Issue:** `#14` (`OPEN`) | **PR:** `#15` (`OPEN`, `DRAFT`)
- **Branch:** `issue/14-longform-timing-boundaries`
- **Exact Tested Source:** `66275bdb512977182ed52fddbb022f2fbac2e670` (`tree=343075870237fe39c7ee813cbcbb1d6e5032769f`)
- **Handoff Carrier:** `b3aa3846682b3f1636fea500fb5a001c63d0ccd7` | **Main Base:** `8f6c03cbc70e84bf1970ba364207a02c9fe22b44`
- **Reused Replay Baseline (`src/` byte-identical):** `99dce3f8bfa1a93670b28473209dbf8bd924a54e:staging/issue14/evidence/20261122T215418Z/`

## Packet Contents
- `source-identity.json` — Git SHAs/trees, diff invariance, host/service state, original job/media digests.
- `validation/tests.md` — Exact-head `compileall`, focused (`32`), full non-GPU (`274`), and Studio (`49`) results.
- `validation/short-asr-probes.jsonl` — `11` bounded Nemotron NVFP4+CTC probes (`6` narrow primary `16–24s`, `2` narrow controls, `3` matched `44–45s` longer-context comparisons), sanitized with token hashes and timing offsets only.
- `validation/ctc-strategy-evidence.md` — Architectural comparison across `90s`, `16–24s`, and `44–45s` contexts.
- `validation/speaker-review.md` — Read-only archived attribution for `W02b/W03b/S05/C2` speakers and `S01/S02/W04c` Riva/Qwen review.
- `validation/metrics-corrections.json` — Corrected recovery counter arithmetic (`legacy = recovered_boundary_units - extended_recovered_units = 4`).
- `review/acceptance-matrix.md`, `review/findings.md`, `COMMAND_SHAPE_FRICTION.txt`, `manifest.json`.
