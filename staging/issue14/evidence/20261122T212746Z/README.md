# Nemovox Issue #14 First Cohesive Slice — Host Verification (`20261122T212746Z`)

> **Scope Notice:** This packet qualifies the initial timing/boundary commit `50b1408f73f3738f2f2ff5e59bbb421e4cb52e2f` (handoff HEAD `6986cf5a4efa33a08564afd8cc02ed5951af827f`) via deterministic CPU suites and offline replay of the `16` archived S04E01 episode chunks. **This is NOT full Issue #14 acceptance or deployment clearance.**

- **Repository / Draft PR / Issue:** `fvanevski/nemovox` · PR `#15` (draft) · Issue `#14` (`OPEN`)
- **Authoritative `main`:** `8f6c03cbc70e84bf1970ba364207a02c9fe22b44` (tree `21ae1c6d1b682f1061ae36f6896cd0ac87db1580`)
- **Tested Implementation Source:** `50b1408f73f3738f2f2ff5e59bbb421e4cb52e2f` (tree `a0731a2b3440118795397cb96fa5fb79dc0d87d7`)
- **Handoff HEAD:** `6986cf5a4efa33a08564afd8cc02ed5951af827f` (`src/` and `tests/` identical to `50b1408f73f3738f2f2ff5e59bbb421e4cb52e2f`)
- **Diagnostic Baseline:** `5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z`
- **Result:** `PARTIAL_A_B` (Phase B CPU/Studio suites `PASS`; Controls `C1`/`C2`/`1327-1329s` `PASS`; Phase C real-episode priority fixes `0/15` resolved + `W04c` `1264.36s` boundary regression observed; Phase D GPU `not needed`)

## Packet Index
- `source-identity.json` — Authenticated refs, hashes, isolated import path, read-only invariance receipt
- `validation/commands.md` — Executed compileall, focused (`27`), non-GPU (`269`), Studio (`49`) suites & replay command
- `validation/metrics.json` — Baseline vs candidate metrics on identical denominators
- `validation/cases.jsonl` — Case-by-case outcomes for `W01..W05`, `S03`, `S04`, `C1`, `C2`, `C_1327_1329` (no copyrighted dialogue)
- `validation/causal-assessment.md` — Mechanistic trace of why tests passed while cached episode cases stayed unresolved or regressed
- `validation/acoustic.md` — `not_attempted` GPU status + baseline cached-WAV anchor provenance
- `review/acceptance-matrix.md` — Consolidated gate table
- `review/findings.md` — Prioritized blocking regression & algorithmic gaps for Central remediation
- `COMMAND_SHAPE_FRICTION.txt` — Friction log (`encountered=no`)
- `manifest.json` — SHA-256 and byte manifest of all published members
