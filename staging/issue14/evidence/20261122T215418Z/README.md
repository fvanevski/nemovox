# Nemovox Issue #14 Corrective A/B Verification (`20261122T215418Z`)

- **Repository / Draft PR / Issue:** `fvanevski/nemovox` · PR `#15` (`OPEN`, `isDraft=true`) · Issue `#14` (`OPEN`)
- **Authoritative `main`:** `8f6c03cbc70e84bf1970ba364207a02c9fe22b44` (`21ae1c6d1b682f1061ae36f6896cd0ac87db1580`)
- **Tested Source:** `866aa734a29e609bd9761e1d7365e32aefd2b2f7` (`d67a87cf75df1be0264101d38d8abae1b8b53f50`)
- **Handoff Carrier:** `05ac52dbb9c70af34e248c287b5ab4bb1a5fb200`
- **Prior Baseline / Failed Candidate:** `5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z` / `50b1408f`
- **Campaign Result:** `FAIL` (Phase B deterministic regression failure at `tests/test_issue14_timing_boundaries.py:64`: `timing_evidence_status == "displaced_sequence_ctc_unverified"` vs emitted `"cross_chunk_ctc_disagreement"`; `1 failed, 273 passed` non-GPU; `49 passed` Studio). Phase C cached-episode replay itself succeeded as `PARTIAL_A_B` (`W04c` & unintended suffix promotions fixed `PASS`, `W05a` `6/6` recovered `PASS`, `W04b` `PASS`, `W02c`/`W03a` `PARTIAL`, all controls `PASS`).

## Packet Contents
- `source-identity.json`, `validation/commands.md`, `validation/metrics.json`, `validation/cases.jsonl`, `validation/causal-assessment.md`, `validation/acoustic.md`, `review/acceptance-matrix.md`, `review/findings.md`, `COMMAND_SHAPE_FRICTION.txt`, `manifest.json`.
