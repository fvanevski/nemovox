# Commands and material results

- Checked out exact candidate `1e0caf42bfa7c7959f70af0c641e9cdcab6e10ba` (`tree 0366372439c3fad3bed253e2f6621a564e0885df`) in the clean isolated detached worktree and verified PR #15 (`OPEN`, `isDraft: true`) and Codex review comment `4685247271`.
- Exported unmodified `src/` from `21bddc1787794ed9f3e9c4403b88a4cc428ee0a5` via `git archive` and executed the 4 parameterized `test_token_dedup_does_not_erase_separately_observed_boundary_repeats` tests plus `test_single_displaced_six_word_match_without_separate_support_still_deduplicates`: `4 failed, 1 passed` (`exit 1`).
- Executed `git diff --check origin/main...HEAD` (exit 0), `git diff --check 21bddc1787794ed9f3e9c4403b88a4cc428ee0a5..HEAD` (exit 0), `python -m compileall -q src tests` (exit 0), and CLI/Studio imports (exit 0).
- Ran `pytest -vv tests/test_issue14_timing_boundaries.py`: `27 passed` (`exit 0`).
- Ran affected 15-file timing/boundary/source/renderer/ASR-integrity pytest suite: `136 passed` (`exit 0`).
- Ran full non-GPU pytest suites in CLI `.venv` (`434 passed`, `exit 0`) and Studio `.venv-studio` (`434 passed`, `exit 0`).
- Replayed `reconcile_chunk_boundaries` (`21bddc1` vs `1e0caf4`) on reconstructed inputs from the retained French (`2,309` lexical observations, `36` ASR observations) and Mandarin (`886` lexical observations, `5` ASR observations) `source-evidence.v1.json` ledgers: exact 6/6 fidelity vs recorded `recording.contract.json` diagnostics and `merged_identical: true` / `diag_identical: true` for both recordings.
- Verified zero owned-service journal activity after the prior `ad76822` run (`latest ts_usec: 1795547203214143`), inactive `nemovox-asr.service`/`nemovox-text-review.service`, zero listeners on `8190`/`8001`, acquirable `model-lifecycle.lock`, and healthy production Studio `/api/health` (`ok`).
