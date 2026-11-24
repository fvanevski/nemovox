# Issue 14 Codex P1 remediation qualification (`1e0caf4`)

**CPU-only deterministic qualification and retained-evidence equivalence verified.** Tested candidate `1e0caf42bfa7c7959f70af0c641e9cdcab6e10ba`, tree `0366372439c3fad3bed253e2f6621a564e0885df` in the existing isolated detached worktree.

## Task 1 — Pre-fix reproduction & CPU qualification (PASS)

- **Isolated pre-fix reproduction against `21bddc1787794ed9f3e9c4403b88a4cc428ee0a5`:** Running the new candidate regressions against `21bddc1`'s unmodified `src/` failed all `4` parameterized `test_token_dedup_does_not_erase_separately_observed_boundary_repeats` cases (`[0.0-SPK_B]`, `[0.0-SPK_A]`, `[0.1-SPK_B]`, `[0.1-SPK_A]` erased the second 6-word occurrence), while `test_single_displaced_six_word_match_without_separate_support_still_deduplicates` passed (`4 failed, 1 passed`, exit `1`).
- **Corrected candidate `1e0caf4` verification:**
  - `tests/test_issue14_timing_boundaries.py`: **27 passed** (`exit 0`), confirming both 6-word occurrences, both original speaker hypotheses/timings, redundant context deduplication, non-promotion of discarded context copies, exact diagnostic counterpart/source index bindings (`left_source_indices`, `right_source_indices`, `left_counterpart_indices`, `right_counterpart_indices`, `resolution: "preserved_both"`), and preservation of the single displaced-duplicate control (`sequence_deduplicated_units: 6`).
  - Affected 15-file timing/boundary/source/renderer/ASR-integrity suite: **136 passed** (`exit 0`).
  - Full CLI `.venv` non-GPU suite: **434 passed** (`exit 0`).
  - Full Studio `.venv-studio` non-GPU suite: **434 passed** (`exit 0`).
  - `git diff --check origin/main...HEAD`, `git diff --check 21bddc1787794ed9f3e9c4403b88a4cc428ee0a5..HEAD`, `python -m compileall -q src tests`, and CLI/Studio module imports all exited `0`.

## Task 2 — Retained-evidence impact verification (EXACT EQUIVALENCE)

Reconstructed pre-deletion lexical units (`source_ordinal` order, text, timing hypotheses, `unit_type`, `join_no_space_prev`, `local_speaker_id`, `global_speaker_hypothesis`) and `diarization_activity_hypotheses` from the retained private `source-evidence.v1.json` ledgers, and replayed `reconcile_chunk_boundaries` on both `21bddc1` (pre-fix) and `1e0caf4` (corrected):

- **French episode (`fr`, `36` effective ASR chunks, `2,309` lexical observations):**
  - `21bddc1` replay matched the recorded `recording.contract.json` `boundary_reconciliation` 100% (`deduplicated_units: 200`, `recovered_boundary_units: 5`, `untimed_preserved_units: 96`, `11` sequence matches, `57` sequence-deduplicated units, exact `sequence_evidence.ambiguities` equality).
  - `1e0caf4` replay produced **identical** output (`merged_count_old == merged_count_new == 2005`, `merged_identical: true`, `diag_identical: true`, `11` sequence matches, `57` sequence-deduplicated units, `0` `separately_corroborated_occurrences` vetoes).
- **Mandarin Studio control (`zh`, `5` ASR chunks, `886` lexical observations):**
  - `21bddc1` replay matched the recorded `recording.contract.json` `boundary_reconciliation` 100% (`deduplicated_units: 63`, `recovered_boundary_units: 1`, `untimed_preserved_units: 202`, `0` sequence matches, `0` sequence-deduplicated units).
  - `1e0caf4` replay produced **identical** output (`merged_count_old == merged_count_new == 654`, `merged_identical: true`, `diag_identical: true`, `0` `separately_corroborated_occurrences` vetoes).
- **Projection limitations:** `source-evidence.v1.json` omits empty-text units (`ordinal_gaps: 0` in both recordings confirms none were present), maps `global_speaker_id == "SPK_UNK"` to `None` (reconstructed as `"SPK_UNK"`), projects `valid_timing` through `timing_status` (`disputed_rows: 0` confirms no disputed timed units), and omits alignment-only metadata not read by `reconcile_chunk_boundaries`. Exact 6/6 fidelity match against recorded diagnostics establishes causal equivalence.

## Prior live qualification linkage & zero inference confirmation

- **No model inference occurred:** Both `nemovox-asr.service` and `nemovox-text-review.service` remained `inactive` with zero new journal entries (latest timestamp `1795547203214143` matches the prior `ad76822` completion), ports `8190`/`8001` have `0` listeners, `model-lifecycle.lock` (`inode` `55650998`) is acquirable, and production `nemovox-studio.service` (`InvocationID=6cd5cf36a6774b0693d37a196c088464`, `MainPID=1121444`) remained healthy (`/api/health` `"status":"ok"`).
- **Retained live qualification:** Because accepted source observations and downstream inputs are provably unchanged, the prior French completion at `ad768220a3a2240889461f0d10994a9a532c0375` (carrier `21bddc1787794ed9f3e9c4403b88a4cc428ee0a5`, packet `20261124T190732Z`) and prior Mandarin Studio control at `eb5bc5243677ede15e7b04c836cfabf84d260955` (carrier `ece833d875b00f754cf2d429fad004c5e70eb14e`, packet `20261124T183928Z`) remain causally valid without repeating GPU runs.

[Metrics](metrics.json) · [Commands](commands.md) · [Command friction](COMMAND_SHAPE_FRICTION.txt) · [Checksums](SHA256SUMS)
