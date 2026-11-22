# Findings & Next Remediation Priorities (`866aa734`)

1. **[BLOCKING TEST FAILURE — Assertion Drift at `tests/test_issue14_timing_boundaries.py:64`]** `boundary_evidence.py:245` emits `"cross_chunk_ctc_disagreement"`, causing `test_long_displaced_duplicate_from_adjacent_excerpts_is_not_rendered_twice` to fail against the outdated `"displaced_sequence_ctc_unverified"` assertion (`1 failed, 273 passed` non-GPU). Requires Central fix (update test expectation or restore constant) and full re-run.
2. **[VERIFIED BOUNDARY GAINS]** Speaker-local streams, `_consistent_runs` discontinuity split, NFKD normalization, `>= 2`-word `_right_continuation`, and exact overlap in `recover_extended_right_context` fixed `W04c`, eliminated `c12:73`/`c03:299` suffix regressions, rescued all `6` `W05a` words, fixed `W04b`, and removed `34` duplicate `W02c` tokens + `18` duplicate `W03a` tokens with `0` harm to controls (`C1`, `C2`, `1327/1329s`).
3. **[REMAINING GAPS FOR FOLLOW-ON ISSUE #14 ITERATIONS]**
   - **CTC Timing Audit (`timing_evidence.py`):** Still quarantines `0/2415` real episode words (`W01a/b/d`, `W02a`, `W03b`, `W04a/c`, `S04` unchanged) because real displacements shift 2+ words or flank >2.5s silences.
   - **Minor `W02c` Inflection Residuals (`3+4+3` tokens):** `sûr`/`sûre` and `venu`/`venue` splits left small fragments around the 34 deduplicated words; `W03a` cross-speaker `7`-token header (`c10:78..85`) stayed below `>= 8` threshold.
   - **Speaker Collapse & Diarization (`W02b`, `W03b`, `D4`):** Intentionally held constant in this timing/boundary slice (`23` speakers).
