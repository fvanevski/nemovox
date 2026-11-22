# Case-by-Case Causal Assessment (`50b1408f` Offline Replay)

Offline replay of all `16` raw episode chunks against exact candidate source `50b1408f73f3738f2f2ff5e59bbb421e4cb52e2f` (`50b1408f` src == `6986cf5a` src) cleanly executed without errors, preserved all negative controls (`C1`, `C2`, `1327/1329s`), and uncovered exact mechanistic gaps explaining why synthetic tests passed while real episode failure cases remained unresolved or regressed slightly.

## 1. CTC Timing Audit (`timing_evidence.py`: `0 / 2415` quarantined)
- **Observed:** `audited_units=2415`, `quarantined_units=0`, `large_gap_count=63` (logged review-only).
- **Affected Unchanged Cases (`W01a`, `W01b`, `W01d`, `W02a`, `W04a`, `W04c`, `S04` -> `FAIL`):**
  - `W01b` (`c00:47`, `c00:61`), `W01d` (`c00:89..90` vs `c00:100..108`), and `W04a` (`c13:78..81` vs `c13:100..102`) are multi-token shifts (`2`, `5`, `4+3` tokens) rather than single flanked tokens.
  - `W01a` (`c00:111`, `82.85s`) and `W04c` (`c13:108`, `1254.72s`) sit across `11-55s` same-speaker gaps (`abs(next_mid - prev_mid) > neighbor_consistency_s=2.5s`), so neither met the tight flanked-neighbor rule.

## 2. Displaced Sequence Reconciliation (`boundary_evidence.py`: `10` matches, `101` deduped, `48` promoted, `7` ambiguities)
- **What changed safely:** Standard near-time overlap deduplications cleanly migrated onto `sequence_evidence` (`c0->c1`, `c1->c2`, `c6->c7`, `c7->c8`, `c8->c9`, `c9->c10`, `c12->c13`), reducing utterance count `234 -> 233` solely by combining `c04` `Hm.` into the following sentence.
- **`W02c` (`c02 <-> c03`, `FAIL`):** In real multi-speaker audio, concurrent turns interleave in chronological token arrays `lt` and `rt`, splitting the `45`-token duplicated passage into `<= 3`-token runs plus one `4`-token run (`short_unanchored_duplicate_candidate`, `relative_ctc_shift_s=39.71`).
- **`W03a` (`c10 <-> c11`, `FAIL`):** Matched `21` tokens (`relative_ctc_shift_s=6.849`), but `SequenceMatcher` gap merging (`<= 2` indices) swept two `SPK_19` interjection tokens inside `chunk_011`'s span, triggering `repeated_or_mixed_voice_sequence`.
- **`W03c` (`c11 <-> c12`, `PARTIAL`):** Deliberately conservative short cross-speaker gate AND speaker interleaving inside `chunk_011` (`0` contiguous 4-gram blocks).
- **`W04b` (`c13 <-> c14` `1253-1255s`, `FAIL`):** Diacritic variation (`c13:106` vs `c14:6` acute accent on third token) left `.casefold()` tokens unequal, capping the match block at `size=3 < 4` so `selected_better_complete_turn` never evaluated `c14`'s complete phrase.
- **REGRESSIONS (`W04c` `c13 -> c14` `FAIL`, plus `c12 -> c13` & `c03 -> c04` suffix promotions):**
  - `W04c`: Match `c13:108..112` vs `c14:12..16` included early-drifted `c13:108` (`1254.61s`, `10s` separated from `c13:109` at `1264.61s`). `_median_mid` masked that outlier (`delta=0.197s`), promoted `c13:109..112`, and deleted `c14:12` at `1264.45s` — dropping the leading token from the `1264.36s` utterance while keeping the `1254.61s` orphan.
  - `c12 -> c13` (`S03` window `1180.0s`): `_continuation` promoted a garbled chunk-edge token (`c12:73`, `+1` unit `2051 -> 2052`) onto the clean `1176-1180s` utterance.
  - `c03 -> c04` (`367-370s`): Same-chunk-left win promoted a single spurious token (`c03:299`) omitted by `chunk_004`.

## 3. Extended Right-Context Recovery (`boundary_evidence.py`: `0` recovered, `12` ambiguities)
- **`W05a` (`c14:248..253` at `1352.01-1353.855s`, `FAIL`):** Even though `chunk_015` had zero tokens in `[1352.01, 1353.86]` and max lexical overlap was `2` tokens (`< 3`), `c15:23` (previous speaker turn end) stopped at `1351.95s`, inside the `group[0]['start'] - 0.30` (`1351.71s`) `nearby` padding, rejecting recovery as `right_overlap_covered_or_lexically_ambiguous`.

## 4. Negative Controls (`C1`, `C2`, `C_1327_1329` -> ALL `PASS`)
- `C1` (`utt-0111` vs `0112`, `29/29` tokens), `C2` (`utt-0176/0177`, genuine repetition + concurrent speaker), and `1327/1329s` repeated `1`-word interjections (`utt-0222/0223`) matched baseline byte-for-byte with zero over-deletion or false merging.

## Next Evidence / Engineering Decision
- **No GPU inference needed (`Phase D not_attempted`):** All five failure mechanisms (`per-speaker matching`, `diacritic normalization`, `mixed-voice gap bridging`, `monotonic token span check before _median_mid`, `nearby end-padding around preceding turn end`) are cleanly diagnosed on CPU from immutable raw archives.
