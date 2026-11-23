# Issue #14 Test Fixture Authority Adjudication

- **Run ID:** `20261123T000342Z`
- **Evaluated Source:** `fvanevski/nemovox@4c32acf659891fe999d2ae6e15f59ccb5750d92f`
- **Rule:** Preserve observable media/product invariants (`KEEP`), decouple tests asserting internal heuristic counters or magic constants (`REWRITE`), and flag unsafe heuristic assumptions blocking better behavior (`RETIRE`). No source or test files were modified in this evidence pass.

## 1. Boundary Matching & Reconciliation (`tests/test_issue14_timing_boundaries.py`, `tests/test_chunking_and_resume.py`)

| Test / Fixture | Classification | Observable Claim (Keep) vs Heuristic Coupling (Rewrite / Retire) |
|---|---|---|
| `test_ctc_audit_only_quarantines_impossible_or_flank_confirmed_outliers` | **KEEP (scope) / REWRITE (threshold specifics)** | **KEEP:** Out-of-excerpt CTC timestamps and flank-contradicted outliers must be quarantined without fabricating timestamps; long pauses must NOT be deleted. Avoid cementing magic `8.0s`/`2.5s` constants as immutable product truth. |
| `test_long_displaced_duplicate_from_adjacent_excerpts_is_not_rendered_twice` | **KEEP** | **KEEP:** Long displaced duplicate across adjacent observation windows (`W03a`-class) must appear once downstream. **REWRITE** if test asserts `sequence_deduplicated_units` internal counter. |
| `test_more_complete_right_excerpt_replaces_truncated_core_without_losing_tail` | **KEEP** | **KEEP:** When right observation captures complete phrase + tail cleanly where left cut mid-phrase, downstream preserves complete phrase once. |
| `test_same_six_word_phrase_by_distinct_speakers_far_apart_is_not_auto_deleted` | **KEEP** | **KEEP:** Same lexical phrase from distinct speakers / distant source times (`>=2s` separation) stays distinct. |
| `test_short_displaced_distinct_speaker_repetition_is_preserved_as_ambiguous` | **KEEP / REWRITE** | **KEEP:** Ambiguous short cross-observation events are not silently deleted (`unresolved` / `preserved`). **REWRITE** assertions tied to exact 4/6/8-token caps. |
| `test_actual_repeated_interjection_in_one_chunk_survives` | **KEEP** | **KEEP:** Genuine intra-observation repetition (`C1`/`C2`) survives downstream. |
| `test_mandarin_character_spans_are_not_deleted_or_extended_by_word_matching` | **REWRITE** | Original test relied on skipping CJK word matching rather than handling CJK character spans properly (`ARCHITECTURE_REDIRECTION.md` explicitly notes this). Rewrite as CJK character-span preservation invariant without banning proper event correspondence. |
| `test_missed_right_context_6_words_recovered_beyond_legacy_cap` / `test_exact_ended_preceding_turn_does_not_block_six_word_boundary_recovery` / `test_extended_recovery_does_not_override_alternate_right_speech` | **REWRITE** | **KEEP observable claim:** Missed boundary speech (e.g. W05a 5w/6w) covered by another clean observation must be recovered without overwriting alternate speech. **REWRITE oracle:** Retire coupling to legacy `>3 word` cap, `sequence_promoted_units` internal field, or fixed left-wins/right-wins heuristics. |
| `test_one_word_or_unknown_voice_cannot_trigger_sequence_based_deletion` / `test_displaced_matching_skips_interleaved_other_voice_without_bridge_deletion` / `test_w04c_discontinuous_offset_does_not_delete_correctly_timed_first_word` | **REWRITE** | **KEEP observable claim:** Never delete another voice's interleaved words or correctly timed adjacent words (`W04b`/`W04c`) due to discontinuous CTC jump. **REWRITE oracle:** Implement via `EventCorrespondence` grouping rather than 3-step token deletion guards. |
| `test_diacritic_variant_selects_complete_neighbor_phrase_without_noise` | **KEEP** | **KEEP:** Accented/complete phrase preserved cleanly without corrupted variant duplicate. |
| `test_same_long_statement_genuinely_repeated_twice_in_each_chunk_stays_distinct` / `test_clean_mid_core_29_token_control_remains_intact` | **KEEP** | **KEEP:** Non-overlapping genuine repeated phrases and clean mid-core speech remain intact (`CTRL_c1`, `CTRL_c2`). |
| `test_boundary_recovery_when_next_chunk_misses_initial_word` (`test_chunking_and_resume.py`) | **REWRITE** | Keep requirement not to lose boundary word; rewrite from chunk-core overwrite heuristic to observation correspondence. |

## 2. ASR Integrity & Pause Planning (`tests/test_issue14_asr_integrity.py`, `tests/test_issue14_pause_chunks.py`, `tests/test_issue14_speaker_review_guards.py`)

| Test / Fixture | Classification | Observable Claim (Keep) vs Heuristic Coupling (Rewrite / Retire) |
|---|---|---|
| `test_issue14_asr_integrity.py` (all 5 tests) | **KEEP** | **KEEP:** Reject `finish_reason=length` even with short cleaned fragment or valid CTC; allow genuine repeated interjections under `stop`; preserve raw failed response and forbid stale checkpoint from hiding `length` on resume. |
| `test_strong_pause_just_before_minimum_core_is_not_skipped` (`test_issue14_pause_chunks.py`) | **REWRITE** | **KEEP:** Planner should pick strong real pauses near target duration instead of forcing mid-speech cuts. **REWRITE:** Remove exact oracle assertion on `detected_pause_early_long` at `64.241427s` or `early_long` heuristic counters so a clean global optimizer can replace it. |
| Remaining 8 tests in `test_issue14_pause_chunks.py` | **KEEP** | **KEEP:** Complete disjoint media coverage, silence parser stream binding and error propagation, fixed policy backward compatibility, fingerprint invalidation on policy/boundary change, fail-closed invalid policy. |
| `test_issue14_speaker_review_guards.py` (all 5 tests) | **KEEP (scope) / REWRITE (0.05 margin oracle)** | **KEEP:** Borderline embedding similarity alone must not merge distinct speakers; translation review must never invert negation (`ne ... plus` etc.). Decouple rigid 0.05 margin constant if stronger turn correspondence is added later. |

## 3. Cross-Stage & Rendering Risks (`src/nvidia_transcribe/subtitles.py`, `tests/test_subtitles.py`, `tests/test_viewing_quality_regressions.py`)

| Component / Test | Classification | Risk & Adjudication |
|---|---|---|
| `subtitles.py::_deduplicate_concurrent_echo_cues` (L367-395) | **RETIRE / FIX CROSS-ISSUE INVARIANT (#14 -> #9)** | Drops any overlapping >=4-word cue with `SequenceMatcher >= 0.85` regardless of speaker/voice or source-event identity! Acoustic check on `CTRL_c2` (`1022-1028s`) and simultaneous/echo scenes confirms two speakers or two genuine source events MUST NOT be deleted downstream purely by English text similarity. |
| `test_c1_a3_translation_echo_dedup_and_mandarin_fragment_cleanup` (`test_viewing_quality_regressions.py`) | **RETIRE (particle deletion) / REWRITE** | Untranslated single-character Mandarin fragment cleanup (`'的'`, `'峰'`) and blind English-clause similarity deduplication compensate for upstream interleaved CTC fragmentation (observed directly in EP166 `525.16s`). Retire silent downstream deletion once upstream source-event correspondence fixes split characters cleanly; meanwhile require explicit source-event proof before suppression. |
| `test_real_vs_display_induced_overlap_and_ass_lanes` (`test_subtitles.py`) | **KEEP** | Keep preservation of distinct real concurrent speaker events without flattening or inventing timing. |
| Any hardcoded absolute local host paths or legacy `utt-XXXX` cross-version identity assertions | **RETIRE** | Portable CPU tests must not depend on host media paths or unstable sequential ordinal IDs across chunkers. |
