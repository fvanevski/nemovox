# Phase A Report: Eight Correctness Repairs & Regressions

All eight defects were repaired in commit `9c7a160` (`Repair eight chunked transcription correctness defects`) and verified with dedicated adversarial CPU unit tests in [tests/test_phase_a_regressions.py](file:///workspace/nvidia-transcribe/tests/test_phase_a_regressions.py) and live GPU comparison runs on RTX 5090.

---

## Summary of Repaired Invariants

| ID | Violated Invariant (Before) | Root-Cause Repair (After) | Verification Test / Live Evidence |
| :--- | :--- | :--- | :--- |
| **A1** | `_compatible_speaker` returned `True` whenever either unit had unknown speaker (`SPK_UNK_*` / `None`), allowing lexical collisions in boundary windows to silently delete simultaneous speech | `_speaker_evidence` returns `"same_known"` only when both IDs resolve identically; unknown speakers never match (`"unknown_speaker_cannot_prove_duplicate"` logged to `boundary_ambiguities`) and both observations survive | `test_a1_unknown_speaker_never_deduplicates_or_deletes_simultaneous_speech`, `test_a1_same_speaker_repeated_words_and_untimed_preservation` |
| **A2** | Subinterval CTC recovery could overwrite recognized proper nouns or insert hallucinated candidate tokens onto original utterances | `_build_original_unit_skeleton` freezes original tokens; `_apply_verified_sequence_timings` uses `difflib.SequenceMatcher` to attach timestamps ONLY to lexically matching original units, logging `substitutions`/`omissions`/`insertions` under `hypothesis_diff` | `test_a2_subinterval_recovery_never_substitutes_proper_nouns_or_inserts_text` |
| **A3** | Untimed units sorted with `float("inf")` reordered or collapsed mixed timed/untimed sequences in boundary reconciliation and utterance grouping | `_assign_monotonic_sort_keys` & `_compute_speaker_sort_keys` derive strictly monotonic interpolation-free ordering anchors between surrounding timed units without mutating `start`/`end` (`None` remains `None`) | `test_a3_a4_mixed_timed_untimed_preserves_order_and_marks_partial_status` |
| **A4** | Partially timed utterances (mix of valid and missing word/char timestamps) were mislabeled `"ctc_aligned"` or omitted untimed gap metadata | `utterances.py` classifies partial utterances as `alignment_status="partial_alignment"` and `timestamp_provenance.timing_status="partial"`, exposing `unit_count`, `timed_unit_count`, `untimed_unit_count`, and `_extract_timing_gaps` | Verified live on `zh_multichunk_synthetic_off` (`6 full, 2 partial`) and `zh_multichunk_synthetic_qwen` (`6 full, 1 partial`) |
| **A5** | Greedy/post-hoc speaker matching forced ambiguous boundary (`< 0.12`) or embedding (`< 0.05`) near-ties into incorrect merges (e.g., child voices) | `reconcile_speakers` filters row-wise and column-wise near-ties (`< 0.12`) and embedding ties (`< 0.05`) BEFORE assignment, keeping ambiguous speakers unmerged under `ambiguous_matches` (`left_unresolved_not_merged`) | `test_a5_ambiguous_boundary_and_embedding_matches_stay_unmerged`; live episode `2 -> 3` pre-solver ambiguous matches caught, `22 -> 23` global speakers |
| **A6** | Cached chunk WAVs & raw responses could go stale or partial after interruption, source file replacement, or config changes | Added `compute_file_sha256`, `validate_wav_file`, atomic `.tmp.wav` -> `os.replace`, atomic JSON/text writes, and invalidation in `recording.py` when source SHA256, audio stream, or config changes (or `--fresh`) | `test_a6_fresh_run_and_replaced_source_invalidate_stale_cached_wav` |
| **A7** | Multilingual word/char joiner inserted spaces into hyphenated/Latin tokens inside CJK or stripped legitimate spaces between English tokens | `annotate_units_with_source_spacing` records `join_no_space_prev` directly from original raw speaker turns + attaches trailing CJK punctuation so `join_multilingual_units` preserves `"Schotterb"` and `"U S A"` faithfully | `test_a7_latin_spans_preserved_inside_chinese_using_source_provenance`; live `zh_fleurs_05_1805` renders `"Scourthorpe四零三路..."` cleanly |
| **A8** | `itertools.permutations` for speaker matching scaled as $O(n!)$, risking hang on >8 speakers per chunk | Replaced with exact polynomial $O(V^3)$ Successive Shortest Path min-cost max-flow `_best_bipartite_assignment` | `test_a8_polynomial_bipartite_assignment_7x7_and_8x8_fast_and_exact` (`7x7` and `8x8` verified exact) |

---

## Live Before vs. After Comparison on Full 23-Minute French Episode

Run on identical checkpoint (`/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`, `chunk_seconds=90.0`, `context_seconds=10.0`, `min_tail_seconds=15.0`, `source_language="fr"`):

| Metric | Prior Run (`fr_full_episode_90s`) | Repaired Run (`fr_full_episode_repaired_90s`) | Interpretation |
| :--- | :---: | :---: | :--- |
| **Duration (s)** | `1379.901` | `1379.901` | Exact match |
| **Total Utterances** | `213` | **`221`** (`+8`) | Prevents over-aggressive deduplication & recovers boundary speech |
| **Global Speakers** | `22` | **`23`** (`+1`) | Column/row near-tie guard blocked 1 ambiguous merge |
| **Total Timed Units** | `2016` (`2016/2016`) | **`2064`** (`2064/2064`) | `+48` preserved timed word units |
| **Ambiguous Matches Logged** | `2` | **`3`** (`stage="boundary_temporal"`, `left_unresolved_not_merged`) | Pre-solver filter caught column-wise tie missed earlier |
| **Unresolved Local Speakers** | `14` | `15` | One additional ambiguous local speaker cleanly preserved separate |
| **Wall Time (s)** | `31.0437` | `31.4241` | Zero meaningful latency overhead (`+0.38s`) |
