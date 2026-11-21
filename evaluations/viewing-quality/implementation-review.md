# Viewing-Quality Remediation — Code & Implementation Review

**Starting Baseline HEAD:** `e8cbc7c83dc88dd5dca3fe96c0e06d52e2081564`  
**Work Packages Implemented:** A (`A1`, `A2`, `A3`), B, C (`C1`, `C2`, `C3`, `C4`), D  
**Target Workspace:** `/workspace/nvidia-transcribe` (`/workspace/pyannote` untouched at `2b40b1c0c17ce324178af765d8d2c79234870acd`)

---

## 1. Root Causes and Implemented Repairs

### Work Package A — Dialogue Ordering, Untimed Audit, and Mandarin Fragment Cleanup
- **A1 (Cross-chunk untimed ordering root cause):** [utterances.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/utterances.py) previously ranked untimed utterances using `float(cidx) * 1000.0` whenever `source_start` was `None`. Because a 90s chunk with 10s overlap covers ~90–100s rather than 1000s, every untimed item from chunk $\ge 1$ sorted after the final chunk's timed timestamps (`cidx * 1000.0 > 1318.88`).
  - `_infer_chunk_windows` computes real chunk time spans `[chunk_start, chunk_end]` from timed utterances (falling back to stride estimates only for completely empty chunks) and passes them via `chunk_windows` in [recording.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/recording.py).
  - `_compute_speaker_sort_keys` and `reconstruct_speaker_utterances` anchor untimed speaker units linearly inside their true chunk window without fabricating `source_start` or `source_end` (both stay `None`).
- **A2 (Deterministic migration & classification):** Added `audit_and_reconstruct_contract` in [utterances.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/utterances.py). Given an existing `recording.contract.json`, it reconstructs chunk windows, sorts utterances into chronological chunk positions, preserves `source_start=None, source_end=None`, and emits an `untimed_audit` array classifying each untimed utterance as `probable_duplicate`, `plausible_independent_missing_timing`, `likely_hallucination_or_loop`, or `uncertain`.
- **A3 (Mandarin stray particle suppression):** Added `cleanup_mandarin_fragments` in [translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py). Suppresses ultra-short stray Mandarin single-character overlap/particle artifacts (`'的'`, `'了'`, `'地'`, `'得'`, `'峰'`, `'我'`, `'在'`, `'是'`, `'吗'`, `'吧'`, `'呢'`, etc. when duration $< 0.65\text{s}$ and adjacent to nearby active turns) with `fragment_suppressed=True` and empty `translated_text`, while explicitly preserving genuine standalone interjections (`'嗯'`, `'啊'`, `'哦'`, `'喂'`, `'好'`, `'是啊'`, `'不'`).

### Work Package B — Cache Correctness and Zero-Inference Re-render
- **Root cause:** Checkpoint reuse in [refinement.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/refinement.py) and [translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py) keyed cached records only by `utterance_id`. Reordering utterances, fixing upstream refined text, editing a prompt, or modifying neighboring context caused stale cache hits.
- **Fix:** Added deterministic SHA-256 fingerprint helpers (`_batch_cache_key` in `refinement.py`, `_translation_batch_cache_key` in `translation.py`) hashing target utterance text, surrounding context window, ordered batch items, glossary digest, language/profile/style/model config, and prompt version. Cache keys use `f"{uid}:{bkey}"`, preserving selective batch reuse for unchanged batches while immediately invalidating changed batches.
- **Zero-inference `--re-render`:** [subtitle_pipeline.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitle_pipeline.py) and [cli.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/cli.py) keep `--re-render` strictly CPU-only (re-reading `translated.contract.json` directly into `subtitles.py` in `< 0.1s`).

### Work Package C — Echo Dedup, Timing Slack, SwiftU Condensation, and 3-Lane ASS
- **C1 (Adjacent merge trailing-clause & echo deduplication):** Added `_dedup_merged_translation` and `_dedup_internal_trailing_clause` inside `_apply_safe_adjacent_regrouping` in [translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py). Prevents trailing clause duplication when `utt-(k+1)` is folded into `utt-k` (verified on `utt-0019/0020` and `utt-0210/0211` in EP166), and `_deduplicate_concurrent_echo_cues` in [subtitles.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitles.py) collapses concurrent identical/high-similarity cues (verified on French `282.69s` duplicate).
- **C2 (Non-negative durations & silence slack extension):** In [subtitles.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitles.py), `build_display_cues` uses floored sub-chunk weights so rounding never forces negative cue duration (`FR cue 113: -0.33s -> positive 0.15s`), and `_schedule_single_lane` includes Pass 2 trailing-silence slack extension to ease high CPS without intruding into following dialogue (`MIN_INTER_CUE_GAP_S = 0.04s`).
- **C3 (Selective high-CPS condensation):** Added `condense_high_cps_cues_with_swiftu` in [translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py) (configurable via `--condense-high-cps`, `--condense-threshold`), using structured JSON output (`max_tokens=4096`) and caching, while retaining unedited English wording in `uncondensed_translated_text`.
- **C4 (3-lane ASS overflow prevention):** Extended `assign_subtitle_lanes` and `format_ass` in [subtitles.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitles.py) to support `ass_max_lanes=3` (`Main`, `Top`, `TopAlt`), cutting silent ASS omissions from `5 -> 0` (FR) and `3 -> 0` (ZH).

### Work Package D — Hardsub-Aware ASS Layout
- Added `ass_layout` (`"standard" | "raised_main"`), `raised_main_margin_v` (default `150` px on 1080p after visual check showed `135` px nearly grazed tall Chinese top strokes), `ass_opaque_box`, and `ass_max_lanes` to `SubtitleConfig`, [subtitle_pipeline.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitle_pipeline.py), and [cli.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/cli.py).

## 2. Modified and Created Source Files

| File | Role |
| :--- | :--- |
| [utterances.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/utterances.py) | Chunk window estimation, chronological untimed ordering, `audit_and_reconstruct_contract` |
| [recording.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/recording.py) | Propagates `chunk_windows` into `reconstruct_speaker_utterances` |
| [refinement.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/refinement.py) | Input/context/config SHA-256 batch cache keys & `cache_hits` summary |
| [translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py) | SHA-256 batch keys, trailing-clause dedup, `cleanup_mandarin_fragments`, `condense_high_cps_cues_with_swiftu` |
| [subtitles.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitles.py) | Echo cue dedup, positive rounding + silence slack extension, 3-lane ASS (`TopAlt`), `raised_main` (`150` px) |
| [subtitle_pipeline.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitle_pipeline.py) | Wires condensation stage and ASS layout options end to end |
| [cli.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/cli.py) | Exposes `--ass-layout`, `--ass-margin-v`, `--ass-opaque-box`, `--ass-max-lanes`, `--condense-high-cps`, `--condense-threshold` |
| [test_viewing_quality_regressions.py](file:///workspace/nvidia-transcribe/tests/test_viewing_quality_regressions.py) | 6 CPU regression tests covering WP A–D invariants |
| [run_viewing_quality.py](file:///workspace/nvidia-transcribe/experiments/subtitle/run_viewing_quality.py) | Non-destructive dual-language regeneration + FFmpeg frame capture harness |
| [analyze_ep166_baseline.py](file:///workspace/nvidia-transcribe/experiments/subtitle/analyze_ep166_baseline.py) | Before/after subtitle analyzer updated to account for intentional `fragment_suppressed` utterances |
