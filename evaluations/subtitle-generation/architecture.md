# Subtitle Generation Architecture

## Overview

`nvidia-transcribe` separates media subtitle generation into four deterministic, resumable stages orchestrated by [`generate_subtitles`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitle_pipeline.py) and exposed via the CLI command `nvidia-transcribe generate-subtitles`.

```
  Media / Existing Recording Contract
                 │
                 ▼
  [Stage 1] Source ASR + Alignment + Global Speakers
            (Nemotron 3.5 NVFP4 + optional Qwen3-ForcedAligner-0.6B fallback)
            -> recording.contract.json (immutable raw_source_text / timestamps)
                 │
                 ▼
  [Stage 2] Contextual Source Refinement
            (SwiftU / NInfer batched sliding window + GlossaryConfig + safety screening)
            -> refined.contract.json + refined.ckpt.json (refined_text preserved separately)
                 │
                 ▼
  [Stage 3] Multilingual-to-English Subtitle Translation
            (SwiftU default or Riva-4B-Instruct-v2 + safe same-speaker regrouping + orphan repair)
            -> translated.contract.json + translation.ckpt.json
                 │
                 ▼
  [Stage 4] Standalone SRT & 2-Lane ASS Subtitle Compositor
            (wrap_subtitle_lines -> build_display_cues -> assign_subtitle_lanes -> SRT/ASS)
            -> <stem>.en.srt, <stem>.en.ass, subtitles.metadata.json
```

## Module Responsibilities & Separation of Concerns

1. **Stage 1 — Qwen Fallback Localization Fix ([alignment.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/alignment.py), [utterances.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/utterances.py))**:
   - `_contiguous_speaker_spans` groups contiguous units by `(chunk_index, speaker)` and detects OOV / fallback gaps using `failed_windows` or mismatched characters.
   - `_join_units_for_aligner` extracts bounded per-span audio windows instead of passing full multi-speaker chunks to `Qwen3-ForcedAligner-0.6B`.
   - `_apply_verified_sequence_timings` validates tight `[min_start_s, max_end_s]` bounds and monotonic ordering (`enforce_neighbor_monotonic=True`), rejecting out-of-window timestamps with `"external_bounds_violation"`.
   - `utterances.py` preserves lexical emission order within the same `(chunk_index, speaker)` rather than sorting globally across mismatched timestamps.

2. **Stage 2 — Contextual Source Refinement ([refinement.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/refinement.py))**:
   - Supports `backend` (`"off" | "swiftu"`) and `profile` (`"off" | "conservative" | "balanced" | "expressive"`) configured via `RefinementConfig` (`batch_size=6`, `context_utterances_before=6`, `context_utterances_after=4`, `max_tokens=8192`, `reasoning_effort="low"`).
   - `OpenAIChatTextClient` speaks OpenAI-compatible JSON-schema chat completions against `http://127.0.0.1:8001/v1` (`ninfer-serve` / SwiftU), gracefully stripping `reasoning_effort` or `response_format` on `400`/`422` if a downstream endpoint rejects them.
   - `GlossaryConfig` (`load_glossary`) loads domain character names, terms, and series notes (e.g., [mcog_fr.yaml](file:///workspace/nvidia-transcribe/configs/glossaries/mcog_fr.yaml)).
   - `screen_refinement_edit` protects against destructive rewrites and length explosions without requiring audio reverification (`audio_verification=False`), preserving `raw_source_text` untouched.

3. **Stage 3 — Contextual Translation & Safe Regrouping ([translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py), [_riva_translate_worker.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/_riva_translate_worker.py))**:
   - Supports `backend` (`"swiftu" | "riva" | "none"`) and `style` (`"natural" | "literal" | "expressive"`) via `TranslationConfig` (`swiftu` default, `reasoning_effort="low"`, `max_tokens=8192`).
   - `_apply_safe_adjacent_regrouping` merges adjacent turns only when `speaker_id` matches and is known, time gap `<= 1.5s`, combined span `<= 12.0s`, and no intervening speaker appears.
   - Orphan-repair pass re-translates with `allow_regrouping_override=False` whenever a batched model response merges an utterance (`translated_text == ""` + `grouped_with_prev=True`) whose merge is vetoed by acoustic/speaker safety rules, preventing silent fallback to raw French/Chinese text.

4. **Stage 4 — Standalone SRT/ASS Compositor ([subtitles.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitles.py))**:
   - `SubtitleConfig` defaults: `max_chars=42`, `max_lines=2`, `min_display_s=1.2`, `max_display_s=6.5`, `target_cps=20.0`, `overlap_threshold_s=0.4`, `play_res=(1920, 1080)`, `font_name="DejaVu Sans"`, `font_size=52`.
   - Separates acoustic bounds (`raw_start`, `raw_end`) from padded display intervals: simultaneous speaker detection uses acoustic timestamps strictly (never display padding) to assign `Main` (bottom `Alignment=2`, Layer `0`) vs `Top` (top `Alignment=8`, Layer `1`) lanes in ASS.
   - Sequential display collisions induced only by min-duration padding are resolved cleanly by clamping or tiny micro-splits, while `>= 3`-way overlaps are flagged honestly as `three_way_overlap_ass_overflow` in `subtitles.metadata.json`.
   - Supports ASR-free `--re-render` from existing `translated.contract.json` (`~0.035s` CPU execution).

## Data Separation & Sidecar Safety

- Four non-destructive artifacts are written per run: `raw_source_transcript.txt`, `refined_source_transcript.txt`, `final_english_subtitles.txt`, and `subtitles.metadata.json`, alongside `refined.contract.json` and `translated.contract.json`.
- Writing next to original media (`--sidecar-dir`) requires explicit `--overwrite-sidecar` before replacing pre-existing `.srt`/`.ass` files.
