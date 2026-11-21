# Campaign Handback: Multilingual Subtitle Generation & Contextual Localization

- **Repository:** `/workspace/nvidia-transcribe`
- **Starting Baseline HEAD:** `3c76089c955513268582c4fefe924c00e098eb12`
- **Reference `/workspace/pyannote` HEAD:** `2b40b1c0c17ce324178af765d8d2c79234870acd` (unmodified)
- **Hardware:** NVIDIA RTX 5090 (32 GB VRAM, Blackwell `sm_120`)

## Executive Outcomes (Work Packages A–E)

1. **Work Package A — Qwen Fallback Localization Repaired & Live Verified:**
   - Root cause fixed in [alignment.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/alignment.py) and [utterances.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/utterances.py): scoped Qwen alignment inputs per contiguous `(chunk_index, speaker)` span and bounded WAV window rather than passing full multi-speaker chunk WAVs; enforced strict `[min_start_s, max_end_s]` and neighbor monotonicity checks (`"external_bounds_violation"`).
   - Live Mandarin GPU check (`zh_multichunk_synthetic_qwen`, `24.182s`) confirmed `7/7` utterances achieved `full` timing and `'钾'` localized cleanly inside `utt-0001` (`[12.24s, 12.48s]`), never migrating into `utt-0000`.

2. **Work Package B — SwiftU Contextual Source Refinement Productionized:**
   - Promoted SwiftU (`/workspace/ninfer/swiftu`) into [refinement.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/refinement.py) with `off | conservative | balanced | expressive` profiles, YAML/JSON glossary support ([mcog_fr.yaml](file:///workspace/nvidia-transcribe/configs/glossaries/mcog_fr.yaml)), overlapping context batches, resumable checkpoints, lightweight edit screening, and strict preservation of `raw_source_text` vs `refined_text`.
   - Fixed `ninfer-serve` `--thinking-budget 16384` JSON truncation by setting default `reasoning_effort="low"` and `max_tokens=8192` (`4.62s` per 6-utterance batch, `0` batch errors).

3. **Work Package C — Translation Comparison & Default Selection:**
   - Downloaded `/workspace/models/Riva-Translate-4B-Instruct-v2` and benchmarked against SwiftU across 20 French+Mandarin cases ([translation-comparison.md](file:///workspace/nvidia-transcribe/evaluations/subtitle-generation/translation-comparison.md)).
   - Selected **`swiftu` (`natural` style)** as the default translator (`23.677s` total; preserved `Scotturb` / `Cape Roca`, idiom `"it's urgent!"`, elliptical `"It would take two"`; `Riva-4B` remains supported via `--translator riva`).
   - Added orphan-repair in [translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py) so vetoed adjacent same-speaker regrouping attempts never emit untranslated source fallbacks.

4. **Work Package D — Standalone SRT & 2-Lane ASS Subtitle Compositor:**
   - Built [subtitles.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitles.py) handling phrase-aware 2-line wrapping (`42x2`), `1.2s–6.5s` duration clamping, acoustic-only overlap classification (never display-padding induced overlaps), `Main` + `Top` ASS dual-speaker lanes, overflow conflict tracking, safe sidecar overwrite guards, and ASR-free `--re-render` (`0.035s`).

5. **Work Package E — CLI & End-to-End French/Mandarin Validation:**
   - Wired `nvidia-transcribe generate-subtitles` into [cli.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/cli.py) and [subtitle_pipeline.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitle_pipeline.py).
   - Generated complete `.en.srt` & `.en.ass` for *Les mystérieuses cités d'or S04E01* (`221` utterances -> `286` cues, `21` simultaneous `Top` cues, `0` unrenderable cues, `0` untranslated orphans) and all 4 Mandarin fixtures; visually verified 6 FFmpeg-rendered PNG screenshots ([render-validation.md](file:///workspace/nvidia-transcribe/evaluations/subtitle-generation/render-validation.md)).

## Key Report Files (`evaluations/subtitle-generation/`)

- [architecture.md](file:///workspace/nvidia-transcribe/evaluations/subtitle-generation/architecture.md)
- [refinement-evaluation.md](file:///workspace/nvidia-transcribe/evaluations/subtitle-generation/refinement-evaluation.md)
- [translation-comparison.md](file:///workspace/nvidia-transcribe/evaluations/subtitle-generation/translation-comparison.md)
- [subtitle-quality-review.md](file:///workspace/nvidia-transcribe/evaluations/subtitle-generation/subtitle-quality-review.md)
- [render-validation.md](file:///workspace/nvidia-transcribe/evaluations/subtitle-generation/render-validation.md)
- [test-results.md](file:///workspace/nvidia-transcribe/evaluations/subtitle-generation/test-results.md)
- [benchmark.json](file:///workspace/nvidia-transcribe/evaluations/subtitle-generation/benchmark.json)
- [artifact-manifest.json](file:///workspace/nvidia-transcribe/evaluations/subtitle-generation/artifact-manifest.json)
