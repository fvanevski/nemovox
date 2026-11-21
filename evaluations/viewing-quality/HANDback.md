# Dual-Language Viewing-Quality Remediation — Handback Report

**Repository:** `/workspace/nvidia-transcribe`  
**Starting Baseline HEAD:** `e8cbc7c83dc88dd5dca3fe96c0e06d52e2081564`  
**Reference Repo (`/workspace/pyannote`):** Untouched at `2b40b1c0c17ce324178af765d8d2c79234870acd`  
**Models Reused:** Nemotron 3.5 NVFP4 (frozen existing ASR contracts) + SwiftU/NInfer (`/workspace/models/Qwen3.8-27B-swift15-uncensored-ajgazin-nvfp4full-dflash2-NInfer-v3`)  
**Immortalized Baselines:** `outputs/subtitle_runs/fr_full_episode/` and `outputs/subtitle_runs/zh_full_episode_ep166/` preserved untouched.

---

## 1. Executive Summary & Acceptance Status

All four approved work packages (**A1–A3**, **B**, **C1–C4**, **D**) were implemented, unit-tested (`69 passed`), regenerated across both full episodes without rerunning ASR, visually verified via FFmpeg frame extraction across 14 timestamps, and packaged into `/workspace/nvidia-transcribe/outputs/handbacks/viewing-quality-evidence.tar.gz`:

- **WP A (Untimed Ordering & Fragments):** Eliminated `float(cidx) * 1000.0` cross-chunk pushing in [utterances.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/utterances.py) and added `audit_and_reconstruct_contract`, chronologically re-anchoring all 41 untimed EP166 utterances (`22` plausible independent, `11` uncertain, `7` probable duplicates, `1` loop) without fabricating timestamps. `cleanup_mandarin_fragments` suppressed `6` stray short particles (`'的'`, `'了'`, `'峰'`, `'我'`), eliminating EP166's raw CJK subtitle leak (`1 -> 0`).
- **WP B (Cache Correctness):** Replaced ID-only cache hits in [refinement.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/refinement.py) and [translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py) with SHA-256 input/context/config fingerprints, enabling safe selective invalidation while preserving `--re-render` (`0.059s` CPU-only).
- **WP C (Duplication, Timing, Condensation, 3-Lane ASS):**
  - Fixed trailing merge clause duplication (`utt-0019/0020`, `utt-0210/0211`) and French `282.69s` concurrent duplicate cue (`1 -> 0`).
  - Fixed French negative cue duration (`cue 113: -0.33s -> positive`, `invalid: 1 -> 0`) and added trailing-silence slack extension.
  - Condensed high-CPS cues selectively via SwiftU (`29/32` ZH, `5/7` FR, zero errors after bumping condensation `max_tokens` to `4096`), cutting Mandarin `> 20 CPS` cues by **`73.8%`** (`107 -> 28`) and French `> 20 CPS` cues by **`35.3%`** (`51 -> 33`).
  - Added 3rd ASS lane (`TopAlt`), eliminating silent ASS omissions (`FR: 5 -> 0`, `ZH: 3 -> 0`).
- **WP D (Hardsub-Aware ASS Layout):** Added `ass_layout="raised_main"` (`raised_main_margin_v=150` px default after inspecting `135` px frames where bottom descenders lightly brushed Hanzi tops), keeping French on `standard` (`42` px) while cleanly separating English cues above Mandarin EP166 hardsubs.

## 2. Before vs. After Key Outcomes

| Metric | French Before -> After | Mandarin EP166 Before -> After |
| :--- | :--- | :--- |
| ASS Omitted Overflow Cues | `5` -> **`0`** | `3` -> **`0`** |
| Invalid / Negative Cue Timing | `1` -> **`0`** | `0` -> `0` |
| Concurrent Duplicate Cues | `1` -> **`0`** | `0` -> `0` |
| Source Leaks (`cjk_in_english`) | `0` -> `0` | `1` -> **`0`** |
| Cues `> 20 CPS` | `51` -> **`33`** (`-35.3%`) | `107` -> **`28`** (`-73.8%`) |
| Cues `> 30 CPS` | `23` -> **`20`** (`-13.0%`) | `37` -> **`14`** (`-62.2%`) |
| `p90` CPS | `27.68` -> **`22.37`** | `30.75` -> **`19.73`** |
| Cues `< 1.0s` | `25` -> `25` | `35` -> **`29`** (`-17.1%`) |

## 3. Deliverable Artifacts
- [implementation-review.md](file:///workspace/nvidia-transcribe/evaluations/viewing-quality/implementation-review.md)
- [untimed-dialogue-audit.md](file:///workspace/nvidia-transcribe/evaluations/viewing-quality/untimed-dialogue-audit.md)
- [cache-regressions.md](file:///workspace/nvidia-transcribe/evaluations/viewing-quality/cache-regressions.md)
- [readability-comparison.md](file:///workspace/nvidia-transcribe/evaluations/viewing-quality/readability-comparison.md)
- [visual-review.md](file:///workspace/nvidia-transcribe/evaluations/viewing-quality/visual-review.md)
- [test-results.md](file:///workspace/nvidia-transcribe/evaluations/viewing-quality/test-results.md)
- [benchmark.json](file:///workspace/nvidia-transcribe/evaluations/viewing-quality/benchmark.json)
- [comparison_metrics.json](file:///workspace/nvidia-transcribe/evaluations/viewing-quality/comparison_metrics.json)
- [artifact-manifest.json](file:///workspace/nvidia-transcribe/evaluations/viewing-quality/artifact-manifest.json)
- Regenerated tracks: `outputs/subtitle_runs/viewing_quality/fr_full_episode/` and `outputs/subtitle_runs/viewing_quality/zh_full_episode_ep166/`
- Evidence archive: `/workspace/nvidia-transcribe/outputs/handbacks/viewing-quality-evidence.tar.gz`
