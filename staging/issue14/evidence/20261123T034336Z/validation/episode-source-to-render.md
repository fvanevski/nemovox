# Opt-In `generate-subtitles --recording-contract` Source-to-Render Validation

All runs executed in fresh private output directories without mutating production jobs, ASR checkpoints, or running services.
Model resources config: `configs/model-resources.example.json` (`riva_local` BF16 offline worker on `cuda:0`, `/workspace/models/Riva-Translate-4B-Instruct-v2`).

## 1. French Episode (`fr_orig16`) Baseline vs Promoted Contract

| Metric | Baseline `fr_orig16` (Riva) | Promoted `fr_orig16` (Riva) | Promoted `fr_orig16` (`none`) |
|---|---:|---:|---:|
| Input source utterances | 234 | 241 | 241 |
| Accepted `audio_review_attested_source` added | 0 | 2 (`H2_opening_faites_voir`, `B_w04_W04c_captain_shout`) | 2 |
| Accepted `model_source_candidate_provisional` added | 0 | 5 (`H1_c01_gap_136_138`, `D_w03a_distant_call`, `B_w04_esteban_tao_question`, `C_w05_S05_omitted`, `C_w05_tail_omitted`) | 5 |
| Quarantined non-speech / colliding proposals | — | 3 (`D_w02b_colliding_turn`, song hallucination `[25,40]`, SFX hallucination `[1200,1216]`) | 3 |
| Source text verbatim preserved across all utterances | yes | yes (True) | yes |
| Renderable SRT/ASS cues | 309 | 320 (+11) | 329 |
| ASS top-lane cues (`lane=1`) | 26 | 26 | 26 |
| Unrenderable cues | 0 | 0 | 0 |
| Consecutive duplicate-looking SRT cues | 6 | 7 | 9 |
| Riva peak VRAM (MiB) / runtime (s) | 8024.58 / 64.862 | 8024.58 / 67.14 | 0 / 0.058 |

### Qualitative & Lineage Findings (FR)
- `CTRL_c2_repeat_pair` (`utt-0177` & `utt-0178`): both legacy turns retained distinctly in promoted contract and subtitle render.
- Missing audited speech recovered: `H2_opening_faites_voir` (`[362.5, 364.2]`, retains direct-review text despite bounded Qwen miss) and `B_w04_W04c_captain_shout` (`[1261.5, 1263.5]`, 8.0s bounded Qwen crop with 3s bilateral context recovered the captain shout where pause-truncated crop emitted a generic exclamation).
- False singing/SFX blocked: song hallucination overlapping `CTRL_song_25_40` and SFX exclamation overlapping `CTRL_sfx_1200_1216` were quarantined and excluded from promoted utterances and rendered SRT/ASS cues.
- Context-word risk on provisional crops: `C_w05_S05_omitted` and `C_w05_tail_omitted` carried neighboring dialogue / spurious proper-noun leakage from ~8–9.5s crop windows; accurately labeled `model_source_candidate_provisional` (`source_text_human_reference_verified=false`) rather than attested truth.

## 2. Mandarin Episode (`zh_ep166`) Full Contract & Bounded Overlap Fixture

| Metric | Bounded ZH Fixture (Riva, `raised_main`) | Full `zh_ep166_promoted` (`none`, `raised_main`) |
|---|---:|---:|
| Input utterances to renderer | 9 (4 promoted + 5 untimed legacy) | 288 (285 legacy incl. 41 untimed + 3 handoff turns) |
| Lineage-verified supersessions | 4 (`utt-0062`..`utt-0065` -> 3 handoff turns) | 0 (raw legacy lacks `source_turn_ids`; 41 untimed untouched) |
| Untimed legacy utterances preserved in contract | 5 | 41 |
| Renderable timed SRT/ASS cues | 6 | 247 |
| Unrenderable cues (no fabricated timing) | 5 (`unrenderable_no_usable_timing`) | 41 (`unrenderable_no_usable_timing`) |
| ASS `raised_main` top-lane cues | 0 | 40 |

### Qualitative Findings (ZH)
- Rapid elder/young speaker handoff `[518.1, 530.2]`: in the lineage-linked EP166 bounded fixture, stray legacy 1-character overlap fragment (`utt-0064`) and truncated overlap turns (`utt-0062`, `utt-0063`, `utt-0065`) were cleanly superseded via matching `source_turn_ids` into the 3 attested turns (`elder` -> `young` -> `elder`) and translated into non-empty English cues.
- Full `zh_ep166` untimed tail: all 41 untimed units remained intact in the promoted recording contract, were rejected by the renderer solely as `unrenderable_no_usable_timing`, and received zero fabricated `source_start`/`source_end` timestamps.
