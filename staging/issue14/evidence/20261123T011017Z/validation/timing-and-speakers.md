# Issue #14 Model Challenge: Timing Alignment & Speaker Diarization

- **Run ID:** `20261123T011017Z`

## 1. Qwen3-ForcedAligner-0.6B (One Resident Batched Session)

- Loaded in `1.3012s` (`1756.3 MiB` peak); batched `9` cases in `2.472s` (`2760.85 MiB` peak).

| Case ID | Condition | Audio Dur | Items | Zero-Dur | Non-Mono | Span (s) | Expected Band (s) | Diagnostic Outcome |
|---|---|---:|---:|---:|---:|---|---|---|
| `T_fr_A_c28_03` | `true_text` | 21.0s | 9 | 5 | 0 | `[1123.0, 1138.04]` | `[1131.5, 1136.0]` | Aligned in batch |
| `T_fr_B_w04_full` | `true_text` | 44.0s | 62 | 4 | 0 | `[1231.2, 1273.92]` | `[1230.5, 1274.0]` | Aligned in batch |
| `T_fr_CTRL_c2` | `true_text` | 14.0s | 18 | 6 | 0 | `[1022.0, 1032.56]` | `[1022.5, 1036.0]` | Aligned in batch |
| `T_fr_H4` | `true_text` | 20.0s | 10 | 2 | 0 | `[920.0, 935.36]` | `[928.5, 936.0]` | Aligned in batch |
| `T_zh_overlap_518` | `true_text` | 18.0s | 35 | 1 | 0 | `[518.0, 529.2]` | `[521.0, 531.0]` | Aligned in batch |
| `FALSE_fr_w04c_on_sfx` | `false_text_on_sfx` | 16.0s | 10 | 7 | 0 | `[1200.0, 1214.72]` | `[]` | **FALSE TEXT ALIGNED MONOTONICALLY** |
| `FALSE_fr_on_H3_sfx` | `false_text_on_sfx` | 20.0s | 7 | 4 | 0 | `[900.0, 918.56]` | `[]` | **FALSE TEXT ALIGNED MONOTONICALLY** |
| `FALSE_fr_wrong_w04c_candidate_on_B_w04` | `false_text_on_real_clip` | 44.0s | 6 | 2 | 0 | `[1230.0, 1233.04]` | `[1261.5, 1263.5]` | **FALSE TEXT ALIGNED MONOTONICALLY** |
| `FALSE_zh_on_fr_sfx` | `false_text_on_sfx` | 16.0s | 12 | 4 | 0 | `[1200.0, 1210.0]` | `[]` | **FALSE TEXT ALIGNED MONOTONICALLY** |

### Key Alignment Findings

1. **True-text alignment fails on unsegmented multi-event clips:** On full `44s` `B_w04`, `Qwen3-ForcedAligner-0.6B` smeared the 5-word `W04c` command uniformly across `[1256.096, 1266.32]` (`1.136s` per word!) right across the `[1262.9, 1264.5]` silence pause onto the female narration band (`1265.18-1266.32s`). On uncropped `A_c28_03` (`21s`), 5/9 words collapsed to `0.0s` and early words drifted to `1123.0s`. On `H4`, the opening filler collapsed to `920.0s` (`8.5s` early).
2. **Pre-bounded subclip test (`B_w04c_bounded` `[1261.0, 1264.0]` & `B_narr_bounded` `[1264.0, 1274.0]`):** When audio is pre-bounded to the active turn, `W04c` aligned at `[1261.0, 1262.92]` (before the silence pause, fixing Nemotron's ~2s late CTC shift!) and narration aligned cleanly at `[1264.08, 1274.0]`, though `3/5` fast `W04c` words still quantized to `0.0s` (`80ms` frame step).
3. **False-text negative controls ALL received `non_monotonic=0`:**
   - `FALSE_fr_w04c_on_sfx` (French dialogue on pure SFX `CTRL_sfx`): monotonic `[1200.0, 1214.72]`, `7/10` zero-duration, `3/10` stretched `0.72s-1.36s`.
   - `FALSE_zh_on_fr_sfx` (Mandarin text on French SFX `CTRL_sfx`): monotonic `[1200.0, 1210.0]`, **8/12 positive-duration characters** (`0.16s-1.147s`).
   - `FALSE_fr_on_H3_sfx` (French greeting on `H3` machinery SFX): monotonic `[900.0, 918.56]`, one word stretched to `7.36s`.
   - `FALSE_fr_wrong_w04c_candidate_on_B_w04`: monotonic `[1230.0, 1233.04]` on wrong speaker turn (`4/6` positive duration).
   - **Implication:** Monotonic forced-alignment timestamps NEVER prove acoustic word existence. Production integration must pre-bound audio to `AudioRegion` speech bands and explicitly reject high `zero_duration` ratios (`>=40%`), uniform linear smearing / stretched word durations (`>1.2s`), and pause-crossing placements.

## 2. Diarization Comparison (`pyannote Community-1` vs `Nemotron-3-Diarization` vs `Nemotron 3.5`)

| Case | True Voices | `pyannote` Clip (reg=exc) | `pyannote` Full 23m | `Nemotron-3` Clip (off / str) | `Nemotron-3` Full 23m (off / str) | `Nemotron 3.5` (`full35` / `zh166`) |
|---|---|---|---|---|---|---|
| `D_w02b` (209-232) | 2 (boy/girl rapid) | 1 spk (collapsed) | 1 spk (collapsed) | **2 spk** (1 turn separated) | 2 spk (off/str) | Collapsed `<spk:0>` |
| `C_w05` (1338-1380) | 2 (narrator/Pichu) | **2 spk** | 4 spk (over-split Pichu) | **2 spk** (clean) | **1 spk (collapsed; Pichu dropped)** | Collapsed tail to `<spk:0>` |
| `B_w04` (1230-1274) | 5 (4 spoken bands) | **3 spk** | **4 spk** | **3 spk** / 4 spk | 3 spk / 4 spk | Mixed `<spk:0..2>` |
| `A_c28_03` (1123-1144) | 3 (male/laugh/child) | 1 spk | 1 spk | **2 spk** (`0.2s`/`0.5s` ov) | 0 spk / 1 spk | c28 rejected |
| `H1` (120-140) | 4 | 1 spk | **3 spk** | 1 spk / **2 spk** | **2 spk** / **2 spk** | 1 spk (`c03` omitted line) |
| `H2` (360-380) | 2 (1 main+sigh) | 1 spk | 3 spk (over-split) | 1 spk / 2 spk | 1 spk / 1 spk | 1 spk |
| `CTRL_song_25_40` | 0 spoken (singing) | 1 spk (6 segs FA) | 1 spk (2 segs FA) | 1 spk (4-5 segs FA) | 0 spk / 1 spk | 7 unpop diar spans |
| `CTRL_sfx_1200_1216` | 0 spoken (SFX) | **0 spk (clean)** | **0 spk (clean)** | 1 spk (2 segs FA) | 1 spk (2-3 segs FA) | 1 diar span |
| `ZH_overlap_518_536` | 2 (rapid handoff) | 1 spk (collapsed) | N/A | 1 spk (collapsed) | N/A | 2 spk (CTC collision `525.16s`) |

### Key Speaker Findings

- **Regular vs Exclusive `pyannote Community-1`:** In all 18 short clips, `regular` and `exclusive` produced identical segment counts (`0s` overlap); on the full 23m episode (`12.06s` on GPU, `2746.24 MiB`), `regular=328` vs `exclusive=320` segments (both `7` speakers), preserving true overlaps only in `regular`.
- **Complementary failure modes:** Whole-episode `Nemotron-3-Diarization` (`2.44s` offline / `27.83s` streaming) saturates its 7-8 speaker cache after ~4.5m and drops/collapses late-episode speakers (`A_c28` missed, `C_w05` Pichu collapsed). Conversely, **scene-bounded `Nemotron-3-Diarization`** (`<=2-5 min`) excels at separating child/parrot turns (`D_w02b`, `C_w05`) without cross-episode cache collapse, while **`pyannote Community-1` (full episode)** is cleaner on `CTRL_sfx` (`0` FA) and tracks cross-episode macro-clusters (`7` speakers in `12s` GPU) despite collapsing rapid child turns (`D_w02b`) and false-triggering on singing (`CTRL_song`).
