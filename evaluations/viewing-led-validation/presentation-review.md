# Presentation, Layout & Reading-Speed Review

## 1. French Layout (`standard`, `MarginV=42`, `ass_max_lanes=3`)
- **Single-speaker (`Main`):** Clean bottom-aligned positioning with `2.5` px black outline + `1.2` px shadow; crisp 2-line wraps across narration and dialogue (`0` lines `> 42` chars, `0` `> 2` lines).
- **2- & 3-Speaker Overlaps (`Top`, `TopAlt`):** Verified in `fr_02a`, `fr_03`, and `fr_04`: `Main`, `Top` (`MarginV=42` top-center), and `TopAlt` (`MarginV=164` top-center soft green tint) remain clearly readable without collision (`0` ASS overflows).
- **Outliers (`>30 CPS`, `<1.0s`):** Reduced `<0.2s` near-zero flashes from `1 -> 0` and `<1.0s` cues from `25 -> 22` by merging `utt-0079/0081` (`"fine."` `0.15s` eliminated) and removing `"The"`, `"goats"`, `"ogre."`. Remaining `18` `>30 CPS` cues (`p90` improved `22.37 -> 20.21`) occur mainly in rapid overlapping interjections (`fr_06` `811–830s`).

## 2. Mandarin EP166 Layout (`raised_main`, `MarginV=150`, `ass_max_lanes=3`)
- **Coexistence with Chinese Hardsubs:** Verified across bright OP frames (`zh_01`), recap cards (`zh_02`), glowing portal interiors (`zh_04`), twilight roofs (`zh_05`), deep blue palace halls (`zh_06..09`), and close-up twilight confrontations (`zh_08`, `val_zh_08_988s_recovered_monologue.png`). The `150` px 1080p bottom margin keeps both 1-line and 2-line English `Main` cues cleanly separated above burned Chinese hardsubs.
- **2- & 3-Speaker Overlaps:** `Main` (bottom raised), `Top` (top), and `TopAlt` (`zh_05` at `563s`) render with zero overlap (`52` `Top` cues, `2` `TopAlt`, `0` ASS overflows).
- **Reading Comfort:** Despite adding `+20` recovered spoken cues (`281 -> 301`), mean CPS improved (`16.40 -> 15.60`), `p95` CPS dropped (`29.86 -> 26.25`), `>30 CPS` cues dropped (`14 -> 12`), `<1.0s` cues dropped (`29 -> 26`), and rapid `<=50ms` swaps dropped (`18 -> 15`).

## 3. SRT vs. ASS Parity
- French: **280** SRT cues = **280** ASS `Dialogue` events (`0` difference, `0` dropped cues).
- Mandarin EP166: **301** SRT cues = **301** ASS `Dialogue` events (`0` difference, `0` dropped cues). Both tracks share exact dialogue coverage while ASS provides speaker-lane and hardsub-clearing positioning.
