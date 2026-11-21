# Viewing-Quality & Localization Review — EP166 Baseline

## 1. Source Refinement & Translation Quality (`balanced` + `natural`)

- **Strong contextual & glossary wins (`94` Mandarin refinements):**
  - `utt-0006`: ASR `"以头轧既星空，到极宗"` -> Refined `"一头扎进星空道极宗"` -> `"plunged headfirst into the murky waters of the Starry Sky Dao Polarity Sect."`
  - `utt-0012`: ASR `"好难而志... 长程... 我白小春... 大聚龟"` -> Refined `"好男儿志在千里... 长城... 我白小纯... 大巨鬼"` -> Natural English idioms & lore terms preserved cleanly.
  - `utt-0015`: `"天道元音"` -> `"天道元婴"` -> `"Heavenly Dao Nascent Soul"`.
  - Scene coverage verified across opening narration (`utt-0004–0015`), Underworld Emperor Palace lore (`utt-0030–0068`), Song Que comedy (`utt-0173–0199`), emotional confrontation (`utt-0205–0225`), and closing departure (`utt-0236–0250`, ending cleanly at `1203.62s` matching external reference `.en.srt` ending `1203.76s`).
- **Mandarin-specific ASR & localization defects observed:**
  1. **Interleaved single-character CTC fragmentation:** `utt-0044` (`"你"` -> `"You..."`), `utt-0064` (`"峰"` -> `"peak"`, rendered alone on Top lane at `525.16s`), `utt-0083` (`"前"` -> `"Front…"` on Top lane at `591.56s`), and `utt-0241` (`"的"` at `1163.64s`, triggered `fallback_empty_translation` -> literal CJK leak `'的'` in SRT/ASS).
  2. **Batch-context echo duplication:** `utt-0185` (`"怎么。"`) absorbed adjacent line `utt-0186` (`"不认我了"`), outputting `"What? Don't recognize me? Don't recognize me?"`.
  3. **Untimed silence-tail hallucination (Chunk 14):** `96.56s` dead silence (`1222.28–1318.84s`) after the `1204–1222s` promo caused Chunk 14 ASR to loop (`16,384` completion tokens, `71.19s`) and emit `34` untimed utterances (`utt-0251..0284`), dropped cleanly as unrenderable (`41` total untimed including `utt-0000..0003` OP and `utt-0198..0200`).

## 2. Visual Screenshot & Viewing-Quality Inspection (`10` FFmpeg ASS Frames)

All `10` PNGs rendered in `outputs/subtitle_runs/zh_full_episode_ep166/review/screenshots/` were directly inspected:

- **Severe bottom-center hardsub collision:** Because *A Will Eternal EP166* carries white burned-in simplified Chinese hardsubs at bottom-center, Stage 4's default `Main` style (`Alignment=2`, `MarginV=42`, transparent background, outline `2.5`) directly overlaps the Chinese text across `shot_01` (`102s`), `shot_03` (`270s`), `shot_04` (`411s`), `shot_06` (`368s`), `shot_09` (`1045s`), and `shot_10` (`1201s`), degrading legibility on bright outlines.
  - **Recommended remediation approaches (without video restoration):** (a) enable opaque/semi-opaque dark backdrop boxes (`BorderStyle=4` or `BorderStyle=3` + `BackColour=&HA0000000`) covering the bottom hardsub strip, or (b) elevate `Main` `MarginV` (`~110–130 px`) / relocate `Main` upward when bottom hardsubs are detected.
- **`Top` lane simultaneous speaker rendering (`shot_04`, `shot_06`, `shot_08`):** Renders in clean unobstructed upper sky (`50` Top-lane events, `43` distinct 2-speaker pairs), confirming multi-speaker placement works well visually except where single-char stray splits (`"peak"`, `"Front…"`) pollute `Top`.
- **Reading-speed burst (`shot_03`, `270s`):** Long compact Mandarin utterance `utt-0015`/`0016` split unevenly against short partial ASR sub-windows produced **`153.45 cps`** (`0.391s–0.535s` display windows for `60–82` English chars), making fast recap lines unreadable.
- **Post-episode boundary (`shot_11`, `1212s`):** Shows `《斩神2》` outro preview cleanly; zero English subtitle shown after the actual episode ending (`1203.62s`).
