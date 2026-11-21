# Dual-Language Readability Comparison (Before vs. After)

## 1. Side-by-Side Subtitle Readability Metrics

| Metric | French Baseline | French After | Mandarin EP166 Baseline | Mandarin EP166 After |
| :--- | :--- | :--- | :--- | :--- |
| Total utterances | `221` | `221` | `285` | `285` |
| Suppressed stray Zh fragments | `0` | `0` | `0` | **`6`** |
| Adjacent merged utterances | `4` | `4` | `15` | `13` |
| SRT renderable cues | `286` | `284` | `307` | `281` |
| ASS displayed cues | `281` | **`284`** | `304` | **`281`** |
| **ASS omitted (overflow) cues** | `5` | **`0`** | `3` | **`0`** |
| **Invalid/negative cue timing** | `1` (`-0.33s` #113) | **`0`** | `0` | `0` |
| **Concurrent duplicate cues** | `1` (`282.69s`) | **`0`** | `0` | `0` |
| **Source-language leaks (`cjk_in_english`)** | `0` | `0` | `1` (`'的'`) | **`0`** |
| Cues `< 1.0s` duration | `25` | `25` | `35` | **`29`** (`-17.1%`) |
| **Median CPS (`p50`)** | `14.31` | `14.35` | `16.67` | **`15.10`** |
| **`p90` CPS** | `27.68` | **`22.37`** | `30.75` | **`19.73`** (`-35.8%`) |
| **`p95` CPS** | `40.24` | **`36.31`** | `39.77` | **`29.86`** |
| Mean CPS | `16.53` | `15.90` | `19.51` | **`16.40`** |
| **Cues `> 20` CPS** | `51` | **`33`** (`-35.3%`) | `107` | **`28`** (`-73.8%`) |
| **Cues `> 25` CPS** | `34` | **`26`** (`-23.5%`) | `69` | **`21`** (`-69.6%`) |
| **Cues `> 30` CPS** | `23` | **`20`** (`-13.0%`) | `37` | **`14`** (`-62.2%`) |
| Condensed cues (SwiftU) | `0` | `5/7` | `0` | **`29/32`** (`0` errors) |
| Line `> 42` chars / `> 2` lines | `0 / 0` | `0 / 0` | `0 / 0` | `0 / 0` |

## 2. Targeted Fixes Verified in Output Contracts

### A. Trailing-Clause Merge Echo Deduplication (`C1`)
- **EP166 `utt-0019 / utt-0020` `[268.541..270.191]`:** Baseline duplicated `"to the forefront— —and what I can grasp, what I can reclaim."` across fast recap boundaries. Regenerated contract folds `utt-0020` cleanly into `utt-0019` with zero clause repetition.
- **EP166 `utt-0210 / utt-0211` `[950.24..951.982]` (`怎么。` + `不认我了。`):** Clean single cue `"Hm? Don't recognize me?"`.
- **French `282.69s` Concurrent Duplicate (`C1`):** Baseline emitted two simultaneous identical cues (`#62` and `#63`: `"The interior of the continent is still unexplored."`). Regenerated output collapses them into one clean `Main` cue at `282.28..285.05s`.

### B. Mandarin Stray Particle Suppression (`A3`)
Six isolated single-character fragments from overlapping dialogue/cutoffs (`< 0.65s`) were suppressed (`fragment_suppressed=True`) so viewers do not see distracting 1-word flashes or raw Hanzi leaks (`'的'`):
- `utt-0052` (`469.88..470.095s`, `'了'`)
- `utt-0073` (`525.16..525.375s`, `'峰'`)
- `utt-0119` (`725.08..725.295s`, `'我'`)
- `utt-0159` (`845.08..845.385s`, `'的'`)
- `utt-0162` (`850.28..850.495s`, `'了'`)
- `utt-0274` (`1163.64..1164.114s`, `'的'`)

### C. Negative Duration & Rounding Fixes + Silence Slack Extension (`C2`)
- **FR cue #113 -> #110 (`504.21..504.36s`):** Baseline emitted invalid negative duration (`start=504.21, end=503.88, -0.33s`) on multi-line proportional sub-chunk rounding. Floor allocation plus strict end clamping gives positive duration `+0.15s`.
- **Silence slack extension (`_schedule_single_lane` Pass 2):** Automatically extends high-CPS cues into empty trailing gaps (e.g., FR last cue #283 extended from `1370.453s` to `1370.970s`; ZH `utt-0004` `103.641s -> 103.833s`).

### D. Selective SwiftU Condensation Examples (`C3`)

| Utterance | Window (s) | Uncondensed Text (Len) | Condensed Text (Len) |
| :--- | :--- | :--- | :--- |
| ZH `utt-0004` | `100.50..103.64` | `"I'm from the finance office. Military types can't be trusted — that's just how it is."` (`85`) | `"Finance office. Soldiers can't be trusted. That's how it is."` (`60`) |
| ZH `utt-0012` | `180.02..188.69` | `"I beat the Blue Dragon Gang, stomped the Heavenly Palace Society, went from a hostage no one had hopes for to a business magnate, built up two cultivation divisions, and by sheer skill, climbed to the top of the Starry Sky Ranking"` (`230`) | `"Crushed the Blue Dragon Gang, stomped the Heavenly Palace Society, from overlooked hostage to magnate, built two cultivation divisions, topped the Starry Sky Ranking"` (`165`) |
| ZH `utt-0021` | `271.77..274.88` | `"Only the life and death of those closest to me—that's where I, Bai Xiaochun, can still fight."` (`93`) | `"Only for those closest to me can I still fight."` (`47`) |
| ZH `utt-0028` | `318.08..319.77` | `"So who exactly is the successor the Underworld Emperor has chosen?"` (`66`) | `"Who's the Underworld Emperor's heir?"` (`36`) |
| FR `utt-0029` | `196.68..205.87` | `"Sure, it's obvious! Right! Let's move to phase two! Put on your masks! We're going down! If we hadn't wasted all that time in the Golden City of Kumla playing with holograms like kids, we would have found the double medallion before Ambrosius. Now, he"` (`251`) | `"Sure, it's obvious! Phase two! If we hadn't wasted time in Kumla's Golden City with holograms, we'd have found the double medallion before Ambrosius. Now, he"` (`157`) |

*(Original wording remains preserved verbatim inside `uncondensed_translated_text` in `translated.contract.json`.)*
