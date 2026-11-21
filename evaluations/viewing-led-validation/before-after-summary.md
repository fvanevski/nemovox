# Pre-Validation vs. Post-Validation Quantitative & Qualitative Summary

Pre-validation tracks (`outputs/subtitle_runs/viewing_quality/`) remain untouched; post-validation outputs reside in `outputs/subtitle_runs/viewing_validation/`.

## 1. Metric Comparison Table

| Metric | French Pre | French Post | Mandarin EP166 Pre | Mandarin EP166 Post |
| :--- | ---: | ---: | ---: | ---: |
| SRT cues | `284` | **`280`** | `281` | **`301`** (`+20` recovered) |
| ASS events | `284` | **`280`** | `281` | **`301`** |
| ASS overflow omissions | `0` | **`0`** | `0` | **`0`** |
| Invalid cue intervals | `0` | **`0`** | `0` | **`0`** |
| Near-zero cues (`<0.2s`) | `1` | **`0`** | `0` | **`0`** |
| Short cues (`<1.0s`) | `25` | **`22`** | `29` | **`26`** |
| Rapid swaps (`<=50ms`) | `26` | **`24`** | `18` | **`15`** |
| Concurrent duplicate cues | `0` | **`0`** | `0` | **`0`** |
| Source leaks (`cjk_in_english`) | `0` | **`0`** | `0` | **`0`** |
| Untimed utterances remaining | `0` | `0` | `41` | **`22`** (`16` active unrenderable after `6` dupes suppressed) |
| Mean CPS | `15.90` | **`15.84`** | `16.40` | **`15.60`** |
| Median (`p50`) CPS | `14.35` | `14.50` | `15.10` | **`15.00`** |
| `p90` CPS | `22.37` | **`20.21`** | `19.73` | **`19.53`** |
| `p95` CPS | `36.31` | `36.31` | `29.86` | **`26.25`** |
| Cues `> 20 CPS` | `33` | **`30`** | `28` | **`27`** |
| Cues `> 25 CPS` | `26` | **`24`** | `21` | **`18`** |
| Cues `> 30 CPS` | `20` | **`18`** | `14` | **`12`** |

## 2. Highlighted Viewing Fixes

- **FR `00:43–00:50`:** Eliminated stray `"The"` (`43.71s`) and `"goats"` (`49.23s`) over the opening French narrator.
- **FR `08:16–08:25`:** Fixed `"Slowly, everything will"` -> 7s-late `"fine."` + fake `"ogre."` into continuous `"Easy now, everything will be fine."`
- **FR `03:42` & `13:25`:** Fixed `"a nat'cal"` -> `"a Naacal"`, `"It's going to rain!"` -> `"It's raining rats!"`.
- **ZH `02:53–02:57`, `05:05–05:39`, `10:33–11:25`, `15:32–15:40`, `16:25–16:35`, `17:05–17:59`:** Recovered **19 missing spoken dialogue utterances** (`+20` subtitle cues) eliminating four multi-second blocking blackouts (`7.8s`, `8s`, `13.5s`, `11s`, `12s`), relocated 7s-early `utt-0243` (`1064.88s` -> `1071.9..1078.8s`), stripped `utt-0019` 8s forward spoiler echo, suppressed `9` confirmed overlap duplicates/noises (including `"Thank you."` splitting the ending speech at `19:55`), and fixed `"Xu Zheng"`, `"unknown chapter"`, `"Victory is mine!"`, `"Underworld Emperor Palace"`, `"will wait"`, and `"Gazing toward you from afar, Master..."`.
