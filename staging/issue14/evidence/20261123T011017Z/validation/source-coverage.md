# Issue #14 Model Challenge: Source Activity & ASR Coverage

- **Run ID:** `20261123T011017Z`
- **Corpus:** 16 independently auditioned French `mcog_s04e01` regions (`42` spoken events, `175.10s` reference speech) + 2 Mandarin `ep166_zh` regions (`4` spoken events, `17.50s` reference speech).
- **WER/CER Status:** `UNVERIFIED` across the full 18-region corpus (no human-verified verbatim reference text in `20261123T000342Z`). Comparisons use `score.py` event-overlap coverage, non-speech false alarms, word-count recovery vs `expected_lexical_words_approx`, and audited omission/hallucination checks.

## 1. Scorer Summary (`evaluations/issue14/model_challenge/score.py`)

| Run Key | Scored | Unscored | Event Hits | Speech Covered | Non-Speech Speech FA |
|---|---:|---:|---:|---:|---:|
| `silero_vad_v6_fr` | 16 | 0 | 28/42 | 119.86/175.10s | 14.52s |
| `firered_vad_fr` | 16 | 0 | 40/42 | 154.03/175.10s | 49.50s |
| `firered_stream_vad_fr` | 16 | 0 | 40/42 | 149.11/175.10s | 48.41s |
| `firered_aed_fr` | 16 | 0 | 40/42 | 146.44/175.10s | 39.92s |
| `pyannote_clip_reg_fr` | 16 | 0 | 38/42 | 132.61/175.10s | 16.98s |
| `pyannote_full_reg_fr` | 16 | 0 | 36/42 | 131.44/175.10s | 16.83s |
| `nem3_diar_clip_off_fr` | 16 | 0 | 37/42 | 131.69/175.10s | 15.70s |
| `nem3_diar_full_off_fr` | 16 | 0 | 23/42 | 100.17/175.10s | 3.86s |
| `nem35_orig16_asr_fr` | 16 | 0 | 36/42 | 124.23/175.10s | 5.96s |
| `nem35_full35_asr_fr` | 13 | 3 | 30/37 | 106.42/160.10s | 5.28s |
| `parakeet_tdt_v3_fr` | 16 | 0 | 30/42 | 81.18/175.10s | 4.36s |
| `qwen3_asr_explicit_fr` | 16 | 0 | 42/42 | 175.10/175.10s | 121.20s |
| `qwen3_asr_auto_fr` | 16 | 0 | 40/42 | 168.60/175.10s | 67.70s |
| `silero_vad_v6_zh` | 2 | 0 | 4/4 | 13.37/17.50s | 3.00s |
| `firered_vad_zh` | 2 | 0 | 4/4 | 13.70/17.50s | 2.75s |
| `firered_aed_zh` | 2 | 0 | 4/4 | 10.07/17.50s | 2.80s |
| `pyannote_clip_reg_zh` | 2 | 0 | 4/4 | 13.03/17.50s | 2.70s |
| `nem3_diar_clip_off_zh` | 2 | 0 | 4/4 | 10.15/17.50s | 3.00s |
| `nem35_zh166_asr_zh` | 1 | 1 | 3/3 | 6.00/10.00s | 2.58s |
| `qwen3_asr_explicit_zh` | 2 | 0 | 4/4 | 17.50/17.50s | 18.50s |
| `firered_asr2_aed_zh` | 2 | 0 | 3/4 | 8.78/17.50s | 2.86s |

## 2. Voice / Singing / Music Activity Discriminators

- **Silero VAD v6 (`t=0.5`):** `28/42` FR event hits (`119.86/175.10s`, `14.52s` FA); `4/4` ZH (`13.37/17.50s`). Clean `0.00s` on `CTRL_song_25_40`, `CTRL_sfx_1200_1216`, `H3_900_920`, but **missed** `A_c28_01`, `A_c28_02`, `CTRL_c2_1022_1036`, `W04c`, and `A_c28_03` heroic speech (`1131.5-1136s`). Detected `H2` omitted speech (`364.66-380.0s`).
- **FireRedVAD (binary VAD):** `40/42` FR event hits (`154.03/175.10s`, `49.50s` FA); `4/4` ZH (`13.70/17.50s`). Recovered `A_c28_01` (`[1108.23, 1110.0]`), `A_c28_03` (`1131.48-1136.79s`), `CTRL_c2`, and `H2` — **but** false-triggered `15.00s` on `CTRL_song_25_40` and `3.99s` on `CTRL_sfx_1200_1216`.
- **FireRedVAD-AED (multi-label `speech` / `singing` / `music`):** `40/42` FR event hits (`146.44/175.10s`, `39.92s` FA); `4/4` ZH (`10.07/17.50s`). **Decisive discriminator:** classified `CTRL_song_25_40` as **`0.00s` speech, `15.00s` singing, `15.00s` music**, while STILL recovering `A_c28_01` (`[1108.29, 1110.0]`), `A_c28_03`, `CTRL_c2`, and `H2`! Still emitted short speech false triggers on non-lexical grunts/groans in `CTRL_sfx_1200_1216` (`3.64s`) and `H3_900_920` (`1.64s`) alongside `music` flags (`16.0s` / `19.78s`).
- **Shared blind spot:** **ALL detectors** (`Silero`, `FireRedVAD`, `FireRedAED`, `pyannote`, `Nemotron-3-Diarization`) missed `W04c` (`1261.5-1263.5s` captain shout inside `B_w04`) and `H2`'s `362.5-364.2s` initial phrase — proving no VAD/diarization model may act as hard audio-exclusion authority.

## 3. French & Mandarin ASR Challengers (Lexical Recovery vs Hallucinations)

| Region | Expected Approx | Nemotron 3.5 (`orig16` / `full35` / `zh166`) | Qwen3-ASR-1.7B (`explicit`) | Parakeet-TDT-0.6b-v3 | FireRedASR2-AED |
|---|---|---|---|---|---|
| `A_c28_01` (1099-1110) | ~4w shout | `orig16`: 2w / `full35`: 0w (c28 rejected) | 2w (recovered shout; `auto`=`""`) | 0w (omitted) | `skipped_unsupported` |
| `A_c28_03` (1123-1144) | ~14w (9w crisp) | `orig16`: 10u / `full35`: 0w (c28 rejected) | 8w garbled FR; **`auto` mis-detected `Chinese`** | **10w** (closer phonetic recovery; minor lexical errors) | `skipped_unsupported` |
| `B_w04` (1230-1274) | ~58w (`W04c`+23w narr) | `full35` c32 **omitted 23w female narration** | **65w** (recovered 23w narration + `W04c`) | **65w** (recovered 23w narration + `W04c`; minor orthography error) | `skipped_unsupported` |
| `C_w05` (1338-1380) | ~103w (incl `S05` 6w) | `full35` c34 omitted `S05` (6w) | **114w** (full coverage; omitted 1 interjection) | **64w (BROKEN: dropped ~50w middle passage & switched to English tokens)** | `skipped_unsupported` |
| `D_w02b` (209-232) | ~58w alternating | `orig16`: 61u / `full35`: 79u | **63w** (complete coverage, no `<spk>` tags) | **28w (BROKEN: dropped first 11s / ~35w & switched to English tokens)** | `skipped_unsupported` |
| `H2` (360-380) | ~33w (8w init omit) | `full35` c09 omitted ~8w opening | **30w** (recovered opening clause) | 30w (recovered opening clause; proper-noun error) | `skipped_unsupported` |
| `CTRL_song_25_40` | 0w (singing only) | `orig16`: 6u / `full35`: 0u | **HALLUCINATED** (6w FR explicit; English lyrics in `auto`) | **0w** (clean negative) | `skipped_unsupported` |
| `CTRL_sfx_1200_1216` | 0w (SFX only) | **4w anticipatory bleed** at 1203.23s | 2 non-lexical exclamations (`auto`: 0w) | **0w** (clean negative) | `skipped_unsupported` |
| `H3_900_920` | 0w (SFX/groan only) | 0u | 1 non-lexical exclamation (`auto`: 0w) | **0w** (clean negative) | `skipped_unsupported` |
| `ZH_ordinary_096_114` | ~25 chars + rock BGM | `zh166`: chunk 1 `alignment_failed` (0 timed, 163 untimed; garbled segment) | **27 chars** (clean Mandarin recovery) | `skipped_unsupported` | **0 chars (`''`, `conf=0.359`) on 18s clip**; recovered 26 chars (`conf=0.613`) ONLY when pre-cropped to `[99.0, 106.5]` |
| `ZH_overlap_518_536` | ~34 chars rapid handoff | `zh166`: 35 chars, CTC boundary collision at `525.16s` | **39 chars** (exact domain terms, `auto=Chinese`) | `skipped_unsupported` | 35 chars (2 domain-term homophone errors) |

Note on `ZH_overlap_518_536`: all 5 activity/diarization models detected `~2.7-3.0s` speech activity inside `518-521s` because an earlier phrase occurs right after `518.0s`; `score.py` records this against the `20261123T000342Z` reference `non_speech` interval without post-hoc relabeling.
