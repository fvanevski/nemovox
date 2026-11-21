# Quality Review: French, Mandarin Chinese, Overlap, Translation, and Long-Form

## 1. French ASR & Diarization (Episode + FLEURS)
### Clean Read French (FLEURS `fr_fr`, 4 clips, 98 ref words)
- **Corpus WER:** **10.20%** (`10` edits across `98` words: `fr_fleurs_00_1987` 0.0%, `fr_fleurs_01_1857` 15.38%, `fr_fleurs_02_1913` 7.69%, `fr_fleurs_03_2004` 15.38%). Errors were primarily minor tense/number/proper-noun spelling (`ajoutait` vs `a ajouté`, `17` vs `dix-sept`, `Machad` vs `Mashhad`).
- **CTC timestamp alignment:** 100% valid word timings across all 4 clips (`error: null`).

### Diagnostic French Episode Excerpts vs WhisperX/Pyannote (`result.json`)
1. **03:16–03:55 (`fr_0316_0355_child_voices`, 39s):**
   - **Pyannote baseline:** Collapsed both child voices (Esteban and Tao) into a single `SPEAKER_04` across all 12 utterances.
   - **Nemotron NVFP4 (`speaker` mode, 2.64s):** Correctly separated `<spk:0>` and `<spk:1>` across turns (`map={'0': 0, '1': 1}`, 7 diarization segments, `114/114` words aligned). Missed the final two words `"de m'avoir"` in both `speaker` and `verbatim` on the 39s crop, though recovered them in the 100s crop.
2. **03:05–04:45 (`fr_0305_0445_cont_100s`, 100s continuous):**
   - **Nemotron (`speaker` mode, 1.86s, 53.8x RT):** Identified 4 distinct speakers (`<spk:0>`..`<spk:3>`, `map={'0': 0, '1': 1, '2': 2, '3': 4}`, 16 diarization segments, `246/246` words aligned). Captured `"Il doit pen- il doit penser à Isabella, Nadia"` and stuttering `"Ben, ben, alors pou- pou- pour pourquoi"` accurately where WhisperX produced `"Il doit penser à Isabella, là, il dira"`.
3. **04:22–04:39 (`fr_0422_0439_rapid_turns`, 17s):**
   - Nemotron assigned `<spk:0>` to the entire clip in text (`map={'0': 1}`), while Sortformer diarization detected 5 intervals. Lexical output was accurate (`"Nous l'équipage ! Je suis sûr qu'Isabella..."`, dropping the initial truncated `"C'est"` at the crop boundary).
4. **Short 2–4s Isolated Excerpts (Critical Failure Mode Observed):**
   - `05:39–05:42` (`3.0s`, `"nuit. On n'a rien à manger"` + non-lexical vocalization): Nemotron **hallucinated English/Spanish** (`"Oh, look at all those things."` in speaker mode, `"Oh, I don't think."` in verbatim, `"No! No lo creo. <ev:laughter>"` in rich mode; LID returned `da`).
   - `22:08–22:11` (`3.0s`, high-pitched `"Alerte ! Alerte !"`): `speaker` and `verbatim` modes omitted `"Alerte ! Alerte !"` completely (emitting only `"Hier, c'est mieux."`), while **`rich` mode** recovered `<ev:breath> <sty:stt:vocal_intensity_pitch> Alerte! Alerte! <sty:end:vocal_intensity_pitch>`.
   - `05:50–05:52` (`2.0s`): `speaker` mode omitted `"Mais oui !"`, keeping `"C'est même comme ça."`, though Sortformer diarization detected 2 speakers (`[350.0, 350.43]` and `[350.53, 352.0]`).
   - **Implication:** Very short (<4s) cropped audio snippets without surrounding context degrade language ID and cause hallucinations/omissions; chunking pipelines should use >=15–60s windows.

## 2. Mandarin Chinese ASR, Timing, and Multi-Speaker Attribution
### Google FLEURS `cmn_hans_cn` (6 clips, 178 reference characters)
- **Corpus CER (auto language):** **12.36%** (`22` edits / `178` chars).
- **Corpus CER (explicit `"Mandarin Chinese"`):** **10.67%** (`19` edits / `178` chars).
- **Per-clip breakdown (auto -> explicit):**
  - `zh_fleurs_00_1906`: **0.00%** / **0.00%** (23/23 chars aligned, `0.21s`)
  - `zh_fleurs_01_2006`: **0.00%** / **0.00%** (transcript `钙、钾等元素属于金属，银和金等元素当然也是金属。` exact, **BUT CTC alignment returned `alignment_failed` (`0/24` timed)** — server log: `Unrepresentable character: '钾'` in the 23,862-token CTC vocabulary!)
  - `zh_fleurs_02_1883`: **2.70%** / **2.70%** (`净` -> `近`, 37/37 chars aligned)
  - `zh_fleurs_03_1734`: **0.00%** / **0.00%** (31/31 chars aligned)
  - `zh_fleurs_04_1890`: **6.90%** / **6.90%** (2-char repetition `大气大气`, 31/31 chars aligned)
  - `zh_fleurs_05_1805`: **51.35%** -> **43.24%** (foreign proper noun `"Scotturb"` transcribed in Japanese Katakana `スコート儿` / `スコートブ` and `辛特拉` -> `新特拉`)
- **Language ID:** `6/6` (`100%`) returned `zh` in `lid` mode.
- **Sequential 2-speaker Mandarin (`zh_2spk_seq`, 27.08s A-B-A):**
  - Attributed `<spk:0> -> <spk:1> -> <spk:0>` with 100% turn consistency; Sortformer intervals matched `[1.9..7.32]`, `[11.42..17.55]`, `[19.94..25.41]`. Loanword `"Scotturb"` became `"Schoolchildren bus"`, and turn 3 containing `'钾'` again triggered `ctc_timestamps.error = "alignment_failed"`.

## 3. Simultaneous / Overlapping Speech
Evaluated on two verified controlled 2-speaker overlap fixtures (2.0s offset, distinct genders):
- **French 2-speaker overlap (`fr_2spk_overlap`, 12.2s, overlap `2.0..8.28s`):**
  - Both `speaker` and `multispeaker_sot` modes recovered **both** concurrent lexical streams via interleaved `<spk:0>`/`<spk:1>` turns (`map={'0': 0, '1': 1}`, `diar=4`) in `0.45–0.46s`, with minor interference on one phrase (`capacités de développement` vs `stade de développement`).
- **Mandarin 2-speaker overlap (`zh_2spk_overlap`, 10.38s, overlap `2.0..9.62s`):**
  - `speaker` mode dropped most of Speaker A (`这并不是告铁`), whereas **`multispeaker_sot`** recovered both Speaker A (`这并不是告别... 而是一个篇章的结束也是新篇章的开始`) and Speaker B (`斯科特四零三路公交车车 停其发车前往新加兰在罗卡角停靠`), with minor homophone errors (`停其`/`新加兰`).

## 4. Direct English Speech Translation
- **French:** Fluent overall, but introduced semantic and proper-noun errors (`"langage mu"` -> `"the dead language"`, `"Nadia"` -> `"the giraffe"`, `"se sont fichus de nous"` -> `"don't care about us"`).
- **Mandarin:** Generally strong on technical sentences (`zh_fleurs_00`, `03`, `04`), but **omitted `"钙"` ("calcium")** in `zh_fleurs_01` (`"Elements such as potassium..."`), mistranslated `"净空"` ("clearance") as `"vertical shaft"` in `zh_fleurs_02`, and hallucinated `"Rockery"` for `"罗卡角"` (Cabo da Roca) in `zh_fleurs_05`.
- Direct translation provides no independent timestamps and can drop or distort domain entities; keep translation as a separate stage operating on verified source transcripts.

## 5. Longer-Form Scaling (90s -> 5 min -> 23 min Full Episode)
- **100s (`fr_0305_0445_cont_100s`):** `1.86s` (`53.8x` RT), `4` speakers, `246/246` words aligned, clean turns.
- **300s (`fr_0000_0500_300s`):** `4.35s` (`69.02x` RT), `3,810` prompt + `826` completion tokens, `finish_reason="stop"`, `517/517` words aligned, **but** speaker labels collapsed to 2 alternating tags (`0`, `1`) across the 5-minute span, and theme-song choir audio induced lyric hallucinations (`Esteban, Tincio, Carlos...`).
- **Full 23-min Episode (`fr_full_episode`, 1379.901s):** Completed in **`17.75s` (`77.72x` RT)** (`17,309` prompt + `3,281` completion tokens, `finish_reason="stop"`, peak VRAM `28,794 MiB`, `1992/1992` words aligned, `7` speakers `0..6`, `229` diarization segments). **However**, long single-pass context triggered severe word-level `<spk:N>` ping-ponging (`<spk:4> ... <spk:2> de <spk:4> la <spk:2> zélogrammes ...`) and occasional acoustic hallucinations (`estimadziao`). Windowed chunking (e.g., 60–120s segments with cross-chunk speaker reconciliation) is strongly recommended over single-pass 23-minute decoding for diarization stability.
