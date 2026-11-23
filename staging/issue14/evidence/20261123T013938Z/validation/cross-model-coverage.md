# Issue #14 Bounded Qwen3-ASR-1.7B Pilot & Coverage-Floor Reconciliation

- **Run ID:** `20261123T013938Z`
- **Model:** `Qwen/Qwen3-ASR-1.7B` @ `7278e1e70fe206f11671096ffdd38061171dd6e5` (`bfloat16`, `cuda:0`, `local_files_only=True`, explicit `language="French"` or `"Chinese"`)
- **WER / CER Status:** `UNVERIFIED` (the 18-region acoustic ledger has no verbatim human reference transcripts; all comparisons below are bounded event/region observations, word/char counts, and positive/negative controls).

## 1. Eight-Clip Positive/Negative Pilot (`pilot_qwen_results.json`)

- **Admission & Execution:** `8/8 ok`, load `4.1073s`, total inference `5.4383s`, peak VRAM `4480.79 MiB`.

| Clip ID | Pinned Lang | Bounds (s) | Status | Elapsed | Words | Chars | Crop SHA-256 (prefix) |
|---|---|---|---|---:|---:|---:|---|
| `P1_c28_rescue` | `French` | `[1131.0, 1136.5]` | `ok` | `2.493s` | `9` | `56` | `150fdbeb51cd` |
| `P2_W04c_captain` | `French` | `[1261.0, 1264.0]` | `ok` | `0.142s` | `4` | `16` | `499b89aed4e8` |
| `P3_H2_start` | `French` | `[362.0, 368.5]` | `ok` | `0.266s` | `4` | `22` | `12ad12c3961c` |
| `P4_W05a_S05` | `French` | `[1344.5, 1354.5]` | `ok` | `1.075s` | `30` | `150` | `676c5c8a7c0f` |
| `P5_CTRL_song` | `French` | `[25.0, 40.0]` | `ok` | `0.263s` | `6` | `33` | `2010643ac270` |
| `P6_CTRL_sfx` | `French` | `[1200.0, 1216.0]` | `ok` | `0.130s` | `2` | `7` | `ebb2ee3a2be3` |
| `P7_ZH_ordinary` | `Chinese` | `[96.0, 114.0]` | `ok` | `0.434s` | `1` | `27` | `5ec5bafc4d45` |
| `P8_ZH_handoff` | `Chinese` | `[518.0, 536.0]` | `ok` | `0.632s` | `1` | `39` | `db007d12639e` |

### Pilot Observations (Candidate Only, Non-Authoritative)
- **`P1_c28_rescue` (`[1131.0, 1136.5]`):** Recovered 9 French words in the c28 shout band where `full35 c28` hit `finish=length` (`0` accepted units).
- **`P2_W04c_captain` (`[1261.0, 1264.0]`):** On an isolated 3.0s crop with no preceding dialogue context, Qwen emitted a 4-word French hallucination — whereas in the 24s coverage-floor window `win[57]` (`[1254.0, 1278.0]`) it recovered the 5-word captain command plus female narration. Isolated ultra-short crops cannot be trusted without surrounding context.
- **`P3_H2_start` (`[362.0, 368.5]`):** Emitted 4 French words covering the opening clause omitted by `full35 c09` (`[362.5, 364.2]` was joint-VAD negative), though opening lexemes showed phonetic distortion.
- **`P4_W05a_S05` (`[1344.5, 1354.5]`):** Emitted 30 French words covering both `W05a` (`1344.8-1346.3s`) and `S05` (`1352.0-1354.1s`, omitted by `full35 c34`).
- **`P5_CTRL_song` (`[25.0, 40.0]`):** **False positive on singing-only control** (`6` French words hallucinated over theme music where `FireRedAED` measured `singing=15.0s, speech=0.0s`).
- **`P6_CTRL_sfx` (`[1200.0, 1216.0]`):** Emitted 2 short non-lexical exclamation tokens over SFX/effort grunts (no anticipatory 27s-ahead dialogue hallucination as occurred in Nemotron `orig16 c13` / `full35 c30`).
- **`P7_ZH_ordinary` (`[96.0, 114.0]`):** Emitted 27 Mandarin characters in the fast-BGM region where `zh_ep166` chunk 1 had `alignment_failed` (`0` timed units).
- **`P8_ZH_handoff` (`[518.0, 536.0]`):** Emitted 39 Mandarin characters across the rapid speaker handoff where `zh_ep166` suffered a `525.16s` CTC boundary collision.

## 2. Full 23-Minute French Coverage-Floor Sweep (`63` Windows, `0.0-1379.901s`)

- **Execution:** `63/63 ok` (`window_s=24.0`, `overlap_s=2.0`), load `2.26s`, total inference `255.3648s`, peak VRAM `4480.79 MiB`, `0` empty windows.
- **Windows with Qwen text where `full35` had `0` timed units:** `7` windows (`win[50]` & `win[51]` inside rejected `c28`, plus `win[18,19,24,25,41]` emitting `1-2` non-lexical exclamation tokens on SFX/groan intervals).

| Region ID | Ref Speech | `full35` Timed Units | Overlapping Qwen Coverage-Floor Windows |
|---|---:|---:|---|
| `A_c28_01_1099_1110` | `True` | `0` | w49[1078-1102:4w], w50[1100-1124:2w] |
| `A_c28_02_1110_1123` | `True` | `0` | w50[1100-1124:2w], w51[1122-1146:11w] |
| `A_c28_03_1123_1144` | `True` | `0` | w50[1100-1124:2w], w51[1122-1146:11w] |
| `A_c28_04_1144_1156` | `True` | `6` | w51[1122-1146:11w], w52[1144-1168:24w] |
| `B_w04_1230_1274` | `True` | `70` | w55[1210-1234:14w], w56[1232-1256:29w], w57[1254-1278:43w] |
| `C_w05_1338_1380` | `True` | `119` | w60[1320-1344:51w], w61[1342-1366:75w], w62[1364-1380:42w] |
| `D_w02b_209_232` | `True` | `79` | w09[198-222:72w], w10[220-244:48w] |
| `D_w03a_985_1002` | `True` | `47` | w44[968-992:34w], w45[990-1014:59w] |
| `CTRL_c1_686_700` | `True` | `54` | w31[682-706:44w] |
| `CTRL_c2_1022_1036` | `True` | `18` | w46[1012-1036:28w], w47[1034-1058:34w] |
| `CTRL_song_25_40` | `False` | `0` | w01[22-46:5w] |
| `CTRL_sfx_1200_1216` | `False` | `4` | w54[1188-1212:25w], w55[1210-1234:14w] |
| `H1_120_140` | `True` | `17` | w05[110-134:32w], w06[132-156:15w] |
| `H2_360_380` | `True` | `38` | w16[352-376:41w], w17[374-398:10w] |
| `H3_900_920` | `False` | `0` | w40[880-904:37w], w41[902-926:1w] |
| `H4_920_940` | `True` | `18` | w41[902-926:1w], w42[924-948:22w] |

### French Sweep Reconciliation Findings
- **Joint-VAD false-negative floor recovery:** `win[57]` (`[1254.0, 1278.0]`, `43` words) covered both `W04c` (`[1261.5, 1263.5]`) and the `23w` female narration (`[1264.4, 1274.0]`) omitted by `full35 c32`; `win[16]` (`[352.0, 376.0]`, `41` words) + `win[17]` (`[374.0, 398.0]`, `10` words) covered `H2` opening (`[362.5, 364.2]`).
- **`c28` incomplete chunk coverage:** `win[50]` (`[1100.0, 1124.0]`, `2w` truncated), `win[51]` (`[1122.0, 1146.0]`, `11w` noisy/garbled shout compared to tight `P1_c28_rescue`), and `win[52]` (`[1144.0, 1168.0]`, `24w`) cover all 4 `A_c28_*` bands.
- **`C_w05` (`win[61]`, `win[62]`):** `win[61]` (`75w`) and `win[62]` (`42w`) recovered `S05` and the closing farewell omitted by earlier challengers.
- **Non-speech false-positive boundary:** `win[00]` (`[0.0, 24.0]`, theme song) emitted singing lyrics; `win[41]` (`[902.0, 926.0]`, `H3`) emitted 1 non-lexical exclamation (`3` chars); `win[18,19,24,25]` emitted `1-2` exclamation tokens in `full35`-empty non-speech slots. `win[54]` (`[1188.0, 1212.0]`) did NOT reproduce Nemotron's `1203.23s` anticipatory dialogue bleed.

## 3. Optional Mandarin EP166 Coverage-Floor Sweep (`60` Windows, `0.0-1318.88s`)

- **Execution:** `60/60 ok` (`window_s=24.0`, `overlap_s=2.0`), load `3.2937s`, total inference `59.3467s`, peak VRAM `4480.79 MiB`, `0` empty windows.
- **Windows with Qwen text where `zh_ep166` had `0` timed units:** `33` windows (primarily chunks `0..4`, `6..8` where `zh_ep166` failed alignment, plus OP/ED non-speech song windows).
- **`ZH_ordinary_096_114` (`win[04]` `[88.0, 112.0]`):** Emitted `27` Chinese characters (`zh_ep166` chunk 1 had `alignment_failed`, `0` timed units).
- **`ZH_overlap_518_536` (`win[23]` `[506.0, 530.0]`, `win[24]` `[528.0, 552.0]`):** Emitted `69` and `31` Chinese characters covering all rapid handoff turns.
- **Rejected `c14` (`[1250.0, 1318.88]`, `win[56..59]`):** Each window emitted only a single 2-char non-lexical particle over ending theme music/credits, corroborating that `zh_ep166 c14`'s `finish=length` generation loop occurred in a non-dialogue tail rather than lost multi-turn dialogue, while confirming pinned-Chinese Qwen emits 2-char filler false positives on music/BGM (`win[00..03]`, `win[56..59]`).
