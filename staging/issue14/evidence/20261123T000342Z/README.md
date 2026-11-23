# Nemovox Issue #14 Architecture-Redirection Acoustic & Evidence Packet

- **Run ID:** `20261123T000342Z`
- **Evaluated Source SHA:** `4c32acf659891fe999d2ae6e15f59ccb5750d92f` (`tree=a4e866f53899fe159c99793ff8a78326a10b26f5`)
- **Controlling Handoff Commit:** `0dac64e2ddcb706db188cb1f3084ea4fe4cc78b4`
- **Authoritative Main Base:** `8f6c03cbc70e84bf1970ba364207a02c9fe22b44`
- **Execution Mode:** Strictly CPU-only offline archive analysis + genuine multimodal WAV listening (`GPU_INFERENCE=forbidden`, `SOURCE_REMOTE_UPDATE=forbidden`, zero source/test/production mutations).

## Key Empirical Answers for Central Architecture Design

1. **`c28` 1099-1156s Independent Acoustic Truth:**
   - `1099-1110s`: Metallic collapse SFX (`1099-1105.5s`), then 1 urgent child/female shout (`1105.5-1110s`).
   - `1110-1123s`: Brief shout/scream continuation (`1110-1112s`), followed strictly by crash SFX, effort groans, and extended villain laughter (`1112-1123s`) — NO long omitted multi-word speech in `1112-1123s`. `c28` `unpopulated` diar spans here are false alarms on screams/laughs.
   - `1123-1144s`: SFX/horse-neigh/music (`1123-1131.5s`), **CRISP 9-word adult male urgent rescue speech (`1131.5-1136s`)**, grunts/music, and urgent child calls (`1141.5-1144s`). `FULL35 c28` (`finish=length`) dropped all of this; **naive `stop` retry `F1` (`44s`) ALSO dropped the entire 9-word speech event with `finish=stop` and `1` token**, whereas `P2` captured it along with 4 false interleaved `spk_1` words.
   - `1144-1156s`: Urgent repeated call (`1144.5-1147s`), brief cry (`1147-1150.5s`), and crisp 4-word male instruction (`1151-1154s`). `FULL35 c29` (`infer_start=1146.141s` after forced cut at `1150.141s`) covered only the tail instruction (`1152.3-1154.5s`) and missed `1144.5-1146s`.

2. **`W04b/W04c` Acoustic Chronology (`1230-1274s`):**
   - `W04b` occurs at `1252-1256s` (`~6w` + scream); crash/fanfare occupies `1256-1261.5s`; **`W04c` (`5w` command) occurs at `1261.5-1263.0s` BEFORE the genuine `[1262.9, 1264.5]` silence pause** (`c31/c32` cut at `1263.700s`). Continuous ~23w female narration spans `1264.4-1274.0s`.
   - **Critical falsification:** `P4` (`1242-1274s`) was internally monotonic yet placed `W04c` +2s late at `1264.44-1266.40s` on top of female narration and silently omitted the female narration (`1` `!` token only).

3. **`W05a/S05` & EOF Tail (`1338-1379.901s`):**
   - `1344.8-1346.3s` (`parrot_voice` turn 1): **5 spoken words** (`F35 c33` & `F3`: 5w; `P5`/`P6`: 4w).
   - `1352.0-1354.1s` (`S05`): **6 spoken words** (1w interjection `1352.0-1352.5s` + 5w question `1352.7-1354.1s`). `F35 c34` omitted all 6w; `F3` recovered 6w + 4 duplicate echo tokens; `P6` got 5w (dropped interjection); `P5` misplaced the interjection to `1346.86s`.
   - `1371.5-1376.8s` (`parrot_voice` reply + `narrator` closing sign-off) is genuine valid speech ending cleanly `3.1s` before EOF (`1376.8-1379.901s` quiet tail). `F3` recovered speaker separation and lexical accuracy (only missing single farewell token `1374.0-1374.7s`), whereas `c34` collapsed both voices to `spk:0` and compressed CTC words into `1370.3->1375.0`.

4. **Held-Out & Controls Decision Impacts:**
   - **All 4 deterministically sampled held-out windows (`H1 120-140`, `H2 360-380`, `H3 900-920`, `H4 920-940`)** yielded decisive calibration data: `H1` confirmed a true-positive unpopulated diar omission (`124.07-125.54s`), while **`H2` proved a false negative** (`FULL35 c09` silently dropped `~8w` at `362.5-368.0s` with `0` `>=1s` unpopulated diar triggers!).
   - **`CTRL_song` (`25-40s`)** produced `3` false `>=1s` unpopulated diar triggers inside singing; **`CTRL_sfx` (`1200-1216s`)** produced a `4`-word anticipatory CTC hallucination (`1203.23s`) in BOTH `orig_16` and `full_35` during pure collapse SFX (`27s` early!).
   - **Mandarin `ep166_zh` cross-language check (`96-114`, `518-536`)** confirmed `525.16s` synthetic ASS Top-lane collision came from a single-character CTC split across a rapid sequential turn handoff (NOT simultaneous speech).

## Packet Contents

- `source-identity.json`
- `validation/perception-method.md`
- `validation/acoustic-event-ledger.jsonl`
- `validation/candidate-observations.jsonl`
- `validation/detector-contingency.json`
- `validation/event-correspondence-matrix.md`
- `validation/pause-boundary-inputs.json`
- `review/fixture-adjudication.md`
- `COMMAND_SHAPE_FRICTION.txt`
- `manifest.json`
