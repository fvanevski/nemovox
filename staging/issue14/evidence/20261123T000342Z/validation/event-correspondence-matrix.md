# Conceptual Policy Comparison & Event-Correspondence Matrix

- **Run ID:** `20261123T000342Z`
- **Policies Compared (Conceptual Audit Only — No Production Implementation):**
  1. **Policy 1 (Existing Core-Ownership + 3-Step Boundary Heuristics):** `chunking.py` + `boundary_evidence.py` (fixed `core` ownership + displaced matching + extended right-context promotion + `finish=length` fail-closed without region-level observation pooling).
  2. **Policy 2 (Event-Centered Correspondence Concept):** `AudioRegion` -> immutable `ASRObservation` pool -> `EventCorrespondence` -> `AcceptedSourceTurn` -> `RecoveryController` as sketched in `staging/issue14/ARCHITECTURE_REDIRECTION.md`.

## Per-Region Outcome Comparison (Grounded in Independently Heard Source Events)

| Region | Heard Speech Events | Policy 1 (`full_35` Baseline) Preserved / Omitted / Dup / Unresolved | Policy 2 (`EventCorrespondence`) Expected With Raw Obs Only vs + Independent Acoustic Activity | Minimum Positive Selection Evidence Required |
|---|---:|---|---|---|
| `A_c28_01 (1099-1110)` | 1 | `0` / `1` / `0` / `0` (silent omission in c27 & c28) | Needs retry (`RecoveryController` sees active `1105.5-1110s` acoustic region with 0 accepted words across c27/c28) | Acoustic voice-activity band `[1105.5, 1110.0]` + lexical output >0 |
| `A_c28_02 (1110-1123)` | 1 (+laugh/SFX) | `0` / `1` (`1110-1112`) + `finish=length` block | P1 captures `[1110.26]`; **raw Nemotron alone** shows 4 unpop diar spans in `1112-1123` (false alarm on laugh/SFX); needs non-speech/laugh discriminator | Speech/SFX discriminator distinguishing `1110-1112` shout from `1112-1123` laugh/groans |
| `A_c28_03 (1123-1144)` | 3 (`9w` @ `1131.5-1136s`) | `0` / `3` / `0` / `0` (c28 failed `length`; naive `stop` retry `F1` still omitted!) | P2 observation covers `1131.5-1136s` cleanly IF hallucinated interleaved `spk_1` is rejected (`c12`-like clean obs wins over `F1` sparse `stop`); **needs active-region coverage gate** so `F1` (`finish=stop`, 1w) does NOT satisfy `[1123, 1144]` | Multi-second acoustic speech region `[1131.5, 1136.0]` vs word-timestamp density (`>1` word/s expected during dense speech) |
| `A_c28_04 (1144-1156)` | 3 | `1` / `2` (`1144.5-1150` missed before c29 infer_start) | Pooling targeted retry `[1123..1156]` with `c29` `[1146.1..1156]` covers all 3 events; correspondence keeps `1144.5-1147` + `1151-1154` non-destructively | Source-interval overlap matching across observations `[c29, P2]`, not core-cutoff truncation |
| `B_w04 (1230-1274)` | 5 (`W04a`, `W04b`, `W04c`, narration) | Corrupted chronology (c31/c32 interleaved W04b/W04c/narration) | **Raw Nemotron alone FAILS W04c timing:** even monotonic `P4` shifted W04c +2s past the `[1262.9..1264.5]` acoustic pause and dropped 23w narration! Combining `-32 dB` acoustic pause anchor `[1262.9, 1264.5]` + `c31` text vs `c32` separates pre-pause W04c from post-pause narration | Independent audio energy/VAD envelope + forced aligner on contested pause-crossing events |
| `C_w05 (1338-1380)` | 8 (`S05` 6w, parrot_voice tail, EOF) | `5` / `2` (`S05` omitted in c34) / `0` / speaker-merged tail | Combining `c33` (`1344.8-1346.3` 5w) + targeted `F3`-like obs recovers `S05` 6w and tail speaker separation (`F3` needs echo suppression on 1351.9/1354.2) | Cross-observation consensus in `[1344.5, 1346.5]`, `[1352.0, 1354.1]`, `[1371.5, 1376.8]` + independent forced aligner for `S05` `1352.0s` |
| `D_w02b (209-232)` | 5 turns (2 voices) | `5` text preserved / `0` voice separation (`spk_0` only) | Event correspondence preserves all ~58w across `c05/c06` without fragile token-bridge rules; voice separation STILL requires diarization/turn-level speaker re-clustering independent of raw `<spk:N>` | Acoustic turn boundaries + voice embeddings on clean audio (raw `<spk:N>` insufficient) |
| `D_w03a (985-1002)` | 3 (with offscreen call) | `3` / `0` / `0` / `0` in `full_35 c25` (`orig_16` duplicated) | Source-interval grouping eliminates `c10/c11` cut duplicate automatically without arbitrary token-length caps | Overlap interval matching |
| `CTRL_c1 (686-700)` | 3 | `3` / `0` / `0` / `0` | `3` / `0` / `0` / `0` | Baseline overlap correspondence |
| `CTRL_c2 (1022-1036)` | 3 (2 genuine repeats) | `3` / `0` / `0` / `0` (merged speakers) | `3` / `0` / `0` / `0` (distinct intra-observation acoustic intervals preserved; NEVER deduplicated) | Disjoint acoustic intervals (`~1023s` vs `~1026s`) forbid echo deduplication |
| `CTRL_song (25-40)` | 0 spoken | `0` / `0` / `0` / `0` (`c00` emitted 0w + 3 false unpop diar triggers) | Needs **singing/music detector** to avoid retrying `[25, 40]` endlessly despite 7 diar spans and 0 words | Speech vs music/singing classifier |
| `CTRL_sfx (1200-1216)` | 0 spoken | Hallucinated 4w bleed (`1203.2s`) in BOTH `orig_16` and `full_35` | **Raw Nemotron alone CANNOT catch `1203.2s` bleed** because audio has loud collapse SFX (`>-32 dB`) + Nemotron emitted confident monotonic CTC words; requires independent speech/SFX classifier or acoustic cross-check | Acoustic speech-probability check (`non_speech_sfx` rejects `1203.2s` bleed) |
| `H1 (120-140)` | 5 | `4` / `1` (`123.5-126s` omitted) + 2w context bleed | Unpop `1.47s` diar + active acoustic band triggers bounded retry recovery of omitted turn; context bleed rejected | Active acoustic band + cross-obs consistency |
| `H2 (360-380)` | 2 | `1` / `1` (`362.5-368s` ~8w silently omitted) | **`0` unpop>=1s diar triggers in `c09`!** Only comparing actual active speech band (`362.5-368s`) vs CTC onset (`368.12s`) detects the omission | Independent speech VAD coverage vs populated CTC intervals (`gap > 2s` during speech-positive audio) |
| `H3 (900-920)` | 0 spoken | `0` / `0` / `0` / `0` (clean negative) | `0` / `0` / `0` / `0` | Clean skip |
| `H4 (920-940)` | 2 | `2` / `0` / `0` / `0` | `2` / `0` / `0` / `0` | Clean merge across `935.06s` pause |

## Architectural Verdict on Required Signal Primitives

**Direct answer:** Operating strictly on `ffmpeg silencedetect` + raw Nemotron responses (`finish_reason`, CTC words, Nemotron diarization) is **NOT sufficient** for a compact, reliable replacement. It fails in four demonstrated failure modes observed in this evidence packet:
1. **Sparse `stop` omissions with 0 unpopulated diar spans:** `F1_c28_wide` (`44s`, 1w, `stop`) and `H2_360_380` (`c09`, `362.5-368s` ~8w omitted, 0 `unpop>=1s` diar spans) + `P4` (`1266.4-1274s` 23w narration omitted under populated-looking diar).
2. **Singing/SFX false triggers:** `CTRL_song` (`25-40s`, 3 `unpop>=1s` diar spans during theme song) and `A_c28_02` (`1112-1123s`, laughs/groans triggering diar).
3. **Pure-SFX anticipatory text hallucination with monotonic CTC:** `CTRL_sfx` (`1203.2s`, identical 4-word hallucination in BOTH `orig_16 c13` and `full_35 c30` over loud SFX).
4. **Internally monotonic yet displaced CTC timing:** `W04c` in `P4` (`1264.44-1266.40s` vs acoustic `1261.5-1263.0s` separated by a real silence pause at `1263.7s`).

Therefore, an event-centered replacement requires **one lightweight independent acoustic-activity / speech-vs-music-vs-SFX detector (e.g., Silero VAD or WebRTC/CNN speech classifier) paired with forced timing alignment only on contested pause-crossing or low-coverage windows**, keeping raw Nemotron responses as immutable `ASRObservation` hypotheses.
