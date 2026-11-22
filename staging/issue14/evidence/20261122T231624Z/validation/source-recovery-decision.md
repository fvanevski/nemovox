# Offline Candidacy & Safe Automatic Recovery Criteria (`4c32acf6`)

*Note: This document records observed probe evidence and proposed safe selection rules for Central's implementation on PR #15. Zero production recovery code was modified locally.*

## 1. Per-Problem Evidence Summary
- **`c28` (`1100.14..1154.14s`, `finish=length` Loop):**
  - `P2_c28_right` (`1123.0..1156.0s`) succeeded cleanly (`finish=stop`, `27` real words across `1123.39–1154.48s`, `2` speaker tags, exact `<=0.06s` alignment overlap with `c29` `utt-0139`).
  - **Negative counterexamples:** Both `P1_c28_left` (`1099.5..1129.0s`, `2` words across `6` diar spans) and `F1_c28_wide` (`1100.1..1144.1s`, `1` word across `7` diar spans) terminated with `finish_reason="stop"` despite unpopulated multi-second diarization spans.
  - **Safe acceptance criteria:** Never accept an incomplete-chunk retry merely because `finish=stop`; require (a) `audit_asr_generation == "complete"`, (b) monotonic CTC alignment without inter-turn jumps, (c) populated diarization coverage (`words_per_diar_span` / `unpopulated_diar_gap < threshold`, rejecting `1–2` words across `6–7` diar spans), and (d) boundary-overlap agreement against `c29` where present (`[1152.38..1154.48]`).
- **`W04b` / `c31` (`1253s` vs `1264s` Swap & Clip):**
  - `P4_w04b_wide` (`1242.0..1274.0s`) resolved BOTH phrases in chronological order (`W04b` `8` tokens `<spk:0>` at `1253.16–1255.55s`, `W04c` `6` tokens `<spk:1>` at `1264.44–1266.40s`).
  - **Counterexamples:** `P3_w04b_narrow` (`end=1266.5s`) dragged trailing `!` onto `1266.08s`; `F2_w04b_c31match` (`start=1231.0s` mid-dialogue) dropped both `1233s` and `1253s`.
  - **Safe acceptance criteria:** Trigger when a chunk's CTC word stream interleaves across a `>=5s` diarization gap or drops tail tokens past `core_end` while the adjacent chunk starts after that speech (`c32` started `1259.7s > 1253s`). Retry window MUST start cleanly in silent gap after `W04a` (`~1238–1242s`) and include `>=8s` post-target right context (`end >= 1274s`).
- **`W05a` / `c34` (`1352.0..1354.0s` Child Turn Omission):**
  - Trigger is directly observable without listening: `c34` contained an **unpopulated `1.18s` diarization segment (`[1352.63..1353.81] spk=1`)** between `u165` (`1351.97s`) and `u166` (`1354.24s`) with `0` transcribed words.
  - Both `P6_w05a_late` (`1345.0..1379.901s`, `1352.72–1353.89s` + `S05` speaker split) and `F3_w05a_c34plus` (`1338.0..1379.901s`, all `6` words `1352.00–1353.94s` + `1344.75s` turn + `S05` split) recovered the missing turn cleanly, using `u165`/`u166`/`u168` as non-loss anchors and guarding against `2`-word adjacent echoes (`F3`).

## 2. Negative Controls & Preserved Scope Limits
- Clean `C1` (`u089`, `29` words) and genuine repetition `C2` (`u130`, `8` tokens) must remain untouched; genuine `Non! Non! Non!` is explicitly allowed by `audit_asr_generation`.
- Do not concatenate local `<spk:N>` tags across independent clips by raw integer ID; cross-clip integration must run `reconcile_speakers` + `reconcile_chunk_boundaries`.
