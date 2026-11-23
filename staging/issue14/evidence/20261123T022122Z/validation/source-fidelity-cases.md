# Issue #14 Bounded Source-Fidelity & Acoustic Identity Cases (`20261123T022122Z`)

- **Direct Audio Review Provenance:** Native `audio/wav` multimodal listening via `read_file` on 10 bounded 16kHz mono PCM WAV clips (`review_evidence_sha256=5295c606ecb8ca5efd5cadaa4337f099f7e891b510f48306ae14f86ebbc9ab98`), cross-checked with `20261123T000342Z` acoustic-event-ledger (`private_acoustic_reference.json` SHA-256 `30292212c209241fca4914fe2d41410c844ab7db5a680d912ba6aea7e1460545`).
- **WER / CER:** `UNVERIFIED` across full episodes (no full-episode verbatim human reference).
- **Zero-Deletion Invariant:** `source_unit_observations_deleted = 0` across all dossiers (`fr_orig16`, `fr_full35`, `zh_ep166`).

## Case Outcomes

1. **`W04c` Captain Shout (`B_w04_W04c_captain_shout`, `[1261.5, 1263.5]`):**
   - **Auditioned:** 5-word French shout heard before the `[1263.5, 1264.4]` silence/music break (`source_text_verified=true`, SHA-256 prefix `251094204a6e`).
   - **Joint-VAD Miss Confirmed:** All 3 detectors (`Silero`, `FireRedAED`, `pyannote`) missed `[1261.5, 1263.5]` (`model_activity_evidence` shows only `music`).
   - **CTC Displacement & Same-Event Resolution:** `full35 c31` & `c32` displaced CTC to `[1264.46, 1266.52]`, yet both parent inference windows (`[1226.2, 1267.7]` and `[1259.7, 1308.6]`) intersect `[1261.5, 1263.5]`. Linking all 10 unit IDs resolves `corr_c2eaf165c404fdbfad509dc7` (`full35`) and `corr_cd068a7f16d40047fbd0eee5` (`orig16`) to `same_audio_event_reviewed_keep_observation_lineage` (`0` deletions).
   - **Short-Crop vs Context Control:** Prior 3s isolated crop `P2_W04c_captain` hallucinated 4 unrelated French words, whereas 24s Qwen floor window `win[57]` (`[1254, 1278]`) covered the shout (`window_only_unlocalized_hypothesis`, `source_content_accepted=false`).

2. **`H2` Opening (`H2_opening_faites_voir`, `[362.5, 364.2]`) & Tail (`H2_tail_ambrosius`, `[375.5, 378.0]`):**
   - **Auditioned:** 2-word opening phrase heard at `[362.5, 364.2]` (`source_text_verified=true`), omitted by `full35 c09` and missed by all 3 CPU detectors; covered by Qwen floor `win[16]` (`[352, 376]`).
   - **Same-Event Correspondence:** Tail event (`[375.5, 378.0]`) links 16 units across `c09`/`c10` (`corr_bdb8e24948694b4a1cee64fa` -> `same_audio_event_reviewed_keep_observation_lineage`).

3. **`c28` Incomplete Chunk Rescue (`A_c28_03_heroic_rescue`, `[1131.5, 1136.0]`):**
   - **Auditioned:** 9-word urgent shout heard at `[1131.5, 1136.0]` (`source_text_verified=true`).
   - **Incomplete-Chunk Boundary:** `full35 c28` (`[1100.14, 1154.14]`) failed `generation_limit_exhausted` (`0` accepted ledger units, `legacy_asr_window_context_ids=[]`), prioritized first (`generation_incomplete`, 3 windows) in recovery planning and covered by Qwen `win[51]` (`[1122, 1146]`). In `orig16`, `c12` preserves all 9 lexical units.

4. **Genuine Repeat `CTRL_c2` (`[1022.8, 1024.7]` & `[1024.8, 1026.8]`):**
   - **Auditioned:** Two distinct spoken instances of the same 4-word French exclamation (`CTRL_c2_repeat_1`, `CTRL_c2_repeat_2`).
   - **Result:** Emits two distinct `source_turn_id`s (`src_turn_8f4e5ab43d9ca455eae1d0d5`, `src_turn_e5e026ce3e6ddc2c92ae69d2`), proving identical text is never merged across distinct events.

5. **Non-Speech Singing/SFX Hallucination & False Correspondence Controls:**
   - **`CTRL_song_25_40` (`[25.0, 40.0]`):** Pure theme song singing (`FireRedAED` reports `singing` + `music`); Qwen `win[01]` and `P5_CTRL_song` emitted French lyrics. Dossier enforces `source_content_accepted=false`, `shared_non_speech_risk=true`, `0` promoted turns.
   - **`CTRL_sfx_1200_1216` (`[1200.0, 1216.0]`):** Pure collapse SFX + grunts; Qwen `P6_CTRL_sfx` emitted 2 exclamation tokens, Nemotron `c30` hallucinated 4 words at `1203.23s` linked by `corr_e7b147c2dc051e2106ced627` (`offset=-15.3643s`) to `c31` (`[1231.20, 1232.82]`). Leaving `c30` unlinked keeps `corr_e7b147c2dc051e2106ced627` unresolved (`preserve_both_until_positive_audio_event_adjudication`); mapping left vs right to distinct events produces `distinct_audio_events_reviewed_preserve_both` with `0` deletions.
   - **Disjoint Owning Window Negative Control:** Linking `full35 c00` (`[0, 46.8]`) to `W04c` (`[1261.5, 1263.5]`) failed closed (`ValueError: Reviewed unit has no parent audio-window overlap with the event`).

6. **Mandarin `ep166` Collision (`525.16s`) & Incomplete `c14`:**
   - **Auditioned (`ZH_overlap_518_536`):** Rapid sequential 3-turn handoff (`ZH_overlap_turn1_elder` `[518.1, 524.5]`, `ZH_overlap_turn2_young` `[525.1, 526.7]`, `ZH_overlap_turn3_elder` `[526.7, 530.2]`), NOT true simultaneous overlap. Preserves all 34 observed characters across distinct turns despite `525.16s` CTC collision (`unit_00d45eae1a9f07ee44a99060` vs `unit_e28d01281b5dcca28542b281`).
   - **Incomplete `c14` (`[1250.0, 1318.88]`):** Strict replay rejects `c14` (`generation_limit_exhausted`); forensic replay preserves `0` units and schedules 4 `generation_incomplete` windows ahead of floor.
   - **Forced Alignment Provisional Gate:** Re-verified `FALSE_zh_on_fr_sfx` receives `structurally_provisional` (`zero_duration_ratio=0.333333 < 0.4`) with `text_verified=false` and `timestamp_is_acoustic_truth=false`.
