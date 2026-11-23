# Positive Same-Event Supersession & Preservation Case Matrix

All checks executed against unmodified Central code (`e3b10a94e0a1`) with `--source-evidence` enabled.

## 1. French (`fr_orig16`) Positive & Guard Cases
- **`A_c28_03_heroic_rescue` (`1131.5-1136.0s`)**: `9/9` lexical units of `utt-0193` map to this reviewed event (`fully_same_event_time_plausible`).
  - Without `supersedes_legacy_utterance_ids`: quarantined (`reviewed_same_event_requires_complete_explicit_supersession`), preventing duplicate subtitle insertion.
  - With `supersedes_legacy_utterance_ids=["utt-0193"]`: cleanly replaces `utt-0193` (`superseded_legacy_utterance_count` increments by 1) and retains reviewed bounds.
- **`B_w04_on_a_reussi_tao` (`1252.0-1255.8s`)**: `5/5` lexical units of truncated `utt-0206` map to this reviewed event (`fully_same_event_time_plausible`).
  - Without `supersedes_legacy_utterance_ids`: quarantined (`reviewed_same_event_requires_complete_explicit_supersession`).
  - With `supersedes_legacy_utterance_ids=["utt-0206"]`: cleanly replaces `utt-0206`.
- **`B_w04_W04c_captain_shout` (`1261.5-1263.5s`) — Partial & Displaced Protection**:
  - `utt-0207` (`1254.61-1254.825s`, `1/1` unit): classified `fully_same_event_timing_displaced_or_missing` (>3s displaced). Preserved in output; attempting explicit supersession quarantines with `['legacy_supersession_audio_bounds_unverified', 'legacy_turn_lacks_positive_same_event_lineage', 'requested_supersession_not_proven_by_immutable_source_units']`.
  - `utt-0208` (`1264.36-1266.575s`, `5/7` units): classified `partial_or_multiple_reviewed_audio_events`. Preserved in output and reported in `partially_attributed_legacy_utterance_ids`; attempting explicit supersession quarantines with `['legacy_turn_lacks_positive_same_event_lineage', 'requested_supersession_not_proven_by_immutable_source_units']`.
- **`CTRL_c2_repeat_pair` (`utt-0177` & `utt-0178`)**: both turns preserved distinctly.
- **`S02` negation (`utt-0070`)**: source negation preserved through Riva translation (no inversion).
- **Controls (`CTRL_song_25_40`, `CTRL_sfx_1200_1216`, `D_w02b_colliding_turn`, `utt-0202` 1203s bleed)**: all non-speech/colliding candidates quarantined; legacy `utt-0202` preserved.

## 2. Mandarin (`zh_ep166`) Full Episode Positive & Guard Cases
- **`ZH_overlap_turn1_elder` (`518.1-524.5s`)**:
  - `utt-0064` (`1/1` unit, `525.16-525.375s`): `fully_same_event_time_plausible` (1-char tail split from `utt-0062` within 3.0s slack). Superseded cleanly when explicitly listed.
  - `utt-0062` (`18/28` units, `515.61-523.991s`): `partial_or_multiple_reviewed_audio_events`. Preserved intact (`partially_attributed_legacy_utterance_ids: ["utt-0062"]`); attempting supersession quarantines with `['legacy_turn_lacks_positive_same_event_lineage', 'requested_supersession_not_proven_by_immutable_source_units', 'reviewed_same_event_requires_complete_explicit_supersession']`.
- **`ZH_overlap_turn2_young` (`525.1-526.7s`)**: `5/5` units of `utt-0063` map cleanly (`fully_same_event_time_plausible`); superseded explicitly.
- **`ZH_overlap_turn3_elder` (`526.7-530.2s`)**: `10/10` units of `utt-0065` map cleanly (`fully_same_event_time_plausible`); superseded explicitly.
- **Untimed Tail (`41` utterances including `utt-0284`)**: `100%` (`41/41`) preserved without fabricated timestamps.
