# Issue #14 Architecture Reconciliation & Next Central Inputs (20261123T013938Z)

## 1. Validated Invariants

1. **Fail-closed source replay:** `source_event_replay` rejects `finish_reason=length` chunks (`fr_full35 c28`, `zh_ep166 c14`) in strict mode even when legacy manifests marked them `completed`, and preserves explicit incomplete gaps in `--forensic-incomplete` mode.
2. **Joint-VAD blind-spot closure via opt-in coverage floor:** Under ALL-detector-negative conditions (`orig16` & `full35` without `--auditioned-misses`), selective activity planning yields `0` requests covering `W04c` (`1261.5-1263.5s`) and `H2` opening (`362.5-364.2s`). Enabling `--coverage-end-s` schedules bounded `24s` (`2s` overlap) windows over the entire media (`63` FR, `60` ZH).
3. **Explicit budget accounting (with priority-ordering nuance):** On `fr_full35` with `--max-requests 128`, `activity_without_timed_source_hypothesis` generates `73` uncoalesced per-detector intervals (priority `2`), so total requests reach `139` (`145` with `6` misses) and `11` (`17`) tail `independent_coverage_probe` windows (`~1138-1379.901s`, including `W04c` `[1254.0, 1278.0]`) go explicitly into `unplanned_due_to_budget` (`status=partial_budget`) instead of silently vanishing.
4. **Source-turn lineage preservation:** `assemble_source_turn_revision` -> `project_source_turn_to_v1_utterance` -> `translate_utterances` -> `build_display_cues` / `assign_subtitle_lanes` / `render_subtitles` preserves `source_turn_ids` across normal, overlapping, and unrenderable cues, blocks cross-event regrouping, suppresses only exact same-event render copies, and keeps v1 checkpoint/resume caching intact.
5. **Forced-alignment truth limit:** Re-checking all `9` batch + `4` bounded GPU cases through `inspect_forced_alignment` confirms `FALSE_zh_on_fr_sfx` still receives `structurally_provisional` (`zero_duration_ratio=0.333 < 0.4`) while `text_verified=False` and `timestamp_is_acoustic_truth=False`.

## 2. Unresolved Decision Inputs for Central

1. **Coalescing overlapping detector activity & priority budgeting:** Currently `source_recovery.py` coalesces only `generation_incomplete` proposals, leaving `73` overlapping multi-detector `activity_without_timed_source_hypothesis` proposals ahead of `independent_coverage_probe` so `--max-requests 128` defers the episode tail (`W04c`) to `unplanned_due_to_budget`. Central should decide whether to coalesce overlapping `activity_without_timed_source_hypothesis` intervals or separate coverage-floor budget from selective-gap budget.
2. **Singing/SFX non-speech gating before Qwen acceptance:** Qwen3-ASR-1.7B coverage sweep recovered `W04c`, `H2`, `c28`, `S05`, and `ZH_ordinary`, but hallucinated French lyrics on `P5_CTRL_song`, `win[00]` (OP song), and 1-2 word/char fillers on SFX/music (`win[18,19,24,25,41]` in FR; `win[00..03,56..59]` in ZH), and hallucinated on isolated 3s crop `P2_W04c_captain`. Upstream `AudioRegion` / `FireRedAED` (`singing`/`music` vs `speech`) + multi-ASR adjudication must gate Qwen candidates before `AcceptedSourceTurn` promotion.
3. **Conservative `fr_orig16` `+1` cue retention:** Re-rendering `fr_orig16` increased cues from `308` to `309` (`+1` previously similarity-suppressed cue). Until upstream cross-chunk event correspondence classifies unresolved ASR echoes, render-time preservation retains it conservatively.
4. **WER / CER:** Remains `UNVERIFIED` pending human-verified verbatim source transcripts for the audited regions.

## 3. Next Cohesive Central Objective

Implement the model-bound `AudioRegion` + multi-ASR reconciliation controller that gates Qwen/Parakeet/Nemotron candidates against `FireRedAED` non-speech classes, resolves cross-chunk ASR echoes into stable `AcceptedSourceTurn` / provisional `source_turn_id` decisions with bounded timing/speaker advisories, and qualifies one new end-to-end episode gate.
