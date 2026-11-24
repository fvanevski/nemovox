# Issue #14 — Audio Source Recovery & Selective Postedit Summary (`20261124T035505Z`)

## Verdict
- **Result**: **PASS** — `apply_audio_source_recovery` (`preview` + `apply`), sequential `riva_local` MT, bounded `ninfer_messages` selective English postedit, and isolated `0o700` SRT/ASS render all succeeded end-to-end on real Run B with zero source/translation regressions on retained rows and full immutability of prior 6/7-component artifacts.

## Summary
1. **Preflight & CPU Gates (`604e8dc9c68c13029d3d883a9c2e1a1322b552de`)**: PR `#15` draft/open, Issue `#14` open, `CENTRAL_SOURCE_SHA=604e8dc9c68c13029d3d883a9c2e1a1322b552de`, `CENTRAL_SOURCE_TREE=ba9aed1e190a94245918d421d8e6978d56d7a5aa`, `UNCHANGED_CONFIG_SHA=e5c5a8a6c3443f7bfbefc25119fd392694eca1dc`. Prior archive verified (`11/11` checksums, `12` unique members); prior GitHub packet verified (`7/7+manifest`). `compileall` exit `0`; focused 7-file `85 passed`; `.venv` full core `508 passed`; `.venv-studio` full Studio `508 passed`; `git diff --check` clean.
2. **Real Run B Audio Source Recovery (`208 -> 209` utterances)**:
   - Four bounded original-media WAV crops (`[76,88]`, `[346,364]`, `[279,310]`, `[336,343]`) verified against `media_fingerprint_sha256`.
   - Recovered `8` spoken events (`w01c_ev_goats_alert` [ASR-absent!], `w01c_ev_question`, `w01c_ev_return`, `s02_ev_escale`, `s02_ev_aventure_q`, `s02_ev_oasis`, `s02_ev_zia`, `s02_ev_naacal_neg`), superseded `7` legacy rows (`utt-0006`, `utt-0050`, `utt-0052`, `utt-0055`, `utt-0056`, `utt-0058`, `utt-0064`), rejected `31` raw units (`22` `same_occurrence_alternate` + `9` `not_spoken_asr_artifact`), preserved all raw observations (`0` deleted), and passed `6/6` fail-closed negative controls.
3. **Sequential Riva + Bounded NInfer Messages Postedit + Isolated Render**:
   - `riva_local` translated all `8` fresh turns cleanly (`14.208s`, peak `8011.82 MiB` -> restored `3110 MiB`), and `assemble_revised_translation` retained all `201` unchanged rows.
   - `postedited_audio_recovery_translation` (`ninfer_messages`, `thinking_budget=0`, `reasoning_effort=none`, `max_tokens=1024`, `json_schema`) reviewed ONLY the `8` target turns across `4` calls (`4/4` `finish_reason=stop`, `9.869s`), refining `2` S02 turns (idiomatic verb sense + natural negative-polarity English) while keeping all `201` retained rows bit-identical (`rejected_polarity=0`; offline inversion & truncation controls preserved primary MT).
   - `ModelLifecycle` restored `nemovox-text-review.service` cleanly (`3112 MiB` used / `29015 MiB` free).
   - Isolated `0o700` render produced `278` renderable cues (`0` overflow, `sidecars={}`), leaving prior accepted 6/7-component artifacts bit-identical.
