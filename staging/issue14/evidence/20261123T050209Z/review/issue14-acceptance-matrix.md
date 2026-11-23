# Issue #14 / PR #15 Integrated Source-Fidelity Acceptance Matrix

- **Admitted PR Head:** `e975cd86856c6fd47bd61a559f6402ad61be6512`
- **Central Source SHA:** `e3b10a94e0a1214856aef7ade1258e1f4a5a4d41` (unchanged on branch; `SOURCE_DIFF=none`)
- **Overall Result:** `PARTIAL` (CPU gate has 1 test-fixture failure; host lineage/context-gate/supersession/Riva phases executed end-to-end)

| Gate / Requirement | Status | Evidence & Finding |
|---|---|---|
| Phase A: `compileall` & `git diff --check` | PASS | Both exited `0` |
| Phase A: Focused (`90` tests) & Full non-GPU (`394` tests) | FAIL (exit `1`) | `89/90` focused & `393/394` full passed; `test_short_fr_and_rapid_zh_model_speech_are_not_blanket_rejected` failed (`ValueError` at `source_promotion.py:164`) due to stale `"fr"` in `dossier()["source_turns_audio_review_attested"][0]["source_language"]` |
| Phase B: FR `orig16` & ZH `EP166` exact unit lineage | PASS | FR `2051/2051` matched (`0` deleted); ZH `2832/2901` matched (`69` unmatched in chunk 14 `utt-0284`, `0` deleted); cross-run `full35` vs `orig16` rejected |
| Phase C: Context-contamination screen (`model_text_scope_exceeds_audio_event_duration`) | PASS | Old `7` accepted FR turns reduced to `3` (`W04c`, `H2`, `H1`); all `4` bloated candidates (`S05`, `W05_tail`, `W04b`, `D_w03a`) quarantined |
| Phase C: Positive same-event supersession & partial/displaced guards | PASS | FR (`utt-0193`, `utt-0206` replaced; `utt-0207`/`0208` protected); ZH (`utt-0063/0064/0065` replaced; `utt-0062` & `41` untimed protected); duplicate reviewed appends blocked |
| Phase C: Bounded local audio listening (`read_file`) | PASS (with findings) | Confirmed `B_w04_esteban_tao_question` (7w) & `C_w05_S05_omitted` (7w); kept `C_w05_tail` & `D_w03a` quarantined; flagged `W04c`/`H2` timing offsets in prior dossier |
| Phase D: Full FR & Full ZH English Riva SRT/ASS renders | PASS | FR `237/312` (Central) & `239/314` (agent-recovered) vs `234/309` baseline & `241/320` unsafe; ZH `285/304` baseline & `285/305` vetted (`41` untimed preserved, `0` CJK leaks) |
| Host GPU & Services Non-Mutation | PASS | Desktop (`kwin_wayland`, `plasmashell`, `Xwayland`), `text-embeddings-router`, `local_embedding`, `studio_api` preserved; VRAM `3105 -> 3103 MiB` |
| `WER_CER` | UNVERIFIED | No independent full-episode human verbatim reference |
