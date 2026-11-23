# Issue #14 Downstream Subtitle Integration Qualification (`20261123T195952Z`)

- **Repository / PR / Issue:** `fvanevski/nemovox` · Issue `#14` (`OPEN`) · Draft PR `#15` (`OPEN`, `issue/14-longform-timing-boundaries`)
- **Admitted Starting PR Head:** `0e7969ef41ca58ea5c2106a677b4b0c514b121a2`
- **Validated Central Source SHA:** `17bb0239f7d8290d43475b06055cd95c2418a1d6` (`tree=1dded4da4ed118632724b69eb83b92f40a80153d`)
- **Source / Test Diff vs Central:** `none`
- **Overall Result:** **`PARTIAL`** (genuine canonical `assemble_revised_translation` ID-permutation blocker exposed and preserved cleanly; all CPU gates, voice overlay, 9-turn Riva translation, raw overlap disposition, and diagnostic end-to-end SRT/ASS render passed cleanly)

## Findings Summary
1. **CPU & Studio Gates (`PYTHONPATH=src:.`):** `compileall` exit 0; focused 6 files **`71 passed`**; full non-GPU `.venv` **`431 passed`**; full non-GPU `.venv-studio` **`431 passed`**; `git diff --check` exit 0; both interpreters verified importing `source_fusion_downstream` from candidate worktree.
2. **Local Acoustic Voice Overlay:** Attached `voice_0`/`voice_1` onto `4` attested `W02b` events (`global speaker_id=SPK_UNK` retained, `[231.95, 232.45]` breath unlabeled/UNRESOLVED, 4 negative controls failed closed).
3. **Preview + 9-Turn Riva Translation:** `assemble_revised_translation(enable=False)` isolated the exact `9` missing turns; `riva_local` under `ModelLifecycle` translated all `9` turns cleanly in `16.065s` (`peak_vram=8017.14 MiB`, restored `3248 MiB`).
4. **Canonical Assembly Blocker vs Diagnostic Render:** Canonical `assemble_revised_translation(enable=True)` rightly raised `ValueError("Cannot reuse translation for changed retained source")` because `utterances.py:660-661` permuted `utt-0148..0152` IDs in `translated.contract.json` (`original_utterance_id`). Under an explicitly diagnostic `original_utterance_id`-rekeyed copy, all `209` retained translations matched `100%` and private `render_subtitles()` improved `W02b`, `W03b` (`unrenderable_cues: 3 -> 2`), and `W04a` (eliminated `-27.3s` SFX and `+20s` displaced cues; `gt_18_cps: 61 -> 60`).
5. **Raw Overlap (`c10:75..78` vs `c11:0..3`):** Emitted `proposed_disposition_not_auto_applied`; single `"Nous avions déjà vu"` heard at `[985.75, 986.90]` in `[984, 1014]` while `[980, 984]` remains uncropped/blocked.
