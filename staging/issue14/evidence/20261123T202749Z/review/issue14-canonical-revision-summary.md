# Issue #14 Canonical Revision Subtitle Qualification (`20261123T202749Z`)

- **Admitted Head:** `a3aa626545d9b7b42d42c927067e99f34b6f50f4`
- **Central Source:** `59a1b6a07613c1d7901f549af9c61db1d5db8696` (`tree=14aabba2a241fdec636970ffb6a2a8bc07fff82e`)
- **Execution Mode:** CPU-only (`NO_NEW_ASR=yes`, `NO_NEW_RIVA=yes`, `NO_NEW_VISION_RUN=yes`)
- **Result:** **`PASS`**

## Verified Highlights
1. **CPU & Studio Gates (`PYTHONPATH=src:.`):** `compileall` exit 0; focused 8 files **`73 passed`**; full `.venv` `-m 'not gpu'` **`436 passed`**; full `.venv-studio` `-m 'not gpu'` **`436 passed`**; both interpreters import `source_fusion_downstream.py` from `/workspace/nemovox-issue14-canonical-revision/src/nvidia_transcribe/`.
2. **Untouched 217-Row `original_utterance_id` Bijection:** Proven `217/217` one-to-one mapping with `0` source mismatches and exact `5` display-renumbered rows (`utt-0148..0152`).
3. **Canonical `assemble_revised_translation(..., enable=True)` (Zero External Rekeying):**
   - `enable=False` returns `report.enabled=False`, `translated_contract=None`, `historical_translation_renumbered_rows=5`.
   - `enable=True` directly on untouched `recording.contract.json` (`a67056da...`) and `translated.contract.json` (`9422b386...`) + `9` existing Riva receipts produces **`218`** utterances (`209` retained + `9` new reviewed), **`209/209`** exact source & English equality, `historical_translation_renumbered_rows=5`, `historical_translation_identity_map_sha256=85ae485bcf77c954781e518b459484669996fb28bd66e7d1cc46e0bb2636b578`.
   - Zero active `grouped_utterance_ids`, `merged_into`, or `probable_duplicate_of` point to superseded IDs; `utt-0151` (historical `utt-0149`) cleanly migrated `"probable_duplicate_of": "utt-0151"` -> `"historical_probable_duplicate_of_superseded": "utt-0148"`.
   - All `6` private fail-closed negative controls raised `ValueError`.
4. **Canonical Private Render (`subtitles.render_subtitles`):**
   - `fr_canonical.en.srt` is **byte-for-byte identical** (`b0fb66ed937c1fd0168a8a33cef8432be8e0830972b24752f565d6e767edb007`) to prior diagnostic `fr_revised_diag.en.srt` (`289` cues, `unrenderable_cues=2` down from `3`, `conflict_cues=54` down from `55`, `gt_18_cps_raw=61` down from `62`, `gt_22_cps=27`).
   - `fr_canonical.en.ass` differs from `fr_revised_diag.en.ass` strictly in `[Script Info] Title: fr_canonical (en)` (`289` dialogue lines, `13` lane-1 top cues, `0` ASS overflow).
   - Preserves `c10:75..78`/`c11:0..3` proposed/unresolved, 3 blocked components, `SPK_UNK` global speaker absence, unverified word-level timing, and `HUMAN_WER_CER_DER=UNVERIFIED`.
