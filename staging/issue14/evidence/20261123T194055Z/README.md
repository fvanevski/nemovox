# Issue #14 Evidence-Fusion & General Source-Event Partition Qualification (`20261123T194055Z`)

- **Repository / PR / Issue:** `fvanevski/nemovox` · Issue `#14` (`OPEN`) · Draft PR `#15` (`OPEN`, `issue/14-longform-timing-boundaries`)
- **Admitted Starting PR Head:** `ba14ef8d2020fe77a0cbf0549f56139b64dbd60d`
- **Validated Central Source SHA:** `420173025dd28837adc91dfdb970227f3795124e` (`tree=3c98b4f72ab7d68924abcb3a5d1fe8fec2fc42b8`)
- **Source / Test Diff vs Central:** `none`

## Qualification Results
1. **CPU & Studio Gates (`PYTHONPATH=src:.`):** `compileall` exit 0; focused 6 Issue #14 files **`70 passed`**; full non-GPU `.venv` **`425 passed`**; full non-GPU `.venv-studio` **`425 passed`**; `git diff --check` exit 0; both interpreters verified importing from candidate worktree.
2. **Real 2,008-Unit French Partition Replay (`plan_reviewed_source_partitions`):** `2,008/2,008` units matched across `217` legacy utterances (`0` duplicates); `213` connected components -> **`6` complete reviewed components** (`8` legacy utterances -> `9` reviewed turns, `94` units) vs **`3` blocked incomplete reviewed components** (`utt-0034` partial 20/26; `utt-0147/0150` `unowned_reviewed_unit_ids` for 4 dropped `c11:0..3` units; `utt-0154/0156` blocked together because `utt-0156` lacks 2/3 unit reviews).
3. **Opt-In Reversible Revision (`build_reviewed_source_revision`):** `enable=False` returned `new_recording=None`; `enable=True` on the `6` complete components created an isolated private v1 contract (`218` utterances = `209` retained + `9` `src_turn_*`, `SPK_UNK`, `source_words=[]`, `0` deleted raw units, original recording immutable); incomplete-component and stale-SHA controls both raised `ValueError`.
4. **Evidence-Fusion Cases, Policy Variants & Optional Probe:** Verified `7` cases (W02b, W04a, W03b, C2/S05, W04c/S02 SFX, ZH music/character grain, live-action vs animated context control), `v1` vs `v2` policy digests, excluded-ancestor laundering fail-closed rejection, and `3/3` serialized `nemovox-text-review.service` calls passing `adjudicate_review` (`3248 -> 30817 -> 3248 MiB`, `inactive`).
5. **Privacy & Non-Mutation:** `HUMAN_WER_CER_DER=UNVERIFIED`. No scripts, screenshots, WAVs, private recordings, or tokens in public GitHub.
