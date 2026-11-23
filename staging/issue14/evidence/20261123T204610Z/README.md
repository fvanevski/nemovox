# Issue #14 Reviewed Revision CLI Qualification (`20261123T204610Z`)

- **Admitted PR Head:** `5bf0e84f4d5077943f2bed50b68a462231ab10ca`
- **Validated Central Source SHA:** `32bf172d44f2f195b518edc2abd9cbba278c98f4` (`tree=5004b634021d31af04b7718f503c60037cb03473`)
- **Mode:** CPU-only (`NO_NEW_ASR_RIVA_VLM_PYANNOTE=yes`)
- **Result:** **`PASS`**

## Verified Summary
1. **Archives & Tests (`PYTHONPATH=src:.`):** Authenticated canonical (`25/25`), GitHub public (`8/8`), default (`56/56`), fusion (`28/28`), downstream (`34/34`), and w02b (`CHECKSUMS.sha256`) archives. Both `.venv` and `.venv-studio` import `reviewed_revision_runner`, `source_fusion_application`, and `source_fusion_downstream` from `/workspace/nemovox-issue14-revision-cli/src/nvidia_transcribe/`. `compileall` exit `0`; focused `8` files **`73 passed`**; `.venv` `-m 'not gpu'` **`441 passed`**; `.venv-studio` `-m 'not gpu'` **`441 passed`**; `git diff --check` exit `0`; `--help` exit `0`.
2. **Plan & Six-Component Selection:** Recomputed `plan_reviewed_source_partitions` over `2008` units matching `plan_sha256=48ea4dfeea98815c20af587227aca415b3832a49a26490026425f11b8ded9ba1`, **`6`** complete ready components (`8` legacy IDs superseded -> `9` reviewed events) and **`3`** blocked incomplete components (`utt-0034`, `utt-0147+utt-0150`, `utt-0154+utt-0156`).
3. **Read-Only Preview & Opt-In `--apply`:** Preview exited `0` (`status="preview"`, `applied=False`) creating zero directories. `--apply` exited `0` (`status="applied_private_output"`), creating the 5 exact artifacts with **`218`** utterances (`209` retained + `9` reviewed), **`289`** renderable cues, **`fr_canonical.en.srt` SHA-256 `b0fb66ed937c1fd0168a8a33cef8432be8e0830972b24752f565d6e767edb007`**, and **`fr_canonical.en.ass` SHA-256 `2ac26cb94315ee33e452971be310056e76b87a8af0f618e5e9d23097d5d3c779`** — **100% bit-for-bit identical** to prior accepted canonical outputs.
4. **Fail-Closed Negatives (`10/10` non-zero exit):** Tested stale plan SHA, blocked component, duplicate ID, stale dossier/recording digests, missing receipt, invalid local voice event, existing output dir, unsafe stem, and malformed media identity; zero partial dirs or mutations.
5. **Residual Boundaries:** Preserved `3` blocked components (`c11:0..3` unowned proposal vs `c10:75..78`), `SPK_UNK` global speaker labels with local acoustic voice labels in contracts, unverified word timing, Riva terms `nacale`/`langage mu` as tuning candidates, and `HUMAN_WER_CER_DER=UNVERIFIED`.
