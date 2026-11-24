# Issue #14 Final Source-Unit Amendment & Long-Form Qualification (`20261124T004243Z`)

- **Admitted PR Head:** `3e15450ef2db7c9952235deead94b80625072a51`
- **Validated Central Source SHA:** `c53f2c3ea4a35c013ed06d7c5d4d72c0e87d9b82` (`tree=69d80d1dc6dbe9fb7c072225932bd112828142cf`)
- **Source / Test Diff vs Central:** `none`
- **Overall Result:** **`PARTIAL`** (Generic amendment & CLI path, acoustic adjudication, 7-component application + 1-turn Riva + 289-cue render, 12+ fail-closed negatives, and complete 23-minute `modelopt_mixed` GPU regression all **PASSED/EXECUTED** cleanly; `PARTIAL` reflects strictly the two genuinely unresolvable residual components `utt-0034` & `utt-0154+0156`, untouched `utt-0151` untimed cue, and live stochastic ASR variance in the fresh 208-utterance full pass rather than any tooling failure.)

## Key Verified Findings
1. **CPU & Studio Gates (`PYTHONPATH=src:.`):** `compileall` exit `0`; focused `9` files **`78 passed`**; `.venv` `-m 'not gpu'` **`446 passed`**; `.venv-studio` `-m 'not gpu'` **`446 passed`**; `git diff --check` exit `0`; both interpreters bound to `/workspace/nemovox-issue14-final/src/nvidia_transcribe/`.
2. **Direct Acoustic Audit of 3 Residual Components (Blind Pass 1 + Unit/CTC Pass 2):**
   - **B1 `utt-0034` (`7cb782e0...`):** Missing `6` units (`[2,70..76]`) belong to an earlier phrase + ~211.5-212.0s chuckle before `w02b_ev_1 [212.05, 215.55]`; **stays blocked**.
   - **B2 `utt-0154+utt-0156` (`3688881d...`):** Missing `2` units (`[11,58],[11,60]`) begin a follow-on clause continuing into `utt-0158` outside fixed `w03b_interroges_eglises_repartant` `source_text`; **stays blocked**.
   - **B3 `w03a_bateau_volant` (`d0df0537...`, `c11:0..3`):** New crop `b3_w03a_979p0_991p0.wav` (`[979.0, 991.0]`, `sha256=1968d6b7...`) directly excludes any competing speech in `[981.2, 985.7]` and confirms ONE occurrence at `[985.75, 989.0]`. Admitted `4` `reviewed_alternate_observation` actions (`raw_unit_observations_deleted=0`).
3. **Amended `7`-Component CLI + Frozen `6`-Component Control:**
   - Frozen `6`-component control reproduced canonical SRT `b0fb66ed...` and ASS `2ac26cb9...` bit-for-bit.
   - `derived_dossier` (`eda34122...`) -> `derived_plan` (`3664f318...`) promotes `w03a_bateau_volant` to **`7` complete components** (`2` remaining blocked), translates the `1` new turn (`src_turn_0767a7666f762d056bb19520`) cleanly via `riva_local` (`15.945s`, `peak=8010.76 MiB`), and runs `apply-reviewed-revision --apply` -> **`217`** utterances (`207` retained + `10` reviewed, `SPK_UNK`, `4` alternates recorded, word timing unverified) and **`289`** cues, reducing **`unrenderable_cues: 2 -> 1`**.
4. **Full 23-Minute Episode GPU Regression (`c53f2c3e`):** Executed under `ModelLifecycle` (`nemovox-asr.service` `quantization=modelopt_mixed`, FP8 audio encoder, CTC timestamps enabled; VRAM restored `3111 MiB` used / `29016 MiB` free): `208` utterances (`127.041s` ASR), `279` cues (`62.839s` sub, `2` unrenderable, `0` ASS overflow). In live stochastic ASR decoding, Chunk 4 omitted `S02` and Chunk 0 omitted `W01c`, underscoring why immutable source ledgers + explicit reviewed CLI revisions remain distinct from ordinary pipeline reruns.
