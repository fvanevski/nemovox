# Issue #14 Decisive W02b Split & Displaced Source-Unit Adjudication (`20261123T185939Z`)

- **Repository / PR / Issue:** `fvanevski/nemovox` · Issue `#14` (`OPEN`) · Draft PR `#15` (`OPEN`, `issue/14-longform-timing-boundaries`)
- **Admitted Starting Head:** `d8cff10d45d4c27bfdec719cf025621e12a9f3d9`
- **Validated Central Source SHA:** `7bb957ed829649d965445c0aee91bf062b698849` (`tree=f42962cfe3986b2af19bd2e23cfb8ea189a09370`)
- **Source / Test Diff vs Central:** `none` (CPU 50/50 + 411/411 core + 411/411 Studio gates reused under unchanged source)

## Key Adjudications
1. **W02b Source Ordinals 100 (`Toi ?`) & 101 (`Parfaitement !`):** **`different` speakers.** Bounded stereo listening + 50ms RMS establish a `350 ms` silent gap (`[220.70, 221.05]`) between `ordinal 100` (`[220.25, 220.70]`, `Voice_B`) and `ordinal 101` (`[221.05, 222.00]`, `Voice_A`), which continues uninterrupted (`100 ms` clause break) into `"Ambrosius n'a pas de nacale..."` (`222.15-231.50`). True decoder PTS cuts at `220.080s` and `220.920s` match cleanly inside the silent gaps.
2. **W02b Final Turn (`ordinals 128..139`, `[232.45, 235.85]`):** **`Voice_A`** (same bright/brisk timbre as `[212.05, 215.55]`, `[218.60, 220.00]`, and `[221.05, 231.50]`). Prior `Voice_B` assignment was an alternation artifact from merging `100/101`. Supported structure across `[212.0, 236.0]`: **`5` speaker-alternation turns / `6` pause-delimited spoken segments / `2` child voices**, with a `0.95s` gap + faint breath/chuckle `[231.95, 232.45]` whose source identity stays explicitly `UNRESOLVED`.
3. **Offline `source_reconciliation` + `source_unit_lineage` Dossier:** Reconciles `13` reviewed spoken turns (`2008/2008` legacy units matched, `0` deletions, `5` eligible legacy utterances `utt-0037/0147/0148/0149/0153`), while accurately blocking `utt-0034`, `utt-0036`, `utt-0150`, `utt-0152`, `utt-0154`, `utt-0156`, `utt-0187`, and `utt-0188` under the all-units-one-event & 3s time-slack invariants.
4. **Epistemic Status:** `HUMAN_WER_CER_DER=UNVERIFIED` (host primary acoustic/PTS qualification only). No screenshots, `.wav`s, scripts, or secrets in GitHub.
