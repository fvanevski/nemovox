# Issue #14 Meaning-First QC & Tiered Reasoning Qualification (20261124T013230Z)

- **Result**: `PARTIAL` (CPU gates + T0 boundary QC replay **PASS** with documented left-context/interleaving bounds; live T1/T2 owned text-model probe **EXECUTED -> 6/6 `unaccepted_model_response`** due to `ninfer-serve` / `OpenAIChatTextClient` `reasoning_effort` HTTP 400 fallback + `--thinking-budget 16384` starving `max_tokens=1280/512`).
- **Admitted PR Head**: `0bf9e7bbe5c7991bb92c9e623d49ea31080f0d69`
- **Validated Central Source**: `622715afc00dd6d215fe27c9b5844f615d762971` (tree `1317b67868baec61852e9d84d087b0e0e3994d5d`)
- **Forensic Private Archive**: `sha256=52568c845ab3e7c2541e3711b8950cccde93a11164cb7eadf3c706d4bf923b8e` (`21/21` checksums, `27` members, `0` duplicates)

## Summary of Findings

1. **CPU Gates**: `.venv` (`453/453`), `.venv-studio` (`453/453`), focused semantic-review/timing/fusion (`66/66`), `compileall`, and `git diff --check` all passed against worktree imports.
2. **T0 Boundary QC Replay (`DEFAULT_SEMANTIC_REVIEW_POLICY` & `configs/meaning-first-review.example.json`)**:
   - Canonical `merged_units` invariance verified on Run A (`2008` units) and Run B (`2002` units); accepted 6/7-component CLI hashes (`out_cli`) unchanged.
   - **S02 (`360s`)**: detects `355.31–359.812s` and `350.0–354.034s` as top `reasoned_text` candidates with corroborating `possible_unrepresented_speech=True` intervals, though `left_lexical_correspondence` clipped at `4` words because `left_all` in `semantic_review.py` is global rather than per-local-voice across interleaved `spk_2` tokens.
   - **W01c/W01a (`90s`)**: detects Chunk 1 left-context `80.2–83.254s` as `reasoned_text`; `77.63–80.0s` cannot be recovered from Chunk 1 (audio begins at `80.0s`), and `61.2–63.8s` background shout / `+35s` late echo sit outside left-context detection scope. `[64.0..67.0]s` is confirmed musical transition.
   - **Negative Controls**: `C1`, `C2`, `W04c`, and Mandarin `zh_ep166` (`3269` char units) produced `0` false promotions and `0` candidates (`W04a` surfaced `1` advisory `light_text` exclamation at `1260s`).
3. **Owned Text Model Probe (T1/T2 under `ModelLifecycle` on `nemovox-text-review.service`)**:
   - Lifecycle acquisition/release verified (`startable -> ready_managed -> startable`), `6` cases queued (`6912` reserved tokens), `Optional T3` skipped (`not_materially_needed`).
   - Call 0 (`reasoning_effort="high"`) triggered HTTP 400 (`invalid prompt | messages 2`) on `ninfer-serve`; `OpenAIChatTextClient` sticky-disabled `reasoning_effort` for all `6` calls, leaving `--thinking-budget 16384` active and filling 100% of `max_tokens=1280` / `512` with `reasoning_tokens` (`content=""`).
   - Exact repro preserved for Central fix (`thinking` / `reasoning_effort` compatibility + non-thinking output reservation).
