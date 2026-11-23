# Issue #14 Independent Timing Anchors, Direct Audio & Bounded Phase 3b AV Evidence (`20261123T183454Z`)

- **Repository**: `fvanevski/nemovox`
- **Issue / PR**: `#14` (OPEN) / Draft PR `#15` (OPEN, `issue/14-longform-timing-boundaries`)
- **Admitted PR Head SHA**: `9162dc6fa9abe2ea73325e74a7e74bd1add67ade`
- **Validated Central Source SHA**: `7bb957ed829649d965445c0aee91bf062b698849` (`tree=f42962cfe3986b2af19bd2e23cfb8ea189a09370`)
- **Source / Test Diff vs Central**: `none`

## Summary of Executed Campaign
1. **CPU & Dual-Venv Gates**: `compileall`, `git diff --check`, focused Issue #14 suite (`50 passed`), core `.venv` non-GPU suite (`411 passed`), and Studio `.venv-studio` non-GPU suite (`411 passed`) all passed using worktree imports (`PYTHONPATH=src:.`).
2. **Offline Cross-ASR Timing Replay**: Replayed `source_challenger_timing` on archived `whisper_large_v3` (`44` candidates, `4` displaced `W03b` c11 spans at `-5.925s, +6.44s, +11.43s, +10.1s`, `1` unmatched `FR_C2_repeat`, `no_automatic_source_mutation=True`), `whisper_large_v3_turbo` (`41` candidates, `4` displaced, missed `FR_W04c_reanchored` fanfare), and Mandarin EP166 control (`0` candidates, character grain preserved).
3. **Bounded Direct-Audio Review**: Audited `12` archived clips plus `3` newly extracted `FR_W04a` clips (`1201-1208s` SFX only proving `-27.3s` early pull, `1229-1238s` true speech, `1247-1256s` proving `+20s` displacement) and verified `W02b`'s `6` alternating child turns (`212.05-235.85s`) via 100ms RMS envelopes. Frozen pre-vision audio baseline (`a0024f7d...`) locked before visual extraction/review.
4. **Bounded Phase 3b Audiovisual Experiment (Evidence-Only)**: Verified multimodal capability of `nemovox-text-review.service` (`chat`, `--vision`) under `ModelLifecycle` (`3105 -> 30675 -> 3103 MiB`, cleanly stopped after run). Executed strictly separated Pass 1 (visual-only) and Pass 2 (constrained AV fusion) across `8` scenes (`16/16` parsed). Vision remains **non-integrated** into production.

## Sanitized File Index
- `manifest.json`
- `source-identity.json`
- `COMMAND_SHAPE_FRICTION.txt`
- `README.md`
- `review/issue14-independent-anchors-summary.md`
- `validation/cpu-gates.json`
- `validation/challenger-timing-replay.json`
- `validation/audio-reviewed-turns.json`
- `validation/audiovisual-experiment.json`
- `validation/cases-22-matrix.json`
- `validation/mandarin-control.json`
- `validation/resource-nonmutation.json`

## Privacy & Non-Mutation
No copyrighted screenshots, contact sheets, `.wav` clips, full transcripts, secrets, or tokens are included in this packet. Full logs, contact sheets, frames, raw Pass 1/Pass 2 prompts/responses, and `.wav` attestations remain host-private in the companion archive.
