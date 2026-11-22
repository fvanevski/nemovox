# Nemovox Long-Form French Subtitle Audit — Baseline Evidence Packet (`20261122T205259Z`)

- **Repository:** `fvanevski/nemovox`
- **Task:** Pre-implementation diagnostic evidence collection for Episode 1 (*Les mystères / MCOG S04E01*, `1379.901s`), prior to remediation design.
- **Execution Classification:** `EVIDENCE ONLY` (read-only audit; zero source, config, job, or media mutations; zero GPU/model runs).
- **Authoritative `main`:** `8f6c03cbc70e84bf1970ba364207a02c9fe22b44` (tree `21ae1c6d1b682f1061ae36f6896cd0ac87db1580`)
- **Handoff Commit (`evidence/mcog-s04e01-baseline-20261122`):** `b32c6c91f821ef0bff0c5a9e9e5942b5a0fd189b`

## Source Epoch Distinction
The audited Studio job (`980af31f014245c4b200e0af8b33ebca`, `2026-11-22 09:41-09:46 PST`) predates the PR #13 / `8f6c03cb` deployment (`12:04 PST`) and did not record a git commit SHA in `job.json`, `manifest.json`, `recording.contract.json`, or `execution.log`. Accordingly, `ORIGINAL_JOB_EXECUTION_SHA` is recorded as `UNVERIFIED`. Offline read-only replay of `parse_ctc_timestamps`, `recover_chunk_alignment`, `normalize_chunk_speakers`, `reconcile_chunk_boundaries`, `reconstruct_speaker_utterances`, `build_display_cues`, and `assign_subtitle_lanes` imported directly from `8f6c03cb` reproduces the archived `234` utterances (`53` boundary deduplications, `4` recovered units) and `308` ASS events (`277` Main, `26` Top, `5` TopAlt) identically.

## Privacy & Omissions
- No full copyrighted transcripts, ASS/SRT payloads, raw `chunk_*.raw.json` files, audio WAV clips, or screenshots are committed.
- Personal host path prefixes are redacted cleanly (`<INPUT_VIDEO>`, `<INPUT_ASS>`, `<INPUT_CATALOG>`, `<INPUT_JOB>`, `<ISOLATED_WORKTREE>`, `<PRIVATE_SCRATCH>`).
- Full per-token private traces and bounded `ffmpeg`-extracted WAV subclips remain strictly in a private `chmod 700` host directory (`PRIVATE_SUPPORT_REQUIRED=no`).

## Case & Artifact Index
- **Priority Windows:** `W01a..W01d` (`00:25-01:25`), `W02a..W02d` (`03:18-05:12`), `W03a..W03c` (`16:20-18:10`), `W04a..W04c` (`20:00-21:10`), `W05a` (`22:25-22:40`).
- **Controls:** `C1` (`11:27-11:36`, clean mid-core baseline), `C2` (`17:03-17:06`, genuine intra-speaker repetition under overlap).
- **Supplementary Cases:** `S01..S05` (fragment Qwen regressions, negation inversion, cross-chunk lexical error mask, `82.35 cps` clamp, tail turn collapse).
- **Files:** `provenance.json`, `baseline-metrics.json`, `cases.jsonl`, `boundary-traces.md`, `speaker-and-translation.md`, `acoustic-spotchecks.md`, `acceptance-matrix.md`, `findings.md`, `COMMAND_SHAPE_FRICTION.txt`, `manifest.json`.
