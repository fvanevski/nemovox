# Issue #14 Public Sanitized Evidence Packet — `20261124T035505Z`

- **Scope**: End-to-end qualification of `apply_audio_source_recovery`, sequential `riva_local` translation, bounded `ninfer_messages` selective English postedit, and isolated `0o700` SRT/ASS rendering (`CENTRAL_SOURCE_SHA=604e8dc9c68c13029d3d883a9c2e1a1322b552de`, `PR_HEAD=604e8dc9c68c13029d3d883a9c2e1a1322b552de`).
- **Outcome**: **PASS** (`85` focused / `508` core / `508` Studio passed; `8` S02/W01c events recovered including ASR-absent turn; `31` raw units explicitly accounted/rejected; `201` retained English rows bit-identical; `4/4` structured JSON postedit calls `stop`; `278` rendered cues).
- **Privacy Screening**: Zero copyrighted media quotes, zero private ASR transcripts, zero WAV crops, and zero credentials are in GitHub. Full recovered contracts, WAV crops, SRT/ASS files, and receipts live only in the offline private archive.

## Contents
- [source-identity.json](source-identity.json)
- [validation/cpu-gates.json](validation/cpu-gates.json)
- [validation/audio-recovery-and-unit-accounting.json](validation/audio-recovery-and-unit-accounting.json)
- [validation/translation-postedit-and-render.json](validation/translation-postedit-and-render.json)
- [review/issue14-audio-source-recovery-summary.md](review/issue14-audio-source-recovery-summary.md)
- [COMMAND_SHAPE_FRICTION.txt](COMMAND_SHAPE_FRICTION.txt)
- [manifest.json](manifest.json)
