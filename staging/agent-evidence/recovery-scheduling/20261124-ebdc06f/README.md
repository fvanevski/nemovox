# Recovery Scheduling Qualification & Evidence Supplement (`20261124-ebdc06f`)

- **Tested source SHA**: `ebdc06f0bd2997c06d73e5e32158e07beb82e3a6` (tree `2865d1773ca385d6ea5a7695cc50a2c9d28fc51a`)
- **Prior qualified source SHA**: `029f969b43c58d53fd27975ff122c097ead04167`
- **Prior evidence carrier SHA**: `25fd9d15c24bd5623a3ba8cf93d132b713fa4bd4`
- **Result**: `FAIL` (literal `scripts/replay_recovery_schedule.py:93` `tuple != list` bounds comparison defect; all CPU/Studio suites, unmodified-core 5-case replay, targeted CPU Whisper crop pair, and 9/9 primary raw-response authentication passed)

## Contents

- `HANDback.md` — Consolidated qualification handback, reproducer, and findings
- `identities.json` — Exact source, carrier, and supplementary archive identities
- `friction.json` — Canonical `COMMAND_SHAPE_FRICTION` for the runner defect
- `logs/` — Native compilation, pytest (focused, full CPU, Studio), `git diff --check`, challenger env verification, and literal runner traceback
- `receipts/` — Challenger dependency verification and CPU-only GPU/service receipts
- `summaries/` — Sanitized 5-case replay + targeted worker summary and 9/9 primary raw-response hash mapping
- `manifest.json` — Exact file inventory with byte sizes, SHA-256 digests, and command exit codes

## Private Supplementary Archive (Host-Local)

- **Path**: `/workspace/nvidia-transcribe/outputs/handbacks/recovery-scheduling-20261124-ebdc06f/recovery-scheduling-supplement-ebdc06f.tar.gz`
- **Bytes**: `51812`
- **SHA-256**: `a512f262035860a9f3aaa9ac78b523b517c1cc673573af05eacc9eee2b4ffc4d`
