# Issue #14 Public Sanitized Evidence Packet — `20261124T023146Z`

- **Scope**: NInfer `/v1/messages` system-wire hoisting fix (`CENTRAL_SOURCE_SHA=9230bde41a418c074140252edb6c46637560adb2`, `PR_HEAD=73d5f099805c45fa7061c84d4e16fee83db900b7`) and bounded T1/T2 meaning-first QC under owned `ModelLifecycle`.
- **Outcome**: System-wire fix & thinking-budget enforcement **VERIFIED PASS**; bounded live batch **PARTIAL** (`6/6` HTTP 200, `0` reasoning starvation, `0/6` validator-accepted `advisory_only` due to prompt/enum/schema/fence gaps documented in [validation/meaning-first-quality.json](validation/meaning-first-quality.json)).
- **Privacy Screening**: Zero copyrighted media transcripts, zero raw model `content` / `reasoning_content`, zero weights, and zero secrets are included in this directory. Full raw receipts live only in the offline private archive.

## Contents
- [source-identity.json](source-identity.json)
- [validation/cpu-gates.json](validation/cpu-gates.json)
- [validation/wire-shape-and-t0-replay.json](validation/wire-shape-and-t0-replay.json)
- [validation/ninfer-messages-live-probe.json](validation/ninfer-messages-live-probe.json)
- [validation/meaning-first-quality.json](validation/meaning-first-quality.json)
- [review/issue14-ninfer-messages-wire-summary.md](review/issue14-ninfer-messages-wire-summary.md)
- [COMMAND_SHAPE_FRICTION.txt](COMMAND_SHAPE_FRICTION.txt)
- [manifest.json](manifest.json)
