# Issue 14 completion qualification — CPU regression gate block

**Blocking CPU regression failure before GPU execution.** Tested application `db8faeaa7432e1f53fbbc254b145453b191551dc`, tree `06cc432bd15fe7a5086a1dd7bab4d412de81960d`.

In an isolated clean worktree at the exact candidate commit, diff checks, `compileall`, and CLI/Studio environment imports exited 0, and all three new subchunk-recovery tests in `tests/test_issue14_asr_integrity.py` passed. However, **all non-GPU test runs failed the existing regression test `tests/test_issue14_asr_integrity.py::test_legacy_completed_checkpoint_cannot_hide_length_finish`**:

- Focused non-GPU suite (23 test files): **279 passed, 1 failed** (exit 1)
- Full CLI `.venv` non-GPU suite: **419 passed, 1 failed** (exit 1)
- Full Studio `.venv-studio` non-GPU suite: **419 passed, 1 failed** (exit 1)

Under the contract's CPU-first gate and stop condition for failed required regressions (without modifying tests or source to force green results), GPU execution was not entered. Neither the French episode resume nor the independent Mandarin Studio job was started.

## Minimal reproduction and root cause

Commit `db8faeaa7432e1f53fbbc254b145453b191551dc` (with parent `9aa89a0fd29433f5ec20ac30e21675df6d43f194`) introduced `"planned_chunks"` in `manifest.json` and updated `src/nvidia_transcribe/recording.py` (`line 278`) to load resume checkpoints from `existing_manifest.get("planned_chunks", existing_manifest.get("chunks", []))`.

1. On a fresh `process_media(...)` run, `manifest.json` now contains both `"chunks"` and `"planned_chunks"`.
2. `test_legacy_completed_checkpoint_cannot_hide_length_finish` replaces `chunks/chunk_000.raw.json` with a `finish_reason="length"` response and updates `manifest["chunks"][0]["raw_response_sha256"]`, leaving `manifest["planned_chunks"][0]["raw_response_sha256"]` unchanged with the original `"stop"` digest.
3. On replay, `recording.py` reads `"planned_chunks"`, detects a digest mismatch between `planned_chunks[0]` and `chunk_000.raw.json`, drops `chunk 0` from `completed_by_idx`, deletes `chunk_000.raw.json`, and calls `client.transcribe(...)` again instead of auditing the suspect checkpoint and raising `IncompleteASRGeneration`.
4. Removing `"planned_chunks"` from the simulated manifest (matching a genuine pre-`9aa89a0` manifest) causes `recording.py` to fall back to `"chunks"`, audit the suspect file, and raise `IncompleteASRGeneration` with zero `transcribe` calls.

## Preserved prior state and host health

- Read-only verification of the prior incomplete French output confirmed all 29 manifest entries (28 `completed`, 1 `incomplete_generation` at chunk 28) and their WAV/raw JSON SHA-256 digests remain intact and unmodified (`config_signature` `65c6a4af9fc2169a0924c60ee38a3afd9f68c6c13584bcc3138f6e995fc91137`).
- No ASR or NInfer services were started; `nemovox-asr.service` and `nemovox-text-review.service` remain inactive, ports 8190 and 8001 have zero listeners, `model-lifecycle.lock` is acquirable (`inode` 55650998), and production `nemovox-studio.service` (`InvocationID=6cd5cf36a6774b0693d37a196c088464`, `MainPID=1121444`) reports `/api/health` status `ok`.

[Original gate matrix](gate-matrix.md) · [Metrics](metrics.json) · [Commands](commands.md) · [Command friction](COMMAND_SHAPE_FRICTION.txt) · [Checksums](SHA256SUMS)
