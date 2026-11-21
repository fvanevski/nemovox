# Nemovox Issue #2 Studio Remediation — Host Integration Handback

**Status:** `PASS` (CPU/API/frontend/browser & B1/B2/I1 verification; live GPU ASR/SwiftU endpoints `UNVERIFIED` as services were not running)

## Authoritative SHAs & Branches

- **Repository:** `https://github.com/fvanevski/nemovox`
- **Issue:** `https://github.com/fvanevski/nemovox/issues/2`
- **Baseline `origin/main`:** `9b4c8b539c67f2f6266fef135a259e92f5477b19`
- **Prior central stage (`origin/central/gui-foundation-20261121`):** `cf74bc8c5a793286bd424b3c3667451a92d8079e`
- **Prior evidence branch (`origin/evidence/gui-foundation-20261121`):** `47e97fc7e52e5d7aae2f4e0dd4e6acefc63aa611` *(preserved untouched)*
- **Final implementation branch (`integration/gui-foundation-20261121`):** `2f0b16e890b2877ec0779818d09d837c79b83086` *(all Central issue #2 commits validated cleanly without requiring additional source commits)*
- **New evidence branch (`evidence/issue-2-studio-20261121`):** branched directly from `2f0b16e890b2877ec0779818d09d837c79b83086`
- **Evidence archive (`staging/agent-evidence/issue-2-studio/issue-2-studio-evidence.tar.gz`):** `1,353,815` bytes (~1.29 MiB), SHA-256 `8285ecd1dab628b11ab62871a3635d22e9c1d5b2b1cf8bbc447f6beb5527fe98`

## Integrated Central Patch (`2f0b16e890b2877ec0779818d09d837c79b83086`)

- `docs/gui/ISSUE2_LOCAL_INTEGRATION.md` — Issue #2 host integration contract and verification checklist
- `src/nvidia_transcribe/studio_api.py` — B1 Linux `/proc` session/process-group descendant tracking & bounded `SIGTERM`/`SIGKILL` teardown (`_group_has_live_members`, `_terminate_process_group`, `JobRunner.shutdown` + FastAPI `lifespan`), honest interrupted-job error message on unclean restart, and B2 contract validation/import (`_read_transcription_contract`, `_format_existing_transcript`)
- `tests/test_studio_api.py` — B1 (`test_cancel_cleans_descendant_after_group_leader_exits`) and B2 (`test_contract_transcription_import_and_empty_contract`) regressions
- `web/src/App.tsx`, `web/src/source-selection.ts`, `web/src/source-selection.test.ts` — I1 active-source invariant (`chooseSourceMethod`, `uploadEpoch`, `sourceMethodRef`, `sourceForSubmission`, `uploadResponseIsCurrent`)

## Verification Highlights

- **B1 Process-Group & Lifespan Shutdown:**
  - Verified unit regression `test_cancel_cleans_descendant_after_group_leader_exits` (`PASSED`).
  - Verified live against Uvicorn (`127.0.0.1:8769`, `--workers 1`) with an owned `NEMOVOX_CLI_PYTHON` wrapper spawning a `SIGTERM`-resistant descendant and exiting early: both `POST /api/jobs/{id}/cancel` and `SIGTERM` to Uvicorn (FastAPI `lifespan` -> `runner.shutdown()`) killed the surviving descendant (`/proc/<pid>` absent) and marked jobs `cancelled`, without signalling unrelated host/GPU processes.
- **B2 Contract-Backed Transcription:**
  - Populated contract (timed + untimed utterances): completes cleanly with downloadable `asr/recording.contract.json` & `asr/transcript.txt` (`200`), preview shows `Transcript cues` with original text (`今天的测试已经准备好了。`, `没有时间的对白也必须保留。`), speaker IDs, and `—` for untimed cues.
  - Empty contract (`{"utterances": []}`): completes cleanly with empty preview cues and `[No utterances in source contract]` in `transcript.txt`.
  - Invalid contract (`{"utterances": [{"id": "broken"}]}`): rejected upfront with HTTP `422` (`Every contract utterance must contain source_text.`) shown in UI `[role="alert"]` before queueing.
- **I1 Active Media-Mode Invariant:**
  - Browser verification via Playwright Chromium + route interception confirmed switching from `Host files` -> `Upload` clears the prior host selection, a delayed `POST /api/media/upload` finishing after switching to `Remote URL` is discarded (`Selected source` remains hidden), and submitting a URL sends `POST /api/jobs` with `source = {"kind": "url", "value": "https://example.org/synthetic-clip.webm"}` only.
- **Subtitle Generation, JASSUB & Multi-Browser Preview:**
  - Subtitles job via real `.venv/bin/python -m nvidia_transcribe.cli generate-subtitles` + ephemeral local OpenAI-compatible mock translator rendered `ASS · libass` overlay on synthetic WebM in Chromium (`--disable-gpu`) and loaded cleanly in Garuda FireDragon `13.13.2` (`HTML text fallback`), preserving state across `?job=...&step=preview` reloads.

## Current Limitations & Out-of-Scope Notes

- **Live GPU ASR / SwiftU inference:** `UNVERIFIED` in this session (no Nemotron vLLM or SwiftU server active on `127.0.0.1:8190` / `8001`; RTX 5090 remained at `3230 MiB / 32607 MiB`, `6%` util).
- **What remains for Central review:** Inspect `integration/gui-foundation-20261121` (`2f0b16e890b2877ec0779818d09d837c79b83086`) and this evidence branch (`evidence/issue-2-studio-20261121`), run any desired PR/late-review gate, and authorize PR creation/merge separately.

## Command-Shape Friction

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events: []
  omitted_sensitive_values=no
```
