# Antigravity CLI: Integrate centrally staged Nemovox Studio

**Repository:** https://github.com/fvanevski/nemovox
**Central branch:** central/gui-foundation-20261121
**Original main baseline:** 9b4c8b539c67f2f6266fef135a259e92f5477b19
**Host workspace:** /workspace/nvidia-transcribe
**GPU:** single RTX 5090, model directory /workspace/models

## Kickoff

This is a host integration, verification and bounded repair handoff. Central ChatGPT has ALREADY authored and staged the FastAPI API/job runner, a three-tab React GUI, typed API client, styles, and API tests.

Do not redesign the project or delegate major implementation back under the guise of wiring. Use central files as implementation authority while confirming the latest exact branch SHAs and local runtime. The target is one working personal application.

## 1. Git

- Read AGENTS.md and docs/gui/ARCHITECTURE.md.
- git fetch origin; read exact full HEAD of origin/main and origin/central/gui-foundation-20261121.
- Preserve unrelated/dirty local changes; do not reset user work.
- Create one working branch based on the central GUI branch, for example agent/gui-foundation-integration-20261121.
- Fix observed build/API/integration issues directly; avoid rewriting ASR, CTC, translation, or subtitle logic.
- No PR or merge to main until central review.

## 2. Studio backend integration

- Create separate .venv-studio and install package with optional [studio] extra.
- Set NEMOVOX_CLI_PYTHON to the existing model-compatible /workspace/nvidia-transcribe/.venv/bin/python.
- Verify ffmpeg/ffprobe, yt-dlp and existing ASR/SwiftU endpoints. Do NOT automatically stop/restart unrelated GPU services.
- Launch uvicorn nvidia_transcribe.studio_api:app on 127.0.0.1:8765 with --workers 1.
- Run tests/test_studio_api.py and all other CPU tests.
- Exercise health, allowlisted media browsing, upload, input/path rejection, job create/query, cancel, log tail, cue preview, bounded artifact download, and interrupted-job persistence.
- Confirm CLI subprocess uses the correct interpreter and does not leave Qwen worker processes behind on cancellation.
- Test one bounded real media job if already available endpoints permit, plus synthetic fixture tests that do not require GPU inference.
- Avoid full French/Mandarin episode processing simply to validate a GUI.

## 3. Frontend integration

- Install web dependencies using Node/npm on host and commit lockfile. npm run build must succeed; handle real package/type mismatches as small integration repairs.
- Verify full three-tab navigation and keyboard accessibility.
- Verify host browser, upload, YouTube/remote URL form, task selection, source language, task-specific settings, glossary selection, and disabled/error states.
- Ensure background task status and recent run history survive browser refresh.
- Preview a source with browser-supported codec and synchronized cues. Check JASSUB ASS preview on supported browser; label HTML fallback honestly.
- Verify Firefox/Firedragon and Chromium if available; GPU-disabled browser support where practical.
- Optional: integrate shadcn registry primitives from web/components.json where useful, preserving the authored UX. Do not rebuild into a generic admin template.
- If MKV is not playable in browser, show a clear fallback; a short, bounded FFmpeg preview proxy may be added if necessary for the test media, but do not re-encode entire episodes.

## 4. Focused acceptance

User can:
1. select server-local video or browser upload/remote URL,
2. select transcription, translation, or subtitle generation,
3. adjust task-appropriate options,
4. submit to the existing pipeline,
5. follow a job and inspect failures,
6. preview source video and synchronized text where supported,
7. download translated contracts and SRT/ASS where generated.

Require existing CPU test suite + Studio API tests green and web TypeScript/Vite build green. No weakening tests/security gates to get a pass. Provide synthetic-data UI screenshots and logs. Report honest limitations if live GPU services unavailable. Avoid overengineering.

## 5. Push code AND evidence to GitHub

The user wants review from GitHub, not manual uploaded handbacks.

1. Commit and push the integration branch to GitHub, report its exact full SHA.
2. Create a separate evidence branch FROM THAT EXACT INTEGRATION SHA, e.g. evidence/gui-foundation-20261121.
3. Automatically create a compact self-contained archive locally at outputs/handbacks/gui-foundation-evidence.tar.gz, including:
   exact SHAs/diffs, changed source/tests, package lock, dependency inventory, test/build/HTTP smoke evidence and exit codes, synthetic-only UI screenshots, sanitized responses, manifests/checksums, and bounded errors.
4. Verify gzip integrity and all payload byte sizes/SHA256. Keep the archive under about 20-25 MiB by pruning images/logs, not by bypassing limits.
5. On the evidence branch, publish:
   staging/agent-evidence/gui-foundation-20261121/gui-foundation-evidence.tar.gz
   staging/agent-evidence/gui-foundation-20261121/manifest.json
   Git ignores tar.gz: use git add -f for the single approved archive.
6. Commit/push the evidence branch; return browser URLs, exact full branch SHAs, archive size and SHA256, and verification status.
7. Do NOT merge an evidence branch containing binary archives to main.

**PUBLIC REPOSITORY SAFETY:** Do not upload TV/anime media, copyrighted episode screenshots/subtitle transcripts, personal file paths, model weights, credentials, secrets, private logs, or full media, even inside archives. Use synthetic demo data and sanitized outputs. Git history is persistent.

One cohesive implementation branch and one evidence branch are sufficient. No PR until central audit.

## 6. Stop/escalate

Resolve routine host integration/test/UX bugs autonomously. Escalate only destructive actions, publicly exposing local services, materially changing the framework/architecture, or losing core subtitles. Stop once working GUI, code branch, test evidence and repository evidence archive are published.
