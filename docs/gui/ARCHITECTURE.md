# Nemovox Studio architecture

## Git authority

Current reviewed source authority was GitHub fvanevski/nemovox main at exact SHA:
9b4c8b539c67f2f6266fef135a259e92f5477b19.

Central GUI implementation is staged on branch central/gui-foundation-20261121. Local integration must inspect fresh branch heads before testing/writing.

## Processing boundary

React/Vite Studio → FastAPI local HTTP → one-job subprocess worker → existing Nemovox Python CLI:
process-media (recording.contract.json) → generate-subtitles (refined and translated contracts, SRT/ASS).

There must be only ONE owner of ASR, diarization, translation, text refinement and subtitle rendering: the existing Python source modules. The web layer only validates and configures requests, submits CLI jobs, retrieves output contracts, and visualizes results.

Use a separate HTTP Studio environment and NEMOVOX_CLI_PYTHON to invoke the model-compatible interpreter. Do not import vLLM into the web process. Do not auto-restart external services.

## HTTP endpoints

| Method/route | Purpose |
|---|---|
| GET /api/health | Backend and local tool availability |
| GET /api/media | Browse allowed server-local media roots |
| POST /api/media/upload | Browser upload into local Studio uploads |
| GET /api/glossaries | Existing local glossary configurations |
| POST /api/jobs | Create validated queued task |
| GET /api/jobs | Recent job history |
| GET /api/jobs/{id} | Job stage and state |
| POST /api/jobs/{id}/cancel | Cancel process group |
| GET /api/jobs/{id}/logs | Bounded log tail |
| GET /api/jobs/{id}/preview | SRT cues or transcript utterances |
| GET /api/jobs/{id}/artifacts | Download links for output files |
| GET /api/jobs/{id}/artifact/{path} | Allowed output file download |
| GET /api/jobs/{id}/media | Source video/audio stream for preview |

Task models are Pydantic in src/nvidia_transcribe/studio_api.py and mirrored as TypeScript interfaces in web/src/api.ts. Keep both in sync.

Background work uses ThreadPoolExecutor(max_workers=1), file-backed atomic job.json, subprocess CLI calls and bounded status/reporting. This is a pragmatic single-user local queue, not a durable distributed job service. Run one Uvicorn worker only. Browser uploads and remote media are stored under ignored outputs/studio.

Media browsing is limited to explicit roots, without arbitrary host-path execution. URL checks reject private or local hosts, but downloader redirects and local-only security assumptions warrant review before any LAN/public exposure.

## UX

- Media & task: browse host paths, upload from browser, remote URL, source language, task.
- Configure: dynamic task options, glossary and names, SwiftU profile/style, ASS layout and acoustic controls.
- Process & preview: progress/history, output file access, synchronized dialogue search/click-to-seek, browser JASSUB where available, HTML overlay fallback.

Design aims for a contemporary media workstation, not a generic enterprise dashboard: clear typography, progressive disclosure, dark subdued palette, responsive controls, easily accessible task state.

## Deferred (outside first integration)

- Audio waveform/spectrogram and per-cue editing.
- Broader library indexing of historic outputs.
- Remote sharing or authenticated multiuser deployment.
- Elaborate background/cluster scheduler.
- Public-service URL hardening.
- Re-encoding entire episodes for browser preview.

Keep API and GUI modifications focused on getting the existing authored foundation working on the host.
