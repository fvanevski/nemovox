# Nemovox Studio architecture

## Git authority

For development and review, use the live repository source and exact Git branch/commit identities as authority. Historical GUI staging branches and integration handoffs do not establish the current source head.

## Processing boundary

React/Vite Studio → FastAPI local HTTP → one-job subprocess worker → existing Nemovox Python CLI:
process-media (original recording contract and occurrence ledger) → generate-subtitles
(recording context, occurrence-directed source-quality assessment and selective
acoustic/visual evidence → canonical source with explicit uncertainty → Riva
primary machine translation → balanced contextual English review with comparative
validation → SRT/ASS). Original ASR remains intact. Legacy CLI text-only source
rewriting is a separate explicit route, incompatible with current source quality;
direct text-model translation and extra cue condensation are explicit options.

There must be only ONE owner of ASR, diarization, translation, text refinement and subtitle rendering: the existing Python source modules. The web layer only validates and configures requests, submits CLI jobs, retrieves output contracts, and visualizes results.

Use a separate HTTP Studio environment and NEMOVOX_CLI_PYTHON to invoke the model-compatible interpreter. Do not import vLLM into the web process. The CLI worker automatically starts/releases only exactly identified Nemovox-owned model units; never restart unrelated external services. See [lifecycle contract](../model-resources/LIFECYCLE.md).

## HTTP endpoints

| Method/route | Purpose |
|---|---|
| GET /api/health | Backend and local tool availability |
| GET /api/media | Browse allowed server-local media roots |
| POST /api/media/upload | Browser upload into local Studio uploads |
| POST /api/media/inspect | Inspect allowed local/uploaded file with FFprobe; return real audio-only tracks |
| GET /api/translation-backends | Read-only Riva checkpoint/worker configuration availability; no live GPU probe |
| GET /api/resource-status | Read-only ASR/text model readiness and managed-unit startability; never starts a model |
| GET /api/glossaries | Optional glossary names, terms, characters and context notes |
| POST /api/jobs | Create validated queued task |
| GET /api/jobs | Recent job history |
| GET /api/jobs/{id} | Job stage and state |
| POST /api/jobs/{id}/cancel | Cancel process group |
| GET /api/jobs/{id}/logs | Bounded log tail |
| GET /api/jobs/{id}/preview | SRT cues or transcript utterances |
| GET /api/jobs/{id}/artifacts | Download links for output files |
| GET /api/jobs/{id}/artifact/{path} | Allowed output file download |
| GET /api/jobs/{id}/media | Source video/audio stream for preview |

Task models are Pydantic in `src/nvidia_transcribe/studio_api.py` and mirrored as TypeScript interfaces in `web/src/api.ts`. Keep both in sync. Model-resource resolution runs in the child CLI using `model_resources.py` and its inherited `NEMOVOX_MODEL_RESOURCES` environment; Studio must not silently promote host environment values into higher-precedence CLI overrides. See [model-resource configuration](../model-resources/CONFIGURATION.md).

Background work uses ThreadPoolExecutor(max_workers=1), file-backed atomic job.json, subprocess CLI calls and bounded status/reporting. This is a pragmatic single-user local queue, not a durable distributed job service. Run one Uvicorn worker only. Browser uploads and remote media are stored under ignored outputs/studio.

Media browsing is limited to explicit roots, without arbitrary host-path execution. URL checks reject private or local hosts, but downloader redirects and local-only security assumptions warrant review before any LAN/public exposure.

## UX

- Media & task: browse host paths, upload from browser, remote URL, source language, task.
- Configure: detailed accessible in-menu and selected-option explanations for ASR, English translation review and style. Local source audio choices are dynamically probed and validated against the selected file; URL audio tracks are unknown until download. Dedicated Riva primary translation and selective English text-model post-edit appear as distinct steps. Current-media Riva jobs automatically use current-source quality; legacy text-only source rewriting is absent from Studio. Fixed 90-second cores, 10-second context, balanced review, and optional Qwen/TitaNet/extra condensation off are the normal defaults. Example glossaries are off by default and show names/terms/context when chosen. Resource identities and role bindings remain host-configured.
- Process & preview: progress/history, output file access, synchronized dialogue search/click-to-seek, browser JASSUB where available, HTML overlay fallback.

Design aims for a contemporary media workstation, not a generic enterprise dashboard: clear typography, progressive disclosure, dark subdued palette, responsive controls, easily accessible task state.

## Deferred (outside first integration)

- Audio waveform/spectrogram and per-cue editing.
- Broader library indexing of historic outputs.
- Remote sharing or authenticated multiuser deployment.
- Elaborate background/cluster scheduler.
- Public-service URL hardening.
- Re-encoding entire episodes for browser preview.

Keep GUI/API code focused on local media workflows: the *shared CLI-owned* job lifecycle manager acquires and releases verified Nemovox-owned services and serializes model stages. UI only displays resource status, never mutates systemd services directly.
