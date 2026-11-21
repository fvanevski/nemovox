# Nemovox Studio

Central-authored GUI implementation staged at https://github.com/fvanevski/nemovox/tree/central/gui-foundation-20261121

This is a local React interface over the existing Nemovox Python CLI, not a replacement processing pipeline.

## Stack
- React 19, TypeScript 6, Vite 8, Tailwind 4
- Radix Tabs, TanStack Query, Lucide icons, Motion
- shadcn/ui registry configuration for accessible component generation
- JASSUB/libass WASM for browser ASS overlay, with HTML text fallback
- FastAPI and a single background CLI worker

## Run on the Garuda host

Use a separate lightweight HTTP environment, NOT the existing vLLM environment.

~~~bash
cd /workspace/nvidia-transcribe
uv venv .venv-studio --python 3.13
uv pip install -p .venv-studio/bin/python -e '.[studio]'
export NEMOVOX_CLI_PYTHON=/workspace/nvidia-transcribe/.venv/bin/python
export NEMOVOX_MEDIA_ROOTS=/path/to/optional/extra/media
.venv-studio/bin/python -m uvicorn nvidia_transcribe.studio_api:app --host 127.0.0.1 --port 8765 --workers 1
~~~

In a second shell:

~~~bash
cd /workspace/nvidia-transcribe/web
npm install
npm run dev
~~~

Then open http://127.0.0.1:5173. The frontend proxies /api to the Python backend. A production build (npm run build) is served by the same backend from web/dist.

The backend does NOT start/stop vLLM or SwiftU. Both model-serving endpoints are reused as configured by the existing CLI.

FFmpeg/ffprobe are required for media processing; yt-dlp is required for remote/YouTube URLs. All model weights continue under /workspace/models.

## Three tabs

1. Media & task: host directory browsing, browser file upload, remote URL, language choice, transcription/translation/subtitle tasks.
2. Configure: task-specific settings, SwiftU refinement, glossary, style, ASS display, speaker/alignment options.
3. Process & preview: job status/history, logs, video where browser-compatible, synchronized cues, ASS preview and downloads.

## Expected limitations of this foundation

- The app indexes new Studio jobs. Historical CLI outputs are not yet automatically included in the GUI job library.
- WebM typically plays in current Firefox/Chromium; MKV or unsupported codecs may require a future short FFmpeg preview proxy.
- Percent shown is a coarse workflow milestone, not per-chunk completion.
- In-flight jobs are marked interrupted if the Studio process restarts. Re-submission is possible; automated cross-job cache reuse is not implemented.
- The HTML subtitle overlay is approximate. JASSUB is the accurate preview when worker/WASM/video support permits.
- The frontend uses Radix Tabs and styled semantic controls. Host-side shadcn component generation is authorized by components.json but has not yet been executed.
- Remote ingestion is intended for a trusted local operator. Keep the API bound to loopback; it is not ready for public deployment.

Architecture and host handoff: ../docs/gui/ARCHITECTURE.md and ../docs/gui/ANTIGRAVITY_HANDOFF.md.
