# Nemovox Studio

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

The backend does not start/stop model services. Studio uses existing CLI stages with **Riva-Translate for primary translation** and a separate configured OpenAI-compatible text model (e.g. SwiftU/NInfer) for source ASR correction and English translation review. The default Riva binding requests `cuda:0`, without silently falling back to CPU. Verify model availability and GPU headroom on the host; the UI reports only configured checkpoint/worker file availability, not a successful inference session. Export `NEMOVOX_MODEL_RESOURCES` before starting Studio; resource identities are configured on the host rather than selected by a model marketplace in the UI. See [resource configuration](../docs/model-resources/CONFIGURATION.md).

FFmpeg/ffprobe are required for media processing; yt-dlp is required for remote/YouTube URLs. All model weights continue under /workspace/models.

## Three tabs

1. Media & task: host directory browsing, browser file upload, remote URL, language choice, transcription/translation/subtitle tasks.
2. Configure: accessible explanations for ASR/refinement/review/style settings; **actual FFprobe audio tracks for a local selected file** (not hard-coded 0–3 choices); dedicated Riva translation followed by optional context-aware English post-edit; descriptive optional glossary previews; ASS layout and speaker/alignment settings. The shipped GUI selects Riva rather than legacy text-model primary translation.
3. Process & preview: job status/history, logs, video where browser-compatible, synchronized cues, ASS preview and downloads.

## Expected limitations of this foundation

- The app indexes new Studio jobs. Historical CLI outputs are not yet automatically included in the GUI job library.
- Selected local/uploaded media are inspected for real audio-only stream indices. Remote URL tracks cannot be shown before download, so track 0 is used. A preexisting recording contract needs no audio track selection.
- Two example series glossaries are included for demonstration but **not selected automatically**; glossary entries are contextual guidance, not enforced replacements.
- WebM typically plays in current Firefox/Chromium; MKV or unsupported codecs may require a future short FFmpeg preview proxy.
- Percent shown is a coarse workflow milestone, not per-chunk completion.
- In-flight jobs are marked interrupted if the Studio process restarts. Re-submission is possible; automated cross-job cache reuse is not implemented.
- The HTML subtitle overlay is approximate. JASSUB is the accurate preview when worker/WASM/video support permits.
- The frontend uses Radix Tabs and styled semantic controls. Host-side shadcn component generation is supported by `components.json` but is not required for running Studio.
- Remote ingestion is intended for a trusted local operator. Keep the API bound to loopback; it is not ready for public deployment.

Current architecture: [Studio architecture](../docs/gui/ARCHITECTURE.md). The original [Antigravity integration handoff](../docs/gui/ANTIGRAVITY_HANDOFF.md) is historical and should not override current source.
