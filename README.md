# nemovox

Multilingual speech transcription, speaker diarization, contextual transcript refinement, translation, and subtitle generation pipeline built on **NVIDIA Nemotron 3.5 Transcribe 30B-A4B NVFP4** (`vLLM` + `nemo_speechlm`) and OpenAI-compatible local reasoning LLMs.

*(Python package and CLI entrypoint remain `nvidia_transcribe` / `nvidia-transcribe`.)*

---

## Key Capabilities

- **Single-Clip & Chunked Recording ASR:** Transcribe short clips (`transcribe`) or full-length audio/video recordings (`process-media`) with chunk-overlap boundary reconciliation, CTC word/character timestamp extraction, optional sub-interval recovery, and resumable manifests.
- **Speaker Tracking & Contracts:** Reconstruct per-speaker utterances across overlapping streams, reconcile speakers across chunks, and emit structured JSON contracts (`recording.contract.json`) that preserve verbatim source Unicode text and honest timestamp provenance without fabricating missing timings.
- **Contextual Refinement & Translation:** Refine noisy ASR output (proper nouns, homophones, domain terminology via YAML/JSON glossaries) and translate into natural target text using local OpenAI-compatible reasoning models (e.g., SwiftU/NInfer) or Riva-Translate workers, with content-addressed batch caching.
- **Post-Processing & Export (`generate-subtitles`):** Render `SRT` and multi-lane `ASS` subtitles (`Main`, `Top`, `TopAlt`), optional hardsub-clearing layout (`--ass-layout raised_main`), selective high-CPS condensation (`--condense-high-cps`), reversible review overlays (`--validation-overlay`), and zero-inference CPU re-rendering (`--re-render`).

---

## Setup

### Requirements
- Python `>= 3.12` (see [`pyproject.toml`](pyproject.toml))
- `ffmpeg` and `ffprobe` on `PATH` for media inspection, chunk extraction, and preview rendering
- For live GPU inference: a running `vLLM` server with `nemo_speechlm` CTC timestamp middleware, plus an optional OpenAI-compatible text endpoint for refinement/translation/condensation

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Host Configuration

Copy the tracked template files to your local ignored config files and set host-specific model paths and URLs:

```bash
cp configs/vllm_rtx5090.env.example configs/vllm_rtx5090.env
cp configs/runtime.env.example .env
```

Launch the ASR serving backend (reads `configs/vllm_rtx5090.env` by default):

```bash
./scripts/start_vllm_server.sh
```

---

## CLI Quick Start

```bash
# 1. Inspect local checkpoint layout and ModelOpt NVFP4 quantization metadata
nvidia-transcribe validate-checkpoint --model-path /path/to/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4

# 2. Single-clip transcription with CTC timestamps
nvidia-transcribe transcribe clip.wav \
  --base-url http://127.0.0.1:8190/v1 \
  --mode speaker \
  --out-json clip.contract.json

# 3. Full recording chunked ASR + speaker reconciliation
nvidia-transcribe process-media episode.webm \
  --output-dir outputs/episode_run \
  --source-language zh \
  --mode speaker

# 4. Refine, translate, and render SRT/ASS from a recording contract
nvidia-transcribe generate-subtitles \
  --recording-contract outputs/episode_run/recording.contract.json \
  --output-dir outputs/episode_subs \
  --source-language zh \
  --target-language en \
  --glossary configs/glossaries/a_will_eternal_zh.yaml \
  --ass-layout raised_main \
  --condense-high-cps

# 5. Fast CPU-only re-render (no GPU/LLM calls) with optional review overlay
nvidia-transcribe generate-subtitles \
  --output-dir outputs/episode_subs \
  --source-language zh \
  --ass-layout raised_main \
  --ass-margin-v 150 \
  --validation-overlay configs/overlays/zh_ep166_validation.json \
  --re-render
```

---

## Repository Layout

- **`src/nvidia_transcribe/`** — Core package:
  - *ASR & media:* `checkpoint.py`, `client.py`, `prompts.py`, `media.py`, `chunking.py`, `parser.py`, `recording.py`
  - *Alignment & speakers:* `alignment.py`, `qwen_aligner.py`, `speakers.py`, `scene_diarization.py`, `utterances.py`, `contract.py`, `metrics.py`
  - *Refinement, translation & output:* `refinement.py`, `translation.py`, `subtitles.py`, `subtitle_pipeline.py`, `cli.py`
- **`configs/`** — Example server/runtime templates (`*.env.example`), domain glossaries (`configs/glossaries/`), and validation overlays (`configs/overlays/`).
- **`scripts/`** — vLLM server launcher (`start_vllm_server.sh`), evaluation runners, and evidence packagers.
- **`tests/`** — CPU unit/regression suite (`-m "not gpu"`) and gated GPU markers (`-m gpu`).
- **`experiments/`** — Modular bakeoff, reliability, and viewing-quality harness scripts.
- **`evaluations/`** — Architecture notes, benchmarks, comparison metrics, and validation reports across development milestones.
- **`outputs/`** *(Git-ignored)* — Local runtime artifacts, audio chunks, transcripts, subtitle tracks, logs, and evidence archives.

---

## Testing

CPU tests are decoupled from live GPU/model endpoints:

```bash
python -m compileall -q src tests experiments
pytest -q -m "not gpu"
```
