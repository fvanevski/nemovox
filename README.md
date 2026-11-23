# nemovox

Multilingual speech transcription, speaker diarization, contextual transcript refinement, translation, and subtitle generation pipeline built on **NVIDIA Nemotron 3.5 Transcribe 30B-A4B NVFP4** (`vLLM` + `nemo_speechlm`) and OpenAI-compatible local reasoning LLMs.

*(Python package and CLI entrypoint remain `nvidia_transcribe` / `nvidia-transcribe`.)*

---

## Key Capabilities

- **Single-Clip & Chunked Recording ASR:** Transcribe short clips (`transcribe`) or full-length audio/video recordings (`process-media`) with chunk-overlap boundary reconciliation, CTC word/character timestamp extraction, optional sub-interval recovery, and resumable manifests.
- **Speaker Tracking & Contracts:** Reconstruct per-speaker utterances across overlapping streams, reconcile speakers across chunks, and emit structured JSON contracts (`recording.contract.json`) that preserve verbatim source Unicode text and honest timestamp provenance without fabricating missing timings.
- **Untouched ASR → dedicated translation → selective English review:** Preserve the unmodified source-language ASR transcript. **No source-language text-model correction is part of the normal Studio workflow.** The Studio subtitle path uses a **dedicated Riva-Translate worker** for initial English machine translation, then uses the configured text model (e.g., SwiftU/NInfer) **only to assess and selectively correct original English translations**, using adjacent raw English cues, untouched ASR text, speaker context and an optional glossary. The initial machine translation remains inspectable; stage-specific caching prevents stale cross-model reuse. Source-language refinement remains an explicit legacy CLI opt-in (`--refinement` non-off). For standalone CLI backward compatibility, primary translation still defaults to `openai_chat` unless `--translator riva` is chosen explicitly; **Studio always sends `riva`** and selects the English review independently.
- **Post-Processing & Export (`generate-subtitles`):** Render `SRT` and multi-lane `ASS` subtitles (`Main`, `Top`, `TopAlt`), optional hardsub-clearing layout (`--ass-layout raised_main`), explicit optional high-CPS condensation (`--condense-high-cps`, disabled by default in Studio to avoid a second English text-model pass), reversible review overlays (`--validation-overlay`), and zero-inference CPU re-rendering (`--re-render`).

---

## Setup

### Requirements
- Python `>= 3.12` (see [`pyproject.toml`](pyproject.toml))
- `ffmpeg` and `ffprobe` on `PATH` for media inspection, chunk extraction, and preview rendering
- For live ASR: an available `vLLM` server with `nemo_speechlm` CTC timestamp middleware. For the Studio's normal English subtitle workflow: a **locally installed Riva-Translate checkpoint and worker interpreter** and an OpenAI-compatible text endpoint for **one selective English review pass**. Source correction and additional condensation are off by default. Mere path presence does not guarantee model compatibility or sufficient GPU memory.

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Host Configuration

Copy the tracked templates into ignored host-local files and set model paths and URLs:

```bash
cp configs/vllm_rtx5090.env.example configs/vllm_rtx5090.env
cp configs/runtime.env.example .env
cp configs/model-resources.example.json configs/model-resources.local.json
# Edit local resource bindings before starting CLI/Studio:
export NEMOVOX_MODEL_RESOURCES="$PWD/configs/model-resources.local.json"
# Optional: source a locally reviewed .env for per-resource overrides.
```

The resource file binds functional roles (ASR, **dedicated Riva translation and English translation review**, optional legacy source refinement/condensation, alignment, embeddings) to installed local resources. The shipped `text_local` binding points to the existing SwiftU/NInfer host endpoint for selective English post-editing. The CLI and Studio use the same resolver, but the Studio uses Riva for primary translation while the legacy CLI text translator remains opt-in. See [model-resource configuration](docs/model-resources/CONFIGURATION.md) and the [Studio subtitle workflow guide](docs/gui/SUBTITLE_WORKFLOW.md). Reading this configuration never starts inference services.

For automatic per-job ASR/reviewer service acquisition, provision the explicitly owned user-systemd units once (not per job) and configure `NEMOVOX_LIFECYCLE_CONFIG` as described in [model lifecycle](docs/model-resources/LIFECYCLE.md). The CLI/Studio then starts and releases required owned resources automatically. The following standalone launcher remains available for explicit diagnostics:

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

# 4. Dedicated Riva English translation followed by contextual text-model review.
# Requires an installed Riva checkpoint/worker and a compatible text endpoint.
# Omit --glossary for unrelated media.
nvidia-transcribe generate-subtitles \
  --recording-contract outputs/episode_run/recording.contract.json \
  --output-dir outputs/episode_subs \
  --source-language zh \
  --target-language en \
  --translator riva --translation-review balanced \
  --style natural

# Legacy compatibility: direct OpenAI-compatible text-model primary translation
# remains an explicit CLI choice, without the new Riva post-edit stage.
# nvidia-transcribe generate-subtitles --recording-contract ... --output-dir ... --translator openai_chat

# 5. Fast CPU-only re-render (no GPU/LLM calls) with optional review overlay
nvidia-transcribe generate-subtitles \
  --output-dir outputs/episode_subs \
  --source-language zh \
  --ass-layout raised_main \
  --ass-margin-v 150 \
  --validation-overlay configs/overlays/zh_ep166_validation.json \
  --re-render
```

### Reviewed source revision (explicit offline opt-in)

The `apply-reviewed-revision` CLI consumes immutable ASR source evidence,
independently reviewed audio-event reconciliation, a digest-pinned **explicit**
selection of complete source-unit components, an original recording, and
revision-specific English translation receipts. It is a separate, model-free
path; the default subtitle generator remains unchanged. This feature requires
host qualification before relying on it for the media library.

```sh
# Default: no writes and no model calls, validates the selected evidence
nvidia-transcribe apply-reviewed-revision \
  --original-recording /private/recording.contract.json \
  --source-evidence /private/source-evidence.v1.json \
  --review-dossier /private/candidate_reconciliation_dossier.json \
  --selection /private/approved-components.json

# Explicitly create one NEW private revised contract and SRT/ASS set
nvidia-transcribe apply-reviewed-revision \
  --original-recording /private/recording.contract.json \
  --source-evidence /private/source-evidence.v1.json \
  --review-dossier /private/candidate_reconciliation_dossier.json \
  --selection /private/approved-components.json \
  --previous-translated /private/translated.contract.json \
  --translation-receipts /private/revision-translation-receipts.json \
  --local-voice-review /private/local-acoustic-voice-review.json \
  --output-dir /private/new-reviewed-output \
  --episode-stem my_episode --apply
```

The selection is `nemovox-reviewed-component-selection/v1` JSON containing
`selected_component_ids`, `expected_plan_sha256`, and
`recording_content_sha256`, `ledger_content_sha256`,
`dossier_content_sha256`. These bind the current recomputed reviewed-unit
partition and three original evidence inputs. Incomplete components are
rejected, and `--apply` refuses an existing output directory. Riva/ASR/VLM
inference and live sidecar export are NOT invoked; new reviewed turns require
pre-existing translation receipts tied to each source revision. Local reviewed
voice tags remain distinct from global speaker identity. No automatic disposal
of ambiguous raw ASR overlaps or invention of word-level timing occurs.

---

## Repository Layout

- **`src/nvidia_transcribe/`** — Core package:
  - *ASR & media:* `checkpoint.py`, `client.py`, `prompts.py`, `media.py`, `chunking.py`, `parser.py`, `recording.py`
  - *Alignment & speakers:* `alignment.py`, `qwen_aligner.py`, `speakers.py`, `scene_diarization.py`, `utterances.py`, `contract.py`, `metrics.py`
  - *Resource resolution and lifecycle:* `model_resources.py` (validated role/resource mappings) and `resource_lifecycle.py` (exact-owned model service acquisition, GPU admission and release)
  - *Refinement, translation & output:* `refinement.py`, `translation.py`, `translation_review.py`, `subtitles.py`, `subtitle_pipeline.py`, `cli.py`
- **`configs/`** — Example model-resource JSON (`model-resources.example.json`), server/runtime templates (`*.env.example`), domain glossaries, and validation overlays.
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
cd web && npm test && npm run build
```

The live ASR and text services are separate validation requirements and are not started by this test suite. For current resource CLI switches, run `nvidia-transcribe generate-subtitles --help`.


### Studio output destinations

The Studio Source tab can select which final files to copy into a host folder.
Set `NEMOVOX_EXPORT_ROOTS` to explicitly approved writable host directories
(or use the built-in `outputs/exports` root). Internal job artifacts, logs and
remote downloads are retained independently of selected exports. See
[Studio output configuration](docs/gui/STUDIO_OUTPUTS.md).
