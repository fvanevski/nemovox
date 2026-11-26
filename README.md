# nemovox

Multilingual speech transcription, speaker diarization, contextual transcript refinement, translation, and subtitle generation pipeline built on **NVIDIA Nemotron 3.5 Transcribe 30B-A4B NVFP4** (`vLLM` + `nemo_speechlm`) and OpenAI-compatible local reasoning LLMs.

*(Python package and CLI entrypoint remain `nvidia_transcribe` / `nvidia-transcribe`.)*

### Development status and compatibility

Nemovox is under active development for personal use. The current supported
workflow takes precedence over historical experimental implementations.
Superseded code, fixtures, schemas, configuration and development evidence
should be removed rather than maintained for backward compatibility. New
acceptance evidence must be generated with the current code, configuration
and model resources. The development contract in [AGENTS.md](AGENTS.md)
governs this transition. The occurrence-centered subtitle workflow was qualified
on the deployed Garuda host at the merged PR #21 source; see the
[acceptance record](https://github.com/fvanevski/nemovox/issues/19#issuecomment-7178926236).
That evidence is specific to the tested source, host configuration and models.

### Subtitle quality objective

Nemovox aims to generate fluent, grammatical, readable subtitles that convey
the original dialogue's intended meaning and make sense in context, not to
maximize verbatim word-for-word correspondence. Idiomatic translation,
condensation, and flexible phrasing are welcome when important meaning,
negation, narrative sequence, and relevant speaker turns survive. Utterance-level
timing can be useful even when individual word timing is uncertain.

Source audio, ASR, CTC, diarization and optional visual observations serve as
separate evidence for **reasoned** correction of problematic dialogue. Lightweight
contextual text QC should handle routine wording; richer audio-aware, and only
selectively audiovisual, reasoning belongs on genuinely ambiguous material.
Correct English phrasing does not certify unheard words as source-acoustic truth.
The normal current-media subtitle workflow performs bounded source quality
review before translation. Text assessments prioritize candidates; separately
extracted acoustic contexts can corroborate a missing turn or an extended or
untimed source event. The same ASR model may be used for both contexts: this
does not establish independent-model agreement or human-verified speech.
Competing or unsupported hypotheses and exhausted budgets remain explicit
in the current job's source-quality report. No historical dossier is needed.

---

## Key Capabilities

- **Single-Clip & Chunked Recording ASR:** Transcribe short clips (`transcribe`) or full-length audio/video recordings (`process-media`) with chunk-overlap boundary reconciliation, CTC word/character timestamp extraction, explicit unresolved timing, and resumable manifests.
- **Speaker Tracking & Contracts:** Reconstruct per-speaker utterances across overlapping streams, reconcile speakers across chunks, and emit structured JSON contracts (`recording.contract.json`) that preserve verbatim source Unicode text and honest timestamp provenance without fabricating missing timings.
- **ASR → current-source quality → Riva → English review:** Preserve original ASR observations, then select acoustically corroborated current source events before dedicated translation. Bounded NInfer Messages reasoning prioritizes review; lightweight English postediting is separate and preserves original MT. CLI and Studio share the pipeline. `--source-quality off` explicitly skips acoustic review; a recording contract without original media cannot be acoustically reviewed.
- **Post-Processing & Export (`generate-subtitles`):** Render `SRT` and multi-lane `ASS` subtitles (`Main`, `Top`, `TopAlt`), optional hardsub-clearing layout (`--ass-layout raised_main`), explicit optional high-CPS condensation (`--condense-high-cps`, disabled by default in Studio to avoid a second English text-model pass), reversible review overlays (`--validation-overlay`), and zero-inference CPU re-rendering (`--re-render`).

---

## Setup

### Requirements
- Python `>= 3.12` (see [`pyproject.toml`](pyproject.toml))
- `ffmpeg` and `ffprobe` on `PATH` for media inspection, chunk extraction, and preview rendering
- For live ASR: an available `vLLM` server with `nemo_speechlm` CTC timestamp middleware. For the Studio's normal English subtitle workflow: a **locally installed Riva-Translate checkpoint and worker interpreter** and a text endpoint supporting **NInfer Messages** for recording context, selective
source-quality adjudication and balanced English review. Legacy text-only source
rewriting and additional condensation are off by default. Mere path presence does not guarantee model compatibility or sufficient GPU memory.

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

The resource file binds functional roles (ASR, **dedicated Riva translation and English translation review**, optional legacy source refinement/condensation, alignment, embeddings) to installed local resources. The shipped `text_local` binding points to the existing SwiftU/NInfer host endpoint for current-source context/adjudication and selective English post-editing. The CLI and Studio use the same resolver and default to dedicated Riva translation and balanced English review in their normal subtitle workflow. See [model-resource configuration](docs/model-resources/CONFIGURATION.md) and the [Studio subtitle workflow guide](docs/gui/SUBTITLE_WORKFLOW.md). Reading this configuration never starts inference services.

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

# 4. Current-source quality, Riva translation, and contextual English review.
# Requires an installed Riva checkpoint/worker and a compatible text endpoint.
# Omit --glossary for unrelated media.
nvidia-transcribe generate-subtitles episode.webm \
  --recording-contract outputs/episode_run/recording.contract.json \
  --output-dir outputs/episode_subs \
  --source-language zh \
  --target-language en \
  --translator riva --translation-review balanced \
  --style natural

# Alternative: direct OpenAI-compatible text-model primary translation
# remains an explicit CLI choice, without the new Riva post-edit stage.
# nvidia-transcribe generate-subtitles --recording-contract ... --output-dir ... --translator openai_chat

# 5. Fast CPU-only re-render (no GPU/LLM calls); an optional operator-authored
# --validation-overlay may be supplied separately.
nvidia-transcribe generate-subtitles \
  --output-dir outputs/episode_subs \
  --source-language zh \
  --ass-layout raised_main \
  --ass-margin-v 150 \
  --re-render
```

---

## Repository Layout

- **`src/nvidia_transcribe/`** — Core package:
  - *ASR & media:* `checkpoint.py`, `client.py`, `prompts.py`, `media.py`, `chunking.py`, `parser.py`, `recording.py`
  - *Alignment & speakers:* `alignment.py`, `qwen_aligner.py`, `speakers.py`, `utterances.py`, `contract.py`, `metrics.py`
  - *Resource resolution and lifecycle:* `model_resources.py` (validated role/resource mappings) and `resource_lifecycle.py` (exact-owned model service acquisition, GPU admission and release)
  - *Refinement, translation & output:* `refinement.py`, `translation.py`, `translation_review.py`, `subtitles.py`, `subtitle_pipeline.py`, `cli.py`
- **`configs/`** — Example model-resource JSON (`model-resources.example.json`), server/runtime templates (`*.env.example`), and domain glossaries. Operator-authored validation overlays are optional inputs.
- **`scripts/`** — vLLM server launcher (`start_vllm_server.sh`), evaluation runners, and evidence packagers.
- **`tests/`** — CPU unit/regression suite (`-m "not gpu"`) and gated GPU markers (`-m gpu`).
- **`staging/agent-evidence/`** — Retained qualification records; not runtime inputs. See the [evidence index](staging/agent-evidence/README.md) for historical development material.
- **`outputs/`** *(Git-ignored)* — Local runtime artifacts, audio chunks, transcripts, subtitle tracks, logs, and evidence archives.

---

## Testing

CPU tests are decoupled from live GPU/model endpoints:

```bash
python -m compileall -q src scripts tests
PYTHONPATH=src:. .venv/bin/python -m pytest -q -m "not gpu"
PYTHONPATH=src:. .venv-studio/bin/python -m pytest -q \
  tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py
cd web && npm test && npm run build
```

The live ASR and text services are separate validation requirements and are not started by this test suite. For current resource CLI switches, run `nvidia-transcribe generate-subtitles --help`.


### Studio output destinations

The Studio Source tab can select which final files to copy into a host folder.
Set `NEMOVOX_EXPORT_ROOTS` to explicitly approved writable host directories
(or use the built-in `outputs/exports` root). Internal job artifacts, logs and
remote downloads are retained independently of selected exports. See
[Studio output configuration](docs/gui/STUDIO_OUTPUTS.md).

### Optional speaker embeddings

The normal CLI and Studio path uses existing ASR speaker tags and reconciliation without TitaNet. To use separately provisioned local TitaNet embeddings, explicitly select `--speaker-embeddings titanet` (or the Studio option). `generate-subtitles` also accepts `--audio-stream` and `--speaker-embeddings`, as does `process-media`. No implicit model download is required by the default path.

### Occurrence-directed source recovery

Current-source quality consumes the recognized-turn/occurrence ledger, schedules bounded localized acoustic observations and preserves explicit unresolved alternatives. Configure the optional `source_challenger` role with `whisper_ct2` on `cuda:0`
using a dedicated worker environment and an existing cached CTranslate2 checkpoint;
see [model resource configuration](docs/model-resources/CONFIGURATION.md).
The independent worker uses CUDA `int8_float16` under the shared lifecycle lease;
there is no automatic CPU fallback. No universal second transcription or word
alignment is required. The qualified default is fixed 90-second cores with
10-second context and a 15-second minimum tail; pause-aware geometry remains
an explicit option, not a universally superior policy.

### Native MOSS recording candidate

The optional `moss_events` ASR resource is available for bounded #24
qualification through the existing recording/subtitle commands. It preserves
native dialogue events and anonymous recording-scoped speakers without
manufacturing lexical timings. See [model-resource configuration](docs/model-resources/CONFIGURATION.md).
Nemotron remains the deployed default until episode-scale qualification and
Central promotion. Source-language selection must be explicit for MOSS.
