# Model-resource configuration

Nemovox resolves **task roles** to **named resources** and then uses an
adapter specific to the required inference contract. Configuring a different
resource does not change an algorithm, install a model, or start a server.

The current host's SwiftU/NInfer text endpoint is a **default binding**, not a
required implementation. All model inference remains local only if the
operator binds resources to locally hosted endpoints and workers. A configured
remote OpenAI-compatible endpoint would send the task's text to that endpoint.

## First-time setup

From the repository root:

```bash
cp configs/model-resources.example.json configs/model-resources.local.json
# Edit the ignored local file to match installed model endpoints/checkpoints.
export NEMOVOX_MODEL_RESOURCES="$PWD/configs/model-resources.local.json"
python -m nvidia_transcribe.cli generate-subtitles --help
```

The `*.local.json` pattern is Git-ignored. Do not commit secrets, host-specific
configuration, model weights, or private transcripts. JSON file format version
is currently `1`. No live service is started by reading this file.

`configs/runtime.env.example` provides optional environment overrides.
It is an example: copying to `.env` does **not** automatically source it.
Explicitly source a locally reviewed env file (or set variables in the launch
environment) before running the CLI or Studio.

## Roles and supported adapters

| Task role | Required adapter | Shipped resource | Function |
|---|---|---|---|
| `asr` | `nemotron_ctc` | `asr_local` | Nemotron/vLLM speech recognition with CTC metadata |
| `source_refinement` | `openai_chat` | `text_local` | Legacy CLI-only explicit source-language correction; off by default, not used in Studio |
| `text_translation` | `openai_chat` | `text_local` | Explicit legacy CLI primary text-model translation |
| `translation_review` | `openai_chat` | `text_local` | Independent English post-edit of dedicated MT using source context |
| `subtitle_condensation` | `openai_chat` | `text_local` | Shorten dense subtitle cues |
| `riva_translation` | `riva_worker` | `riva_local` | Dedicated Riva translation worker |
| `source_challenger` | `whisper_ct2` | none (optional explicit binding) | Independent, selectively admitted CUDA acoustic evidence |
| `forced_alignment` | `qwen_forced_aligner` | `qwen_local` | Optional selective Qwen forced alignment |
| `speaker_embedding` | `titanet` | `titanet_cpu` | NeMo speaker embeddings on CPU |

The same named `openai_chat` resource may serve optional legacy source correction, selective English
translation review and optional extra subtitle condensation, or these roles may bind distinct
compatible servers/models. `text_translation` remains for **explicit legacy CLI
text-model primary translation**. For the normal Studio workflow, `riva_translation`
supplies a dedicated machine-translation worker. `--translation-review balanced`
then asks the text model to correct supported English errors *after* Riva, using
source text, neighboring utterances and optional glossary. Original MT is retained
separately in `machine_translated_text`. Model resource resolution is read-only; required model services are automatically started for job stages by the separate [lifecycle manager](LIFECYCLE.md), when explicitly cooperatively owned;
Nemotron CTC, Qwen alignment, Riva and NeMo embeddings retain specialized adapters.

## Resource schema

A version-1 JSON document has `version`, optional `resources`, and optional
`roles` objects. Each named resource has an `adapter` and adapter-appropriate
fields:

| Field | Meaning |
|---|---|
| `adapter` | One of `nemotron_ctc`, `openai_chat`, `riva_worker`, `whisper_ct2`, `qwen_forced_aligner`, `titanet` |
| `endpoint` | HTTP(S) base URL, e.g. `http://127.0.0.1:8001/v1`; required for served ASR/text models |
| `model` | Served model ID for ASR/text, or NeMo model name for speaker embedding |
| `checkpoint_path` | Local model checkpoint/directory for applicable adapters |
| `worker_python` | Python interpreter path for Riva, Whisper and Qwen worker processes |
| `device` | Adapter-specific execution target: Qwen `cuda:0` or explicit `cpu`; shipped Riva `cuda:0` (explicit `cpu` or `auto` supported as opt-in and may be slower); Whisper `cuda:0` only; TitaNet `cpu` only |
| `api_key_env` | Environment **variable name** holding a token, never the token value |

Example: keep the current SwiftU/NInfer binding for correction, but assign
independent locally hosted models for translation and condensation:

```json
{
  "version": 1,
  "resources": {
    "translation_local": {
      "adapter": "openai_chat",
      "endpoint": "http://127.0.0.1:8200/v1",
      "model": "translation-served-id"
    },
    "condensation_local": {
      "adapter": "openai_chat",
      "endpoint": "http://127.0.0.1:8300/v1",
      "model": "condensation-served-id"
    }
  },
  "roles": {
    "source_refinement": "text_local",
    "text_translation": "translation_local",
    "translation_review": "text_local",
    "subtitle_condensation": "condensation_local"
  }
}
```

The ports and model IDs in this example are **illustrative**, not verified
services. A compatible endpoint must be available or startable by the job-owned lifecycle manager. Other roles retain
their built-in bindings unless overridden. The `resources` object merges
settings into shipped named resources or defines new resources; `roles` binds
each functional task to a resource name. All roles are validated at load time.
The `openai_chat` adapter supports OpenAI-compatible `/chat/completions` for
explicit text-model routes. Current-media source quality and its English review
use the NInfer Messages wire protocol; the selected endpoint must support that
protocol and structured responses. Mere server reachability does not establish
model quality or JSON behavior.

For an authenticated endpoint, use e.g. `"api_key_env": "TEXT_API_TOKEN"`
on the chosen resource and set `TEXT_API_TOKEN` securely in the execution
environment. The variable's value is not written to resource fingerprints.
The Nemotron client also retains its legacy `VLLM_API_KEY` fallback when a
resource-specific `api_key_env` is not set.

For ASR, `checkpoint_path` is optional when matching CTC vocabulary metadata
is unavailable. New named ASR resources without a checkpoint continue native
CTC association and occurrence-directed recovery, but do **not** run vocabulary-based
OOV classification; recording provenance stores `model_path: null`, and
alignment diagnostics report `ctc_vocabulary_status: not_configured`.
To enable that classification, explicitly bind the **matching** checkpoint.
Changing the built-in ASR model through JSON `asr_local.model`, `ASR_MODEL`,
or `--model` clears its inherited checkpoint unless a matching path is
explicitly rebound; `process-media --model-path` supplies an override for
that invocation. An endpoint-only change preserves the existing binding.

## Precedence and CLI examples

The effective precedence is:

1. Built-in defaults.
2. Named resource definitions and task bindings from the host JSON file.
3. Environment overrides of **built-in** resources.
4. Explicit CLI arguments for the invoked command.

For built-in `text_local`, `TEXT_BASE_URL` / `TEXT_MODEL` override
JSON defaults. `ASR_BASE_URL` / `ASR_MODEL` similarly override
built-in `asr_local`. These environment variables do not silently rewrite
new, independently named resources.

```bash
# Normal current-media source-quality, dedicated MT and English review:
nvidia-transcribe generate-subtitles episode.webm \
  --model-resources configs/model-resources.local.json \
  --recording-contract outputs/episode_run/recording.contract.json \
  --output-dir outputs/episode_subs \
  --translator riva --translation-review balanced \
  --style natural

# Legacy direct text-model translation remains a separate explicit choice:
# nvidia-transcribe generate-subtitles ... --translator openai_chat

# Select explicitly named text models for one invocation.
nvidia-transcribe generate-subtitles \
  --model-resources configs/model-resources.local.json \
  --recording-contract outputs/episode_run/recording.contract.json \
  --output-dir outputs/episode_subs \
  --refinement-resource text_local \
  --translation-resource translation_local \
  --condensation-resource condensation_local \
  --fresh-refinement --fresh-translation

# One-off override applied to all text roles (including English translation review).
nvidia-transcribe generate-subtitles \
  --recording-contract outputs/episode_run/recording.contract.json \
  --output-dir outputs/episode_subs \
  --text-url http://127.0.0.1:8001/v1 --text-model chat
```

`--asr-resource` selects a named ASR/CTC resource for `transcribe`,
`process-media`, or `generate-subtitles`; `--base-url` and `--model`
override that resource's served endpoint/ID for one invocation.
`--model-resources` is available on these three commands and takes precedence
over `NEMOVOX_MODEL_RESOURCES` for the config-file location.

## Compatibility and migration

- Use `--text-url`, `--text-model`, `TEXT_BASE_URL`, and `TEXT_MODEL` for generic text resources. The CLI flags apply to all text roles for that invocation. Use `--translator openai_chat` for explicit direct text-model translation. Retired SwiftU-specific selectors and Python function aliases are unsupported; retired environment settings fail explicitly rather than silently selecting another endpoint.
- The `transcribe` single-clip command now resolves the same `asr` role as
  recording ASR, so its **unqualified default** is the role's local served ID at
  `http://127.0.0.1:8190/v1`. Previously the single-clip parser alone used
  `http://127.0.0.1:8000/v1` and a local checkpoint path as its model ID.
  To preserve that old invocation, specify `--base-url
  http://127.0.0.1:8000/v1 --model /workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`,
  or configure a named ASR resource for that server.
- Omitted `--qwen-device` defers to the selected Qwen resource's device
  (shipped default `cuda:0`). Explicit `--qwen-device cpu` remains available,
  and managed ASR handoff retains its existing CUDA/ownership restrictions.
- Existing re-render mode does not invoke inference and uses already translated
  contracts, regardless of current model bindings.

## Studio host configuration

Studio is a CLI subprocess orchestrator, not a second inference pipeline.
Export NEMOVOX_MODEL_RESOURCES and credential variables before starting it.
Current-media Riva subtitle jobs use the translation_review resource for
bounded NInfer Messages source advice and English postedit, with ASR acquired
separately for acoustic challenges. The CLI default is Riva and balanced
review; alternative primary translators default to review off. Configure
the existing owned model units and resource bindings; source review requires
a compatible NInfer Messages endpoint, not merely a Chat Completions URL.

## Validation, caches, and safety

```bash
python -m compileall -q src scripts tests
PYTHONPATH=src:. .venv/bin/python -m pytest -q -m "not gpu"
PYTHONPATH=src:. .venv-studio/bin/python -m pytest -q \
  tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py
cd web && npm test && npm run build
```

`src/nvidia_transcribe/model_resources.py` validates known roles, required
adapter fields, HTTP(S) endpoint syntax, role/adapter compatibility, and
adapter-specific allowed fields/devices. Unsupported settings are rejected
rather than accepted and silently ignored (including requesting GPU TitaNet
speaker embeddings, which currently run on CPU only).
Required Riva/Qwen worker paths are validated lazily when the CLI requests
those workers. A file existing on disk does not prove runtime compatibility.
A missing configured token fails before the applicable inference call.
Invalid or unavailable selected resources are not silently swapped to a
different model or device.

Cache/checkpoint identities include effective resource fingerprints or
associated backend settings, including the selected CTC metadata checkpoint
(or its absence). Source correction, dedicated machine translation, English
post-editing and optional condensation retain separate checkpoint identities.
Post-edit caching also includes the raw machine translation, source context,
glossary digest, style/profile and reviewer identity. Do not carry unrelated
historical ASR, speaker, translation or review outputs across model changes.
`--fresh`, `--fresh-refinement`, and `--fresh-translation` provide
explicit recomputation controls. `--re-render` intentionally reuses an
existing translated contract instead of repeating model inference.

For programmatic Riva translation with condensation enabled, supply a
compatible `SubtitlePipelineConfig.condensation=TranslationConfig(...)`.
A Riva-specific `TranslationConfig` is not an OpenAI chat model; the pipeline
rejects ambiguous implicit reuse. `--translator none` skips condensation
even if `--condense-high-cps` was specified.

Reading configuration never starts, stops, loads, unloads or downloads a model. An actual CLI/Studio processing stage now automatically acquires/releases *verified Nemovox-owned* user units, but never changes an unrelated service's GPU allocation. See [model lifecycle](LIFECYCLE.md). Qwen's optional
managed-ASR flow still requires explicit ownership/lease admission. Do not
use production/private media for smoke testing without authorization.

**Verification boundary:** The integrated normal path, including live ASR,
CUDA challenger, Riva and native Messages review, was qualified on the deployed
PR #21 source; see the [acceptance record](https://github.com/fvanevski/nemovox/issues/19#issuecomment-7178926236).
A different source, configuration, worker environment or model requires
qualification of the affected behavior. Do not treat historical receipts or
path-only readiness as proof of inference on a changed host.

## Optional independent acoustic challenger

The shared source-quality scheduler can request bounded independent faster-whisper evidence after the managed Nemotron scope releases. Configure the optional `source_challenger` role explicitly in the host-local resource JSON, using an existing complete multilingual CTranslate2 checkpoint and a dedicated Nemovox-owned Python environment with faster-whisper installed. No model or environment is downloaded automatically. Example additional entries (replace the placeholder paths with authenticated host paths):

```json
{
  "resources": {
    "whisper_local": {
      "adapter": "whisper_ct2",
      "checkpoint_path": "/absolute/path/to/cached/large-v3-ct2",
      "worker_python": "/workspace/nvidia-transcribe/.venv-challenger/bin/python",
      "device": "cuda:0"
    }
  },
  "roles": {"source_challenger": "whisper_local"}
}
```

Install the dedicated worker dependencies using the checked-in `requirements-challenger.txt` with a supported Python (the qualified host used 3.12.14). For example, after creating or selecting the dedicated environment, run `uv pip install --python /workspace/nvidia-transcribe/.venv-challenger/bin/python -r requirements-challenger.txt` and `uv pip check --python /workspace/nvidia-transcribe/.venv-challenger/bin/python`. Do not install these into another project's environment. The PyAV pin is intentional: the observed faster-whisper 1.2.1 invocation failed with av 19.0.2 because its audio decoder passed the removed `metadata_errors` argument; av 18.0.0 passed the original CPU control and subsequent deployed CUDA qualification. This is a tested dependency combination, not a claim about all future releases.

Merge these entries into the existing version-1 configuration; do not replace unrelated roles. Worker execution uses CUDA int8_float16 and local-files-only admission. The former `whisper_cpu` adapter is unsupported. If migrating an older host configuration, change only the challenger adapter and device, preserving its authenticated checkpoint and worker paths. The worker uses the shared ModelLifecycle CUDA lease after owned Nemotron and text stages have released their scopes. No CPU fallback, CUDA visibility override, unrelated service eviction or alternate checkpoint is allowed. A blocked lease, unsupported CUDA compute type, loader failure, or mismatched actual device receipt leaves explicit unresolved evidence. Qualify CUDA library availability and actual worker execution on a new or changed host; the deployed PR #21 job exercised this CUDA path. Earlier CPU evidence does not establish CUDA readiness. Missing or unusable resources are explicit unresolved/error outcomes, not silent model substitution. The source-quality report distinguishes admitted tasks, acquisition receipts, resource deferral, occurrence resolutions and unresolved candidates. Qwen remains optional timing-only work; it cannot supply independent recognition of submitted wording.

Primary recording no longer launches the former automatic per-chunk CTC subinterval recovery. CTC errors preserve recognized text, explicit unresolved timing and native observations for the shared scheduler; optional Qwen uses the audited admission path. The former automatic native retry callback and its switch have been removed. Localized Qwen audio extraction and occurrence-directed acoustic recovery remain supported. Current-media source quality requires the original recording contract and its occurrence ledger to match their completed-run manifest digests; regenerate incompatible retained recordings before acoustic review. Offline re-rendering of a translated contract is unaffected.

### Rebuilding the qualified CUDA challenger environment

On the qualified Garuda host CTranslate2 4.9.1 required cuBLAS 12 during actual inference; `/opt/cuda/lib64` exposed CUDA 13. The existing CUDA 12.8 files resolved the issue. Check the actual wheel's `ctranslate2.libs` directory and ELF RPATH with `readelf -d` before recreating links. The qualified targets were `/usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.3.14` and `libcublasLt.so.12.8.3.14`; link them as `libcublas.so.12` and `libcublasLt.so.12` in that environment's `ctranslate2.libs` only, without overwriting existing regular files. Revalidate SONAME, real target path, installed cuDNN 9, `uv pip check`, and actual cached-model inference both with `LD_LIBRARY_PATH` unset and with the host's CUDA 13 path. Do not assume a capability probe establishes working inference. Recheck after a wheel/environment rebuild; no system driver changes or model downloads are necessary for this qualified setup. Apply resource changes only through an authorized, reversible host deployment; do not recreate a qualified environment merely to repeat historical setup.

### MOSS native-dialogue candidate

For #24 qualification, add a resource to a private copy of the effective
configuration (not the deployed registry):

```json
{"adapter":"moss_events","model":"OpenMOSS-Team/MOSS-Transcribe-Diarize","checkpoint_path":"/absolute/admitted/snapshot","worker_python":"/absolute/isolated/venv/bin/python","device":"cuda:0"}
```

Bind it to the asr role or select with --asr-resource. No endpoint/model
CLI overrides, implicit downloads, CPU fallback or CTC diagnostics. Use
process-media or generate-subtitles with explicit source language. Studio uses
the same private registry in its child environment. MOSS uses one complete
recording request, independent of legacy chunk controls, with a duration-scaled
generation budget and explicit cap failure. Legacy source-quality on, source
rewriting, alignment fallback and speaker embeddings are not applicable;
auto source quality reports the native path as not_run, not reviewed. Select
source language explicitly in Studio. Same-job ASR resume binds input bytes,
stream, model assets, worker/runtime, adapter and raw generation. No new
resource default or service deployment until qualified.
