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
| `adapter` | One of `nemotron_ctc`, `openai_chat`, `riva_worker`, `qwen_forced_aligner`, `titanet` |
| `endpoint` | HTTP(S) base URL, e.g. `http://127.0.0.1:8001/v1`; required for served ASR/text models |
| `model` | Served model ID for ASR/text, or NeMo model name for speaker embedding |
| `checkpoint_path` | Local model checkpoint/directory for applicable adapters |
| `worker_python` | Python interpreter path for Riva/Qwen worker processes |
| `device` | Adapter-specific execution target: Qwen `cuda:0` or explicit `cpu`; shipped Riva `cuda:0` (explicit `cpu` or `auto` supported as opt-in and may be slower); TitaNet `cpu` only |
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
The `openai_chat` adapter expects OpenAI-compatible `/chat/completions` and
structured text responses; mere server reachability does not prove model
quality or JSON behavior.

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

For built-in `text_local`, the legacy `SWIFTU_BASE_URL` / `SWIFTU_MODEL`
override JSON defaults, while `TEXT_BASE_URL` / `TEXT_MODEL` override
the legacy env values. `ASR_BASE_URL` / `ASR_MODEL` similarly override
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

- `--swiftu-url` and `--swiftu-model` remain aliases of `--text-url`
  and `--text-model`. Their semantics explicitly apply to **all text roles**
  for that invocation.
- `--translator swiftu` and the prior SwiftU-named Python function exports
  remain compatibility aliases. Prefer `--translator openai_chat` and generic
  text-inference names in new integrations.
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
python -m compileall -q src tests experiments
pytest -q -m "not gpu"
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

**Verification boundary:** Issue #7 implementation has no host qualification
at the time of authoring. Prior issue #5 test results are tied to their earlier
exact source and do not verify these changes. Fresh CPU/TypeScript/UI tests
and bounded host checks are required. Live ASR/Riva/text inference is UNVERIFIED
until safely exercised on the actual configured host without interrupting
shared GPU processes.

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

Install the dedicated worker dependencies using the checked-in `requirements-challenger.txt` with a supported Python (the qualified host used 3.12.14). For example, after creating or selecting the dedicated environment, run `uv pip install --python /workspace/nvidia-transcribe/.venv-challenger/bin/python -r requirements-challenger.txt` and `uv pip check --python /workspace/nvidia-transcribe/.venv-challenger/bin/python`. Do not install these into another project's environment. The PyAV pin is intentional: the observed faster-whisper 1.2.1 invocation failed with av 19.0.2 because its audio decoder passed the removed `metadata_errors` argument; av 18.0.0 passed the actual CPU control. This is a tested dependency combination, not a claim about all future releases.

Merge these entries into the existing version-1 configuration; do not replace unrelated roles. Worker execution uses CUDA int8_float16 and local-files-only admission. The previous whisper_cpu adapter is superseded; change only the configured challenger adapter and device, preserving its local checkpoint and worker paths. The worker uses the shared ModelLifecycle CUDA lease after owned Nemotron and text stages have released their scopes. No CPU fallback, CUDA visibility override, unrelated service eviction or alternate checkpoint is allowed. A blocked lease, unsupported CUDA compute type, loader failure, or mismatched actual device receipt leaves explicit unresolved evidence. Qualify CUDA library availability and the real worker on the installed host before promotion; prior CPU evidence is not CUDA qualification. Missing or unusable resources are explicit unresolved/error outcomes, not silent model substitution. The source-quality report distinguishes admitted tasks, acquisition receipts, resource deferral, occurrence resolutions and unresolved candidates. Qwen remains optional timing-only work; it cannot supply independent recognition of submitted wording.

Primary recording no longer launches the former automatic per-chunk CTC subinterval recovery. CTC errors preserve recognized text, explicit unresolved timing and native observations for the shared scheduler; optional Qwen uses the audited admission path. The former `process-media --no-subinterval-recovery` switch is therefore removed. Direct low-level alignment experiments can still use the explicit helper independently of the normal recording workflow. Current-media source quality requires the original recording contract and its occurrence ledger to match their completed-run manifest digests; regenerate incompatible retained recordings before acoustic review. Offline re-rendering of a translated contract is unaffected.
