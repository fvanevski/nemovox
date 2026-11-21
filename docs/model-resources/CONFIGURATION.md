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
| `source_refinement` | `openai_chat` | `text_local` | Contextual correction of source-language text |
| `text_translation` | `openai_chat` | `text_local` | Contextual text translation/localization |
| `subtitle_condensation` | `openai_chat` | `text_local` | Shorten dense subtitle cues |
| `riva_translation` | `riva_worker` | `riva_local` | Dedicated Riva translation worker |
| `forced_alignment` | `qwen_forced_aligner` | `qwen_local` | Optional selective Qwen forced alignment |
| `speaker_embedding` | `titanet` | `titanet_cpu` | NeMo speaker embeddings on CPU |

The same named `openai_chat` resource may serve all three text roles, or each
may bind a different compatible server/model. The `riva_translation` role is
selected by `--translator riva`; it is **not** a drop-in OpenAI chat model.
Condensation after Riva uses the separate `subtitle_condensation` text role
when requested. Nemotron CTC, Qwen alignment, and NeMo speaker embeddings keep
specialized adapters and safety requirements.

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
| `device` | Adapter-specific execution target, e.g. `cuda:0` (Qwen) or `auto` (Riva) |
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
    "subtitle_condensation": "condensation_local"
  }
}
```

The ports and model IDs in this example are **illustrative**, not verified
services. A compatible server must already be available. Other roles retain
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
# Use the host config's task bindings.
nvidia-transcribe generate-subtitles \
  --model-resources configs/model-resources.local.json \
  --recording-contract outputs/episode_run/recording.contract.json \
  --output-dir outputs/episode_subs \
  --translator openai_chat --condense-high-cps

# Select explicitly named text models for one invocation.
nvidia-transcribe generate-subtitles \
  --model-resources configs/model-resources.local.json \
  --recording-contract outputs/episode_run/recording.contract.json \
  --output-dir outputs/episode_subs \
  --refinement-resource text_local \
  --translation-resource translation_local \
  --condensation-resource condensation_local \
  --fresh-refinement --fresh-translation

# One-off override applied to all three text roles (including independent ones).
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
  and `--text-model`. Their semantics now explicitly apply to **all three**
  text roles for that invocation.
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
Export `NEMOVOX_MODEL_RESOURCES` and any credential variables **before**
starting the FastAPI Studio service. Child CLI jobs inherit that environment
and perform the same role resolution as standalone CLI commands. Studio's
`translator` selection chooses `openai_chat` (default), `riva`, or legacy
`swiftu`; Studio does not currently offer per-job resource ID selectors in its
UI. Configure model identities/bindings in the host JSON instead.

Keep the Studio server bound to loopback and separate its HTTP Python
environment from the inference environment; see `web/README.md`.

## Validation, caches, and safety

```bash
python -m compileall -q src tests experiments
pytest -q -m "not gpu"
cd web && npm test && npm run build
```

`src/nvidia_transcribe/model_resources.py` validates known roles, required
adapter fields, HTTP(S) endpoint syntax and role/adapter compatibility.
Required Riva/Qwen worker paths are validated lazily when the CLI requests
those workers. A file existing on disk does not prove runtime compatibility.
A missing configured token fails before the applicable inference call.
Invalid or unavailable selected resources are not silently swapped to a
different model or device.

Cache/checkpoint identities include effective resource fingerprints or
associated backend settings. Do not carry unrelated historical ASR, timing,
speaker, translated or condensed outputs across models without reevaluation.
`--fresh`, `--fresh-refinement`, and `--fresh-translation` provide
explicit recomputation controls. `--re-render` intentionally reuses an
existing translated contract instead of repeating model inference.

For programmatic Riva translation with condensation enabled, supply a
compatible `SubtitlePipelineConfig.condensation=TranslationConfig(...)`.
A Riva-specific `TranslationConfig` is not an OpenAI chat model; the pipeline
rejects ambiguous implicit reuse. `--translator none` skips condensation
even if `--condense-high-cps` was specified.

Configuration alone does not start/stop, load/unload, or download any model,
and must not change another service's GPU allocation. Qwen's optional
managed-ASR flow still requires explicit ownership/lease admission. Do not
use production/private media for smoke testing without authorization.

**Verification boundary:** CPU and frontend checks from the prior local
integration handback apply to their recorded exact source and local test
patch. Any subsequent code or documentation change requires fresh tests on
its own exact head. Live text/ASR inference was unavailable during that
campaign and remains unverified until separately exercised safely.
