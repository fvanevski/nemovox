# AGENTS.md — Nemovox local development and evaluation

## Development Contract: Clean Forward Development

Nemovox is a pre-release personal-use application. The operator's development
objective is one clean, efficient, usable current application, not a collection
of historically compatible incarnations. This section governs future design,
implementation, test selection and evaluation; it supersedes earlier
compatibility/fixture-retention instructions in development handoffs and
historical evidence documents.

- No historical implementation, schema, fixture, output, model choice, CLI alias,
  configuration option, intermediate revision format, or previous run has a
  mandatory retain status merely because it previously existed or passed tests.
  When a design is superseded, remove the obsolete implementation and all
  consumers, tests, fixtures, flags, documentation and configuration that exist
  only to support it. Do not retain a compatibility shim, legacy mode, duplicate
  pathway or migration system unless it has a concrete current use or the
  operator explicitly chooses it. Existing tests for valid current behavior
  remain requirements; replace obsolete tests with coverage of the successor's
  actual invariants, never weaken tests to conceal a regression.
- Optimize for fluent, concise, readable target-language subtitles preserving
  intended meaning, consequential polarity, facts, narrative chronology and
  meaningful speaker distinctions. Verbatim source-word fidelity, exact
  word-level timing and complete recovery of immaterial background speech are
  not objectives. Natural restructuring and condensation are legitimate.
- Converge on one normal CLI/Studio workflow: ASR with source/timing evidence,
  cheap detection of material uncertainty, bounded contextual reasoning and
  selective independently supported acoustic (and when useful audiovisual)
  adjudication, canonical source events, dedicated translation, lightweight
  contextual English review, and rendering. Ordinary use must not require
  hand-authored episode-specific review dossiers or historical revision trees.
  This is the target architecture, not a claim that it is already integrated.
  Do not fabricate acoustic evidence, speaker identity, source speech, timing,
  or model receipts. Keep unresolved consequential uncertainty explicit.
- Keep original model observations, current-run input identity and selected
  source/MT/review evidence only to the extent needed for current correctness,
  reproducibility, useful diagnostics and honest quality assessment. These are
  current-run invariants, not requirements to retain outputs of superseded
  experiments forever. Current-version resume/rerender can remain useful;
  cross-version job/checkpoint compatibility is not required. Incompatible
  cached state must be invalidated or regenerated, not silently reused.
- Qualify the actual current code/config/model state from fresh inputs in a
  clean workspace after substantive consolidation. Historical runs may explain
  failure mechanisms, but are not current acceptance evidence, mandatory
  golden output, or a basis for continued design without fresh confirmation.
  Remove superseded development evidence from the active working tree and
  evaluation corpus; no requirement to maintain a parallel historical archive.
  Do not rewrite shared Git history. Scope physical deletion to identified
  disposable project-generated development artifacts; never infer permission
  to delete original media, models, user library outputs, user configuration,
  unrelated repositories, active jobs, or unrelated services.
- Consolidate substantive work in appropriately sized changes on the existing
  PR. Central owns architectural decisions and implementation; local agents
  perform host integration and bounded mechanical cleanup from an explicit
  reviewed inventory. No serial historical replay campaign, artificial
  exact-output preservation gate, or test-count target.

## Workspace Boundaries
- This repository (normally `/workspace/nvidia-transcribe`, but isolated worktrees are supported) is the Nemovox development workspace. Core ASR defaults to NVIDIA Nemotron 3.5 Transcribe 30B-A4B NVFP4, while model resources may be configured by role.
- **Do not modify `/workspace/pyannote`** or install dependencies into `/workspace/pyannote/.venv`. Treat `/workspace/pyannote` and its outputs as read-only reference material.
- **Do not modify, move, rename, or redownload** the local checkpoint at `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`.

## Local GPU & Runtime Boundaries
- Host GPU: single NVIDIA RTX 5090 (32 GB VRAM, Blackwell sm_120).
- Do not stop, restart, or unload external GPU services automatically. Check `nvidia-smi` before GPU runs.
- Use the dedicated `.venv` under `/workspace/nvidia-transcribe/.venv` (Python 3.13 + vLLM + NeMo Speech plugin).
- For the default Nemotron ASR server, load only the existing local checkpoint at `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`. An explicitly configured alternative model resource must be validated for its adapter and required CTC contract; do not trigger implicit remote weight downloads.
- Verify active ModelOpt mixed NVFP4/FP8 quantization (`quantization=modelopt_mixed`) in server logs; silent BF16 dequantization is invalid for 32 GB feasibility.
- Keep CTC timestamp capture (`--middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware`, `VLLM_USE_V2_MODEL_RUNNER=0`, `--trust-request-mm-kwargs`, and `mm_processor_kwargs={"capture_ctc_timestamps": true}`) enabled when evaluating word/character timestamps and diarization.

## Provenance & Evaluation Rules
- Preserve source-language transcription and original Unicode text verbatim alongside any English translation; never overwrite source transcripts with translations.
- Bind current-source quality decisions to the exact recording ledger and separate acoustic observations. Canonical source may differ from ASR only with positive observation support; preserve prior hypotheses and expose unresolved ambiguity. Do not equate same-model repeat observations with independent-model or human confirmation.
- Convert excerpt-relative timestamps to absolute media timestamps using recorded crop offsets (`source_offset_s`).
- Never fabricate missing timestamps, confidence scores, or speaker mappings.
- Keep CPU-only unit tests strictly separated from live GPU inference tests (`-m "not gpu"`).
- For bounded NInfer meaning-first QC, opt in to `managed_text_client(..., wire_protocol="ninfer_messages")` on the existing owned text service. The installed NInfer Chat Completions parser ignores nonstandard `thinking_budget`; do not claim it was enforced because the JSON payload contained that field. Anthropic Messages accepts `thinking: {type: disabled}` for T1 or `{type: enabled, budget_tokens: N}` with N >= 1024 and separate final-answer capacity for T2. Keep these version-bound to the QC policy and never silently fall back to unbounded model thinking.
- For isolated-worktree Studio tests, prefix `PYTHONPATH=src` (including when using the separate `.venv-studio` interpreter) and verify imports resolve to the candidate worktree rather than the main checkout. A green test using main's source is not candidate validation.
- Do not assume `HF_HUB_OFFLINE=1` blocks downloads performed by WhisperX/torchaudio through `torch.hub`. Check ALL required aligner checkpoints in the actual cache before loading an aligner, and do not call a loader that can fetch missing assets in an offline/cached-only campaign. Record any unexpected download as command/runtime friction without treating a downloaded model as an admitted test input.
- Exclude model weights, full-length media, `.venv`, large audio/logs, and secrets from Git commits and evidence archives.
- Treat `configs/model-resources.example.json` as the public template and `configs/model-resources.local.json` (or `NEMOVOX_MODEL_RESOURCES`) as host-specific configuration. Preserve role/adapter compatibility, deterministic override precedence and credential indirection (`api_key_env`); never silently substitute other model endpoints or CPU devices.
- Resource selection alone is not lifecycle authority. The CLI/Studio job pipeline MUST automatically acquire and release **explicitly Nemovox-owned** model services under verified unit/resource/endpoint/model identity and cooperative locks. Never stop/evict unrelated GPU services, bypass ownership admission, downgrade device, or download checkpoints. See `docs/model-resources/LIFECYCLE.md`; Qwen's separate managed-ASR handoff remains lease-gated.
- Keep `README.md`, `web/README.md` and `docs/model-resources/CONFIGURATION.md` consistent with final tested CLI and Studio behavior. Remove superseded aliases and paths when their current use is no longer justified.
