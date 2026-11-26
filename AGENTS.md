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
- Promote broadly applicable mechanisms, not episode-specific patches. A
  failing example is not dispositive against a useful evidence channel;
  evaluate cross-scenario benefit, regression risk and cost. Use complete
  preceding and following dialogue at adaptable scene scale, variable audio
  crop geometry, independent acoustic hypotheses and visually observed
  source-language captions where useful. Preserve the distinction between
  lexical meaning, occurrence, timing and speaker evidence. Configurable
  bounds are resource controls, not assumed universally optimal chunk sizes.
- Consolidate substantive work in appropriately sized changes on the existing
  PR. Central owns architectural decisions and implementation; local agents
  perform host integration and bounded mechanical cleanup from an explicit
  reviewed inventory. No serial historical replay campaign, artificial
  exact-output preservation gate, or test-count target.

## Workspace Boundaries
- This repository (normally `/workspace/nvidia-transcribe`, but isolated worktrees are supported) is the Nemovox development workspace. Recording recognition uses the local MOSS native-event worker, configured by role.
- **Do not modify `/workspace/pyannote`** or install dependencies into `/workspace/pyannote/.venv`. Treat `/workspace/pyannote` and its outputs as read-only reference material.
- **Do not modify, move, rename, or redownload** the local checkpoint at `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`.

## Local GPU & Runtime Boundaries
- Host GPU: single NVIDIA RTX 5090 (32 GB VRAM, Blackwell sm_120).
- Do not stop, restart, or unload external GPU services automatically. Check `nvidia-smi` before GPU runs.
- Use the repository's dedicated interpreter for orchestration and isolated worker interpreters for MOSS, Riva and optional CPU acoustic assessment. Admit the exact model assets, code, runtime and device; never download implicitly or substitute CPU for CUDA.
- MOSS performs a recording-level request; explicit spoken language is required. Do not invent lexical timestamps or speaker identities. No automatic smaller-window retry.

## Provenance & Evaluation Rules
- Preserve source-language transcription and original Unicode text verbatim alongside any English translation; never overwrite source transcripts with translations.
- Bind current-source quality decisions to the exact recording ledger and separate acoustic observations. Canonical source may differ from ASR only with positive observation support; preserve prior hypotheses and expose unresolved ambiguity. Do not equate same-model repeat observations with independent-model or human confirmation.
- Convert excerpt-relative timestamps to absolute media timestamps using recorded crop offsets (`source_offset_s`).
- Never fabricate missing timestamps, confidence scores, or speaker mappings.
- Keep CPU-only unit tests strictly separated from live GPU inference tests (`-m "not gpu"`).
- For bounded NInfer meaning-first QC, opt in to `managed_text_client(..., wire_protocol="ninfer_messages")` on the existing owned text service. The installed NInfer Chat Completions parser ignores nonstandard `thinking_budget`; do not claim it was enforced because the JSON payload contained that field. Anthropic Messages accepts `thinking: {type: disabled}` for T1 or `{type: enabled, budget_tokens: N}` with N >= 1024 and separate final-answer capacity for T2. Keep these version-bound to the QC policy and never silently fall back to unbounded model thinking.
- For isolated-worktree CPU and Studio tests, run from the candidate root with `PYTHONPATH=src:.` and the intended interpreter via `python -m pytest` (including the separate `.venv-studio` interpreter). Verify package and root-level `scripts.*` imports resolve to the candidate worktree rather than the main checkout. A green test using main's source is not candidate validation.
- Inspect all required local model assets before invoking workers; offline environment variables are not proof against every loader's implicit downloads.
- Exclude model weights, full-length media, `.venv`, large audio/logs, and secrets from Git commits and evidence archives.
- Treat `configs/model-resources.example.json` as the public template and `configs/model-resources.local.json` (or `NEMOVOX_MODEL_RESOURCES`) as host-specific configuration. Preserve role/adapter compatibility, deterministic override precedence and credential indirection (`api_key_env`); never silently substitute other model endpoints or CPU devices.
- Resource selection alone is not lifecycle authority. The CLI/Studio job pipeline MUST automatically acquire and release **explicitly Nemovox-owned** model services under verified unit/resource/endpoint/model identity and cooperative locks. Never stop/evict unrelated GPU services, bypass ownership admission, downgrade device, or download checkpoints. See `docs/model-resources/LIFECYCLE.md`.
- Keep `README.md`, `web/README.md` and `docs/model-resources/CONFIGURATION.md` consistent with final tested CLI and Studio behavior. Remove superseded aliases and paths when their current use is no longer justified.

## Native dialogue and source assessment

MOSS is the normal recording backend. Assess event coverage, including both recording boundaries, without equating uncovered time with omitted speech. Optional CPU speech activity is investigative evidence, not a correction or speaker authority. Preserve coherent native events; acquire additional acoustic evidence only when justified by material uncertainty.
