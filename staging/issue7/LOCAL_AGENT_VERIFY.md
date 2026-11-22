# PR #8: Exact-source host qualification and UI verification

## Local-agent kickoff

Work in `fvanevski/nemovox` on issue #7 / PR #8, **not** merged issue #5.
Central ChatGPT has already authored the substantive code, architecture, tests,
and documentation. Your task is to execute a **single consolidated host
verification**, diagnose any failures precisely, and publish an auditable
evidence handback. Do **not** redesign, change source logic, or make new PRs.

```text
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=7
PR_NUMBER=8
TARGET_BRANCH=issue/7-studio-subtitle-workflow
TESTED_SOURCE_SHA=f38a1f734642cf361bbb64f3020a9711f10e9b3a
BASE_MAIN_SHA=8205d7783230272c39a669c3f51d89bca50669e6
LOCAL_AGENT_PRIMARY=evidence
LOCAL_AGENT_FALLBACK=build (only when the actual harness forbids test/scratch execution, with explicit role boundary)
SOURCE_EDITS=forbidden
REMOTE_SOURCE_UPDATES=forbidden
MERGE=forbidden
ISSUE_CLOSURE=forbidden
GPU_SERVICE_MUTATION=forbidden
HOST_INSTALLATION=read-only or isolated disposable test venv only
TARGET_ENVIRONMENT=Garuda Linux, Python 3.13 project CLI env + Studio frontend Node/npm
GPU_INTENSIVE_VALIDATION=optional read-only/available-only, never required for CPU PASS
```

## 1. Admit the exact production source

Fetch the branch without rebasing, resetting, force-pushing, or modifying an
existing operator worktree. Create a clean detached worktree at
`f38a1f734642cf361bbb64f3020a9711f10e9b3a` (suggested scratch `/workspace/nemovox-issue7-evidence`).
Prove full `git rev-parse HEAD`, `git status --porcelain=v1`, and module
import paths. A single evidence-only commit after qualification may advance
the issue branch; before such a commit, establish tree equivalence of all
source, tests, config and docs with the tested SHA. Never infer it from a
short SHA or previous tests. If branch/main changed, report `STALE`, do not
merge or reconcile branches.

Before actions, inspect the installed OpenCode/Hermes tooling and permissions
and use the supported host terminal route. The Evidence role may lack
permission for tests that write caches; if so, use an explicitly scoped Build
validation session with source edits and production mutations still forbidden.
Do not weaken runtime guard or tool policies merely to run tests.

## 2. Deterministic verification (CPU only)

Use existing environments and the repository's actual CLI/package setup.
Record interpreter/node/npm versions, commands, return codes, and any
environment setup. Work from the repository root except the frontend step.

```bash
python -m compileall -q src tests experiments
pytest -q tests/test_translation_review.py tests/test_studio_api.py \
  tests/test_model_resources.py tests/test_resource_integration.py \
  tests/test_translation.py
pytest -q -m "not gpu"
cd web
npm test
npm run build
```

If `python` or `pytest` points outside the actual project env, run the
equivalent exact `.venv/bin/python -m pytest` and record the effective
command. Use existing `web/node_modules` if compatible; if dependencies
must be provisioned, use only an isolated test environment and a committed
lockfile. Never update the dependency manifest, lockfile, test assertions,
or project settings to obtain PASS. Local tests must not touch models/services.

## 3. Behavioral invariant probes

Validate, using deterministic fake clients and controlled metadata:

- `studio_api.Options` retains old persisted-job compatibility but the
  frontend's new default is `translator=riva`,
  `translation_review=balanced`; CLI legacy
  `openai_chat|swiftu` remains explicit and functional.
- `process-media` and Studio media selection use **audio-only** ordinal
  indices. FFprobe reports actual tracks, languages, titles and channels
  for authorized local/uploaded files; no spurious 0..3 list. Invalid
  index and unauthorized path are rejected server-side. Switching source
  resets invalid selection and an old source's asynchronous inspection
  result must not appear as current; remote URLs do not fabricate metadata;
  a recording contract does not request an audio stream.
- Riva primary translation must precede the target-language text-model
  review; `SwiftU`/NInfer is not invoked as the **primary** translator on the
  GUI's normal path. Source-language refinement remains separate.
- Raw Riva English survives in `machine_translated_text`; review can modify
  only final target text with validated per-ID status/reason. Source text,
  speaker IDs, timing and utterance IDs are invariant. Malformed/incomplete
  reviewer output or missing/duplicate Riva results cannot fabricate an
  accepted translation or fallback to a different model.
- Changing text model, style, glossary, source/context or Riva output
  invalidates the appropriate review cache; re-render uses the saved
  contract and makes no model calls.
- Localization style affects the English **post-edit**, not raw Riva MT.
  A disabled post-edit is clearly indicated as rendering style ineffective.
- Configuration help covers all four actual ASR modes with tradeoffs.
  Menu descriptions work for hover, keyboard, and touch-visible
  selection. The optional French/Mandarin example glossaries display
  proper titles, names/terms/notes and are *off* by default. No private
  host paths, credentials or audio content appear in the evidence.

When suitable existing local media is already within Studio's approved
roots, run one bounded read-only FFprobe against a single-track and/or
multi-track example. Do not search unrelated private media or consume a
user-selected video beyond metadata. If no approved file or host-side
selection is available, record `REAL_MEDIA_STREAM_PROBE=UNVERIFIED`.
You may inspect already-present host Riva and text-service configuration
read-only to distinguish `configured` from `live/inference-verified`.
**Never start/stop/restart/evict models, alter GPU allocation, or run
full-episode inference.** Missing live services are `UNVERIFIED`, not FAIL
or PASS for deterministic tests.

## 4. Documentation and integration audit

Compare current CLI `generate-subtitles --help`, JSON resources,
Studio Options/TypeScript `JobOptions`, UI descriptions, API endpoints,
`README.md`, `web/README.md`, `docs/gui/SUBTITLE_WORKFLOW.md`,
`docs/gui/ARCHITECTURE.md`, and
`docs/model-resources/CONFIGURATION.md` to the exact checked-out source.
Report broken configuration examples, misleading backends and omissions.
Identify semantic/API regressions as `BLOCKING` or `IMPORTANT`
with file+line, reproduction, and precise expected behavior. Do not
silently patch errors, reduce tests, or claim live Riva success.

## 5. Evidence packet and publication

Return one self-contained `.tar.gz` evidence archive including:

- `manifest.json`: schema, exact source SHA, repository+PR, worktree
  identity, UTC run ID, relative member paths, byte counts and SHA256;
- `source/identity.json` including full Git SHA, worktree status, imported
  package and validation environment;
- `validation/`: command logs with commands and exit codes (compile,
  focused, CPU suite, frontend test/build, selected behavior/FFprobe probes);
- `review/`: concise documented acceptance matrix, unresolved blockers,
  documentation audit, applicability of live unavailable services;
- `COMMAND_SHAPE_FRICTION.txt` reporting requested vs effective commands,
  role/gate denials, workaround and actual side effects, or explicit none.

Archive suggestion:
`/workspace/nvidia-transcribe/outputs/handbacks/nemovox-pr8-issue7-<UTC_RUN_ID>.tar.gz`.
Keep archives, weights, media, credentials and raw private paths out of Git.
Use `staging/issue7/evidence/<UTC_RUN_ID>/` on the **existing PR branch**
for the sanitized small evidence members, with one evidence-only commit if
the remote source head is still as admitted. Verify the published commit
diff contains only that evidence root, never production source/tests/docs.
Do not open another PR. The operator can attach the private archive in
ChatGPT to enable Central member-hash audit.

Return a short handback with:
```text
RESULT=PASS|FAIL|BLOCKED|STALE
TESTED_SOURCE_SHA=<40 hex>
CPU_COMPILE=...
CPU_FOCUSED=...
CPU_FULL=...
WEB_TEST=...
WEB_BUILD=...
AUDIO_PROBE=...
TRANSLATION_PIPELINE=...
DOC_AUDIT=...
SOURCE_CHANGES=no
BLOCKING_FINDINGS=[...]
UNVERIFIED_ITEMS=[...]
EVIDENCE_BRANCH_SHA=<40 hex or none>
EVIDENCE_ROOT=<repo-relative>
EVIDENCE_ARCHIVE_PATH=<host path>
EVIDENCE_ARCHIVE_SHA256=<64 hex>
EVIDENCE_ARCHIVE_BYTES=<int>
COMMAND_SHAPE_FRICTION=...
```

## Stop conditions

Stop and return host evidence if tests fail, source or branch moves,
permissions reject the execution route, optional model dependencies are
unavailable, or the scope would require substantive source changes. A
semantic fix remains Central-owned. Do not close issue #7 or ask Codex
for review; Central will independently review the exact tested head and
handle PR lifecycle after the handback.
