# PR #8: Exact-source verification after Codex P2 remediation

**Execute once in a clean isolated worktree; do not replay earlier host campaigns.**
The previous 179-CPU / 10-Vitest PASS was for the superseded source
`4d560467fdcf97c0b01b54a9b7cf711e44317eaf`. Its preserved packet is
still valid history, but cannot qualify this amended production source.

```text
REPO_ID=fvanevski/nemovox
ISSUE=7
PR=8
BRANCH=issue/7-studio-subtitle-workflow
TESTED_PRODUCTION_SHA=15101b37ae970e5c5ec6711343abe728701ecd30
BASE_MAIN_SHA=8205d7783230272c39a669c3f51d89bca50669e6
CODEX_REVIEWED_OLD_HEAD=31a3459b3d5a32bcaef72ebaea00fc1b2e8ae997
HARNESS=OpenCode/Hermes Evidence role; scoped Build only for test execution if required
SOURCE_EDITS=forbidden
GPU_SERVICE_CHANGES=forbidden
CREATE_PR=forbidden
MERGE_OR_CLOSE=forbidden
```

## Scope and exact authority

Central independently reviewed three Codex P2 threads on old head:
- [English Riva en→en unsupported](https://github.com/fvanevski/nemovox/pull/8#discussion_r4656219970)
- [combobox highlight not exposed to assistive technology](https://github.com/fvanevski/nemovox/pull/8#discussion_r4656219972)
- [one malformed glossary disables entire catalog](https://github.com/fvanevski/nemovox/pull/8#discussion_r4656219974)

Five Central commits on the existing issue branch address these findings
and provide regressions; the complete current-source diff changes only 13
specified source/test/docs files, with no unrelated production campaign:
`translation.py`, `subtitle_pipeline.py`, `studio_api.py`, `refinement.py`,
`cli.py`, `App.tsx`, `ExplainedSelect.tsx`,
`config-options.ts`, `test_translation_review.py`,
`test_studio_api.py`, `ExplainedSelect.test.tsx`,
`config-options.test.ts`, `docs/gui/SUBTITLE_WORKFLOW.md`.

Fetch the current remote branch, verify it contains full SHA `15101b37ae970e5c5ec6711343abe728701ecd30`,
and work from a **detached isolated worktree** pinned exactly to this source.
Before running, prove the new branch HEAD differs from the tested source
only by this handoff or evidence files. Do not rely on the name of a local
developer branch. Do not mutate host venvs, package installs, checkpoints,
model processes, or service configuration.

Earlier host qualification showed that its shared Python venv has an editable
install pointing to a different checkout. Set `PYTHONPATH` to the **actual
new detached worktree's `src/`**, verify the imported package file, and
log effective commands. Reuse already installed `web/node_modules` only
when clean and without modifying the source or lockfile; if a temporary
symlink is used, remove it and prove a clean worktree afterward.

## Required checks

```bash
# Run in the clean detached exact-SHA worktree:
export PYTHONPATH="$(pwd)/src"
python -c 'import nvidia_transcribe, yaml; print(nvidia_transcribe.__file__, yaml.__version__)'
python -m compileall -q src tests experiments
python -m pytest -q tests/test_translation_review.py tests/test_studio_api.py tests/test_translation.py tests/test_model_resources.py tests/test_resource_integration.py
python -m pytest -q -m "not gpu"
(cd web && npm test && npm run build)
```

Use the existing trusted host venv's absolute interpreter for these Python
commands. Never substitute a global Python whose dependencies differ. Record
command/executable/source identity, all exit codes, counts and warnings.

### Invariants to verify from source + tests

1. Explicit English source and English target, or **auto-detected English**
   in a recording contract, preserve exact source transcript text and do
   not invoke Riva, a source-language text model, or English post-edit.
   Records explicitly report `bypassed_same_language` /
   `skipped_same_language`. SRT still renders and offline rerender works.
   Verify English CLI contract processing with a deliberately
   **nonexistent** Riva worker/model path is successful without service use.
   If source language is unknown/non-English, Riva requirements still
   apply; no silent model or device fallback.
2. Studio submission with explicitly English source and target does not
   require Riva files; UI offers the same English bypass without hiding the
   media/ASR controls. Non-English Studio still requires dedicated Riva
   and uses the original MT followed by one bounded English review.
3. `ExplainedSelect` has select-only combobox semantics, labeled popup,
   linked option IDs via `aria-activedescendant`, keyboard Arrow/Home/End/
   Enter/Escape behavior, updates the spoken description of the highlighted
   option, and restores focus after a pointer selection. Existing SSR/unit
   tests and TypeScript build must pass. If a suitable browser DOM harness is
   **already installed**, run one bounded keyboard/assistive-attribute DOM
   smoke; otherwise explicitly mark interactive screen-reader behavior
   `UNVERIFIED` instead of installing new tooling or claiming full proof.
4. Optional malformed `.yaml`/`.json` glossaries (`null`, scalar,
   broken syntax, invalid list-field shape) are rejected/individually
   skipped; valid neighbors remain discoverable. Valid shipped French
   and Mandarin glossary metadata still appears. Do not swallow unexpected
   package/runtime import failures to fake a PASS.
5. Re-run preserved core invariants: untouched ASR by default, audio-only
   FFprobe track mapping, invalid audio index/path authorization, primary
   Riva→selective English review, cache/rerender, PyYAML dependency,
   `condense_high_cps=false` in both Studio/UI (explicit opt-in only).
   No additional GPU inference necessary; previously observed real
   single-track probe remains earlier-source evidence since media parser
   has not changed.

## Evidence and publication (no duplicate operator upload)

Publish a **complete small sanitized evidence packet** under a *new* path
`staging/issue7/evidence/<UTC_RUN_ID>/` on the **same PR branch**,
as one evidence-only commit. Include exact-SHA identity, source-vs-current-head
equivalence, complete test logs, regressions/acceptance matrix, current
findings and `COMMAND_SHAPE_FRICTION.txt`; add manifest SHA256+byte counts.
Do not edit production source, rewrite older evidence, add secrets, raw
media, model weights or personal host logs. If an evidence-only push races
with a branch update, stop as `STALE` rather than force-pushing.
Optional local `tar.gz` backup may be created without operator involvement.

**The handback must explicitly decide whether an upload is needed:**

```text
RESULT=PASS|FAIL|BLOCKED|STALE
TESTED_SOURCE_SHA=15101b37ae970e5c5ec6711343abe728701ecd30
PYTHON_IMPORT_IDENTITY=...
CPU_COMPILE=...
CPU_FOCUSED=...
CPU_FULL=...
WEB_TEST=...
WEB_BUILD=...
ENGLISH_SOURCE_NO_INFERENCE=...
ENGLISH_CLI_MISSING_RIVA=...
GLOSSARY_ERROR_ISOLATION=...
COMBOBOX_ACCESSIBILITY=... (interactive DOM UNVERIFIED if not run)
SOURCE_EDITS=no
BLOCKING_FINDINGS=[...]
UNVERIFIED_ITEMS=[...]
GITHUB_EVIDENCE_STATUS=COMPLETE|INCOMPLETE|BLOCKED
GITHUB_EVIDENCE_ROOT=staging/issue7/evidence/<UTC_RUN_ID>
GITHUB_EVIDENCE_COMMIT=<full 40-char SHA>
PRIVATE_ARCHIVE_EXISTS=yes|no
PRIVATE_ARCHIVE_ADDS_REQUIRED_EVIDENCE=yes|no
PRIVATE_ARCHIVE_ADDITIONAL_CONTENT=<specific items or none>
PRIVATE_ARCHIVE_UPLOAD_REQUIRED=yes|no
REQUIRED_OPERATOR_ACTION=NONE|UPLOAD_ARCHIVE
COMMAND_SHAPE_FRICTION=...
```

If all acceptance evidence is already on GitHub,
set `PRIVATE_ARCHIVE_ADDS_REQUIRED_EVIDENCE=no`,
`PRIVATE_ARCHIVE_UPLOAD_REQUIRED=no`, `REQUIRED_OPERATOR_ACTION=NONE`
and do **not** prompt the operator to attach the host archive.
If material required evidence cannot be published, explain exactly what is
missing, why it matters, and whether it needs an operator upload.

Stop and return native diagnostics on any failure; do not relax tests,
make source changes, request a second Codex review, resolve threads,
approve, merge, or close issue. Central handles adjudication and final
exact-head review.
