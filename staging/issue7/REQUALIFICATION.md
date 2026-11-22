# PR #8: focused exact-source requalification after Central corrections

**This handoff supersedes the source identity in the earlier `LOCAL_AGENT_VERIFY.md`.
Do not run new tests against the old source.**

```text
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=7
PR_NUMBER=8
BRANCH=issue/7-studio-subtitle-workflow
TESTED_PRODUCTION_SHA=4d560467fdcf97c0b01b54a9b7cf711e44317eaf
BASE_MAIN_SHA=8205d7783230272c39a669c3f51d89bca50669e6
PREVIOUS_QUALIFIED_SOURCE_SHA=e824321d5680e61044cdbccc9564ca25d2b2f9e9
PREVIOUS_EVIDENCE_COMMIT_SHA=198f92cf5f3a3920bf54c9a81ea477cb2bb3bc36
SOURCE_EDITS=forbidden
GPU_SERVICE_CHANGES=forbidden
FRESH_PR=forbidden
MERGE_OR_ISSUE_CLOSURE=forbidden
LOCAL_HARNESS=OpenCode/Hermes evidence role; use scoped Build for host tests if mandatory
```

## Why this requalification is required

Central independently inspected the published [prior evidence](https://github.com/fvanevski/nemovox/tree/198f92cf5f3a3920bf54c9a81ea477cb2bb3bc36/staging/issue7/evidence/20261122T020859Z) and source, accepted the original deterministic tests at `e824321d5680e61044cdbccc9564ca25d2b2f9e9`, then corrected four narrowly bounded defects at `4d560467fdcf97c0b01b54a9b7cf711e44317eaf`:

1. `pyproject.toml` now explicitly requires `PyYAML>=6.0` because CLI and Studio both load YAML glossaries; the previous host's installed package masked the missing project requirement.
2. `studio_api.Options.condense_high_cps` now defaults **false**, consistent with the GUI and the single-pass English-review contract. Explicit `true` remains an opt-in.
3. Studio's running-job stage label no longer says `Refining` when source refinement is prohibited.
4. Studio glossary hint refers only to target-language English review. Two new CPU regressions cover the dependency declaration and default/explicit condensation, including persisted `job.json`.

All previously tested media inspection and translation code is unchanged. Re-test **this** source and audit the four-file diff. Do not reproduce the original host campaign wholesale or modify the shared GPU services.

## Source and evidence admission

On the Garuda Linux host fetch the current branch, prove the fetched ref contains the exact `4d560467fdcf97c0b01b54a9b7cf711e44317eaf` object and prove it is the latest production source. Run tests from a clean detached worktree at `4d560467fdcf97c0b01b54a9b7cf711e44317eaf` without changing existing operator checkouts, config, or data. Because the installed editable Python package otherwise imports `/workspace/nvidia-transcribe/src`, **set `PYTHONPATH` to the detached worktree's `src` and verify `nvidia_transcribe.__file__`** before and during tests. Record exact full `git rev-parse HEAD`, worktree status, Python/Node/npm versions and dependency provenance.

The earlier handback used a temporary `web/node_modules` symlink pointing to the existing host dependency directory and removed it afterward. You may repeat that bounded read-only dependency reuse only if the worktree is clean afterward and installed packages/lockfiles are not modified. Do not elevate permissions or change host application services.

If current remote branch content changes from the exact admitted state, establish whether it is handoff/evidence-only by an exact Git diff before continuation; if source changes, report STALE and stop.

## Commands, primary validation

Use the existing host venv, not an arbitrary interpreter. From the exact worktree:

```bash
export PYTHONPATH=/workspace/nemovox-issue7-requal/src
/workspace/nvidia-transcribe/.venv/bin/python -c 'import nvidia_transcribe, yaml; print(nvidia_transcribe.__file__); print(yaml.__version__)'
/workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests experiments
/workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_studio_api.py tests/test_translation_review.py tests/test_model_resources.py tests/test_resource_integration.py tests/test_translation.py
/workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
cd web
npm test
npm run build
```

If you choose a different isolated worktree path, **set PYTHONPATH to the actual checked-out src directory**, not the example path above. Record effective commands, exit codes and test totals. Do not use PYTHONPATH from the production worktree. Do not modify tests/assertions to get green. Check `pyproject.toml` metadata with `tomllib` and the installed PyYAML version, including whether the package's runtime requirement is satisfied.

## Acceptance checks

- Omitted `condense_high_cps` in `JobRequest` resolves to `false`, and persisted `job.json` retains `false`.
- Explicit `true` still survives parsing, persistence and existing runner flag mapping. No second text-model condensation pass is requested by default.
- UI initial state remains `refinement=off`, `translator=riva`, `translation_review=balanced`, `condense_high_cps=false`. The backend rejects non-off Studio source refinement. Original ASR, Riva translation, selective English review, per-ID provenance and offline rerender regressions continue to pass.
- CLI and Studio YAML examples load from their declared `PyYAML` dependency; no hidden reliance on a manually provisioned unrelated venv package.
- Updated GUI hint and progress strings accurately describe the workflow. No streaming/service changes, media scanning or live inference necessary.
- Reconcile earlier FFprobe single-track result and synthetic multitrack test as **previous-source evidence**; the unchanged audio inspection code needs no extra real-media read for these four corrections. Do not represent real multi-track or live ASR/Riva/text/GPU/JASSUB as newly verified.

## Evidence and handback

Create one bounded self-contained `tar.gz` with `manifest.json` (SHA256 for members, exact source, command and environment identities), `source/identity.json`, full command logs, focused acceptance matrix, four-file diff audit, and `COMMAND_SHAPE_FRICTION.txt`. No weights, raw media, secrets or private paths. Publish **sanitized** small evidence members under `staging/issue7/evidence/<NEW_UTC_RUN_ID>/` on this same PR branch as one evidence-only commit, with compare/readback proving no source/test/config change. Keep the prior evidence packet untouched. Return an operator-accessible local archive pathname and SHA256 for optional attachment to central review.

```text
RESULT=PASS|FAIL|BLOCKED|STALE
TESTED_SOURCE_SHA=4d560467fdcf97c0b01b54a9b7cf711e44317eaf
PYTHON_IMPORT_IDENTITY=PASS|FAIL
PYYAML_DECLARED_AND_INSTALLED=PASS|FAIL
CONDENSATION_DEFAULTS=PASS|FAIL
CPU_COMPILE=...
CPU_FOCUSED=...
CPU_FULL=...
WEB_TEST=...
WEB_BUILD=...
DOC_COPY_AUDIT=...
SOURCE_EDITS=no
BLOCKING_FINDINGS=[...]
UNVERIFIED_ITEMS=[...]
EVIDENCE_BRANCH_SHA=<full sha>
EVIDENCE_ROOT=staging/issue7/evidence/<UTC_RUN_ID>
EVIDENCE_ARCHIVE_PATH=<host path>
EVIDENCE_ARCHIVE_SHA256=<sha256>
COMMAND_SHAPE_FRICTION=...
```

**Stop:** Any substantive defect, source mismatch, unmet declared dependency, permission blocker or failed regression: return native evidence without bypassing. Central owns source remediation and exact-head independent review. Do not request Codex, formalize approval, merge or close issue #7.
