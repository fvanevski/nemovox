# Issue #5: first integrated resource-modularization slice (Central-authored)

> **Execution authority:** This is a candidate implementation at an exact GitHub issue-branch head, not yet host-qualified. The local Build agent performs one consolidated integration and evidence campaign. No PR, merge, service takeover, or issue closure is authorized by this document.

## Objective and Central design (immutable unless an evidence-backed defect requires escalation)

Prove a useful end-to-end **task role -> model resource -> specialized adapter** abstraction through the existing CLI/Studio and subtitle pipeline, retaining the current host model defaults. Do not initiate a second general AI-router or GPU scheduler design.

| Task role | Adapter | Default binding | Notes |
|---|---|---|---|
| `asr` | `nemotron_ctc` | `asr_local` | vLLM + NeMo CTC timestamp semantics remain required |
| `source_refinement` | `openai_chat` | `text_local` | OpenAI chat HTTP; default host resource is SwiftU/NInfer |
| `text_translation` | `openai_chat` | `text_local` | May map to an independent text model |
| `subtitle_condensation` | `openai_chat` | `text_local` | May map independently, including when translation uses Riva |
| `riva_translation` | `riva_worker` | `riva_local` | Language-pair-aware dedicated worker, not an OpenAI chat adapter |
| `forced_alignment` | `qwen_forced_aligner` | `qwen_local` | Optional isolated worker and existing explicit GPU handoff policy |
| `speaker_embedding` | `titanet` | `titanet_cpu` | Existing CPU speaker embedding algorithm |

New `model_resources.py` loads typed defaults and JSON host overrides; command-specific CLI overrides are last. `NEMOVOX_MODEL_RESOURCES` selects an ignored host file; `TEXT_BASE_URL`/`TEXT_MODEL` and legacy `SWIFTU_*` only override the **built-in** `text_local` resource (new names win), while `ASR_*` overrides built-in `asr_local`. Explicit `--text-url`/`--text-model` or legacy `--swiftu-*` force the specified field for all three text-task resolutions. Individual `--*-resource` options select named compatible text resources. The CLI and Studio child both resolve the same environment/configuration; Studio does not reinterpret environment values as command-line overrides.

Python module names, CLI entrypoint, current raw Unicode/speaker/CTC/timing contracts, corrective prompts, forced-aligner safety policy, and existing OpenAI-compatible HTTP request semantics are retained. Legacy `swiftu` translator and callable aliases remain accepted. Resource fingerprints distinguish cache/checkpoint identity without publishing credentials. Configuration resolution must never start or stop models or services.

### Source anchors

- `src/nvidia_transcribe/model_resources.py` and `configs/model-resources.example.json` (new)
- `src/nvidia_transcribe/cli.py`, `studio_api.py`, `subtitle_pipeline.py`, `recording.py`
- `src/nvidia_transcribe/refinement.py`, `translation.py`, `speakers.py`, `qwen_aligner.py`
- `web/src/App.tsx`, `web/src/api.ts`, `configs/runtime.env.example`
- `tests/test_model_resources.py`, `tests/test_resource_integration.py`, `tests/test_studio_api.py`

Issue acceptance and exact live repository source take precedence over this document if inconsistent. Read `AGENTS.md`, issue #5, relevant source, and `web/README.md` before execution. Historical evaluation artifacts are NOT acceptance authority for the current code.

## Local Build execution sequence

1. **Admit source identity.** Start from the existing host workspace only if clean and safe. Fetch the issue branch from `origin` **before** claiming local SHA/object identity; authenticate `refs/remotes/origin/issue/5-model-resources` against the full kickoff `TARGET_SOURCE_SHA`. Prefer isolated, collision-free worktree if the ordinary checkout is dirty or on another task. No reset/clean/stash/force/rebase. Verify GitHub issue #5 remains open and branch head is unchanged; stop `STALE` on remote drift.
2. **Inspect & qualify candidate.** Check file inventory, standard library/python version, package setup and user-host `AGENTS.md`. Run `python -m compileall -q src tests`, focused CPU tests `pytest -q tests/test_model_resources.py tests/test_resource_integration.py tests/test_studio_api.py`, then entire `pytest -q -m "not gpu"` using the repository's correct existing environment. Use source-installed Python from the authorized repository venv, not the unrelated PyAnnote environment. Diagnose deterministic failures, including our newly staged code. Check output contracts, legacy aliases, role config precedence, Riva/Qwen safety invariants and all caching signatures. There are no GitHub Actions workflows currently configured for this repo: do not fabricate a trusted CI receipt.
3. **Frontend.** From `web/`, run `npm test` and `npm run build` with existing dependencies, or `npm ci` in isolated workspace only if needed and permitted. Validate consistency of translator names between Pydantic Studio API, TypeScript types and selections. Do not redesign UI.
4. **Host resource qualification.** Resolve model roles using the *actual* host-appropriate ignored JSON configuration or a disposable copy of `configs/model-resources.example.json`, without replacing an existing user configuration. Validate `load_model_resources().public_bindings()`, the effective adapter/model IDs, environment/CLI precedence and missing/incompatible path behavior. When independent text resources do not exist on the host, use CPU fakes for split-role behavior; a real shared SwiftU/NInfer binding is sufficient for optional live smoke.
5. **Bounded live smoke, only if safely available.** Inspect running endpoint/process/GPU availability first. Reuse already-running local text inference, send one synthetic text-correction completion with low output bound and capture sanitized status/model identity/latency; optionally inspect an already-running ASR endpoint without loading/evicting a 30B model or processing a full episode. Do not start/stop/restart vLLM, embeddings, Qwen, Riva or shared services; no automated managed ASR lifecycle. Do not send private media or model prompts to remote cloud inference. Treat unavailable live paths as `UNVERIFIED`, not `PASS`. GPU is NOT required for CPU/GUI acceptance.
6. **Fix only demonstrated within-slice integration defects.** The Build agent may repair obvious import/type/syntax mistakes, narrow glue miswiring, test-fixture errors, safe packaging and compatibility issues under the Central-authored role/adapter/precedence design, then rerun the affected regressions and broader checks. Any consequential change to model semantics, role contracts, security/GPU policy, audio/timing, translation safety, or unknown compatibility tradeoff must be reported with a proposed exact patch for Central rather than unilaterally redesigned. Avoid speculative hardening or unrelated cleanup. **Do not push source edits remotely**; deliver full changed-file diff and tests in the evidence handback. Repeat once when needed within the same campaign rather than asking for separate micro-handoffs.
7. **Final state.** Freeze exact tested source SHA, executable environment identity, effective configuration with secrets/host-private paths redacted, tested options, and all residuals. No PR creation/merge/issue closure. Central will independently review source and evidence, apply source fixes and conduct the **final documentation audit after implementation stabilizes**, then move toward one consolidated PR.

## Evidence matrix and delivery

| Criterion | Required shareable artifact |
|---|---|
| Branch/identity | `source/identity.json`: full GitHub branch SHA, local HEAD, diff status, checked worktree |
| Typed roles, precedence, failures | `validation/model_resources.txt`: focused pytest log, plus sanitized role/adapter/fingerprint map and config failure cases |
| CLI/Studio, legacy compatibility | `validation/integration.txt`: exact focused tests and Studio child env/CLI routing evidence |
| Cache/provenance | `validation/cache_identity.txt`: tests or bounded key equality/inequality receipts for refinement, translation, condensation, ASR config; no source media |
| Baseline suite & frontend | `validation/cpu_full.txt`, `validation/web_test.txt`, `validation/web_build.txt`: commands, exit codes, bounded output |
| Host adapter limits | `host/resource_smoke.json`: consulted endpoints and confirmed or UNVERIFIED status; synthetic request only, no secrets |
| Changes/risks | `review/full_candidate_diff.txt` (against immutable base), `review/proposed_local_diff.txt` (if any), `review/findings.md` (blocking / nonblocking / doc gaps) |
| Reproducibility & command friction | `manifest.json` with criterion/path/SHA-256/bytes/source identity and `COMMAND_SHAPE_FRICTION.txt` with canonical schema |

Use an isolated private evidence-staging root under ignored `outputs/handbacks/` and a **single allowlisted, sanitized, operator-attachable `.tar.gz`** named `nemovox-issue5-resource-integration-<UTC_RUN_ID>.tar.gz`. Check every member against an explicit manifest, SHA-256 and byte count; exclude symlinks, traversal, private paths, tokens, environment dumps, transcripts, model outputs, massive logs and downloaded weights. Reopen tar and verify member checksums; compute final full archive SHA-256 *outside* the archive. Return its complete path and hash in the conversational handback. If safe public publication is possible, publish one additional sanitized **text-only** evidence packet under `staging/issue5/evidence/<UTC_RUN_ID>/` on the same issue branch, with one non-force evidence-only commit after remote re-read. Do **not** commit the private `.tar.gz` to public GitHub. If Git guards/permissions or privacy prohibit publication, report `EVIDENCE_PUBLICATION=BLOCKED` and attach the archive through operator handback; do not bypass guards.

### GitHub permission boundaries

`LOCAL_AGENT_PRIMARY=build`; `SOURCE_REMOTE_UPDATE=forbidden`; `EVIDENCE_REMOTE_UPDATE=allowed` for a single sanitized evidence-only branch commit; `EVIDENCE_ROOT=staging/issue5/evidence`; `EVIDENCE_PACKET_REQUIRED=yes`; `EVIDENCE_ARCHIVE_REQUIRED=yes`; `PR_CREATE=forbidden`; `MERGE=forbidden`; `PRODUCTION_SERVICE_MUTATION=forbidden`.

Host read-only diagnostic commands, source-preserving isolated workspace setup, CPU tests, frontend build, safe local temp config creation, and scoped local repairs are authorized. Do not modify user media, `/workspace/pyannote`, model checkpoints, production service units, global tooling, or unrelated repositories.

### Handback fields

Return `RESULT=PASS|FAIL|BLOCKED|STALE`; `SOURCE_BASE_SHA`; `TARGET_SOURCE_SHA`; `TESTED_SOURCE_SHA`; `LOCAL_SOURCE_CHANGES=yes|no`; `SOURCE_PATCH_SHA256`; `EVIDENCE_PUBLICATION=PASS|BLOCKED|SKIPPED`; `EVIDENCE_BRANCH_SHA`; `EVIDENCE_ROOT`; `CPU_FOCUSED`; `CPU_FULL`; `WEB_TEST`; `WEB_BUILD`; `TEXT_LIVE_SMOKE`; `ASR_LIVE_SMOKE`; `GPU_SERVICE_MUTATION=no|yes`; `UNVERIFIED_ITEMS`; `BLOCKING_FINDINGS`; `NONBLOCKING_FINDINGS`; `DOC_AUDIT_FINDINGS`; `EVIDENCE_ARCHIVE_REQUIRED=yes`; `EVIDENCE_ARCHIVE_STATUS=READY|BLOCKED`; `EVIDENCE_ARCHIVE_PATH`; `EVIDENCE_ARCHIVE_SHA256`; `EVIDENCE_ARCHIVE_BYTES`; `EVIDENCE_MANIFEST`; `EVIDENCE_REQUIRED_ARTIFACTS`; `EVIDENCE_INCLUDED_ARTIFACTS`; `EVIDENCE_MISSING_ARTIFACTS`; `EVIDENCE_PRIVACY_CHECK`; `EVIDENCE_MEMBER_CHECK`; plus the complete canonical `COMMAND_SHAPE_FRICTION` block:

```text
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events:
    - scenario=<short repository/task scenario key>
      requested_shape=<bounded submitted command/invocation, sanitized>
      effective_shape=<post-hook/runtime command|same-as-requested|unknown>
      rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
      event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
      execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
      correction_or_annotation=<bounded corrective evidence>
      replacement_shape=<corrected requested/effective shape|none>
      recipe_reference=<GHDEV-10 key|none>
      resource_reference=<plugin/repository recovery resource|none>
      contract_conflict=<yes|no|uncertain>
      durable_gap_candidate=<yes|no|uncertain>
  omitted_sensitive_values=<yes|no>
```

When no friction occurred: `encountered=no`, `events=[]`, `omitted_sensitive_values=no`. Record initial failures and eventual recoveries separately, including any harness-command rewriting.

**Stop/escalate** only for: stale remote head/race; material unresolved design or security/semantic ambiguity; a policy or host guard refusing a required write or tool; unauthorized service mutation risk; inability to verify the target source or safely produce an honest handback. Otherwise complete the integrated objective in this campaign, classify residuals precisely, and return full evidence rather than serial requests for permission.
