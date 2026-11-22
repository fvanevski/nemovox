# Issue #10 / PR #11: Riva automatic-device GPU admission correction and source-bound host qualification

## Mission and authority

This is the **single consolidated continuation** for the Codex P1 finding on PR #11 at prior head `2dd2e29e015db2f2744b08ae96311b5dbabf196d`, discussion `#discussion_r4658943795`. Central has authored the fix and regression tests in the same branch as this instruction file. **Do not open another PR or redesign model lifecycle.**

- Repository: `fvanevski/nemovox`; issue `#10`; draft PR `#11`; branch `issue/10-auto-model-resources`.
- Primary OpenCode agent: `build` in a **genuinely fresh session**.
- Working repository normally `/workspace/nvidia-transcribe`; a fresh isolated worktree is preferable for testing.
- Expected source head: the full SHA in the operator's accompanying kickoff, which must equal the current remote PR head on admission. The existing `main` base was `8fee1e177db9742dbdb38c4a22bdd663c20ac80f`; re-read live GitHub state rather than assuming it remains unchanged.
- `SOURCE_REMOTE_UPDATE=forbidden`; `EVIDENCE_REMOTE_UPDATE=allowed` **only for one public-safe, evidence-only commit on this existing branch**. `MAIN_WRITE=forbidden`; `NEW_PR=forbidden`; `MERGE=forbidden`; `ISSUE_CLOSE=forbidden`; `PRODUCTION_STUDIO_PROMOTION=forbidden`.
- `EVIDENCE_PACKET_REQUIRED=yes`; `EVIDENCE_ROOT=staging/issue10/evidence/<fresh-UTC-run-id>`; `EVIDENCE_ARCHIVE_REQUIRED=no` if that GitHub packet contains all safe required proof; `PRIVATE_ARCHIVE_UPLOAD_REQUIRED=no` unless actual necessary private/unpublishable review material requires one.

## Exact implementation already supplied by Central

- In `src/nvidia_transcribe/translation.py`, the Riva subprocess is now inside `lifecycle.worker(resource_name=riva_resource_name)` when the request is `cuda*` **or** `auto`, before automatic device resolution can choose CUDA. Explicit `cpu` keeps its existing route.
- In `tests/test_translation.py`, central-added CPU mocks prove `auto` and explicit CUDA acquire/hold/release the lease around the subprocess, explicit CPU does not claim a GPU lease, denied admission prevents process launch, and failed worker launch unwinds the lease.
- Do not broaden to a GPU manager redesign, device-policy refactor, F-02 timeout tuning, SIGKILL campaign, service replacement, or unrelated quality changes. The previously validated crash cleanup and ASR auth paths remain provenance. If a *demonstrated material regression* blocks the stated acceptance, diagnose and address it within the authorized **host test environment/config**; report any needed application semantic change to Central with a concrete reproducer and proposed patch, **do not silently modify source**.

## Phase A: isolate and validate the corrected source (CPU only)

1. Read `AGENTS.md`, `docs/model-resources/LIFECYCLE.md`, this file, the PR/issue and the Codex finding. Confirm actual current remote head and base, current branch identity, source/test diffs and no unexpected changes. On this Build task, safely `git fetch` the exact issue branch **before** local SHA/object proof; authenticate fetched head and create an isolated candidate worktree or use an independently clean exact-head worktree. Never reset/stash/clean/rebase the operator's existing workspace or change production checkout. If head differs from kickoff, STOP `STALE`.
2. Inspect interpreter/package provenance: execute from candidate worktree with `PYTHONPATH=<exact-candidate>/src`; verify the imported `nvidia_transcribe.translation.__file__` is the candidate, not another installed editable source. Use the proven host Python venv(s) rather than installing new CUDA/PyTorch/vLLM dependencies to satisfy a test.
3. Run and retain exit statuses and bounded full logs for:
   - `python -m compileall -q src tests experiments`
   - `python -m pytest -q tests/test_translation.py tests/test_resource_lifecycle.py tests/test_translation_review.py tests/test_resource_integration.py tests/test_model_resources.py tests/test_studio_api.py`
   - `python -m pytest -q -m 'not gpu'`
   - from `web/`: `npm test`, `npm run build` (reuse authenticated dependencies; if missing, install only inside candidate, without rewriting a lockfile).
   Use the exact venv executable with `PYTHONPATH` and candidate `cwd` for Python; do not accidentally validate `main`. Do not weaken tests or make missing a required test an optional skip.
4. Independently confirm the P1 invariants from the actual execution trace: on uncached Riva `auto`, lease entered **before** worker invocation, retained through completion and released after; a denied lease prevents the worker subprocess from starting; explicit CUDA is protected; explicit CPU preserves the explicit CPU behavior; English-to-English bypass and cached translation do not unnecessarily launch a worker. Review the resulting patch rather than accepting a mock PASS as host GPU proof.

## Phase B: bounded host GPU integration (only after Phase A PASS)

1. Refresh host state: canonical model-lifecycle lock and owner, configured resources and checkpoint identities, user-owned Nemotron and review systemd units/markers, current NVIDIA VRAM/process consumers, actual Studio/API bindings and shared inference lease/admission. Use the host's existing safe Nemovox setup. Existing external services and `nemovox-studio.service` on production `:8765` are READ-ONLY. No new weights, no external-unit stops, no implicit CPU downgrade, no arbitrary PID kill, no model/port substitutions. Use one **canonical shared** lifecycle lock across all host controllers of the managed units.
2. Exercise real uncached `device=auto` Riva translation with an authorized short controlled French recording/recording contract through exact-candidate CLI or isolated Studio (not production). Require worker-returned `device_used=cuda:0`, original MT retained, bounded GPU usage/readiness and no oversized concurrent ASR or text-review residency. Use a fresh isolated output/cache key; a cached pass is not a GPU pass. Never claim a mocked `subprocess.run` proved real CUDA. If enough safe host capacity exists, also run one **short foreign-language Studio → automatic Nemotron → Riva auto CUDA → English review → timed SRT/ASS/preview** and one direct CLI parity path. Reuse earlier proven host fixture/config/owned units and keep candidate Studio on a distinct loopback-only port; do not duplicate the already accepted SIGKILL stress scenario or churn provisioning.
3. Verify actual lease contention without interfering with any other consumer. A bounded controlled competitor holding the canonical lock must cause an uncached `auto` Riva worker request to block/admit only after release or return typed `resource_lease_timeout`, **without** starting an unleased CUDA subprocess. Prefer a deterministic process/trace probe using a test-controlled admission window, with host limits and no interference with real unrelated jobs. If unsafe to induce on the live host, retain the focused CPU proof and mark the real contention observation `UNVERIFIED`.
4. At final readback verify managed units/processes and GPU memory returned to the intended safe state, external GPU jobs untouched, production Studio still running its original source/config/port, and required translation/English-review semantics unchanged. F-02's earlier status-poll versus `_systemctl()` timeout diagnostic remains a **nonblocking recorded limitation**, not a new remediation campaign. If shared inference admission prevents running GPU inference while the interactive agent is alive, do not bypass it: finish CPU/host setup, exit and give **one minimal executable post-exit native job invocation** using the existing worker/service, with its bounded log/readback mechanism. Report `READY_FOR_FINAL_OPERATOR_RUN` rather than falsely claiming live PASS.

## Phase C: publish one complete sanitized evidence packet

Criterion → artifact → primary observation → identity / privacy:
- **Exact tested source/no drift:** `source/identity.json` and `source/remediation.diff`; exact remote SHA/tree, complete Central remediation diff, candidate import path, final Git status; public-safe.
- **Admission regression and failure unwind:** `validation/cpu-tests.txt` and `validation/admission-trace.md`; real test commands/exits, mocked events ordered acquire→subprocess→release, denied admission and explicit CPU; public-safe.
- **Existing deterministic coverage:** `validation/frontend.txt`; actual Vitest/types/build outputs and exits; public-safe.
- **Real CUDA auto and lease contention:** `validation/host-riva-auto.md`; actual worker JSON `device_used`, bounded lease/worker timings, VRAM/process observation and real/fixture classification; sanitized public.
- **Candidate Studio/CLI and service safety:** `validation/foreign-studio-cli.md` and `validation/service-invariance.md`; source-bound stage transitions, model/unit/port identity, output file type/preview, before/after unrelated processes, cleanup, and historical-scoped evidence reuse; sanitized public.
- **Independent review readiness:** `review/findings.md`, `review/acceptance-matrix.md`, `COMMAND_SHAPE_FRICTION.txt`; material residuals, missing observations, F-02 classification, per-gate PASS/PARTIAL/FAIL/BLOCKED; public-safe.

Publish under `staging/issue10/evidence/<UTC-run-id>/` with `manifest.json` containing an exhaustive allowlist, SHA-256, byte counts, criterion mapping and evidence class for every other member (no circular self hash). Freeze/check the packet; inspect every proposed public file for secrets, private media, raw transcripts, model paths with sensitive data, logs and tokens. Keep raw sensitive logs/recordings/backups private. If GitHub alone cannot carry material private evidence, create **only one** minimal secure private archive for that missing content, verify its members/digest, and request operator attachment explicitly.

**One publication commit maximum**: permit only `staging/issue10/evidence/<UTC-run-id>/...` paths on the *existing* issue branch, after fresh remote head compare, non-force push and remote SHA/readback. No source or test changes in that evidence commit. A race/mismatch is `STALE`, not a force/rebase invitation. Return exact `TESTED_SOURCE_SHA` and resulting `GITHUB_EVIDENCE_COMMIT`; certify that the intervening diff is evidence-only and the tested source tree is unchanged. When publication is blocked, return a complete private-safe evidence handback without inventing a commit.

## Stop and handback

Proceed autonomously across the **one** CPU→GPU→evidence campaign. Repair safe test scratch/setup and reversible local *candidate-only* host configuration inside the prior #10 authorization without re-requesting approval. Never widen to production, credential changes, external GPU/managed service takeover, schema/test weakening, or semantic repository rewrites. Stop for exact-head drift, a demonstrated blocking implementation defect, unsafe host shared lease/resource overlap, or a genuinely out-of-scope/destructive decision.

Return one structured handback:

```text
RESULT=PASS|PARTIAL|FAIL|BLOCKED|STALE|READY_FOR_FINAL_OPERATOR_RUN
REPO=fvanevski/nemovox
ISSUE=10
PR=11
TESTED_SOURCE_SHA=<40-hex>
BASE_SHA=<40-hex>
SOURCE_REMOTE_UPDATE=forbidden
UNCOMMITTED_SOURCE_CHANGES=<none|exact files>
CPU_COMPILE=<command, exit, count>
CPU_FOCUSED=<command, exit, count>
CPU_FULL=<command, exit, count>
FRONTEND=<command, exit, count/build>
RIVA_AUTO_ADMISSION=<PASS|FAIL|UNVERIFIED plus actual proof>
RIVA_LEASE_CONTENTION=<PASS|FAIL|UNVERIFIED plus simulated/real distinction>
RIVA_REAL_CUDA_AUTO=<PASS|FAIL|BLOCKED|UNVERIFIED; returned device>
STUDIO_CLI_FOREIGN_E2E=<PASS|FAIL|BLOCKED|UNVERIFIED; scope>
EXTERNAL_RESOURCES_UNTOUCHED=<PASS|FAIL|UNVERIFIED>
PRODUCTION_STUDIO_UNCHANGED=<PASS|FAIL|UNVERIFIED>
F02_DISPOSITION=<prior nonblocking/any newly material evidence>
EVIDENCE_PACKET_REQUIRED=yes
GITHUB_EVIDENCE_ROOT=<path|none>
GITHUB_EVIDENCE_COMMIT=<40-hex|none>
EVIDENCE_REQUIRED_ARTIFACTS=<list>
EVIDENCE_INCLUDED_ARTIFACTS=<list>
EVIDENCE_MISSING_ARTIFACTS=<list + reasons>
EVIDENCE_PRIVACY_CHECK=<PASS|FAIL>
EVIDENCE_INTEGRITY_CHECK=<PASS|FAIL>
EVIDENCE_ARCHIVE_REQUIRED=<yes|no>
PRIVATE_ARCHIVE_ADDS_REQUIRED_EVIDENCE=<yes|no>
PRIVATE_ARCHIVE_UPLOAD_REQUIRED=<yes|no>
CENTRAL_REVIEW_ATTACHMENT_REQUIRED=<yes|no>
BLOCKING_FINDINGS=<list|none>
UNVERIFIED_ITEMS=<list|none>
REQUIRED_OPERATOR_ACTION=<NONE|exact post-exit or attachment action>
```

Append the canonical `COMMAND_SHAPE_FRICTION` schema, **even if empty**:

```text
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events:
    - scenario=<scenario>
      requested_shape=<sanitized issued shape>
      effective_shape=<actual shape|same-as-requested|unknown>
      rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
      event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
      execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
      correction_or_annotation=<bounded fact>
      replacement_shape=<corrected shape|none>
      recipe_reference=<key|none>
      resource_reference=<source|none>
      contract_conflict=<yes|no|uncertain>
      durable_gap_candidate=<yes|no|uncertain>
  omitted_sensitive_values=<yes|no>
```
When none: `encountered=no`, `events=[]`, `omitted_sensitive_values=no`. Send results directly in conversation with exact GitHub evidence location. The host PASS is **not** merge authority: Central will independently inspect the final tested source, new evidence and stale/current review inputs, then decide the next GHDEV gate.
