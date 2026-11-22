# Issue #10: source-bound CPU qualification and automatic host model-resource integration

This is one consolidated Build-agent campaign for draft PR #11. Central authored the model lifecycle, CLI/Studio integration, tests, ownership-unit contract, and documentation. **Do not open another PR or implement a new resource architecture.** Source authority is the exact current PR head, which must be freshly retrieved at execution time. Deployment to the existing production Studio service is NOT authorized before merge.

## Identity and environment

```text
REPOSITORY=fvanevski/nemovox
ISSUE=10
PR=11
BRANCH=issue/10-auto-model-resources
BASE_MAIN_AT_HANDOFF=8fee1e177db9742dbdb38c4a22bdd663c20ac80f
STAGING_HANDOFF=staging/issue10/HOST_VERIFY.md
PRIMARY_AGENT=build
CPU_ONLY_PHASE=required
GPU_LIVE_PHASE=required_if_safe_and_CPU_passes
GPU_MODEL_SERVICES=only_exact_owned_Nemovox_units
PRODUCTION_STUDIO_SERVICE_MUTATION=forbidden
MAIN_BRANCH_WRITE=forbidden
NEW_PR=forbidden
PR_MERGE=forbidden
ISSUE_CLOSURE=forbidden
```

## F-01 credential parity remediation (requalification of changed source)

Previously qualified source `8d6f70f161c9100d4e64fca49da5139c4d109615`
passed 207 CPU tests, 11 frontend tests/build and isolated French Studio/CLI
on-demand ASR -> Riva -> English-review E2E. Its live probe initially received
HTTP 401 on ASR `/v1/models` when the host's `VLLM_API_KEY` was present but
the ASR resource lacked `api_key_env`. A host-only explicit setting resolved
that test. Central has now aligned ASR readiness with
`NemotronVLLMClient`'s `VLLM_API_KEY` fallback, without altering
explicit credential enforcement or text-model authorization.

The candidate **source SHA changes** (both credential parity and crash-recovery controller); rerun Phase A completely and confirm
new credential regressions: ASR default key, no key/EMPTY parity, explicit
ASR override precedence, missing explicit token fail-closed, and no leak to
the text review endpoint. Continue Phases B/C safely, but **do not**
unnecessarily repeat the entire long foreign-language E2E if exact fresh
evidence already proves every pipeline dependency other than the changed ASR
readiness header: execute a targeted authenticated ASR start/readiness/CTC
probe from both initially stopped unit and exact candidate CLI, **without**
`api_key_env` in the isolated ASR mapping while `VLLM_API_KEY` is
available to the candidate and the ASR unit. Verify successful credentialed
`/v1/models`, exact served-model ID, one short ASR request with CTC timings,
proper unit release, no unrelated service mutation, and healthy final state.
Source changes now include supervised SIGKILL recovery in Studio and lifecycle, so a **fresh full foreign-language Studio E2E is required**. Reuse the earlier evidence only as comparative baseline, never as new source-bound qualification. Capture both the original failure and
new successful automatic startup, distinguish host-only credentials from
tracked configuration, and preserve old PASS evidence as historical rather
than pretending it is source-bound to the new SHA.

## Phase A: exact-head inspect, CPU and frontend verification

1. Launch in a fresh OpenCode Build session from the existing Nemovox workspace, typically `/workspace/nvidia-transcribe`. Read repository `AGENTS.md`, `docs/model-resources/LIFECYCLE.md`, `docs/model-resources/CONFIGURATION.md`, `web/README.md` and this exact staging handoff. Read the live PR #11 head, `main` and issue #10. **Fetch the PR branch first** and verify the fetched object matches the remote exact full SHA before creating a clean detached test worktree. Never reset, switch, clean, stash, or fast-forward the operator's production checkout.
2. Read the complete candidate source and compare to base; verify all changed paths are related to #10. Test the actual imported package identity under the test interpreters using `PYTHONPATH=<isolated-worktree>/src`; do not accidentally test editable `main` instead. Preserve separate .venv (CLI/GPU) and .venv-studio (API) roles.
3. Run `python -m compileall -q src tests experiments`; `pytest -q tests/test_resource_lifecycle.py tests/test_translation_review.py tests/test_studio_api.py tests/test_model_resources.py tests/test_resource_integration.py`; `pytest -q -m 'not gpu'`; `cd web && npm test && npm run build` (`tsc --noEmit && vite build`). Report exact commands, exit codes, dependency identities and output. Existing front-end build artifacts must be from the tested worktree; `npm ci` only if necessary and never mutate a tracked lockfile or install in PyAnnote.
4. **Previously reported Phase A blocker:** at tested source `ea64f4e33f25e2a71413fcba1655e1b513290f92`, the full suite had 1 failed / 205 passed because the legacy `test_single_clip_asr_role_default_and_legacy_explicit_endpoint` mocked the inference client but not the newly introduced lifecycle boundary. Central corrected the **test isolation** rather than adding a production bypass: simulate a healthy exact-model *unowned* readiness endpoint (real lifecycle still active), verify no systemd mutation, and add a separate regression proving a missing unowned endpoint fails closed before inference. Confirm both tests run under the new exact source SHA. Do not silently skip this test or loosen production authorization. Prior evidence `staging/issue10/evidence/20261122T035837Z` is a FAIL record, **not qualification** for new source.
5. Review test outcomes rather than weakening assertions. Focus the shared lifecycle contract: fixed exact model/endpoint marker; no unit takeover; stage lock contention; signal cleanup; start failure; VRAM admission; missing endpoint; cached stages; English-source bypass; Riva then English review; failure preserves MT but does NOT mark a requested review complete; `/api/resource-status` must be read-only. If tests reveal a substantive source flaw, return one coherent failure packet for Central correction; do not redesign or patch production source. Pure environment/dependency setup and mechanical scratch/test-run work is authorized.

## Phase B: one-time host provisioning (only after Phase A passes)

1. Discover actual `systemd --user` service inventory, owning user, working directory, unit contents, current GPU memory, actual NInfer/SwiftU/text model and how it is launched, `model-resources.local.json` bindings, vLLM/Nemotron NVFP4 CTC checkpoint, `.venv` and `.venv-studio`, and health of `:8190`, `:8001`, `:8000`. Inspect without dumping secrets or user media; do not assume a listener on `:8000` serves Nemotron. Check all current user jobs/processes before touching resources.
2. Provision/update ONLY explicitly authorized `nemovox-asr.service` and `nemovox-text-review.service` user units, if they do not already exist with correct trusted ownership. Use repository `systemd-user/` templates as examples, **not** a guessed or placeholder executable. For text unit derive the real installed host launch command and served model from runtime evidence; if it cannot be established safely, stop and report `BLOCKED_BY_ENV`. Keep units un-enabled at login/boot and initially stopped. Ensure all four `NEMOVOX_MANAGED_RESOURCE`, `NEMOVOX_RESOURCE_NAME`, `NEMOVOX_RESOURCE_MODEL`, `NEMOVOX_RESOURCE_ENDPOINT` markers exactly match their selected model resources. Set `NEMOVOX_MODEL_RESOURCES`, `NEMOVOX_LIFECYCLE_CONFIG`, and lock location in an ignored host-only test env; set GPU admission thresholds from measured usage, not speculation. Confirm no conflicts with the existing managed-Qwen ASR handoff or shared GPU services. Record backup and reversible restoration for any unit/config change.
3. Avoid altering the existing production `nemovox-studio.service` or port 8765. For the candidate, launch a **separate loopback-only** Studio server (e.g. `127.0.0.1:8766`) using exact tested source via `PYTHONPATH`, an isolated job/upload root, and its built `web/dist`. Ensure the Studio subprocess CLI interpreter inherits the exact candidate's `PYTHONPATH`. Preserve existing host model checkpoints, media, ignored config, venvs and external services. No model weight downloads, implicit BF16 conversion, CPU fallback or port/interface exposure.

## Phase C: real on-demand acquisition and end-to-end proof

1. Begin with both **owned** Nemovox ASR and review units stopped, and the Riva worker idle. Run read-only `/api/resource-status` and verify it reports `startable` without starting models. Probe incorrect/unowned unit rejection using a harmless isolated mock, not by changing real external units. Record status/clock/GPU/process IDs before any stage.
2. Submit one bounded, genuine French (or other supported foreign-language) recording through candidate Studio to generate subtitles with `translator=riva` and `translation_review=balanced`. Choose a short, authorized controlled source clip (prefer existing FLEURS test clips or noncommercial synthetic French speech); do not publish copyrighted audio/frames or transcripts. The application must **automatically** start the exact configured Nemotron server, wait for readiness, transcribe with required CTC support, release it at the safe stage boundary, run the Riva CUDA worker, then automatically start/reuse the exact configured English-review server. Validate translated contract, unchanged ASR and MT copies, review metadata, speakers/timing, real SRT/ASS, preview/download, and no blanket retranslation. Capture stage-specific wall time, peak/free VRAM, model/port/PID/unit identities and start/stop transitions. Any manual `systemctl start`, manual launch of a server, or arbitrary GPU-service eviction **during the job** invalidates the autonomous-acquisition acceptance.
3. Run one separate direct CLI `process-media` + `generate-subtitles` case with both managed services initially stopped, and confirm equivalent resource acquisition. If safe, run a bounded cancellation during an owned model stage, verify TERM cleanup and exact service/resource state; record inability to prove SIGKILL recovery as `UNVERIFIED` rather than claiming PASS. Verify source-cache resume and offline rerender/English contract don't start unnecessary models. Confirm no unrelated GPU services were touched, and all smoke-created owned services are restored to intended post-test state.
4. Independently inspect the running candidate Studio UI in Chromium/Firedragon if available, with an actual preview and `/api/resource-status` showing stop/start changes; check that resource blocks are actionable rather than generic `exit 1`, and required review failures do not appear `complete`.
5. **If real GPU services cannot be safely provisioned or tests show a source bug, do not declare success.** Retain evidence and return a complete classified blocker; do not introduce an extra development PR or silently downgrade models. Do not promote candidate source into production 8765 under this premerge qualification campaign.

## Phase C required: supervised SIGKILL/orphan-recovery check

This candidate also changes the Studio supervisor's resource cleanup and the
shared lifecycle lock. Thus the earlier French E2E and CLI parity evidence
at source `8d6f70f161c9100d4e64fca49da5139c4d109615` is historical.
Run complete Phase A and fresh Phase C French Studio ASR -> Riva -> review
from exact current source, including the newly corrected default ASR
`VLLM_API_KEY` readiness path **without** `api_key_env` on the isolated
ASR binding. Keep the existing production Studio on port 8765 untouched.

When testing real systemd units, share **one canonical model-lifecycle lock**
between every Studio/CLI controller that can manage the same services. An
isolated candidate job root does NOT authorize a distinct model-stage lock
while any other controller uses a different path for those units. Separate
locks are allowed only for mock-only CPU tests that do not mutate systemd.
If live controllers cannot share the same lease path and ownership identity,
stop host validation and report a precise blocker rather than bypassing it.

Under safe isolated Studio ownership, run one **controlled SIGKILL of only
the candidate CLI subprocess** while a managed Nemotron or text unit is
active. Do not SIGKILL the systemd model unit, GPU driver, production Studio,
or unrelated services. The Studio controller must detect child exit,
automatically reconcile the idle exact-owned unit, return GPU memory to
baseline, and report the job as failed/interrupted rather than complete.
Also confirm a fresh candidate Studio startup reconciles a deliberately
abandoned *owned* model unit, while mismatched markers, external clients,
and an actively held stage lease prevent any stop. Inspect controller logs
for unrecovered errors or orphaned PIDs. If this cannot be tested safely,
report UNVERIFIED; do not declare full crash recovery PASS.

Standalone direct CLI SIGKILL has no independent always-running controller:
immediate reclaim is explicitly unverified/not guaranteed until a Studio
supervisor starts and reconciles. Verify the actual bounded behavior.
## Phase D: review-ready evidence publication

Publish a single sanitized packet under `staging/issue10/evidence/<UTC_RUN_ID>/` on the SAME existing issue branch, using a non-force exact-head precondition. A branch-head move invalidates prior source-bound qualification; rebind comparison to exact source SHA and ensure the evidence commit contains ONLY staging files. Preserve all earlier evidence.

Required inspectable text/JSON members: `manifest.json` with per-file SHA-256 and sizes; `source/identity.json`; `validation/cpu-tests.md`, `validation/web-tests.md`, `validation/live-stage-transitions.md`, `validation/gpu-usage.md`, `validation/studio-and-cli-e2e.md`, `validation/cancellation-and-recovery.md`, `validation/negative-ownership.md`; `promotion/unit-configuration.md`, `promotion/rollback.md`; `review/acceptance-matrix.md`, `review/findings.md`, `COMMAND_SHAPE_FRICTION.txt`. Include bounded synthetic-only screenshots if safe; keep raw host logs, secrets, authentic media, full private transcripts, model weights and private backups OFF GitHub. State precisely which files have supporting direct output vs agent claims; include exit codes and exact controller/process identity.

Report `PRIVATE_ARCHIVE_EXISTS`, `PRIVATE_ARCHIVE_ADDS_REQUIRED_EVIDENCE`, `PRIVATE_ARCHIVE_ADDITIONAL_CONTENT`, `PRIVATE_ARCHIVE_UPLOAD_REQUIRED`; no redundant upload if GitHub packet sufficient.

Return:

```text
RESULT=PASS|PARTIAL|FAIL|BLOCKED
PR=11
ISSUE=10
TESTED_SOURCE_SHA=<full SHA>
GITHUB_EVIDENCE_COMMIT=<full SHA or none>
GITHUB_EVIDENCE_ROOT=staging/issue10/evidence/<UTC_RUN_ID>
CPU_COMPILE=...
CPU_FOCUSED=...
CPU_FULL=...
FRONTEND_TEST_BUILD=...
HOST_MANAGED_UNITS_PREPARED=...
UNIT_IDENTITY_MARKERS=...
RESOURCE_READINESS_AND_STARTUP=...
ASR_CTC_E2E=...
RIVA_CUDA_E2E=...
ENGLISH_REVIEW_E2E=...
CLI_PARITY=...
GUI_LIVE_STATUS=...
GPU_CAPACITY_AND_HANDOFF=...
ENGLISH_AND_RERENDER_NO_MODEL_START=...
CANCELLATION_RECOVERY=...
UNOWNED_SERVICES_UNTOUCHED=...
SOURCE_EDITED=no
BLOCKING_FINDINGS=[...]
UNVERIFIED_ITEMS=[...]
PRIVATE_ARCHIVE_UPLOAD_REQUIRED=yes|no
COMMAND_SHAPE_FRICTION=<exact bounded events or none>
```

A host evidence PASS is not merge authority. Central must inspect exact-head candidate source and evidence, resolve substantive defects, and perform a late adversarial Codex pass only after otherwise clean review. Subsequent merge and production host promotion require separate authorization.
