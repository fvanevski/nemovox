# Nemovox issue #1: merged-main host promotion and controlled acceptance

## Authority / exact target

- Repository: `fvanevski/nemovox`, issue [#1](https://github.com/fvanevski/nemovox/issues/1), PR [#4](https://github.com/fvanevski/nemovox/pull/4) (squash-merged).
- **DEPLOY_SOURCE_REF**: `refs/heads/main`, exact commit `7b4112cc31784953abbb041fd0b5d7997851841d`; tree `3d2cc429f9b49da3079df7d1fc178f6fd09d7987`.
- **REVIEWED_PR_HEAD**: `1702608f82808fca30fb3eee1b62954230e6da32`, same tree. Squash parent: `242c418f55cd2afb796ee2c4fa742a10ff04f7a3`.
- **PRIOR_DEPLOYMENT_BASELINE** (from accepted Nov 21 report; host must independently reobserve): `/workspace/nvidia-transcribe` on `main@242c418...`; `nemovox-studio.service` enabled/active, binds `127.0.0.1:8765`, runs `.venv-studio/bin/python -m uvicorn nvidia_transcribe.studio_api:app --workers 1`, with `PYTHONPATH=/workspace/nvidia-transcribe/src`, CLI `.venv/bin/python`, persistent jobs/uploads under `outputs/studio`, `Linger=yes`.
- Studio systemd unit previously staged: `staging/agent-evidence/nemovox-studio-service/nemovox-studio.service` at `e739cd604088e65a1a2d674ee563fe3a83d5dc73`, installed SHA-256 previously reported `118c256bc94768b52c2f286c8b3cbdc9cf28423718413bdd6e9e3ba2107342d3`. Do not assume today's installed file matches without checking.
- **This document lives on a separate handoff branch. Never deploy or fast-forward production to that branch's commit. Deploy only the exact merged `main` object above.**
- Source and issue authority: re-read `main`, `AGENTS.md`, `docs/alignment/ISSUE1_QWEN_DESIGN.md`, `web/README.md`, CLI, recording, Studio API and current issue #1 before execution.
- Source was independently approved; PR #4 merged with identical source tree. There are **no configured GitHub Actions** on the merged commit. Prior host CUDA evidence binds to PR head, so run fresh post-merge host acceptance on merged `main`.
- **LOCAL_AGENT_PRIMARY=build** (OpenCode on Garuda). This is an authorized **host deployment**, not GitHub source development or a new PR. Build can perform safe preparation, read-only diagnostics, ordinary dependency compatibility checks, guarded source fast-forward, idle-only Studio service stop/start and validation. Local fixes may repair *only* host-bound environment/install problems inside the stated scope; material production code/ownership architecture changes return to Central.

## Scope and non-goals

Deliver in one campaign:
1. A controlled promotion of the exact merged `main` tree into the current live Studio checkout without losing user media/jobs/venvs/config.
2. Fresh process verification for the running Studio GUI and CLI import binding, including persistent jobs, frontend/API health and unchanged service startup config.
3. Focused exact-main CPU and **one bounded real CUDA BF16** Qwen alignment replay with recorded Mandarin ASR responses if eligible, covering default-off, `subtitle` admission, Qwen single-session reuse, OOV timing recovery and no leakage. Do not need a live Nemotron endpoint for recorded-ASR replay.
4. **Read-only determination** of whether the host now has a genuinely Nemovox-owned, cooperatively leased ASR vLLM unit suitable for a later reversible managed handoff. Report exactly what's missing, not a broad redesign. No live stop/unload/restore of an ASR model is authorized by this assessment.

Never change `/workspace/pyannote`, model weights, vLLM/SwiftU/embedding services, their ports/launchers/units, `nemovox-studio.service` contents, GitHub source/issue/PR states, Studio job definitions or existing subtitle/media artifacts. No `git reset --hard`, `git clean`, stash, force operations, unbounded filesystem sweep, full TV episode benchmark, new weights/dependencies download, independent GUI instance on the live job directory, or synthetic ownership markers/lease/service for the purpose of a passing test.

## Phase A. Inspect / admit exact source and live process

### A1. Source and filesystem identity

- Use `/workspace/nvidia-transcribe` as **LIVE_ROOT**. Verify `git remote get-url origin` matches `fvanevski/nemovox`, branch `main`, current SHA, parent/ancestor relationship, expected old tree, index/worktree status (tracked and potentially conflicting untracked files), and subdirectories/venvs/output locations. Never overwrite unexplained tracked/untracked drift; if dirty, handle clean detached qualification only, report `LIVE_PROMOTION_BLOCKED` and do not switch live root or reconfigure the unit.
- **Fetch before proving newly remote SHA locally**: execute `git -C /workspace/nvidia-transcribe fetch --no-tags origin refs/heads/main`, then independently `git -C /workspace/nvidia-transcribe rev-parse FETCH_HEAD` and `git -C /workspace/nvidia-transcribe cat-file -t 7b4112cc31784953abbb041fd0b5d7997851841d`; `git -C /workspace/nvidia-transcribe rev-parse 7b4112cc31784953abbb041fd0b5d7997851841d^{tree}`. Must prove commit `7b4112...` and tree `3d2cc4...`. Re-fetch immediately before promotion and fail `STALE` if `main` moved.
- Record **PRIOR_LIVE_COMMIT_IDENTITY**, branch, tree, worktree status, service unit SHA-256, unit MainPID and service environment identity, prior job-root directory, job count, enabled/active/linger state, health `GET /api/health` and GUI `GET /`. Capture unit/env without publishing secrets. Keep private rollback snapshot of service file/config and old commit/ref/cleanliness; do not copy raw media, jobs or venv to evidence.
- Confirm `/workspace/nvidia-transcribe/web/dist/index.html` exists and backend has valid static assets. PR #4 changed **no `web/` files**, so there is no need for a frontend `npm ci`/rebuild if the accepted bundle is present and healthy. Rebuild only if missing/corrupt, with existing lockfile and ordinary frontend tests; do not alter lockfile.
- Preflight Studio job activity **from all** `outputs/studio/jobs/*/job.json` (not only `GET /api/jobs`, which returns at most 50). States `queued` or `running` block stopping/restarting/fast-forwarding live source; wait for natural completion or return `HOST_BUSY_DEFERRED`. Do not cancel or interrupt an active job to finish the deployment. If job metadata is unreadable, fail closed and report exact bounded evidence.
- Confirm `systemctl --user show nemovox-studio.service -p MainPID,FragmentPath,ExecStart,Environment`, `systemctl --user is-enabled/is-active`, `loginctl show-user "$USER" -p Linger`, `ss -ltnp` on `127.0.0.1:8765`. Do not copy raw environment values to the public packet. Neither a service enabled unit nor a source checkout proves that its live process has loaded new code.

### A2. Isolated test checkout / initial validations before service disruption

- Create/reuse one **collision-safe, disposable** clean detached worktree for exact merged main (e.g. `mktemp -d -p /workspace nemovox-main-accept-XXXXXXXX`, then `git -C /workspace/nvidia-transcribe worktree add --detach <empty-worktree-path> 7b4112cc31784953abbb041fd0b5d7997851841d`; if `mktemp` creates the target, remove only this freshly created empty directory before `worktree add`, not user contents). Prove worktree commit/tree and that imported `nvidia_transcribe` resolves to this candidate `src`, not the prior live site package.
- With working directory at this checkout and `PYTHONPATH=<acceptance-worktree>/src`, run `/workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe`, focused `tests/test_qwen_session_and_policy.py tests/test_qwen_fallback.py tests/test_studio_api.py`, then full `python -m pytest -q -m "not gpu"` in **both** `.venv` and `.venv-studio`. Report actual counts/exits; PR-head 120+120 is historical expectation, not a substitute. Do not change code/tests/assertions to get green.
- Verify merged CLI interface with `python -m nvidia_transcribe.cli process-media --help`, `generate-subtitles --help`, and `recover-qwen-asr --help`. Verify default `alignment_fallback=off`, `qwen_policy=subtitle`, `qwen_device=cuda:0` (only when Qwen explicitly selected), `qwen_gpu_mode=available`, and that `managed` remains separately opt-in.
- If an isolated test package/library problem exists, distinguish unavailable optional GPU dependencies from Python source test failure. No reinstall of vLLM/NeMo/PyTorch or upgrade of GPU stack in this campaign.

### A3. Bounded GPU Qwen acceptance, pre- or post-promotion as practical

- Read `nvidia-smi`, `ss` and existing GPU consumers; preserve desktop, embeddings and unrelated inference. Use existing offline Qwen3-ForcedAligner checkpoint and Qwen-isolated worker interpreter at `outputs/modular-eval/.venv/bin/python`. If insufficient VRAM or worker/model absent, record `GPU_BLOCKED` with source evidence, not CPU substitution and not GPU service shutdown.
- Prefer the existing **controlled recorded-Nemotron-response replay harness/fixtures** from previous issue-1 evaluation in the isolated workspace (previous evidence branch `evidence/issue-1-codex-20261121@0160067f55592496e1b4e1dcbdf91108352b5def` is a baseline, not freshly executed proof). Record fixture/checkpoint identity, real `process_media` execution with the merged code, one `off` control and one `qwen --qwen-policy subtitle --qwen-gpu-mode available --qwen-device cuda:0` multiple-span Mandarin OOV test. A second `fine` replay is optional if it adds diagnostic value, not necessary for deployment acceptance.
- Verify at least two eligible spans in **one** worker process/model load where fixture permits, source offsets and exact `钾` OOV timings, source text and native-timed neighbor preservation; admitted/skipped reasons, total/timed/untimed units, invalid/zero-duration rejects, `native-alignment.checkpoint.json`, elapsed load/inference/wall, `device_used=cuda:0`, `observed_dtype=torch.bfloat16`, `model_load_count=1`, `align_count>=2`, peak memory and pre/post `nvidia-smi`. Confirm owned Qwen worker terminated/no residual CUDA allocation. Recorded ASR replay is **not** proof of live Nemotron throughput.
- Avoid introducing a new production ASR service solely for the test. Do not run a full episode. Reuse bounded available samples only; no media or private transcripts in public evidence.

## Phase B. Promote exact source to live installation and start fresh Studio process

Only after Phase A source/test preflight PASS and fresh all-jobs idle checks:

1. Re-fetch `refs/heads/main` from origin and prove `FETCH_HEAD == 7b4112...`. Confirm LIVE_ROOT is still the originally inspected clean `main` checkout and ancestor; verify no source, service, active-job or occupied-port drift. Stop if an unrelated actor changed state; do not rebase, reset, stash, clean or force.
2. Capture old active/enabled status, PID, unit hash, job state and rollback point, then `systemctl --user stop nemovox-studio.service` **once** while idle; prove it became inactive and old process group is gone, with no active CLI descendants. Keep other services untouched. If stop fails or any old child remains, STOP and restore/check host state before proceeding.
3. While Studio is stopped, `git -C /workspace/nvidia-transcribe merge --ff-only FETCH_HEAD` (after the single exact main fetch; no intermediate fetch), or equivalent safe non-force fast-forward to the **exact** expected main commit. Re-read LIVE_ROOT HEAD/tree, clean tracked status and working source hashes. Existing `.venv`, `.venv-studio`, `outputs`, `web/dist`, optional `studio.env` and user service unit must remain untouched. If the live checkout is not a clean ancestor, do not modify it: report `LIVE_PROMOTION_BLOCKED`.
4. Verify `.venv-studio/bin/python` and the configured `NEMOVOX_CLI_PYTHON` import `nvidia_transcribe` from `/workspace/nvidia-transcribe/src` (without creating a second Studio runner). Check module/CLI flags and Qwen worker script+checkpoint path now in the live working tree. If editable bindings are broken, only narrow environment repair to bind the already accepted checkout is allowed, with rollback and no dependency upgrade; otherwise STOP.
5. Start the unchanged unit: `systemctl --user start nemovox-studio.service`, then verify `is-enabled=enabled`, `is-active=active`, new `MainPID` and Python argv/PYTHONPATH, exclusive listener `127.0.0.1:8765` (not `0.0.0.0`), `GET /api/health` HTTP 200/status ok, `GET /`/actual hashed JS+CSS assets HTTP 200, and `GET /api/jobs`. Check a previously completed job ID still returns its prior state/artifact via `GET /api/jobs/{id}` (prior synthetic ID `9be92af3ba144e6eb984ee43af0807de` if present; never invent an ID). Preserve counts and job records. If no existing synthetic job is accessible, one controlled CPU-only contract-import job is permitted through the normal API, but **no GPU/vLLM service start**.
6. Record new PID/command environment source, unit SHA and journal relevant to stop/start, no stale CLI children, unchanged unrelated services, and no inadvertent injected `ASR_BASE_URL`/`SWIFTU_BASE_URL` changes. Full reboot is **not required**; `Linger=yes` and enablement are configuration evidence, not proof of startup across a real reboot.
7. **Recovery on failure:** if source fast-forward or new process startup fails, do not claim deployed. With Studio stopped and no jobs, if original live checkout was clean and only the agent's known merge changed it, safely return to the pinned old **source tree** using `git -C /workspace/nvidia-transcribe switch --detach <PRIOR_LIVE_COMMIT_IDENTITY>` (not a destructive reset), then start and verify previous healthy Studio; report detached-head rollback as such, do not conceal it. If the old checkout contains newly discovered edits, never overwrite; preserve state, stop and return explicit escalation/recovery commands. Do not overwrite existing user artifacts.

## Phase C. Read-only managed-ASR feasibility: do not induce a takeover

On the **currently observed host**, without stopping/restarting any model server, determine whether the issue's explicit `--qwen-gpu-mode managed` can be safely exercised in a future controlled run:

- Enumerate plausible local ASR instances and owners, **not only** `127.0.0.1:8190`: `systemctl --user list-units --type=service --all`, `systemctl --user list-unit-files`, bounded `systemctl --user show/cat <actual-specific-unit>`, `ss -ltnp`, `pgrep -af 'vllm|nemotron|nemo_speechlm'`, `nvidia-smi`, read-only ASR `/v1/models` for actual known listener when authorized. Do not infer an instance from a stale pathname or earlier memory.
- Identify **exact, existing** `nemovox-*.service` (except Studio), its actual MainPID/InvocationID/ExecStart/FragmentPath, model and port, and exact unique systemd environment assignments `NEMOVOX_MANAGED_ASR=1` and `NEMOVOX_ASR_LEASE_PATH=<existing-file>`. Use the merged `gpu_handoff._has_exact_cooperative_markers` parser when suitable. Record only redacted marker **presence/identity verdict**, never secret environment values.
- Check that lease file **already exists**, is private/user-owned and has a real owner/consumer protocol; trace known ASR callers' lease participation and whether the service is exclusive to this workflow. A lease file name or marker alone is not evidence of participating consumers. Report suspected shared/nonparticipating clients and non-Nemovox units as **not eligible**. Establish whether current free VRAM actually necessitates a takeover in the typical loaded-ASR scenario; current idle-only free VRAM cannot prove it.
- Check existing configured command-line endpoints and recorded prior missing `8190` ASR only as *historical context*, not current assertion. If no owned unit/lease, return `MANAGED_GPU_BLOCKED: no cooperatively owned ASR unit/lease` plus **minimal concrete requirements** and whether an existing launcher/service can be adapted through a separate Central decision. Do not create, edit or activate a `nemovox-asr.service`, lease file, vLLM unit, job controller, or modify a live ASR/embedding/SwiftU instance. If preconditions actually hold, mark `MANAGED_ASR_ELIGIBLE_FOR_LIVE_SMOKE` but **LIVE_RELEASE_RESTORE_NOT_TESTED**, with exact acceptable unit/model/lease facts and a separately scoped reversible test recommendation. This is a read-only feasibility assessment, not permission to interrupt an active model service.
- Recheck that the two unaffected services `nemovox-studio.service` and `local_embedding.service` remain healthy after assessment.

## Evidence / outputs / one handback

Create a sanitized, self-contained operator-attachable `.tar.gz` (target <= 20 MiB, exclude user media, weights, transcript bodies, complete job metadata/paths, secrets, unredacted unit env, large logs, private backups) containing:
- `HANDback.md`, `manifest.json`, `SHA256SUMS.txt`, `source-provenance.md`, `host-promotion.md`, `runtime-health.md`, `cpu-tests.md`, `gpu-alignment.md`, `managed-asr-feasibility.md`, `logs/*` bounded safe excerpts and a criterion->source->test->result matrix;
- exact merged `main` commit/tree, prior/final live checkout head/tree, process PID before/after, unit hash before/after, enabled/active/linger, loopback health/asset checks and job persistence, CPU tests, CUDA worker metrics if executed, no-CPU-substitution, managed feasibility with explicit evidence/source and unverified portions, original and final unrelated-service identity;
- private rollback location described without contents, controls that stopped/deployed source only after idle, and evidence of cleanup/absence of orphan workers;
- `COMMAND_SHAPE_FRICTION` ledger (even if none).

Publish one **evidence-only branch** `evidence/issue-1-postmerge-host-20261121` based directly on **exact merged-main commit** `7b4112cc31784953abbb041fd0b5d7997851841d` (not this handoff branch, not an unverified local commit). Place browsable reports, manifest, SHA256SUMS, archive and bounded logs under `staging/agent-evidence/issue-1-postmerge/`; one consolidated evidence commit where practical; no PR, issue mutation or automatic branch deletion. Use `git add -f` only for the exact allowlisted evidence paths Git would otherwise ignore. Verify archive SHA, gzip integrity, manifest-member hashes, privacy scan and GitHub exact ref/files readback. If publication is unavailable, preserve one local attachable archive and return path+hash, not fabricated GitHub provenance.

Return a *single consolidated* handback with independent dispositions:

```text
POSTMERGE_SOURCE_IDENTITY=PASS|FAIL|STALE
HOST_PROMOTION=PASS|BLOCKED|FAIL|ROLLED_BACK
STUDIO_FRESH_PROCESS=PASS|BLOCKED|FAIL
CPU_REGRESSIONS=PASS|FAIL|BLOCKED
GPU_QWEN_AVAILABLE=PASS|GPU_BLOCKED|FAIL
MANAGED_ASR_FEASIBILITY=MANAGED_GPU_BLOCKED|MANAGED_ASR_ELIGIBLE_FOR_LIVE_SMOKE|UNKNOWN
LIVE_ASR_RELEASE_RESTORE=NOT_TESTED
ISSUE1_CLOSURE_RECOMMENDATION=READY_FOR_CENTRAL_ASSESSMENT|REQUIRES_OWNERSHIP_INTEGRATION|BLOCKED
GITHUB_MAIN_SHA=...
MERGED_MAIN_TREE=...
PRIOR_LIVE_SHA=...
FINAL_LIVE_SHA=...
SERVICE_PID_BEFORE=...
SERVICE_PID_AFTER=...
ARCHIVE_PATH=...
ARCHIVE_SHA256=...
EVIDENCE_BRANCH_SHA=...
UNRESOLVED_RISKS=...
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events: [<actual bounded records or empty>]
  omitted_sensitive_values=<yes|no>
```

**Stop/escalation boundaries:** A dirty/drifted live checkout, active Studio job, lost service-owned child, persistent startup failure, test failure requiring semantic changes, contaminated evidence, unexplained ASR ownership, or an irreversible live service effect must be surfaced promptly and **not** worked around by bypassing a guard. Ordinary bounded host repair may be completed autonomously where reversible. Do not mark issue #1 closed; Central adjudicates tracker updates/closure and any later managed-ASR integration from evidence.