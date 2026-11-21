# Nemovox issue #1: real loaded-ASR / Qwen available-mode coexistence qualification

## Purpose and authority

This is **one bounded host-capacity experiment**, not a feature implementation or generic GPU-reconfiguration campaign. Determine whether the existing RTX 5090 can simultaneously hold the real **Nemotron 3.5 Transcribe NVFP4 vLLM ASR** server and a real **Qwen3 ForcedAligner-0.6B CUDA BF16** worker while processing subtitle-relevant Mandarin speech in **`--qwen-gpu-mode available`**. Separately measure whether temporary removal of **the exact Docker container `tei-qwen3-embedding`** makes a blocked configuration viable.

The user **expressly authorizes temporarily stopping `tei-qwen3-embedding` for forced alignment when needed**. That authorization is narrowly limited to the identified container and includes **restarting and verifying it after the experiment if originally running**. Embeddings are unnecessary for this pipeline, but still verify container identity, preexisting health, restart supervisor, and absence of unexpected external consumers before stopping.

```text
REPO=fvanevski/nemovox
ISSUE=1 (OPEN)
MERGED_PR=4
SOURCE_BRANCH=main
SOURCE_SHA=7b4112cc31784953abbb041fd0b5d7997851841d
SOURCE_TREE=3d2cc429f9b49da3079df7d1fc178f6fd09d7987
LIVE_ROOT=/workspace/nvidia-transcribe
STUDIO_UNIT=nemovox-studio.service
ASR_ENDPOINT_EXPECTED=http://127.0.0.1:8190/v1
ASR_LAUNCHER=scripts/start_vllm_server.sh
ASR_CONFIG=configs/vllm_rtx5090.env (host-local, ignored by Git)
QWEN_WORKER_PYTHON=/workspace/nvidia-transcribe/outputs/modular-eval/.venv/bin/python
QWEN_CHECKPOINT=/workspace/models/Qwen3-ForcedAligner-0.6B
OPTIONAL_DOCKER_CONTAINER=tei-qwen3-embedding
COEXISTENCE_LANE_A=ASR+QWEN+TEI_CURRENT
COEXISTENCE_LANE_B=ASR+QWEN+TEI_STOPPED_IF_NEEDED
```

**Central-owned contract:** existing `_qwen_align_worker.py` checks actual `torch.cuda.mem_get_info(0)` **against 2,600,000,000 free bytes** before CUDA initialization and does not silently use CPU; historical idle-Qwen peak `1,900,171,776 B` is allocated PyTorch bytes, **not** full resident process footprint. The existing managed-ASR controller's `min_free_vram_mb=2600` is an independent setting; do not conflate its unit with Qwen's byte threshold. Historic loaded-ASR estimate ~3.4 GiB free (at GPU utilization 0.78) is only a hypothesis. **Coexistence PASS requires measured Qwen load/inference AND ASR functionality while both processes remain resident.** Health endpoints alone, free-VRAM arithmetic, or an idle-only replay are insufficient.

Current accepted host-evidence baseline: `evidence/issue-1-postmerge-host-20261121@10491fcd8ef9f42e861a3675810a4bf60712da92`. The prior handback verified merged-main deployment and CUDA replay with **ASR not running**. Do not reuse that result as coexistence evidence. Read current `main`, `AGENTS.md`, issue #1, `docs/alignment/ISSUE1_QWEN_DESIGN.md`, `scripts/start_vllm_server.sh`, the host's sanitized ASR config parameters, `gpu_handoff.py`, `qwen_aligner.py`, `recording.py`, `cli.py`, and previous replay fixture contracts before proceeding. Remote `main` must still equal pinned source; if it moved, STOP as `STALE`; no implicit checkout update.

## Local-agent setup and allowed mutation

**Use OpenCode primary Build** from an isolated already existing `/workspace/nemovox-issue1` checkout if available; otherwise let Build prepare an isolated clean checkout without changing the live `main` branch. There is **no code or PR development** here. Source and testing imports must resolve to the deployed exact `main` tree. Do not install/upgrade vLLM/torch/CUDA/qwen-asr, rebuild media/models, alter `configs/vllm_rtx5090.env`, or change `GPU_MEMORY_UTILIZATION` merely to pass.

Allowed host mutations, after safety checks:
- Launch the **existing configured local ASR launcher** as a **bounded, agent-owned test process** if no ASR is running and no other owner/session uses its port. Track launcher PID/PGID and its children, and tear down **only those test-owned processes** afterward. Do **not** create/reconfigure a systemd ASR unit or permanent service.
- Optionally `docker stop --time 30 <EXACT_VERIFIED_CONTAINER_ID>` on `tei-qwen3-embedding`, **only if running and the conditional lane requires its VRAM**, with guaranteed `docker start <SAME_CONTAINER_ID>` and observed health restoration before handback. The exact container ID and original state must be recorded privately first. Do not stop other containers, `local_embedding.service`, `nemovox-studio.service`, SwiftU, LiteLLM, OpenViking, unrelated vLLM instances or any active Studio job.
- Generate *small disposable* scratch WAV/JSON/logs and record real metrics; stop test-owned worker and ASR process groups and restore the TEI state. Scratch must not modify older evidence/recordings.
- If existing ASR is operator-owned/shared rather than test-owned, **do not kill/restart it**. A coexistence trial is permitted only when service ownership/usage and memory admission are safe; otherwise return `ASR_OWNERSHIP_BLOCKED`. No live managed-ASR release/restore, lease creation or handoff is authorized.

## Phase A: preflight, ownership, and baseline

1. Verify `git -C /workspace/nvidia-transcribe rev-parse HEAD` and `HEAD^{tree}`; read-only verify dirty status. If a working tree is drifted, use a clean isolated worktree with `PYTHONPATH=<candidate>/src` for the test, not source reset. Read live Studio jobs (all `outputs/studio/jobs/*/job.json`, not truncated API) and skip if an active job or competing ASR inference exists. Verify `nemovox-studio.service` and `local_embedding.service` active without restarting.
2. Capture `nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits`; bounded `nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits`, and `ss -ltnp` on actual ASR/TEI/Studio ports. Record GPU UUID so all readings are from the same physical card. Record other active GPU consumers and how/why they will be preserved. Metrics should distinguish `MiB`, `GiB`, and decimal bytes.
3. Inspect **only the exact Docker name** (e.g., `docker container inspect --type container tei-qwen3-embedding`), prove the returned `.Name == "/tei-qwen3-embedding"` and capture exact ID, `.State.Running`, `.State.Health.Status` if defined, `.HostConfig.RestartPolicy`, compose/supervisor labels, GPU device requests and ports. Capture only non-sensitive allowlisted properties; **never publish raw environment variables, image-registry auth, mounts or secrets**. Inspect current connections to the mapped TEI port. User says embeddings are unused by the alignment workflow; do not assume every unrelated client is quiescent. If lifecycle supervisor or active clients make stopping unreliable, mark `TEI_STOP_BLOCKED` and do not stop.
4. Inspect real `configs/vllm_rtx5090.env` **locally only**, sanitizing all credentials. Bind the actual `MODEL_PATH`, `SERVED_MODEL_NAME`, `HOST`, `PORT`, `GPU_MEMORY_UTILIZATION` and Qwen interpreter/checkpoint to host evidence. Require local offline checkpoint, `HOST=127.0.0.1`, no conflicting listener, vLLM NVFP4+FP8/CTC config, and a predictable free-workspace location. Missing/unsafe config is a stop, not authorization to edit or download.
5. If ASR already running, determine its **real owner/process** and whether it is safe to run a bounded Qwen test alongside it; do not start a second ASR. If ASR absent, start one agent-owned instance using **the existing `scripts/start_vllm_server.sh` and actual host config unchanged**. Put it in a dedicated, uniquely identifiable process group, record PID/PGID and child identities privately, install an agent/supervisor cleanup that survives ordinary tool/script failure (trap + explicit final reconciliation), and use a dedicated finite startup deadline; no detached orphan launcher. A failed start must not leave orphan processes or a stolen port. No model download.
6. Verify `127.0.0.1:<ACTUAL_ASR_PORT>/v1/models` reports the exact served model; check startup output for NVFP4/modelopt_mixed rather than BF16 expansion, CTC middleware, and actual VRAM after steady load. Run one **short, existing authorized Mandarin audio** Nemotron ASR smoke (bounded ~8-40s of audio, not a full episode), with CTC capture, and verify a valid response. Treat this as warmup before Qwen; capture VRAM **after** warmup, not only after first model load. Preserve source/transcript privately; publish only bounded status/length/CTC presence, not content. If no suitable clip is safely available, classify actual inference health as `ASR_INFERENCE_UNVERIFIED` rather than claim success.
7. If ASR cannot start with TEI active due to measured GPU memory reservation, and stopping TEI is safe, progress directly to conditional **Lane B** (stop TEI while original ASR is not running, then start test-owned ASR once more). Do not endlessly retry or alter the vLLM memory fraction. If ASR already running, skip this startup variation.

## Phase B: actual simultaneous residency and Qwen alignment

**Lane A (TEI original running state)** is attempted if the loaded, warmed ASR is healthy and available VRAM makes Qwen admission plausible. If TEI was originally stopped or absent, report `LANE_A_TEI_NOT_RUNNING` instead of inventing a TEI-on comparison.

1. Keep Nemotron ASR **running and GPU-resident** throughout Qwen initialization and alignment. Require free CUDA bytes at/above the worker admission threshold and enough observed headroom for the Qwen model; do not force a below-threshold attempt. If resource pressure is unsafe, label `LANE_A_RESOURCE_BLOCKED` and evaluate Lane B.
2. Run `QwenAlignerSession` / existing recorded-ASR replay harness with **real** `cuda:0`, BF16, offline Qwen checkpoint and two eligible Mandarin OOV spans in one session; two-span `钾` fixture from prior evidence is suitable. Use native `process_media` integration where feasible; do not mock Qwen or change source admission rules. An existing real end-to-end Nemotron-ASR-then-Qwen short job is preferable when its CTC returns an eligible span; if the live ASR emits complete CTC, use the controlled replay **while ASR is still resident** to demonstrate the Qwen path, explicitly labeling it `RECORDED_ASR_WITH_LIVE_ASR_RESIDENT`. No claim of native end-to-end OOV fallback without that observed sequence.
3. Take `nvidia-smi` samples **while both Nemotron and the Qwen worker are alive**, not only before/after. Record PIDs, per-process GPU memory if visible, actual GPU free/used/minimum, Qwen `torch.cuda.mem_get_info` at admission (where observable), `cuda_allocated_bytes`, peak allocated bytes, startup and align times, model_load_count, align_count, error statuses and invalid/zero-duration/misaligned units. Verify no silent CPU substitution. Read `/v1/models` during Qwen residence and run a short **post-Qwen** Nemotron inference smoke after worker teardown to catch evicted/broken ASR state; retain ASR process identity and CTC functionality.
4. Bound attempts: **one Qwen cold session per lane**, with up to two or three short alignments and finite timeouts. Failures/worker OOM are evidence, not permission to lower safety thresholds or repeatedly retry. If ASR exits, hangs, or OOMs, record `COEXISTENCE_FAILED`; do not continue stressing GPU. If agent started ASR, stop only that owned group and verify cleanup.

**Lane B (authorized TEI stopped)** only if Lane A is blocked/unsafe/unsuccessful due to memory, or ASR cannot load with TEI active. No need to disturb TEI when Lane A fully passes and the user did not request a comparative TEI-off optimization.

1. Verify that the exact TEI Docker container remains identical to preflight, was originally running, has no disqualifying active consumers, and can be restored under its present supervisor. Record baseline health and client activity, then `docker stop --time 30 <EXACT_CONTAINER_ID>`; check actually stopped, not auto-restarted. If originally stopped, do not start/stop it.
2. Measure VRAM before/after stopping TEI (MiB), capture exact freed amount and whether TEI itself occupied GPU resources. **Do not stop `local_embedding.service` or another container**. If stopping TEI freed no meaningful VRAM, record it and avoid claiming the workaround helps.
3. With Nemotron **still resident** if already running, rerun the same real Qwen admission and two-span BF16 benchmark. If ASR could not launch in Lane A, start the test-owned ASR from unchanged launcher with TEI stopped, confirm warmup and then run Qwen. Keep all source/audio/model choices identical to Lane A. Recheck `/v1/models` and short post-Qwen ASR inference.
4. **Restore TEI even on exception, timeout or cancellation**. First close/verify Qwen worker gone. If the ASR process was agent-created for this test, stop and verify it before restarting TEI (avoid restoration OOM). Then `docker start <SAME_CONTAINER_ID>` **only if originally running**, verify `Running=true` and its health/readiness at least as good as preflight (if initially unhealthy, do not assert healthy). If ASR was preexisting and must remain running, ensure restart-safe headroom was verified **before stopping TEI**, and do not stop that ASR as a workaround. If TEI restoration fails, return `TEI_RESTORE_FAILED` and explicit escalation; never report PASS.
5. Do not change Docker restart policy, image, compose project, volumes or systemd units; do not issue `docker compose down`, `rm`, `prune`, `pull`, or broad container lifecycle commands. If an external supervisor spontaneously restarts TEI during a lane, treat lane as contaminated, restore desired original state and stop testing.

## Phase C: cleanup, decision, and issue scope

- Always verify **both** GPU workers/ASR process identities and resident GPU memory after cleanup. An **agent-owned ASR** that was absent at baseline must be stopped and its port closed. A **preexisting ASR** must stay running, same owner/model identity, and pass health/inference checks. `tei-qwen3-embedding` must return to its original running/stopped state and original ID, health no worse than baseline. `nemovox-studio.service`, `local_embedding.service` and unrelated services must retain expected state. If any resource/process is unrecovered, mark non-PASS and escalate with specific safe recovery information, not a generalized cleanup.
- Distinguish **loaded-ASR Qwen coexistence** from whether a managed handoff is necessary. `COEXIST_PASS_TEI_ON` means both real models worked with TEI running; `COEXIST_PASS_TEI_OFF` means both worked **only** with TEI safely stopped and restored; `COEXIST_BLOCKED_GPU` means insufficient safe VRAM with the authorized option exhausted; `COEXIST_FAIL` means an actual CUDA/ASR correctness or health failure; `INCONCLUSIVE` means required tests could not be safely executed. The classification must rest on real observed results, not prior memory arithmetic.
- If coexistence succeeds, recommend **available-mode operational procedure** (TEI-off if necessary), without inventing a lease or asserting managed mode works. If it fails, identify measured deficit and the smallest necessary next integration decision; only then discuss a Nemovox-owned ASR lifecycle/lease. An individual one-off live test does not prove indefinite stability: explicitly state workload duration and known limitations.
- **Issue #1 stays OPEN** pending Central review of results and closure scope. Do not create another implementation PR, alter source, or change issue/GitHub review state.

## Evidence requirements

Publish **one operator-attachable sanitized archive** (target <20 MiB) and one evidence-only branch `evidence/issue-1-asr-qwen-coexistence-20261121` created **directly from pinned `main@7b4112cc31784953abbb041fd0b5d7997851841d`**, containing `staging/agent-evidence/issue-1-coexistence/` reports. No PR. Evidence branch must not contain production modifications or user media.

Required human-readable files: `HANDback.md`, `source-and-runtime.md`, `asr-and-gpu-samples.md`, `qwen-results.md`, `tei-lifecycle.md`, `coexistence-decision.md`, `manifest.json`, `SHA256SUMS.txt`, bounded structured sanitized logs and archive. Include one **timeline**: baseline -> ASR load/warmup -> Lane A admission/load/align or block -> conditional TEI stop -> Lane B load/align or block -> Qwen teardown -> test-owned ASR teardown -> TEI restoration -> final host/service/VRAM check. Include actual `nvidia-smi` times/values, model/port/launcher identities, Qwen errors, ASR pre/post response status, container exact-ID matching verdict, original/final state and health, and clean cleanup. Keep raw unit/container environments, secrets, audio, full transcripts and raw client requests PRIVATE. Calculate `SHA256SUMS` for all manifest members and archive, verify `gzip -t`, member inventory, privacy scan, and GitHub ref/file readback; if publication unavailable attach local archive path + verified sha and label publication unverified.

**Return a single consolidated handback** (no mid-campaign micro-handoffs):
```text
SOURCE_SHA=...
GPU_UUID=...
ASR_SOURCE=PREEXISTING|AGENT_OWNED|NONE
ASR_MODEL_ID=...
ASR_LOADED_AND_WARMED=PASS|BLOCKED|FAIL
ASR_CTC_PRE_QWEN=PASS|NOT_TESTED|FAIL
ASR_CTC_POST_QWEN=PASS|NOT_TESTED|FAIL
LANE_A_TEI_ON=COEXIST_PASS|RESOURCE_BLOCKED|FAIL|NOT_APPLICABLE|INCONCLUSIVE
LANE_B_TEI_OFF=COEXIST_PASS|RESOURCE_BLOCKED|FAIL|NOT_NEEDED|NOT_APPLICABLE|INCONCLUSIVE
QWEN_DEVICE_OBSERVED=...
QWEN_DTYPE_OBSERVED=...
QWEN_MODEL_LOADS=...
QWEN_ALIGN_CALLS=...
GPU_FREE_MIB_WITH_ASR=...
GPU_FREE_MIB_MIN_DURING_QWEN=...
GPU_FREE_MIB_GAIN_FROM_TEI_STOP=...|NOT_MEASURED
TEI_CONTAINER_ID_PREFIX=...
TEI_ORIGINAL_STATE=...
TEI_FINAL_STATE=...
TEI_RESTORE=PASS|NOT_NEEDED|FAIL
ASR_FINAL_STATE_AND_OWNER=...
STUDIO_AND_OTHER_SERVICES=UNCHANGED|FAIL
OVERALL_COEXISTENCE=COEXIST_PASS_TEI_ON|COEXIST_PASS_TEI_OFF|COEXIST_BLOCKED_GPU|COEXIST_FAIL|INCONCLUSIVE
LIVE_MANAGED_HANDOFF=NOT_TESTED
ISSUE1_RECOMMENDATION=CLOSE_WITH_DOCUMENTED_OPTIONAL_MODE_LIMITATION|OWNERSHIP_INTEGRATION_REQUIRED|REASSESS
EVIDENCE_COMMIT_SHA=...
ARCHIVE_PATH=...
ARCHIVE_SHA256=...
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events: [<actual command/tool mismatches, or empty>]
  omitted_sensitive_values=<yes|no>
```

**Stop/escalate conditions:** uncontrolled external GPU users, unavailable clean model identity, TEI lifecycle uncertain, OOM that threatens existing services, ASR health loss, non-owned process requiring stop, anonymous/background workers, active Studio job, ambiguous remote SHA, raw private data leaking to public evidence, or inability to restore TEI. Local agent may resolve routine bounded host diagnostics without redesigning the code or resource protocol.