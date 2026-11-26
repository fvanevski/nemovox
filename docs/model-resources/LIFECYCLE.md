# Automatic model-resource acquisition

## Contract and ownership

The **CLI is the lifecycle authority** for both direct commands and GUI-submitted
jobs; Studio starts the same CLI as a child. The resource registry selects the
ASR/review model identity and endpoint. The separate lifecycle service mapping
authorizes **only exact Nemovox-owned, systemd --user** services to start/stop.
An already healthy endpoint without exact cooperative ownership may be *used*
but is never stopped, evicted or reconfigured.

**No per-job operator startup is required once the appropriate owned units are
provisioned.** This replaces prior manual-startup guidance for owned resources.
Unrelated inference servers, ASR sessions, non-Nemovox GPU users, checkpoint
directories and other projects remain outside lifecycle authority.

Default managed service identities:
- `asr_local`: `nemovox-asr.service`, bound to
  `http://127.0.0.1:8190/v1` and the configured Nemotron served model.
- `text_local`: `nemovox-text-review.service`, bound to
  `http://127.0.0.1:8001/v1` and served-model ID `chat`.
- Riva translation remains an ephemeral `cuda:0` subprocess started only
  when uncached Riva translation is required.

Copy `configs/model-lifecycle.example.json` to the ignored local path
`configs/model-lifecycle.local.json`, edit names and **measured VRAM
admission budgets**, and export `NEMOVOX_LIFECYCLE_CONFIG` in the Studio
service environment and the direct CLI environment. A non-default resource
identity requires an exact corresponding `services` entry. Without that,
an unavailable endpoint fails with `managed_unit_not_configured`.
If the environment variable is absent, built-in names/units apply.

Each managed user unit **must** have all four exact assignments, as read by
`systemctl --user show <unit> --property=Environment`:

```text
NEMOVOX_MANAGED_RESOURCE=1
NEMOVOX_RESOURCE_NAME=<named resource binding>
NEMOVOX_RESOURCE_MODEL=<exact served-model ID>
NEMOVOX_RESOURCE_ENDPOINT=<exact /v1 endpoint>
```

Unit names must match `nemovox-*.service`. Unmarked/unknown/mismatched services
must never be started or stopped even if a port matches. A healthy **unowned**
endpoint can be borrowed for a stage but not stopped to make room later.

Two templates are included under `systemd-user/`:
- `nemovox-asr.service.example` invokes the repository's NVFP4/CTC launcher
  with the host's ignored `configs/vllm_rtx5090.env`.
- `nemovox-text-review.service.example` is **deliberately non-runnable**
  until host integration discovers the installed text model, its actual
  executable, quantization, served ID and compatible startup command.
  Never guess a server launcher, silently use the vLLM agent on a different
  port, or change quantization merely to make a smoke test pass.

The local Build integrator should discover and provision the two correct
units once, perform `systemctl --user daemon-reload`, and leave them
**stopped**. Subsequent task execution starts/health-checks/stops owned units
without manual user intervention. Record the final exact unit SHA/identities,
startup and GPU telemetry, and rollback plan as host-bound evidence. For a
service currently owned by another controller, do not overwrite its unit.

## Stage coordination

The manager serializes local model stages through a file lock
(`~/.local/state/nemovox/model-lifecycle.lock` by default, override via
`NEMOVOX_LIFECYCLE_LOCK`). It opens resources at the stage that requires
them: ASR for transcription; Riva's isolated worker for uncached MT; text model
**lazily** on an uncached recording-context, source-adjudication or English-review
request (or an explicitly selected legacy refinement/translation/condensation
request). The independent Whisper CUDA worker, when configured and admitted,
uses a separate shared-lease scope between served model phases. Unit readiness requires `/v1/models` advertising
the exact selected ID; port openness alone is insufficient. For Nemotron ASR,
health readiness and inference use the **same** credentials: if the ASR
resource explicitly declares `api_key_env`, that environment variable is
required; otherwise both inherit the legacy `VLLM_API_KEY` value (or the
client's `EMPTY` placeholder when unset). This fallback applies **only**
to the `nemotron_ctc` adapter; it is never sent to the English-review
endpoint. When the ASR service inherits `VLLM_API_KEY` from the systemd user
manager, ensure the CLI/Studio child can access the same secret via a
protected host environment or explicitly set `api_key_env` on the ASR
resource. Do not store secret values in repository configuration or evidence.

On stage exit the manager stops only an owned, identity-matching unit whose
policy says `stop_after_stage=true`, after checking for other active TCP clients.
For missing GPU capacity or an unowned/busy service it reports a typed
`ResourceLifecycleError`; it must not attempt CPU fallback or evict another
application. The Riva CUDA worker needs a bounded free-VRAM admission check.
When no model call is needed (English source contract, cached text pass, or
offline rerender), no service is started.

The normal foreign-language workflow, stage acquisition/release and cache-only
continuation/rerender were exercised on the deployed PR #21 source. See the
[acceptance record](https://github.com/fvanevski/nemovox/issues/19#issuecomment-7178926236).
That qualification is not proof of every cancellation, crash, contention or
capacity scenario on every host; retain the targeted controls below for changes
affecting those boundaries.

## Status and failure interpretation

- `GET /api/resource-status` is a read-only snapshot with
  `ready_managed`, `startable`, `ready_external`,
  `managed_unit_not_configured`, `managed_model_mismatch`,
  `manager_unavailable` and related states. It does **not** load a model.
- A job whose required model cannot be acquired fails visibly with
  `RESOURCE_BLOCKED` and a machine-readable reason in the CLI log.
  The GUI surfaces the blocked resource reason rather than inventing
  a transcription failure.
- If an enabled English post-review encounters request errors, the original
  Riva machine translations remain in `translated.contract.json`, but the
  required review stage does **not** count as a completed reviewed job.

## Validation and deployment gate

For changes affecting the live lifecycle or public inference route, qualify the
affected behavior, not just static tests or synthetic English contract jobs.
With owned managed ASR and review units initially **stopped**, a bounded real
foreign-language Studio job exercises ASR, source quality, selective evidence,
Riva, English review and SRT/ASS; a fresh CLI continuation checks reuse. Record
exact model/CTC and process identity, stage transitions, memory, output
semantics and unrelated service preservation. Exercise cancellation/recovery
and contention controls when those boundaries change, and revalidate affected
behavior after authorized promotion. Publish sanitized receipts and keep
private artifacts local. Documentation-only changes do not require another
inference campaign when implementation and configuration identity are proven.

## Recovery after an unclean model-worker exit

A Python SIGTERM cancellation unwinds managed service cleanup. SIGKILL
cannot execute Python finally blocks and previously could strand an ASR
or English-review systemd service, consuming GPU memory until manual repair.

At startup, and after every supervised CLI subprocess exits, the owning
Studio controller now invokes ModelLifecycle.reconcile_idle_owned().
The reconciler takes the shared stage lock non-blockingly; a concurrently
running CLI keeps its lease and cannot be preempted. The reconciler checks
the currently configured resource name, model, endpoint and exact Nemovox
ownership markers, rechecks systemd MainPID/InvocationID, and refuses to stop
a unit with active external clients. Only owned units configured with
`stop_after_stage=true` are eligible for automatic idle cleanup.

`GET /api/resource-status` remains read-only and does not perform recovery.
This protects passive GUI queries from causing resource mutations.

A direct standalone CLI that is itself SIGKILLed without a running Studio
controller cannot instantly reclaim its service: no external watchdog is
installed. The next admitted Studio startup/reconciliation can recover such
an abandoned idle unit. Do not claim immediate standalone SIGKILL recovery.

Verify the supervised SIGKILL path on the host with a short synthetic
model job, and verify recovery is skipped while another CLI owns the lease,
while external clients are connected or when unit markers change.

### Independent Whisper CUDA phase

The explicitly configured `whisper_ct2` challenger acquires the same CUDA worker lease as Riva/Qwen, only after the managed acoustic endpoint scope exits. Owned ASR and text units must retain `stop_after_stage: true` for sequential resource release. The worker uses one subprocess for the bounded batch; its exit releases its CUDA context before the later scene/text phase. Capacity denial remains explicit rather than falling back to CPU or stopping an external process. The initial admission threshold is the existing 8500 MiB worker minimum, subject to the measured capacity of the actual host.
