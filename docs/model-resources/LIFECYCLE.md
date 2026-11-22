# Automatic model-resource acquisition (Issue #10)

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
- \`asr_local\`: \`nemovox-asr.service\`, bound to
  \`http://127.0.0.1:8190/v1\` and the configured Nemotron served model.
- \`text_local\`: \`nemovox-text-review.service\`, bound to
  \`http://127.0.0.1:8001/v1\` and served-model ID \`chat\`.
- Riva translation remains an ephemeral \`cuda:0\` subprocess started only
  when uncached Riva translation is required.

Copy \`configs/model-lifecycle.example.json\` to the ignored local path
\`configs/model-lifecycle.local.json\`, edit names and **measured VRAM
admission budgets**, and export \`NEMOVOX_LIFECYCLE_CONFIG\` in the Studio
service environment and the direct CLI environment. A non-default resource
identity requires an exact corresponding \`services\` entry. Without that,
an unavailable endpoint fails with \`managed_unit_not_configured\`.
If the environment variable is absent, built-in names/units apply.

Each managed user unit **must** have all four exact assignments, as read by
\`systemctl --user show <unit> --property=Environment\`:

\`\`\`text
NEMOVOX_MANAGED_RESOURCE=1
NEMOVOX_RESOURCE_NAME=<named resource binding>
NEMOVOX_RESOURCE_MODEL=<exact served-model ID>
NEMOVOX_RESOURCE_ENDPOINT=<exact /v1 endpoint>
\`\`\`

Unit names must match \`nemovox-*.service\`. Unmarked/unknown/mismatched services
must never be started or stopped even if a port matches. A healthy **unowned**
endpoint can be borrowed for a stage but not stopped to make room later.

Two templates are included under \`systemd-user/\`:
- \`nemovox-asr.service.example\` invokes the repository's NVFP4/CTC launcher
  with the host's ignored \`configs/vllm_rtx5090.env\`.
- \`nemovox-text-review.service.example\` is **deliberately non-runnable**
  until host integration discovers the installed text model, its actual
  executable, quantization, served ID and compatible startup command.
  Never guess a server launcher, silently use the vLLM agent on a different
  port, or change quantization merely to make a smoke test pass.

The local Build integrator should discover and provision the two correct
units once, perform \`systemctl --user daemon-reload\`, and leave them
**stopped**. Subsequent task execution starts/health-checks/stops owned units
without manual user intervention. Record the final exact unit SHA/identities,
startup and GPU telemetry, and rollback plan as host-bound evidence. For a
service currently owned by another controller, do not overwrite its unit.

## Stage coordination

The manager serializes local model stages through a file lock
(\`~/.local/state/nemovox/model-lifecycle.lock\` by default, override via
\`NEMOVOX_LIFECYCLE_LOCK\`). It opens resources at the stage that requires
them: ASR for transcription; Riva's isolated worker for uncached MT; text model
**lazily** on the first uncached refinement/legacy translation/review/optional
condensation request. Unit readiness requires \`/v1/models\` advertising
the exact selected ID; port openness alone is insufficient.

On stage exit the manager stops only an owned, identity-matching unit whose
policy says \`stop_after_stage=true\`, after checking for other active TCP clients.
For missing GPU capacity or an unowned/busy service it reports a typed
\`ResourceLifecycleError\`; it must not attempt CPU fallback or evict another
application. The Riva CUDA worker needs a bounded free-VRAM admission check.
When no model call is needed (English source contract, cached text pass, or
offline rerender), no service is started.

This is a **candidate implementation**, not proof of live completeness.
The scope still needs deterministic and real-host verification of cancellation,
SIGTERM/SIGKILL recovery, service contention, exact ASR CTC middleware,
measured VRAM, repeated job reuse, and real foreign-language translation.

## Status and failure interpretation

- \`GET /api/resource-status\` is a read-only snapshot with
  \`ready_managed\`, \`startable\`, \`ready_external\`,
  \`managed_unit_not_configured\`, \`managed_model_mismatch\`,
  \`manager_unavailable\` and related states. It does **not** load a model.
- A job whose required model cannot be acquired fails visibly with
  \`RESOURCE_BLOCKED\` and a machine-readable reason in the CLI log.
  The GUI surfaces the blocked resource reason rather than inventing
  a transcription failure.
- If an enabled English post-review encounters request errors, the original
  Riva machine translations remain in \`translated.contract.json\`, but the
  required review stage does **not** count as a completed reviewed job.

## Validation and deployment gate

The PR must not be merged or issue #10 closed based only on static tests or
synthetic English contract jobs. With managed ASR and review units initially
**stopped**, run a real small French or other supported foreign-language
Studio job through ASR→Riva→English review to SRT/ASS, then a direct CLI job.
Prove exact model/CTC and host process identity, finite resource transition
times, cancellation/recovery, GPU reservations, output semantics, reuse, and
that unrelated GPU services remain unchanged. Publish sanitized evidence
on GitHub and revalidate again after promoting a reviewed/merged source.
