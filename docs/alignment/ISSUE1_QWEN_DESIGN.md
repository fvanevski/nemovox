# Issue #1: selective CUDA forced alignment and safe ASR handoff

## Authority and goals

Source authority: `fvanevski/nemovox` issue #1 and current GitHub `main`. At design staging, `main@242c418f55cd2afb796ee2c4fa742a10ff04f7a3`; **re-read before all consequential operations**. This document describes the Central-authored candidate on `issue/1`; it does not claim host qualification.

The objective is subtitle coverage, not character timestamp perfection. All existing CTC/subinterval timings, original spoken text, speaker identity, source offsets and honest untimed states remain authoritative. `alignment_fallback=off` remains the default and must allocate zero Qwen resources.

## Two-phase processing invariant

1. The recording pipeline finishes all chunk ASR, LID and **native CTC/subinterval recovery**, checkpointing raw responses and an independent `native-alignment.checkpoint.json` when Qwen opt-in is selected. The ASR vLLM model must still be available for every native request.
2. Only after native work is finished, a bounded `AdmissionPolicy` scans untimed speech spans. Each candidate must be meaningful, uniquely anchored in a finite localized window, non-overlapping with mixed speakers, and short enough to align. Default `subtitle` skips low-value isolated character gaps in otherwise timed cues; `fine` is opt-in.
3. If there are zero eligible spans, no worker is launched and no service is touched. Otherwise one isolated `QwenForcedAlignerAdapter` owns one JSONL `QwenAlignerSession` across all chunks/candidates, with the model loaded once and responses validated before attaching timings. The same lexical matching and monotonic bounds validation in `alignment._apply_verified_sequence_timings` rejects invalid results without rewriting source text.
4. The worker is deterministically shut down **before** the pipeline normalizes speakers/reconstructs utterances, and **before** any managed ASR vLLM restoration is adjudicated as complete. The final contract includes candidate/admission diagnostics and worker telemetry.

Files: `recording.py`, `selective_alignment.py`, `qwen_aligner.py`, `_qwen_align_worker.py`, `gpu_handoff.py`, `cli.py`, `subtitle_pipeline.py`, focused CPU tests.

## Reusable worker and device policy

- Worker remains in `outputs/modular-eval/.venv` (qwen-asr dependencies), separate from the Nemotron inference venv. Default offline model path is `/workspace/models/Qwen3-ForcedAligner-0.6B`; no automatic checkpoint download.
- JSONL protocol v1 uses `init -> align* -> shutdown` with request IDs and a bounded response/time. On any protocol error, deadline or cancel, kill the owned worker process group; do not silently retry each request with a freshly loaded model. On Linux the worker also registers a parent-death SIGKILL, so cancellation of Studio's separately sessioned CLI parent cannot orphan Qwen CUDA allocations. This parent-death guard does **not** restart ASR.
- GPU requested means `cuda:0` BF16. `auto` within the adapter is an alias for **CUDA required**, never a CPU fallback. Actual CUDA tensor allocation and observed parameter dtype, when accessible through the backend, are reported separately from the dtype request. `cpu` is a distinct **explicit** selection, never selected implicitly.
- Time the model startup/load and each alignment request; report actual device, observed/requested dtype, CUDA allocated/peak memory, model load count, align count, worker failure status, and validated zero-duration/out-of-bounds units.

## Managed vLLM mode: ownership is a hard gate

`--qwen-gpu-mode available` (default) never controls another process. If free CUDA memory is insufficient, report deferred/blocked. `--qwen-gpu-mode managed` is a separate explicit opt-in and requires `--qwen-owned-asr-unit` and `--qwen-asr-lease-path`, both identifying an **already known Nemovox-owned ASR user unit**.

The managed controller refuses to stop a service unless all of the following hold:

- Service is an exact `nemovox-*.service` **user** unit (never `nemovox-studio.service`), loaded, active and has a valid MainPID and invocation identity.
- The service unit includes the operator-provisioned `NEMOVOX_MANAGED_ASR=1` and `NEMOVOX_ASR_LEASE_PATH=<exact private path>` markers. The lease file exists, is user-owned/private and can be locked exclusively without waiting. **All supported ASR consumers must participate in this lease discipline.** Nonparticipating clients cannot be made safe by examining a PID or guessing a port.
- Live process command and `/v1/models` identify the *same* Nemotron ASR vLLM model/loopback port requested by the pipeline. Unknown, shared or externally managed services are not touched.
- Native ASR has finished and a checkpoint has been written. The controller checks actual free VRAM first; it does not stop a service if CUDA is already available.

Once admitted, journal original unit identity/config before stop; stop only that unit; verify stopped state and sufficient freed VRAM; launch bounded Qwen session; teardown; restart original active unit with existing unit configuration; and verify `/v1/models` returns the original model. Restore is attempted after normal return, worker failure, timeout, ordinary CLI SIGTERM cancellation (Python main-thread signal unwind), and ambiguous stop. An interrupted pending journal blocks **new automatic handoff** even if VRAM is free. Uncatchable SIGKILL, power loss and fatal process termination cannot guarantee immediate ASR restoration and require explicit journal/identity-checked recovery; never claim them as automatically recovered. Explicit `recover-qwen-asr` can restore only after revalidating unit executable/hash, fragment path, cooperative marker, lease and original active state.

**Important boundary:** a user-specified unit name or a private lease file alone is not proof that no unrelated applications depend on the model. Host qualification must check launcher/consumer participation and status. If existing launcher/lease semantics do not satisfy this contract, report `BLOCKED_SAFE_HANDOFF` and leave vLLM alone. Do not silently add markers or create a competing service controller as a local-agent workaround. Central must evaluate any material launcher integration separately.

## CLI examples (controlled, not autorun)

Off is ordinary default:
```bash
python -m nvidia_transcribe.cli process-media media.wav --output-dir out/
```

GPU-required optional selective alignment:
```bash
python -m nvidia_transcribe.cli process-media media.wav --output-dir out/ \
  --alignment-fallback qwen --qwen-policy subtitle --qwen-device cuda:0 \
  --qwen-gpu-mode available
```

Fine alignment is opt-in: `--qwen-policy fine`. Explicit CPU, when desired, is `--qwen-device cpu`. Managed mode additionally requires `--qwen-gpu-mode managed --qwen-owned-asr-unit <verified-unit> --qwen-asr-lease-path <existing-private-lease>`.

For a **previously interrupted** and independently identified managed unit, only operator-authorized recovery:
```bash
python -m nvidia_transcribe.cli recover-qwen-asr \
  --asr-unit <verified-unit> --base-url http://127.0.0.1:8190/v1 \
  --model <exact-served-model-id> --lease-path <private-lease> \
  --journal <original-asr-output-dir>/qwen-asr-handoff.journal.json
```

## Evidence / limitations

- Historic OOV and warm/cold benchmark numbers in `evaluations/` are **not** directly comparable to new throughput. Benchmark identical bounded Mandarin and multi-span inputs with off vs qwen on this exact source head; report wall time, useful utterance/cue recovery, invalid/zero durations, device/dtype, worker model loads and VRAM telemetry. Warm-model benefit must be demonstrated, not extrapolated.
- Real managed handoff may be **blocked** by missing service ownership/lease, which is a correct non-destructive result; synthetic lifecycle tests still required. Do not mark full managed-mode acceptance PASS without a real qualified handoff.
- Optional native-HF comparison is permitted only if compatible locally and bounded. Do not make Torch compile or a universal GPU scheduler prerequisites or enlarge the issue for them.
- `Studio`'s existing systemd user service remains on merged `main`; do not swap its interpreter to an unreviewed branch or restart it for issue #1 tests. Host tests run in isolated worktrees and outputs.
