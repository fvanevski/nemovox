# Managed ASR Ownership, Service Audit & Lifecycle Verification

## Read-Only Host Eligibility Audit (`MANAGED_GPU_BLOCKED`)

Read-only inspection of the host prior to GPU runs established:

1. **ASR endpoint `127.0.0.1:8190`:** Not listening (`curl` failed to connect; `ss -ltnp` showed no listener on `8190`).
2. **Systemd user units (`systemctl --user list-units --type=service --all`):**
   - `nemovox-studio.service` is `loaded active running` on `127.0.0.1:8765` (serves from `/workspace/nvidia-transcribe` on `main`; explicitly excluded by `ManagedASRConfig` and `AGENTS.md`).
   - `local_embedding.service` is `loaded active running` on `127.0.0.1:8004` (`text-embeddings-router`, 1728 MiB VRAM).
   - **No** `nemovox-*.service` ASR user unit exists in `~/.config/systemd/user/`.
3. **Cooperative markers / lease infrastructure:** `~/.config/nemovox/` does not exist; no `NEMOVOX_MANAGED_ASR=1` or `NEMOVOX_ASR_LEASE_PATH` environment markers exist on any unit.

**Disposition:** **`MANAGED_GPU_BLOCKED`** — live `--qwen-gpu-mode managed` handoff was **not** executed, no service markers or units were fabricated, and all running host services (`nemovox-studio.service`, `local_embedding.service`) remained active and undisturbed throughout qualification.

Meanwhile, `nvidia-smi` confirmed `29077 MiB` free VRAM on the RTX 5090, so real CUDA BF16 qualification proceeded safely under `--qwen-gpu-mode available` (`released=False`).

## Verified Lifecycle & Safety Controls (CPU / Isolated Tests)

All lifecycle and failure-recovery paths were verified via `tests/test_qwen_session_and_policy.py` (107/107 CPU tests passing in both `.venv` and `.venv-studio`):

- **Non-owned / Studio / live-client rejection:** `test_managed_config_and_owner_rejects_studio_unmarked_and_live_clients` confirms `nemovox-studio.service` and non-`nemovox-*.service` units fail `ManagedASRConfig`; missing `NEMOVOX_MANAGED_ASR=1` fails with `asr_not_cooperatively_owned`; active `ss` TCP connections on the ASR port fail with `asr_has_live_connections`.
- **Busy lease:** `test_managed_gpu_busy_lease_never_stops_service` confirms `asr_busy` aborts before `systemctl stop`.
- **Available VRAM bypass:** `test_available_gpu_never_stops_asr` confirms `managed_asr_handoff` returns `cuda_already_available` without acquiring lease or calling `systemctl`.
- **Error restoration & journaling:** `test_managed_gpu_restores_after_failure_and_journals` verifies `stop -> failure -> start -> /v1/models health -> journal phase="restored"`.
- **SIGTERM unwind restoration:** `test_managed_sigterm_restores_original_asr_and_handler` verifies main-thread `SIGTERM` raises `KeyboardInterrupt`, restores the unit + `/v1/models` health, restores the prior `SIGTERM` handler, and writes `phase="restored"`.
- **Interrupted journal re-entry gate & `recover-qwen-asr`:** `test_interrupted_managed_journal_blocks_reentrant_takeover` and `test_pending_journal_blocks_even_when_gpu_has_capacity` prove `pending_restore` blocks automatic takeover even when VRAM is free; `test_recovery_refuses_changed_asr_unit_without_start` proves `recover_pending_managed_asr` refuses to start a modified unit (`asr_unit_changed_since_handoff`).
- **Linux kernel parent-death cleanup:** `test_linux_parent_death_sigkill_reaps_harmless_worker` proves a real forked Python worker armed with `prctl(PR_SET_PDEATHSIG, SIGKILL)` is immediately killed by the kernel when its parent receives `SIGKILL`.
