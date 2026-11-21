# Read-Only Managed-ASR Feasibility Audit (`7b4112cc31784953abbb041fd0b5d7997851841d`)

Raw read-only audit trace: [`logs/managed_asr_readonly_audit.txt`](logs/managed_asr_readonly_audit.txt).

## Verdict

- **`MANAGED_ASR_FEASIBILITY=MANAGED_GPU_BLOCKED: no cooperatively owned ASR unit/lease`**
- **`LIVE_ASR_RELEASE_RESTORE=NOT_TESTED`**

## Observed Host Evidence

1. **Systemd user units & running processes:**
   - Only one `nemovox-*.service` unit exists: `nemovox-studio.service` (`active`, `MainPID=1006972`, explicitly excluded by `ManagedASRConfig`).
   - Other model/proxy user units (`local_embedding.service` active on `127.0.0.1:8004`, `llama-cpp-embed.service`, `llama-cpp-reranker.service`, `llama-server.service`, `vllm-tools-shim.service`) are non-Nemovox services, have `has_exact_markers=False` under `gpu_handoff._has_exact_cooperative_markers`, and are ineligible.
   - `127.0.0.1:8190` has no active listener and `pgrep -af 'vllm|nemotron|nemo_speechlm'` shows no live Nemotron ASR process.
2. **Existing launcher & callers:**
   - `scripts/start_vllm_server.sh` + `configs/vllm_rtx5090.env` starts Nemotron vLLM interactively (`exec .venv/bin/vllm serve ... --host 127.0.0.1 --port 8190 --gpu-memory-utilization 0.78`), not under a `nemovox-*.service` user unit, with no `NEMOVOX_MANAGED_ASR=1` / `NEMOVOX_ASR_LEASE_PATH` assignment and no lease locking.
   - `src/nvidia_transcribe/studio_api.py` runs CLI jobs with `--alignment-fallback` (`off|qwen`), defaulting to `--qwen-gpu-mode available`, and does not pass `--qwen-gpu-mode managed` or participate in an ASR lease protocol.
3. **VRAM necessity in loaded-ASR scenario:**
   - While the idle host currently has `29,075 MiB` free VRAM (`3052 MiB` used by `text-embeddings-router` + desktop), historical loaded-ASR logs (`outputs/logs/vllm_ep166.log`) show `configs/vllm_rtx5090.env` (`gpu_memory_utilization=0.78`) reserves `24.47 GiB` (`20.13 GiB` weights/non-torch + `2.2 GiB` peak activation + `2.14 GiB` KV cache + `0.08 GiB` CUDA graph), leaving only ~`3.4 GiB` free on startup (or less with concurrent inference consumers) against Qwen's `min_free_vram_mb=2600` MiB gate and `~1.77 GiB` (`1,900,171,776 B`) peak CUDA allocation. Thus a managed takeover path is genuinely motivated when Nemotron ASR is loaded, even though idle VRAM does not trigger it.

## Minimal Concrete Requirements for Future Live Managed Takeover

1. **Dedicated user unit (`nemovox-asr.service`)**: Adapt `scripts/start_vllm_server.sh` into a dedicated systemd user unit (`nemovox-asr.service`) with `NEMOVOX_MANAGED_ASR=1` and `NEMOVOX_ASR_LEASE_PATH=<private-0600-lease-path>`.
2. **Exclusive lease participation**: Ensure **all** supported ASR clients/launchers (CLI, `studio_api.py`, and any interactive/external callers targeting `127.0.0.1:8190`) acquire shared/exclusive `flock` on that lease file, or isolate the unit to exclusive Nemovox workflows.
3. **Separate Central authorization**: Any launcher/Studio integration or unit creation requires a separate Central decision before running a real reversible stop/Qwen/restore smoke test.
