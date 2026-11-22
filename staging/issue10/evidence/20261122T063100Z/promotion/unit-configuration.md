# Host Unit Configuration & Runtime Environment (`20261122T063100Z`)

Reused verified Nemovox-owned `systemd --user` units (`UnitFileState=static`, initially `inactive`, stopped cleanly post-test):
- `/home/filip/.config/systemd/user/nemovox-asr.service`: `Type=exec`, `WorkingDirectory=/workspace/nvidia-transcribe`, `ExecStart=/workspace/nvidia-transcribe/scripts/start_vllm_server.sh`, `KillMode=control-group`, `Restart=no`, `TimeoutStopSec=30`, markers `NEMOVOX_MANAGED_RESOURCE=1`, `NEMOVOX_RESOURCE_NAME=asr_local`, `NEMOVOX_RESOURCE_MODEL=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`, `NEMOVOX_RESOURCE_ENDPOINT=http://127.0.0.1:8190/v1`.
- `/home/filip/.config/systemd/user/nemovox-text-review.service`: `Type=exec`, `WorkingDirectory=/workspace/ninfer`, `ExecStart=/workspace/ninfer/build/rtx5090-release/apps/ninfer-serve /workspace/models/Qwen3.8-27B-swift15-uncensored-ajgazin-nvfp4full-dflash2-NInfer-v3/qwen3_8_27b_swift15abl_nvfp4full-dflash2.ninfer --model-id chat --host 127.0.0.1 --port 8001 ...`, `KillMode=control-group`, `Restart=no`, `TimeoutStopSec=30`, markers `NEMOVOX_MANAGED_RESOURCE=1`, `NEMOVOX_RESOURCE_NAME=text_local`, `NEMOVOX_RESOURCE_MODEL=chat`, `NEMOVOX_RESOURCE_ENDPOINT=http://127.0.0.1:8001/v1`.

## Candidate Runtime Settings (`73760ee8f969a5927ac10cd8985de93f33f71e6a`)
- **`NEMOVOX_MODEL_RESOURCES`:** Exact copy of `configs/model-resources.example.json` with **no** `api_key_env` on `asr_local`, while `VLLM_API_KEY` was non-empty in both `systemd --user` and candidate Studio/CLI environments.
- **`NEMOVOX_LIFECYCLE_CONFIG`:** `min_free_vram_mb: 26000`, `startup_timeout_s: 300`, `stop_after_stage: true` for `asr_local` and `text_local`.
- **Canonical shared lock (`NEMOVOX_LIFECYCLE_LOCK` unset):** `~/.local/state/nemovox/model-lifecycle.lock`, shared identically across candidate Studio, Studio CLI subprocesses, direct CLI runs, and recovery/protection probes.
- **Production Studio (`127.0.0.1:8765`):** Untouched (`MainPID=2922624`, HTTP `200`).
