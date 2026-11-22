# Host Unit Configuration & Runtime Discovery (`20261122T054544Z`)

## Pre-Existing Host State Verified Before Provisioning
- `nemovox-asr.service`: `ABSENT` (not present in `~/.config/systemd/user/`).
- `nemovox-text-review.service`: `ABSENT` (not present in `~/.config/systemd/user/`).
- `nemovox-studio.service` (production): `active` on `127.0.0.1:8765` (`MainPID=2922624`, `UnitFileState=enabled`) — left untouched throughout.
- `/workspace/ninfer/swiftu` confirmed the live host `ninfer-serve` command (`./build/rtx5090-release/apps/ninfer-serve` + 27B `.ninfer` checkpoint on `127.0.0.1:8001`, `--model-id chat`, auth disabled).
- `/workspace/nvidia-transcribe/scripts/start_vllm_server.sh` confirmed the host Nemotron ASR launch script (port `8190`, `NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`, `quantization=modelopt_mixed`, `kv_cache_dtype=fp8`, CTC middleware enabled).

## Provisioned User Units (Un-enabled `UnitFileState=static`, Stopped When Idle)

### `/home/filip/.config/systemd/user/nemovox-asr.service`
```ini
[Unit]
Description=Nemovox explicitly managed ASR model
[Service]
Type=exec
WorkingDirectory=/workspace/nvidia-transcribe
Environment=NEMOVOX_MANAGED_RESOURCE=1
Environment=NEMOVOX_RESOURCE_NAME=asr_local
Environment=NEMOVOX_RESOURCE_MODEL=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4
Environment=NEMOVOX_RESOURCE_ENDPOINT=http://127.0.0.1:8190/v1
ExecStart=/workspace/nvidia-transcribe/scripts/start_vllm_server.sh
KillMode=control-group
Restart=no
TimeoutStopSec=30
```

### `/home/filip/.config/systemd/user/nemovox-text-review.service`
```ini
[Unit]
Description=Nemovox explicitly managed English translation review text model
[Service]
Type=exec
WorkingDirectory=/workspace/ninfer
Environment=NEMOVOX_MANAGED_RESOURCE=1
Environment=NEMOVOX_RESOURCE_NAME=text_local
Environment=NEMOVOX_RESOURCE_MODEL=chat
Environment=NEMOVOX_RESOURCE_ENDPOINT=http://127.0.0.1:8001/v1
ExecStart=/workspace/ninfer/build/rtx5090-release/apps/ninfer-serve /workspace/models/Qwen3.8-27B-swift15-uncensored-ajgazin-nvfp4full-dflash2-NInfer-v3/qwen3_8_27b_swift15abl_nvfp4full-dflash2.ninfer --model-id chat --host 127.0.0.1 --port 8001 --max-context 262144 --context-capacity 7GiB --kv-dtype k8v4 --max-concurrency 1 --prefill-chunk 4096 --prefix-reuse --max-tokens 32768 --thinking-budget 16384 --no-preserve-thinking --spec dflash2 --draft-tokens 7 --proposal-head optimized --vision --host-cache-capacity 0B --memory-headroom 1GiB --pending-timeout 120s --request-log-jsonl /tmp/nemovox-text-review-ninfer.jsonl
KillMode=control-group
Restart=no
TimeoutStopSec=30
```

## Isolated Candidate Runtime Environment (Ignored Host-Only Files)
- Lifecycle config (`NEMOVOX_LIFECYCLE_CONFIG`): `asr_local` -> `nemovox-asr.service` (`startup_timeout_s=300`, `stop_after_stage=true`, `min_free_vram_mb=26000`); `text_local` -> `nemovox-text-review.service` (`startup_timeout_s=300`, `stop_after_stage=true`, `min_free_vram_mb=26000`).
- Resource override (`NEMOVOX_MODEL_RESOURCES`): based on `configs/model-resources.example.json` with `"api_key_env": "VLLM_API_KEY"` on `asr_local` (see Finding F-01 in `review/findings.md`).
- Candidate Studio ran temporarily on `127.0.0.1:8767` (port `8766` occupied by `/workspace/mcp_servers/graph`) with isolated `jobs/`, `uploads/`, and `state/model-lifecycle.lock`, and was stopped cleanly at verification end.
