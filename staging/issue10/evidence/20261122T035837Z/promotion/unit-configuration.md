# Host Unit Configuration Discovery (`20261122T035837Z`)

**Status:** `NOT_PROVISIONED` — Phase B provisioning did not proceed because Phase A (`pytest -q -m "not gpu"`) failed.

## Read-Only Host Discovery Notes (HOST_ATTESTED, No Mutations)
- Production Studio (`nemovox-studio.service`, `127.0.0.1:8765`) was left completely untouched.
- Host text-model launcher discovered read-only: `/workspace/ninfer/swiftu` invokes `./build/rtx5090-release/apps/ninfer-serve` from `/workspace/ninfer` with `/workspace/models/Qwen3.8-27B-swift15-uncensored-ajgazin-nvfp4full-dflash2-NInfer-v3/qwen3_8_27b_swift15abl_nvfp4full-dflash2.ninfer` (`--model-id chat --host 127.0.0.1 --port 8001 ...`) after `systemctl --user start litellm-proxy.service` (which is already `active (running)`).
- Both `/workspace/ninfer/build/rtx5090-release/apps/ninfer-serve` and the NVFP4 `.ninfer` model file exist on disk.
- No `nemovox-asr.service` or `nemovox-text-review.service` unit was created or started.
