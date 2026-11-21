#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${1:-$ROOT_DIR/configs/vllm_rtx5090.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing server env file: $ENV_FILE" >&2
  echo "Copy $ROOT_DIR/configs/vllm_rtx5090.env.example to $ENV_FILE and configure MODEL_PATH." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

export CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export MAX_JOBS="${MAX_JOBS:-2}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${ROOT_DIR}/.cache/flashinfer}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${ROOT_DIR}/.cache/vllm}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$VLLM_CACHE_ROOT" "${ROOT_DIR}/outputs/logs"

export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel

EXTRA_ARGS=()
if [[ "${ENABLE_MTP:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":8}')
fi

exec "${ROOT_DIR}/.venv/bin/vllm" serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --use-replayssm \
  --mamba-backend flashinfer \
  --mamba-cache-mode none \
  --mm-processor-cache-gb 0 \
  --no-async-scheduling \
  --limit-mm-per-prompt '{"audio":{"count":1,"length":57600000}}' \
  --reasoning-parser nemotron_v3 \
  --trust-request-mm-kwargs \
  --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware \
  "${EXTRA_ARGS[@]}"
