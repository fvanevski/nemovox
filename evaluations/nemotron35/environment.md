# Environment & Runtime Verification

## Host & Hardware
- **OS:** Garuda Linux (x86_64)
- **Host RAM:** 31 GiB physical + 47 GiB swap
- **GPU:** NVIDIA GeForce RTX 5090 (32,607 MiB VRAM, Blackwell `sm_120`)
- **NVIDIA Driver / CUDA UMD:** `620.73.04` / CUDA `13.5`
- **CUDA Toolkit (`nvcc`):** `/opt/cuda/bin/nvcc` — `Cuda compilation tools, release 13.5, V13.5.37`
- **Pre-existing GPU processes:** `text-embeddings-router` (~1,760 MiB), Wayland/Xwayland/Plex (~576 MiB); left untouched (`~28,728–29,046 MiB` free prior to vLLM launch).

## Isolated Python Environment
- **Virtual environment:** `/workspace/nvidia-transcribe/.venv`
- **Python:** `3.13.15` (via `uv 0.13.9`)
- **Vendored NeMo Speech checkout:** `/workspace/nvidia-transcribe/vendor/nemo-speech`
  - Git SHA: `99c252a77e7a1f57319618539882faa0d9c1abd2`
  - `uv.lock` SHA-256: `bae85cf8de466b5cc27ededeaa9d5ca81970246a140c8ef52517720217807278`
- **Key installed packages** (full freeze in `pip_list.txt`):
  - `vllm==0.32.0`
  - `nemo-toolkit==3.1.0+99c252a` (editable from `vendor/nemo-speech`)
  - `torch==2.13.0+cu130`
  - `transformers==5.18.0`
  - `flashinfer-python==0.7.0.post1`, `flashinfer-cubin==0.7.0.post1`, `flashinfer-jit-cache==0.7.0.post1+cu130` (`sm120f` included)
  - `nvidia-cudnn-cu13==9.20.0.48`
  - `av==19.0.2`, `opencv-python-headless==5.0.0.93`
  - `nvidia-transcribe==0.1.0` (editable from `/workspace/nvidia-transcribe`)

## Local Checkpoint Identity
- **Authoritative path:** `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`
- **HF download metadata revision:** `66c6adc4020179567074d0399cda2a4eeb2bb347`
- **Model type / architecture:** `nemo_speechlm` / `Nemotron3_5TranscribeForConditionalGeneration`
- **Quantization:** `modelopt` `MIXED_PRECISION` (`93` NVFP4 decoder/attention/MoE targets, `182` FP8 Mamba/encoder/MTP targets)
- **Bundled CTC head:** `model-ctc-bf16.safetensors` (`201,897,998` bytes, `23,862`-token vocabulary)
- **Total checkpoint size (excluding `.cache`):** `21,577,427,198` bytes (`20.10 GiB`), `20,584,095,517` parameters.

## vLLM Server Profile & Startup Verification
Launched via `scripts/start_vllm_server.sh` with `configs/vllm_rtx5090.env`:
```bash
export VLLM_PLUGINS=nemo_speechlm
export VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
export VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=180
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel
```
Key flags: `--dtype bfloat16 --kv-cache-dtype fp8 --gpu-memory-utilization 0.78 --max-model-len 32768 --max-num-batched-tokens 2048 --max-num-seqs 4 --enable-chunked-prefill --no-enable-prefix-caching --use-replayssm --mamba-backend flashinfer --mamba-cache-mode none --mm-processor-cache-gb 0 --no-async-scheduling --limit-mm-per-prompt '{"audio":{"count":1,"length":57600000}}' --reasoning-parser nemotron_v3 --trust-request-mm-kwargs --middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware`.

Verified server log lines:
- `quantization=modelopt_mixed`
- `Using fp8 data type to store kv cache`
- `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`
- `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`
- `FP8 audio encoder: 128 Linears built from quantization_config`
- `[NeMoSpeechLM] CTC timestamp inputs bound to vLLM encoder-cache identity.`
- `[NeMoSpeechLM] CTC timestamps enabled from checkpoint config: /workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4/model-ctc-bf16.safetensors`
- `Model loading took 19.28 GiB memory and 37.818456 seconds`
- `GPU KV cache size: 363,178 tokens, Maximum concurrency for 32,768 tokens per request: 11.08x`

## Operational Friction Notes
1. **Port 8000 occupied:** Host had a pre-existing service on `127.0.0.1:8000`; server bound to `127.0.0.1:8190`.
2. **Host RAM OOM risk without pre-downloaded FlashInfer kernels:** First cold start attempted on-the-fly `nvcc` compilation of `fp4_gemm_cutlass_sm120` on a 31 GiB RAM host; installing `flashinfer download-kernels` (`flashinfer-cubin` + `flashinfer-jit-cache` `sm120f`) eliminated `fp4_gemm_cutlass_sm120` JIT compilation and enabled clean startup.
3. **Bearer auth:** Host environment had `VLLM_API_KEY` set (`EMPTY`), causing vLLM to enforce `Authorization: Bearer ...` on `/v1/*` routes; updated `NemotronVLLMClient` to always send the header.
