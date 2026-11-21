# Source & Host Runtime Identity

- **Repository:** `fvanevski/nemovox` (Issue #1, Merged PR #4)
- **Pinned `main` Commit SHA:** `7b4112cc31784953abbb041fd0b5d7997851841d`
- **Pinned `main` Tree SHA:** `3d2cc429f9b49da3079df7d1fc178f6fd09d7987`
- **Handoff Reference Commit:** `c01d306e6f542b6ab7dd93768a7259667ab399d9` (`handoff/issue-1-asr-qwen-coexistence-20261121`)
- **Live Checkout (`/workspace/nvidia-transcribe`):** Clean on `main@7b4112cc31784953abbb041fd0b5d7997851841d` (tree `3d2cc429f9b49da3079df7d1fc178f6fd09d7987`).
- **Physical GPU:** `GPU 0`, `NVIDIA GeForce RTX 5090`, UUID `GPU-c75b2559-b8e1-ebf9-fe82-65cd2b3ca4c0`, `32,607 MiB` total VRAM.
- **ASR Launcher & Config (unchanged):**
  - Launcher: `scripts/start_vllm_server.sh`
  - Config: `configs/vllm_rtx5090.env` (`MODEL_PATH=/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`, `SERVED_MODEL_NAME=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`, `HOST=127.0.0.1`, `PORT=8190`, `GPU_MEMORY_UTILIZATION=0.78`, `MAX_MODEL_LEN=32768`, `MAX_NUM_BATCHED_TOKENS=2048`, `MAX_NUM_SEQS=4`, `ENABLE_MTP=0`).
- **Qwen Worker & Checkpoint (unchanged):**
  - Python: `/workspace/nvidia-transcribe/outputs/modular-eval/.venv/bin/python`
  - Checkpoint: `/workspace/models/Qwen3-ForcedAligner-0.6B`
  - Worker admission threshold: `2,600,000,000` free CUDA bytes (`torch.cuda.mem_get_info(0)` in `src/nvidia_transcribe/_qwen_align_worker.py`).
- **Preserved Host Services:**
  - `nemovox-studio.service`: `active` throughout (zero active Studio jobs).
  - `local_embedding.service`: `active` throughout.
  - Docker container `/tei-qwen3-embedding` (`64c86ba8a443dddafc9ae268d7381bdbfe538f25aa74c2214d1d4863bad77b84`, `ghcr.io/huggingface/text-embeddings-inference:120-1.9`, `127.0.0.1:8005->8080/tcp`): `running`, `healthy`.
