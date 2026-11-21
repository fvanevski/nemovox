# Evaluation Environment & Isolation

## Hardware & Host Software
- **Host GPU:** Single `NVIDIA GeForce RTX 5090` (`32607 MiB` total VRAM, Blackwell `sm_120`)
- **NVIDIA Driver / CUDA UMD:** `620.73.04` / CUDA `13.5` (CUDA runtime inside venv wheels: `cu132`, PyTorch `2.15.0+cu132` with `sm_120` native kernels)
- **Host RAM:** `32,753,596 kB` (~31.2 GiB total)
- **Storage:** `/workspace` (`186G` free), `/home` (`19G` free) — Hugging Face caches (`HF_HOME=/workspace/models/.hf-cache`, `HF_HUB_CACHE=/workspace/models/.hf-cache/hub`) and model checkpoints stored strictly under `/workspace/models/`.
- **Co-resident GPU services:** External `text-embeddings-router` (`~1760 MiB`) + Wayland desktop compositor (`~3.37 GiB` idle total) left undisturbed throughout. No persistent vLLM server was kept resident during modular Transformers evaluations.

## Demonstrated Dependency Incompatibility & Environment Split

An initial single-environment resolution attempt using `uv pip install --dry-run "qwen-asr==0.0.6" "transformers>=5.18.0"` failed cleanly:

```text
error: No solution found when resolving dependencies
  cause: Because qwen-asr>=0.0.6 depends on transformers==4.57.6 and you require
         qwen-asr==0.0.6, we can conclude that you require transformers==4.57.6.
         And because you require transformers>=5.18.0, we can conclude that your
         requirements are unsatisfiable.
```

Because `nvidia/Nemotron-3-Diarization` requires `transformers>=5.18.0` (`model_type="nemotron3_diarization"`, architecture `Nemotron3DiarizationForAudioFrameClassification`), while `qwen-asr==0.0.6` hard-pins `transformers==4.57.6`, two isolated Python `3.12.14` environments were created under `/workspace/nvidia-transcribe/outputs/modular-eval/`:

1. **Production Nemotron 3.5 env (untouched):** `/workspace/nvidia-transcribe/.venv` (Python `3.13.15`, Torch `2.13.0+cu130`, vLLM + NeMo Speech).
2. **Modular ASR + Aligner env:** `/workspace/nvidia-transcribe/outputs/modular-eval/.venv` (Python `3.12.14`):
   - `qwen-asr==0.0.6`, `transformers==4.57.6`, `torch==2.15.0`, `accelerate==1.12.0`, `huggingface-hub==0.36.2`, `librosa==1.0.1`, `soundfile==0.14.0`, `jiwer==4.0.0`, `nagisa==0.2.11`, `soynlp==0.0.493`, `sox==1.5.0`.
3. **Modular Diarization env:** `/workspace/nvidia-transcribe/outputs/modular-eval/.venv-diar` (Python `3.12.14`):
   - `transformers==5.23.0`, `torch==2.15.0`, `torchaudio==2.11.1`, `accelerate==1.16.0`, `huggingface-hub==2.4.0`, `librosa==1.0.1`, `soundfile==0.14.0`, `scipy==1.18.1`.

Full frozen package manifests (`pip_freeze_asr.txt`, `pip_freeze_diar.txt`) are preserved in the evidence archive. All model loads ran offline (`TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 local_files_only=True`) from `/workspace/models/`.
