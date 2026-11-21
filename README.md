# NVIDIA Nemotron 3.5 Transcribe NVFP4 Evaluation

Standalone evaluation and prototype client for **NVIDIA Nemotron 3.5 Transcribe 30B-A4B NVFP4** on an NVIDIA RTX 5090 (32 GB VRAM) using vLLM + NVIDIA NeMo Speech (`nemo_speechlm` plugin).

## Boundaries

- Local model checkpoint (read-only): `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`
- Baseline comparison reference (read-only): `/workspace/pyannote` (commit `2b40b1c0c17ce324178af765d8d2c79234870acd`)
- Isolated Python 3.13 environment: `/workspace/nvidia-transcribe/.venv`
- Vendored upstream runtime checkout (ignored by Git): `/workspace/nvidia-transcribe/vendor/nemo-speech`

## Structure

- `src/nvidia_transcribe/`: checkpoint validation, prompt templates, vLLM OpenAI-compatible client, CTC timestamp & speaker-tag parser, canonical multilingual transcript adapter, evaluation metrics (WER/CER/alignment).
- `tests/`: CPU-only unit tests (never load the 30B checkpoint).
- `scripts/`: server launcher, audio extraction, benchmark runner, and evidence packager.
- `configs/`: server and evaluation configurations.
- `evaluations/nemotron35/`: reports, benchmarks, and artifact manifest.
