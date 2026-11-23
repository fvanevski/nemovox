# AGENTS.md — Nemovox local development and evaluation

## Workspace Boundaries
- This repository (normally `/workspace/nvidia-transcribe`, but isolated worktrees are supported) is the Nemovox development workspace. Core ASR defaults to NVIDIA Nemotron 3.5 Transcribe 30B-A4B NVFP4, while model resources may be configured by role.
- **Do not modify `/workspace/pyannote`** or install dependencies into `/workspace/pyannote/.venv`. Treat `/workspace/pyannote` and its outputs as read-only reference material.
- **Do not modify, move, rename, or redownload** the local checkpoint at `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`.

## Local GPU & Runtime Boundaries
- Host GPU: single NVIDIA RTX 5090 (32 GB VRAM, Blackwell sm_120).
- Do not stop, restart, or unload external GPU services automatically. Check `nvidia-smi` before GPU runs.
- Use the dedicated `.venv` under `/workspace/nvidia-transcribe/.venv` (Python 3.13 + vLLM + NeMo Speech plugin).
- For the default Nemotron ASR server, load only the existing local checkpoint at `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`. An explicitly configured alternative model resource must be validated for its adapter and required CTC contract; do not trigger implicit remote weight downloads.
- Verify active ModelOpt mixed NVFP4/FP8 quantization (`quantization=modelopt_mixed`) in server logs; silent BF16 dequantization is invalid for 32 GB feasibility.
- Keep CTC timestamp capture (`--middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware`, `VLLM_USE_V2_MODEL_RUNNER=0`, `--trust-request-mm-kwargs`, and `mm_processor_kwargs={"capture_ctc_timestamps": true}`) enabled when evaluating word/character timestamps and diarization.

## Provenance & Evaluation Rules
- Preserve source-language transcription and original Unicode text verbatim alongside any English translation; never overwrite source transcripts with translations.
- Chronology/untimed-audit passes may change a translated row's display `id` while preserving its source identity in `original_utterance_id`. Join retained translations to their immutable original source identity, validate a one-to-one mapping and unchanged source fields, and explicitly remap grouping/duplicate references. Never assume the translated display `id` is a stable ASR-source owner or silently attach English to the wrong source turn.
- Convert excerpt-relative timestamps to absolute media timestamps using recorded crop offsets (`source_offset_s`).
- Never fabricate missing timestamps, confidence scores, or speaker mappings.
- Keep CPU-only unit tests strictly separated from live GPU inference tests (`-m "not gpu"`).
- For isolated-worktree Studio tests, prefix `PYTHONPATH=src` (including when using the separate `.venv-studio` interpreter) and verify imports resolve to the candidate worktree rather than the main checkout. A green test using main's source is not candidate validation.
- Do not assume `HF_HUB_OFFLINE=1` blocks downloads performed by WhisperX/torchaudio through `torch.hub`. Check ALL required aligner checkpoints in the actual cache before loading an aligner, and do not call a loader that can fetch missing assets in an offline/cached-only campaign. Record any unexpected download as command/runtime friction without treating a downloaded model as an admitted test input.
- Exclude model weights, full-length media, `.venv`, large audio/logs, and secrets from Git commits and evidence archives.
- Treat `configs/model-resources.example.json` as the public template and `configs/model-resources.local.json` (or `NEMOVOX_MODEL_RESOURCES`) as host-specific configuration. Preserve role/adapter compatibility, deterministic override precedence and credential indirection (`api_key_env`); never silently substitute other model endpoints or CPU devices.
- Resource selection alone is not lifecycle authority. The CLI/Studio job pipeline MUST automatically acquire and release **explicitly Nemovox-owned** model services under verified unit/resource/endpoint/model identity and cooperative locks. Never stop/evict unrelated GPU services, bypass ownership admission, downgrade device, or download checkpoints. See `docs/model-resources/LIFECYCLE.md`; Qwen's separate managed-ASR handoff remains lease-gated.
- Keep `README.md`, `web/README.md` and `docs/model-resources/CONFIGURATION.md` consistent with final tested CLI and Studio behavior, preserving legacy aliases where supported.
