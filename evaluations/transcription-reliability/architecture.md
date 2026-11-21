# Architecture: Reliability Repairs, Selective Alignment Fallback, and Experimental Evidence Modules

## 1. Production Pipeline vs. Experimental Boundaries

The production chunked transcription pipeline in `/workspace/nvidia-transcribe/src/nvidia_transcribe/` enforces strict separation between **authoritative recording output** and **experimental advisory modules**:

```text
Media / Audio
   │
   ▼  media.py (SHA256 source validation, atomic .tmp.wav extraction, config/hash cache invalidation)
Chunk Windows (chunking.py: unknown_speaker non-dedup, preserved untimed ordering)
   │
   ▼  client.py -> vLLM (Nemotron 3.5 NVFP4 modelopt_mixed, CTC timestamp middleware)
Primary Parse (parser.py: verbatim source Unicode + raw_text spacing provenance)
   │
   ▼  alignment.py
   ├─► Native Subinterval CTC (non-destructive SequenceMatcher; never substitutes text)
   └─► Optional alignment_fallback="qwen" (qwen_aligner.py -> _qwen_align_worker.py in modular-eval/.venv)
          │  • Skipped on mixed-speaker overlap (_has_mixed_speaker_overlap)
          │  • Runs ONLY on remaining untimed spans
          │  • validate_qwen_items rejects zero-duration / out-of-bounds without clamping
          ▼
Speakers & Utterances (speakers.py polynomial MCMF + pre-commit ambiguity filtering;
                       utterances.py explicit partial_alignment / timing_status="partial", CJK/Latin spacing)
   │
   ▼
Authoritative Recording Contract (recording.contract.json)
   │
   ├──► [EXPERIMENTAL ONLY] refinement.py (PatchProposal status="proposed_only", never auto-applied)
   └──► [EXPERIMENTAL ONLY] scene_diarization.py (authoritative_override_applied=False, supplemental evidence)
```

---

## 2. Environment Isolation & Cache Configuration

- **Main Production Environment (`/workspace/nvidia-transcribe/.venv`):** Python `3.13.15`, `vLLM` (with `nemo.collections.speechlm2.vllm.salm.ctc_timestamp_middleware` and `--reasoning-parser nemotron_v3`), PyTorch + NeMo TitaNet.
- **Qwen Aligner Subprocess Runtime (`/workspace/nvidia-transcribe/outputs/modular-eval/.venv`):** Python `3.12.14`, `transformers==4.57.6`, `qwen-asr==0.0.6`. Invoked via JSON stdin/stdout over subprocess so zero dependency conflicts touch the main vLLM environment. Auto-selects CUDA when `>= 2600 MiB` free VRAM is available (successfully observed using `cuda:0`, `torch.bfloat16` co-resident with vLLM) or falls back cleanly to CPU.
- **Scene Diarization Runtime (`/workspace/nvidia-transcribe/outputs/modular-eval/.venv-diar`):** Python `3.12.14`, `transformers==5.23.0` (`Nemotron-3-Diarization`), strictly isolated for scene-bounded GPU runs.
- **Server Cache Optimization (`scripts/start_vllm_server.sh`):** Configured explicit workspace `VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${ROOT_DIR}/.cache/vllm}"` alongside `FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${ROOT_DIR}/.cache/flashinfer}"` and pre-populated it with the warm autotune and `torch.compile` caches (`126 MiB`) so subsequent vLLM launches reuse compiled artifacts without rebuilding.

---

## 3. Production & Experimental Module Map

| Module | Role | Mutates Authoritative Output? |
| :--- | :--- | :--- |
| `src/nvidia_transcribe/chunking.py` | Boundary deduplication & reconciliation | Yes (production — fixed A1, A3) |
| `src/nvidia_transcribe/alignment.py` | Native CTC recovery + selective `ExternalForcedAligner` routing | Yes (production — fixed A2, A7; wired Phase B) |
| `src/nvidia_transcribe/utterances.py` | Intra-speaker stream reconstruction & partial timing status | Yes (production — fixed A3, A4, A7) |
| `src/nvidia_transcribe/speakers.py` | Polynomial MCMF assignment & pre-commit ambiguity filters | Yes (production — fixed A5, A8) |
| `src/nvidia_transcribe/media.py` | Source SHA256, WAV header verification, atomic segment writes | Yes (production — fixed A6) |
| `src/nvidia_transcribe/recording.py` | Pipeline orchestration, cache invalidation, `alignment_fallback` flag | Yes (production — fixed A6; wired Phase B) |
| `src/nvidia_transcribe/qwen_aligner.py` & `_qwen_align_worker.py` | Isolated `Qwen3-ForcedAligner-0.6B` adapter & strict validator | Yes, ONLY untimed spans when `alignment_fallback="qwen"` |
| `src/nvidia_transcribe/refinement.py` | Suspicious-span localization & `PatchProposal` generation | **No** (`status="proposed_only"`) |
| `src/nvidia_transcribe/scene_diarization.py` | Scene planning & non-destructive diarization evidence reconciliation | **No** (`authoritative_override_applied=False`) |
