# Chunked Nemotron Transcription Core — Architecture

## 1. Overview & Motivation

Single-request long-form inference on Nemotron 3.5 Transcribe 30B-A4B NVFP4 exposes four production failure modes that motivated this chunked recording-processing architecture:

1. **Long-form speaker tag destabilization:** In the single-pass 23-minute French episode (`outputs/responses/longform/fr_full_episode.raw.json`), continuous Sortformer `diarization` stayed smooth (`229` segments across speakers `0`–`6` via NeMo's internal Arrival-Order Speaker Cache), whereas the LLM decoder drifted into word-by-word `<spk:N>` ping-ponging (`481` speaker switches across `1992` words). Because vLLM serves stateless OpenAI-compatible requests without exposing persistent cross-request speaker-cache hooks, bounding inference to overlapping chunks (`90s` core + `10s` context on each side) restores stable LLM speaker-turn generation.
2. **Interleaved overlap fragmentation:** For simultaneous speech (especially Mandarin Chinese `multispeaker_sot` / CTC character alignment), `ctc_timestamps.words` is globally sorted by start timestamp, interleaving characters/words from `<spk:0>` and `<spk:1>` (`32` broken 1–2 character fragments on `zh_2spk_overlap`). Reconstructing utterances **per speaker stream first** instead of cutting on global speaker changes eliminates artificial fragmentation while preserving true overlap coordinates.
3. **Chunk-local speaker IDs:** Independent vLLM chunk requests reset `<spk:0>`, `<spk:1>`, ... per request. Speaker identity must be reconciled explicitly across chunks using boundary evidence and acoustic speaker embeddings without collapsing uncertain speakers or merging simultaneously speaking voices.
4. **All-or-nothing CTC forced alignment failures:** Any single OOV character in a chunk's transcript (notably CJK characters absent from the `5679`-character CTC tokenizer vocabulary, e.g., `'钾'`, `'诫'`, `'轱'`) aborts alignment across the entire request in upstream `ctc_serving.py`. The pipeline preserves the exact recognized source text, recovers valid timings on non-OOV sub-intervals, isolates OOV regions with Sortformer diarization bounds (`alignment_status="diarization_bounded_only"`), and provides an `ExternalForcedAligner` extension seam.

## 2. Module Boundaries

All recording-processing modules live under `src/nvidia_transcribe/` and build directly on `NemotronVLLMClient.transcribe` and `parser.py`:

| Module | Responsibility |
| --- | --- |
| [`media.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/media.py) | `ffprobe` stream inspection (`MediaInfo`, bounded SHA-256 fingerprint) and deterministic `ffmpeg` PCM-16kHz mono extraction (`extract_wav_segment`) for chunk windows and speaker crops. |
| [`chunking.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/chunking.py) | Exact gap-free core interval planning (`plan_chunks`) with configurable left/right context and `min_tail_seconds` tail merging; overlap-window deduplication and boundary omission recovery (`reconcile_chunk_boundaries`). |
| [`alignment.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/alignment.py) | CTC vocabulary inspection (`load_ctc_vocabulary_chars`, `find_ctc_oov_characters`), Sortformer diarization interval grouping (`group_diarization_windows`), partial sub-interval CTC recovery (`recover_chunk_alignment`), and `ExternalForcedAligner` protocol. |
| [`utterances.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/utterances.py) | Script-aware multilingual lexical joining (`join_multilingual_units` for CJK vs French/Latin punctuation/hyphens/apostrophes) and per-speaker timeline reconstruction (`reconstruct_speaker_utterances`). |
| [`speakers.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/speakers.py) | Chunk-local speaker normalization (`chunk_XXX:spk_Y` / `chunk_XXX:unknown`), concurrent-speaker `cannot_link` graph, Stage 1 temporal+lexical boundary matching, Stage 2 CPU `TitaNetSpeakerEmbedder` (`nemo:titanet_small:cpu`) clustering, and collision/ambiguity logging (`reconcile_speakers`). |
| [`contract.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/contract.py) | Canonical recording schema (`build_recording_contract`, `nemotron-recording-contract/v1`) plus preserved Phase-1 segment contract helpers (`build_transcript_contract`, `_group_timed_units_by_speaker_runs`, `to_pyannote_compatible_segments`, `normalize_language_code`, `detect_script_language_hint`, `extract_events_from_raw_text`). |
| [`recording.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/recording.py) | End-to-end orchestration (`process_media`), language resolution/provenance (`_resolve_recording_language`), incremental `manifest.json` checkpointing & crash-safe resume, sub-interval re-inference callback, and output serialization (`recording.contract.json`, `transcript.txt`). |
| [`cli.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/cli.py) | `nvidia-transcribe process-media` subcommand alongside existing `validate-checkpoint` and `transcribe`. |

## 3. Data & Control Flow

```
  Media (.webm/.mp4/.wav)
          │
          ▼
   media.probe_media() ──► MediaInfo (duration, stream metadata, bounded SHA-256)
          │
          ▼
   chunking.plan_chunks() ──► ChunkSpec[0..N-1] (exact core [core_start, core_end], request [req_start, req_end])
          │
          ▼  (for each pending chunk; completed chunks resumed from chunks/chunk_XXX.raw.json + manifest.json)
   extract_wav_segment() ──► NemotronVLLMClient.transcribe(..., source_offset_s=req_start)
          │                                            │
          │                                            ▼
          │                              alignment.recover_chunk_alignment()
          │                                  ├─ error=null & valid monotonic timings ──► ok
          │                                  ├─ OOV/failure + subinterval_runner     ──► partial_subinterval_recovery
          │                                  └─ unrecoverable / OOV sub-window      ──► diarization_bounded_only
          │
          ▼  (all chunks collected)
   recording._resolve_recording_language()  (explicit | model_detected | ambiguous)
          │
          ▼
   speakers.normalize_chunk_speakers() + speakers.reconcile_speakers()
     ├─ Stage 1: Adjacent boundary window (diarization IoU + token overlap)
     └─ Stage 2: Optional CPU TitaNet cosine similarity with cannot_link co-occurrence checks
          │
          ▼
   chunking.reconcile_chunk_boundaries()
     ├─ Drop duplicated overlap tokens using temporal proximity + lexical + speaker compatibility
     └─ Recover boundary tokens dropped near chunk boundary inside neighbor context
          │
          ▼
   utterances.reconstruct_speaker_utterances()
     ├─ Group timed + untimed units per global speaker
     └─ Split within-speaker on acoustic pauses (>1.2s), sentence punct (>0.35s), max duration
          │
          ▼
   contract.build_recording_contract() ──► recording.contract.json + transcript.txt + manifest.json
```

## 4. Separation of Concerns (Upstream Alignment)

Per NVIDIA's upstream design findings:

- **ASR chunking window** (`[req_start, req_end]`) is decoupled from **utterance segmentation** and **global speaker identity**.
- **Full-recording raw provenance vs derived reconciliation:** Every chunk retains its original recognized text (`raw_text`), raw Sortformer `diarization` activity intervals, CTC alignment diagnostics, and local-to-global speaker mappings without overwriting source text or conflating successful CTC alignment with correct speaker attribution.
- **CPU TitaNet isolation:** `TitaNetSpeakerEmbedder` runs strictly on CPU (`map_location="cpu"`, `torch.device("cpu")`) so it consumes **0 MiB GPU VRAM** alongside the 27.8 GiB vLLM NVFP4 server on the 32 GB RTX 5090.
