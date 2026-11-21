# Canonical Recording Contract (`nemotron-recording-contract/v1`)

The chunked recording pipeline emits `recording.contract.json` using `build_recording_contract` in [`src/nvidia_transcribe/contract.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/contract.py). The Phase-1 single-clip schema (`nemotron-eval-contract/v1` via `build_transcript_contract`) remains intact for single-clip tools and compatibility tests.

## Top-Level Schema

```json
{
  "schema_version": "nemotron-recording-contract/v1",
  "source": {
    "path": "/path/to/media.webm",
    "duration_s": 1379.901,
    "format_name": "matroska,webm",
    "size_bytes": 313494885,
    "audio_stream_count": 1,
    "selected_audio_stream": 0,
    "audio_codec": "opus",
    "sample_rate_hz": 48000,
    "channels": 2,
    "fingerprint_sha256": "..."
  },
  "source_language": "fr",
  "language_provenance": "explicit",
  "model_provenance": {
    "served_model": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
    "model_path": "/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
    "request_mode": "speaker",
    "language_detection": {"explicit_input": "fr"},
    "total_wall_time_s": 31.0437,
    "resumed_chunks": 0,
    "chunk_request_latencies_s": [1.1027, "..."]
  },
  "processing": {
    "chunk_config": {
      "chunk_seconds": 90.0,
      "context_seconds": 10.0,
      "min_tail_seconds": 15.0
    },
    "chunk_count": 16,
    "boundary_reconciliation": {
      "deduplicated_units": 67,
      "recovered_boundary_units": 4,
      "ambiguities": [],
      "dedup_examples": []
    },
    "chunks": [
      {
        "chunk_index": 0,
        "core_start_s": 0.0,
        "core_end_s": 90.0,
        "request_start_s": 0.0,
        "request_end_s": 100.0,
        "alignment_status": "ok",
        "local_speakers": ["chunk_000:spk_0", "..."],
        "speaker_mapping": {"chunk_000:spk_0": "SPK_00"},
        "raw_text": "...",
        "diarization": [{"start": 0.0, "end": 12.3, "speaker": "0"}]
      }
    ]
  },
  "speakers": [
    {
      "global_speaker_id": "SPK_00",
      "local_speaker_ids": ["chunk_000:spk_0"],
      "chunk_indices": [0],
      "estimated_speech_duration_s": 16.785,
      "identity_status": "provisional_single_chunk"
    }
  ],
  "utterances": [
    {
      "id": "utt-0000",
      "speaker": "SPK_00",
      "start": 4.08,
      "end": 18.97,
      "alignment_status": "ctc_aligned",
      "language": "fr",
      "text": "...",
      "chunk_indices": [0],
      "local_speakers": ["chunk_000:spk_0"],
      "words": [
        {
          "text": "Enfant",
          "start": 4.08,
          "end": 4.5,
          "alignment_status": "ctc_aligned",
          "speaker": "SPK_00",
          "local_speaker": "chunk_000:spk_0",
          "chunk_index": 0
        }
      ],
      "uncertainty": {"untimed_unit_count": 0}
    }
  ],
  "events": [],
  "alignment_diagnostics": [],
  "speaker_reconciliation": {
    "raw_local_speaker_count": 44,
    "reconciled_global_speaker_count": 22,
    "unknown_speaker_count": 0,
    "evidence": [],
    "ambiguous_matches": [],
    "unresolved_speakers": [],
    "collisions_prevented": [],
    "concurrent_speaker_pairs": [],
    "embedding_reconciliation": {
      "enabled": true,
      "backend": "nemo:titanet_small:cpu",
      "embedded_clusters": 24
    }
  },
  "processing_warnings": []
}
```

## Key Contract Invariants

1. **Language & Script Integrity:**
   - `source_language` is normalized (`fr`, `zh`, `en`, `unknown`, ...), never silently defaulting non-French audio to `"fr"`.
   - `language_provenance` distinguishes `"explicit"`, `"model_detected"` (with `detected_votes` and `script_hints` inside `model_provenance.language_detection`), `"ambiguous"`, or `"unknown"`.
   - `utterances[*].text` always preserves the verbatim source-language Unicode string; no machine translation replaces source text.
2. **Absolute Coordinates & Overlap Preservation:**
   - All `start` / `end` values on `utterances`, `words`, `chunks[*].diarization`, and `events` are absolute recording timestamps in seconds (with `source_offset_s` added upon chunk receipt).
   - Utterances are grouped per speaker timeline first and emitted sorted by `(start, end, id)`. Simultaneous speakers retain overlapping `[start, end]` intervals rather than interleaving word-by-word fragments.
3. **Explicit Timing Uncertainty (No Fabrication):**
   - Units from successful chunk alignment carry `"alignment_status": "ctc_aligned"`.
   - Units recovered via non-OOV sub-interval re-inference carry `"alignment_status": "subinterval_ctc_recovered"`.
   - Units inside an unrecoverable OOV or failed alignment interval carry `start: null`, `end: null`, `"alignment_status": "diarization_bounded_only"`, and explicit utterance-level `uncertainty: {"untimed_unit_count": N, "diarization_bounds": [...], "oov_characters": [...]}` plus chunk-level entries in `alignment_diagnostics`.
4. **Explicit Speaker Uncertainty & Provenance:**
   - `speakers[*].identity_status` explicitly distinguishes `"reconciled"` (linked across >= 2 chunks via temporal/lexical overlap or embedding similarity above threshold), `"provisional_single_chunk"` (seen in a single chunk with insufficient non-colliding evidence), and `"unknown"` (never silently collapsed into `SPK_00`).
   - `speaker_reconciliation` audits all `evidence`, `ambiguous_matches`, `unresolved_speakers`, `collisions_prevented`, and `concurrent_speaker_pairs`.
5. **Multimodal / Non-Speech Events:**
   - Non-speech tags (e.g., `<music>`, `<laughter>`) are extracted into top-level `events` instead of polluting spoken `utterances[*].text`.
