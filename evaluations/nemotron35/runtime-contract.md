# Runtime & Experimental Transcript Contract

## Verified vLLM Request Contract
- **Endpoint:** `POST /v1/chat/completions`
- **Required flags for deterministic ASR/translation:** `temperature=0.0`, `top_p=1.0`, `seed=0`, `stream=false`, `chat_template_kwargs={"enable_thinking": false}`, `skip_special_tokens=false`, `spaces_between_special_tokens=false`.
- **CTC timestamp opt-in:** `mm_processor_kwargs={"capture_ctc_timestamps": true}` (requires server `--trust-request-mm-kwargs`, `VLLM_USE_V2_MODEL_RUNNER=0`, and `--middleware nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware`).
- **Observed server `ctc_timestamps` schema:**
  - `words`: list of `{"word": str, "start": float, "end": float, "speaker": str}` (relative seconds to clip start; note `speaker` is `"0"`, `"1"`, etc. when `<spk:N>` tags are present).
  - `diarization`: list of `{"speaker": int, "start": float, "end": float}` (10 ms Sortformer activity segments).
  - `speaker_tag_to_diarization_speaker`: mapping `{"0": int, ...}` linking transcript tags to Sortformer diarization speaker IDs.
  - `error`: `null` on success, or `"not_enabled"` / `"no_capture"` / `"multiple_audio"` / `"capture_unavailable"` / `"alignment_failed"`.

## Experimental Multilingual Transcript Contract (`nemotron-transcript-contract/v1`)
Implemented in `src/nvidia_transcribe/contract.py` (`build_transcript_contract`):
- Top-level fields:
  - `schema_version`: `"nemotron-transcript-contract/v1"`
  - `source_language`: explicit language code or `"auto"`
  - `source_text`: source-language transcript preserved verbatim (never overwritten by translation)
  - `raw_text`: unmodified Nemotron response including `<spk:N>`, `<ev:*>`, `<sty:*>` tags
  - `english_translation`: optional separate English translation string (`null` unless translated)
  - `utterances`: contiguous same-speaker runs (`A -> B -> A` stays 3 runs), each containing:
    - `id`, `source_language`, `source_text`, `speaker_id`
    - `source_start`, `source_end` (offset-adjusted absolute media timestamps, or `null` if unavailable)
    - `source_words`: timed word units (non-CJK) with `raw_start`, `raw_end`, `start`, `end`, `valid_timing`, `timing_error`
    - `source_characters`: timed CJK character units with the same fields
    - `timestamp_provenance`: `backend`, `source_offset_s`, `timing_status`, `invalid_unit_count`
  - `diarization_intervals`: Sortformer intervals with offset-adjusted `start`/`end` and `raw_start`/`raw_end`
  - `speaker_mapping`: `speaker_tag_to_diarization_speaker`
  - `timestamp_provenance`: top-level alignment backend, `error`, `invalid_timing_count`, `unit_count`
  - `model_provenance`: served model, mode, finish reason, usage tokens, latency, audio path
  - `uncertainty`: `null` (not invented; Nemotron vLLM response provides no calibrated word confidence scores)

## Compatibility with `/workspace/pyannote` Subtitle Pipeline
Tested read-only via `scripts/test_pyannote_subtitle_compat.py` against `/workspace/pyannote/pyannote_app/utils/subtitles.py:build_subtitles`:
- **French (`fr_0316_0355_nemotron.srt` / `.ass`):** Adapted cleanly via `to_pyannote_compatible_segments(contract)`, preserving absolute `00:03:16,670` timestamps and producing valid SRT/ASS with 0 overflow.
- **Mandarin (`zh_sample_nemotron.srt` / `.ass`):** Required passing pre-built `subtitle_utterances` alongside `segments` so `build_subtitle_utterances` did not join CJK characters with ASCII spaces, and required note that `subtitles.py:_format_cue_lines` wraps lines on whitespace (which does not split unspaced CJK lines >42 chars). Future subtitle pipeline work needs CJK-aware line wrapping and character-level cue splitting.
