# CTC Alignment Diagnostics & Failure Recovery

## 1. Root Cause of Upstream Mandarin Alignment Failures

NVIDIA's upstream CTC timestamp middleware (`ctc_serving.py`) runs CTC forced alignment over the entire recognized request transcript as one sequence. Inspection of the model's checkpoint tokenizer confirms `5679` valid CTC characters (`load_ctc_vocabulary_chars` in [`src/nvidia_transcribe/alignment.py`](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/alignment.py)), while common/technical simplified Chinese characters — including `'钾'` (potassium), `'诫'`, and `'轱'` — are **absent (`OOV`)**.

When even a single OOV character appears in the generated transcript:
- Upstream returns `ctc_timestamps.error = "alignment_failed"` with empty word/char arrays (`words=[]`), wiping out fine-grained word timestamps for the entire chunk even when ASR recognition itself is 100% accurate.
- Conversely, upstream Sortformer acoustic `diarization` intervals (`start`, `end`, `speaker`) are **still populated** in the response.

## 2. Recovery Policy (`src/nvidia_transcribe/alignment.py`)

`recover_chunk_alignment` applies a non-destructive 4-step policy:

1. **Preserve verbatim source text:** `raw_text` and spoken character sequences are never edited, transliterated, or dropped to satisfy the CTC aligner.
2. **Validate CTC timings first:** Checks `ctc_timestamps.error`, empty word lists on non-empty speech, and monotonic/non-negative finite timestamps (`invalid_timing_count`).
3. **Inspect OOV & root cause:** Checks characters against `tokenizer.json` CTC vocabulary (`failure_cause="ctc_vocab_oov"` when `oov_characters` is non-empty; otherwise `"ctc_alignment_failed"`).
4. **Partial Sub-Interval CTC Recovery (`partial_subinterval_recovery`) + Honest Degradation:**
   - Groups non-empty Sortformer `diarization` segments into bounded sub-windows via `group_diarization_windows`.
   - Re-submits each sub-window through `subinterval_runner` (preserving `capture_ctc_timestamps=True`) so neighboring non-OOV sentences regain exact word/character timestamps (`alignment_status="subinterval_ctc_recovered"`).
   - If only one sub-window fails due to `'钾'`, it is isolated cleanly; failing units retain `start=null, end=null` with `alignment_status="diarization_bounded_only"`, chunk-level `diarization_interval_count`, and explicit `oov_characters` without fabricating linear interpolation.
   - Exposes an `ExternalForcedAligner` protocol (`align(audio_path, text, language, source_offset_s)`) in `alignment.py` as a clean seam for future external CJK aligners.

## 3. Observed GPU Results (`benchmark.json`)

| Run / Case | Raw Text Preserved | CTC Error / OOV Chars | Status | Timed Units | Result Summary |
| --- | --- | --- | --- | --- | --- |
| French full episode (16 chunks) | Yes (100%) | `null` / `[]` | `ok` (16/16 chunks) | `2016 / 2016` | Zero alignment failures across 23 minutes |
| `zh_fleurs_01_2006` (8.28s) | `"钙、钾等元素属于金属，银和金等元素当然也是金属。"` | `"alignment_failed"` / `["钾"]` | `diarization_bounded_only` | `0 / 24` | Exact text preserved; `3` diarization intervals kept; no fake timestamps |
| `zh_2spk_overlap` (10.38s) | Dual-stream Mandarin text preserved | `"alignment_failed"` / `["诫", "轱"]` | `diarization_bounded_only` | `0 / 55` | Overlap split cleanly into `SPK_00` & `SPK_01` utterances with honest `null` bounds |
| `zh_multichunk_synthetic` Chunk `1` | All Chunk-1 sentences preserved | `null` / `[]` | `ok` | `87 / 87` | Clean CTC alignment after chunk boundary |
| `zh_multichunk_synthetic` Chunk `0` | All Chunk-0 sentences (incl. `'钾'`) preserved | `"alignment_failed"` / `["钾"]` | **`partial_subinterval_recovery`** | **`83 / 107`** | Re-inference recovered `utt-0001`–`utt-0003` (`83` units); isolated `[11.68, 17.84]` `'钾'` utterance (`utt-0000`, `24` units) as `diarization_bounded_only`; total recording coverage rose from **`87/194` to `170/194`** |
