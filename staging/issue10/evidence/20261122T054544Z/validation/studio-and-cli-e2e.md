# Phase C: Studio UI & CLI End-to-End Parity (`20261122T054544Z`)

Note: Verbatim clip transcripts are omitted from public GitHub evidence per `staging/issue10/HOST_VERIFY.md`; structural, provenance, and behavioral checks are reported below.

## 1. Studio E2E Verification (`http://127.0.0.1:8767`, Job `0b4da9e0eb6148c1aefabc9c7f2b846a`)
- **Input:** `fr_fleurs_00_1987.wav` (`duration_s=10.2`, `fr -> en`, `asr_mode=speaker`, `translator=riva`, `translation_review=balanced`, `formats=["srt","ass"]`).
- **Browser UI checks (headless Chromium):**
  - Configure view displayed `role="status"` resource box showing `ASR: startable` and `English review: startable`.
  - During active runs, recent jobs and live progress reflected running stage without spurious failures.
  - Completed preview (`/?job=0b4da9e0eb6148c1aefabc9c7f2b846a&step=preview`) rendered `ASS · libass` indicator, `Complete` / `100%`, `2` subtitle cues, and all `10` download links (`/api/jobs/0b4da9e0eb6148c1aefabc9c7f2b846a/artifact/...`), with `0` browser console errors and `0` uncaught page errors.
  - HTTP `GET` verified downloads for `fr_fleurs_00_1987.en.srt` (`renderable_cues=2`), `fr_fleurs_00_1987.en.ass` (`ass_displayed_cues=2`, `ass_overflow_cues=0`, `conflict_cues=0`), `raw_source_transcript.txt`, and `subtitles.metadata.json`.
- **Contract provenance & immutability:**
  - `asr/recording.contract.json`: `model_path="/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"`, `utterance_count=1`, `speaker_count=1` (`SPK_00`), `source_start=2.112`, `source_end=8.672`, `timestamp_provenance.backend="nemotron_ctc"`, `timing_status="full"`, `unit_count=19`, `timed_unit_count=19`, `untimed_unit_count=0`, `unit_statuses=["ctc_aligned"]`, `len(source_words)=19`.
  - `output/translated.contract.json`: `raw_source_text == asr_u0["source_text"]` (`True`), `refined_text == asr_u0["source_text"]` (`True`), `machine_translated_text` preserved intact, `translation_review_status="unchanged"`, `api_calls=1`, `errors=[]`.

## 2. Direct CLI Parity (`fr_fleurs_01_1857.wav`, `duration_s=8.28`)
- `process-media ... --source-language fr`: `rc=0`, `total_wall_time_s=87.9998`, `utterance_count=1` (`SPK_00`, `0.76 -> 7.615`), `timing_status="full"`, `timed_unit_count=23`, `unit_statuses=["ctc_aligned"]`.
- `generate-subtitles --recording-contract ... --source-language fr --target-language en --translator riva --translation-review balanced --formats srt,ass`: `rc=0`, `total_wall_time_s=19.947`, `translation.backend="riva"` (`device_used="cuda:0"`, `peak_vram_mib=8014.75`), `translation_review.status="completed"` (`reviewed=1`, `api_calls=1`, `errors=[]`), `renderable_cues=2`, `ass_displayed_cues=2`, `conflict_cues=0`.
- `output/translated.contract.json` preserved `raw_source_text == asr_u0["source_text"]` (`True`) and `machine_translated_text` verbatim.
