# Phase C: Studio UI & CLI E2E Parity (`20261122T063100Z`)

Verbatim transcript strings are omitted per privacy rules.

## 1. Studio E2E (`job_id=e184776289884fe2882b9d1d46e0e7d6`, `fr_fleurs_00_1987.wav`)
- **UI & API verification (headless Chromium on `127.0.0.1:8767`):**
  - Configure step `role="status"` rendered `ASR: startable` and `English review: startable`.
  - Active job transitioned cleanly to `state="complete"` (`112.279s`); UI displayed `Complete`, `2` subtitle cues, `0` browser console errors, `0` uncaught page errors, and all `10` downloadable artifacts returning HTTP `200`.
- **Provenance & contract checks:**
  - `asr/recording.contract.json`: `timestamp_provenance.backend="nemotron_ctc"`, `timing_status="full"`, `unit_count=19`, `timed_unit_count=19`, `untimed_unit_count=0`, `unit_statuses=["ctc_aligned"]`, `len(source_words)=19`, `speaker="SPK_00"`.
  - `output/translated.contract.json`: `raw_source_text == asr_u0["source_text"]` (`True`), `refined_text == asr_u0["source_text"]` (`True`), `review.status="completed"`, `review.reviewed=1`, `review.api_calls=1`, `review.errors=[]`.
  - `output/subtitles.metadata.json`: `renderable_cues=2`, `ass_displayed_cues=2`, `ass_overflow_cues=0`, `conflict_cues=0`.

## 2. Direct CLI Parity (`fr_fleurs_01_1857.wav`)
- `process-media` + `generate-subtitles` produced identical autonomous lifecycle transitions without `api_key_env` on `asr_local`: `timing_status="full"`, `unit_count=23`, `timed_unit_count=23`, `unit_statuses=["ctc_aligned"]`, `len(source_words)=23`, `speaker="SPK_00"`, `raw_source_text == asr_u0["source_text"]` (`True`), `refined_text == asr_u0["source_text"]` (`True`), `renderable_cues=2`, `ass_displayed_cues=2`, `conflict_cues=0`.
