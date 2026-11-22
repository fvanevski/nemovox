
# Real French ASR -> Riva CUDA -> English Review -> Subtitle/Export E2E

- **Candidate Server:** `127.0.0.1:8769` (isolated `jobs`, `uploads`, `exports` under `/tmp/nemovox-issue12-phasec`)
- **Input:** `/workspace/nvidia-transcribe/outputs/audio/overlap/fr_fleurs_00_1987.wav` (`10.2s`, `16kHz` mono `pcm_s16le`, French)
- **Job ID:** `dc322f312df744bebd6f794d765cf24c`
- **Options:** `task=subtitles`, `source_language=fr`, `target_language=en`, `translator=riva`, `translation_review=balanced`, `formats=[srt, ass]`
- **Export Plan:** destination `/tmp/nemovox-issue12-phasec/exports`, selected `["final_english", "srt"]`
- **Terminal State:** `complete`

## Recorded Stage History & Model Identities
- **ASR (`asr_local` / `nemovox-asr.service`):** `complete` (`90.60s` wall including cold start & release), model `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`
- **Primary Translation (`riva_local`):** `complete` (`10.80s` Studio stage; `11.061s` CLI translation; `device_used=cuda:0`, `dtype=torch.bfloat16`, `load_s=3.8704`, `infer_s=1.8026`, `peak_vram_mib=8013.55`, `cache_hits=0`)
- **English Review (`text_local` / `nemovox-text-review.service`):** `complete` (`8.40s` Studio stage; `7.81s` CLI review; `status=completed`, `profile=balanced`, `api_calls=1`, `cache_hits=0`)
- **Condensation:** `skipped` (`Optional condensation disabled`)
- **Render:** `complete` (`2` timed cues rendered to `.srt` and `.ass`)
- **Export:** `complete` (`2 saved`)

Structured `NEMOVOX_EVENT` lines emitted by the single `generate-subtitles` CLI invocation:
```
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "translation", "status": "running", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "translation", "status": "complete", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "review", "status": "running", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "review", "status": "complete", "detail": "completed"}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "condensation", "status": "skipped", "detail": "Optional condensation disabled"}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "render", "status": "running", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "render", "status": "complete", "detail": ""}
```

## Source Preservation & Deliverables Verification
- **Source French (`recording.contract.json` == `translated.contract.json`):** `L'accident a eu lieu en terrain montagneux, et il semblerait que cela ait été causé par un incendie malveillant.`
- **Riva CUDA MT (`machine_translated_text`):** `The accident occurred in mountainous terrain, and it appears to have been caused by arson.`
- **Reviewed English (`translated_text`, status `unchanged`):** `The accident occurred in mountainous terrain, and it appears to have been caused by arson.`
- **External `nemovox-dc322f312df744bebd6f794d765cf24c/` files:** `['final_english_subtitles.txt', 'fr_fleurs_00_1987.en.srt']` (exact SHA-256 match to internal originals: `final_english_subtitles.txt` `3d006157a105284b84b2cbc46b6e671a041a30cdf85c8f056c8dba19c4a6fef9`, `fr_fleurs_00_1987.en.srt` `4391edb830a0dd4e4091734833ef275b19dbf0d97f6f76d7e018d5560d95de58`)
- **Internal `asr/` retained:** `['chunks', 'manifest.json', 'recording.contract.json', 'transcript.txt']`
- **Internal `output/` retained:** `['final_english_subtitles.txt', 'fr_fleurs_00_1987.en.ass', 'fr_fleurs_00_1987.en.srt', 'raw_source_transcript.txt', 'refined.contract.json', 'refined_source_transcript.txt', 'subtitles.metadata.json', 'translated.contract.json', 'translation-review.checkpoint.json', 'translation.checkpoint.json']`
- **Chromium Browser Verification:** `Export status: complete · 2 saved`, `2 cues`, `Processing log (retained after completion)` with `Copy full log` -> `"Full log copied."`, preserved across `page.reload()` and `Browse all runs` reopen; `0` page errors, `0` console errors.
