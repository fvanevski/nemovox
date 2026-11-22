
# Short French CUDA Smoke E2E (`231c643546df46f48c182001fe258e5d`)

- **Input:** `/workspace/nvidia-transcribe/outputs/audio/overlap/fr_fleurs_00_1987.wav` (`10.2s`, `fr -> en`, `task=subtitles`, `translator=riva`, `translation_review=balanced`, `formats=[srt, ass]`)
- **Selected Exports:** `["final_english", "srt"]`
- **Terminal State:** `complete`
- **UI Stage Panel:** `['complete\nasr', 'complete\nPrimary translation', 'complete\nEnglish review\ncompleted', 'skipped\ncondensation\nOptional condensation disabled', 'complete\nRender subtitles', 'complete\nexport\ncomplete']`
- **Translation (`riva_local`):** `device_used=cuda:0`, `dtype=torch.bfloat16`, `load_s=4.2343`, `infer_s=1.9665`, `peak_vram_mib=8013.55`, `cache_hits=0`
- **English Review (`text_local`):** `status=completed`, `profile=balanced`, `api_calls=1`, `cache_hits=0`
- **Subtitles:** `renderable_cues=2`, `ass_displayed_cues=2`
- **External `nemovox-231c643546df46f48c182001fe258e5d/`:** `['final_english_subtitles.txt', 'fr_fleurs_00_1987.en.srt']` (exact SHA-256 match: `final_english_subtitles.txt` `3d006157a105284b84b2cbc46b6e671a041a30cdf85c8f056c8dba19c4a6fef9`, `fr_fleurs_00_1987.en.srt` `4391edb830a0dd4e4091734833ef275b19dbf0d97f6f76d7e018d5560d95de58`)
- **Internal `asr/` & `output/` retained:** `asr=['chunks', 'manifest.json', 'recording.contract.json', 'transcript.txt']`, `output=['final_english_subtitles.txt', 'fr_fleurs_00_1987.en.ass', 'fr_fleurs_00_1987.en.srt', 'raw_source_transcript.txt', 'refined.contract.json', 'refined_source_transcript.txt', 'subtitles.metadata.json', 'translated.contract.json', 'translation-review.checkpoint.json', 'translation.checkpoint.json']`
- **Source & MT preservation:**
  - `source_text`: `L'accident a eu lieu en terrain montagneux, et il semblerait que cela ait été causé par un incendie malveillant.`
  - `machine_translated_text`: `The accident occurred in mountainous terrain, and it appears to have been caused by arson.`
  - `translated_text`: `The accident occurred in mountainous terrain, and it appears to have been caused by arson.`

Structured `NEMOVOX_EVENT` lines:
```
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "translation", "status": "running", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "translation", "status": "complete", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "review", "status": "running", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "review", "status": "complete", "detail": "completed"}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "condensation", "status": "skipped", "detail": "Optional condensation disabled"}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "render", "status": "running", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "render", "status": "complete", "detail": ""}
```
