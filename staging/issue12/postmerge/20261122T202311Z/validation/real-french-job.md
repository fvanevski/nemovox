
# Phase B3 Live Production French Media Acceptance (`d55d30020bbd448ba36f4dda58f9bc1a`)

- **Clip:** `fr_fleurs_00_1987.wav` (`10.2s`, `fr -> en`, `task=subtitles`, `translator=riva`, `translation_review=balanced`, `formats=[srt, ass]`)
- **Selected Automatic Exports:** `["final_english", "srt"]`
- **Terminal State:** `complete`
- **UI Stage History:** `['complete\nasr', 'complete\nPrimary translation', 'complete\nEnglish review\ncompleted', 'skipped\ncondensation\nOptional condensation disabled', 'complete\nRender subtitles', 'complete\nexport\ncomplete']`
- **Model Resource Lifecycle & Telemetry:**
  - `asr`: `complete` (`nemovox-asr.service` acquired and released cleanly)
  - `translation`: `complete` (`riva_local`, `device_used=cuda:0`, `dtype=torch.bfloat16`, `load_s=4.2196`, `infer_s=1.8405`, `peak_vram_mib=8013.55`, `cache_hits=0`)
  - `review`: `complete` (`text_local` / `nemovox-text-review.service` acquired and released cleanly, `status=completed`, `profile=balanced`, `api_calls=1`, `cache_hits=0`)
  - `condensation`: `skipped` (`Optional condensation disabled`)
  - `render`: `complete` (`renderable_cues=2`, `ass_displayed_cues=2`)
  - `export`: `complete` (`2 saved`)
- **External `nemovox-d55d30020bbd448ba36f4dda58f9bc1a/` files:** `['final_english_subtitles.txt', 'fr_fleurs_00_1987.en.srt']`
  - `final_english_subtitles.txt`: `sha256=3d006157a105284b84b2cbc46b6e671a041a30cdf85c8f056c8dba19c4a6fef9` (matched internal)
  - `fr_fleurs_00_1987.en.srt`: `sha256=4391edb830a0dd4e4091734833ef275b19dbf0d97f6f76d7e018d5560d95de58` (matched internal)
- **Internal Archive Retained:** `asr=['chunks', 'manifest.json', 'recording.contract.json', 'transcript.txt']`, `output=['final_english_subtitles.txt', 'fr_fleurs_00_1987.en.ass', 'fr_fleurs_00_1987.en.srt', 'raw_source_transcript.txt', 'refined.contract.json', 'refined_source_transcript.txt', 'subtitles.metadata.json', 'translated.contract.json', 'translation-review.checkpoint.json', 'translation.checkpoint.json']`
- **Source & MT Preservation:**
  - `source_text`: `L'accident a eu lieu en terrain montagneux, et il semblerait que cela ait été causé par un incendie malveillant.`
  - `machine_translated_text`: `The accident occurred in mountainous terrain, and it appears to have been caused by arson.`
  - `translated_text`: `The accident occurred in mountainous terrain, and it appears to have been caused by arson.`
- **Post-Run Model Units & VRAM:** `nemovox-asr.service=inactive (dead)`, `nemovox-text-review.service=inactive (dead)`, GPU before `NVIDIA GeForce RTX 5090, 3105 MiB, 29022 MiB, 32607 MiB` -> after `NVIDIA GeForce RTX 5090, 3105 MiB, 29022 MiB, 32607 MiB`, `nemovox-studio.service` unchanged (`MainPID=1121444`).

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
