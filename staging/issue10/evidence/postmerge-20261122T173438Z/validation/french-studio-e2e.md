# Phase 3: Fresh Production Studio French E2E Acceptance (postmerge-20261122T173438Z)

Verbatim transcript text is omitted per privacy rules.

- **Job ID:** `96a0b61881844253bc21c72a093bc32f` (uncached cold submission to `POST http://127.0.0.1:8765/api/jobs`)
- **Fixture:** `/workspace/nvidia-transcribe/outputs/audio/overlap/fr_fleurs_00_1987.wav`
- **Request Options:** `source_language=fr`, `target_language=en`, `asr_mode=speaker`, `translator=riva`, `translation_review=balanced`, `refinement=off`, `style=natural`, `formats=["srt","ass"]`
- **Final State:** `state="complete"`, `stage="Complete"`, `progress=100`, `wall_s=119.094`, `error=null`.

## Stage Transitions, PIDs & GPU Telemetry
1. `09:29:36` — `nemovox-asr.service` started automatically (`MainPID=4183661`, `EngineCore pid=4185349`), loading `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` (`quantization=modelopt_mixed`, `kv_cache_dtype=fp8`, `model-ctc-bf16.safetensors` enabled, `20.13 GiB` weights + `2.14 GiB` KV cache).
2. `09:31:03` — ASR `Application startup complete.` -> transcribed French audio -> `09:31:10..09:31:12` `nemovox-asr.service` stopped cleanly (`ActiveState=inactive`).
3. `t=97.92s..106.42s` — Uncached Riva worker (`PID=4194179`) ran under canonical lock while both `nemovox-asr.service` and `nemovox-text-review.service` were `inactive` (`MainPID=0`). GPU used peaked at `11662 MiB` (`20465 MiB free`), worker reported `device_used="cuda:0"`, `dtype="torch.bfloat16"`, `load_s=4.3333`, `infer_s=1.7548`, `peak_vram_mib=8013.55`, `cache_hits=0`.
4. `09:31:25` — `nemovox-text-review.service` started `ninfer-serve` (`PID=2007`, `weights 18.0 GiB`, `model chat`, `listening on http://127.0.0.1:8001`) -> `09:31:33` `req#1 done | total 777 ms` -> `09:31:33` stopped cleanly.

## Output Contracts, Artifacts, Preview & Browser Smoke
- `asr/recording.contract.json`: `timestamp_provenance.backend="nemotron_ctc"`, `timing_status="full"`, `unit_count=19`, `timed_unit_count=19`, `untimed_unit_count=0`, `invalid_unit_count=0`, `alignment_status="ctc_aligned"`, `speaker_id="SPK_00"`.
- `output/translated.contract.json`: `translation.device_used="cuda:0"`, `raw_source_preserved=True` (`raw_source_text == asr source_text`), `machine_mt_preserved=True` (`machine_translated_text` retained), `translation_review.status="completed"`, `reviewed=1`, `api_calls=1`, `errors=[]`.
- `output/subtitles.metadata.json`: `renderable_cues=2`, `ass_displayed_cues=2`, `ass_overflow_cues=0`, `unrenderable_cues=0`, `conflict_cues=0`.
- `GET /api/jobs/96a0b61881844253bc21c72a093bc32f/artifacts`: all `10` artifacts (`recording.contract.json`, `transcript.txt`, `refined.contract.json`, `translated.contract.json`, `raw_source_transcript.txt`, `refined_source_transcript.txt`, `final_english_subtitles.txt`, `subtitles.metadata.json`, `.srt`, `.ass`) returned HTTP `200`.
- `GET /api/jobs/96a0b61881844253bc21c72a093bc32f/preview`: `kind="subtitles"`, `cue_count=2`.
- Playwright Chromium (`http://127.0.0.1:8765/`): rendered `Complete`, Configure tab rendered `"ASR: startable"` and `"English review: startable"`, `0` console errors, `0` page errors.
