# Phase B: Candidate Studio & Direct CLI Foreign-Language E2E (20261122T170806Z)

Verbatim transcript strings are omitted per privacy rules.

## 1. Candidate Studio E2E (`127.0.0.1:8767`, `job_id=ef867313f6d14cbdabb1ffedba7d2e3e`)
- **Fixture:** `/workspace/nvidia-transcribe/outputs/audio/overlap/fr_fleurs_00_1987.wav` (`source_language=fr`, `target_language=en`, `asr_mode=speaker`, `translator=riva`, `translation_review=balanced`, `style=natural`, `formats=["srt","ass"]`).
- **Riva binding:** `GET /api/translation-backends` returned `{"riva": {"configured": true, "device": "auto", ...}}`.
- **Pre-run readiness:** `GET /api/resource-status` returned `asr.state="startable"`, `translation_review.state="startable"`.
- **Stage transitions (`journalctl --user` + sampler):**
  1. `09:03:34` — `nemovox-asr.service` started Nemotron 30B-A4B NVFP4 (`quantization=modelopt_mixed`, `kv_cache_dtype=fp8`, `model-ctc-bf16.safetensors` enabled); `09:04:56` `Application startup complete.` -> `09:05:05..09:05:07` stopped cleanly.
  2. `t=93.64s..104.29s` — Riva `device=auto` subprocess ran (`device_used="cuda:0"`, `peak_vram_mib=8013.55`) with both systemd units `inactive`.
  3. `09:05:21` — `nemovox-text-review.service` started `ninfer-serve` (`weights 18.0 GiB`, `model chat`, `listening on http://127.0.0.1:8001`) -> `09:05:29` `req#1 done | total 804 ms` -> `09:05:29` stopped cleanly.
- **Job & UI checks:**
  - Job reached `state="complete"`, `progress=100` in `117.014s`.
  - All `10` downloadable artifacts returned HTTP `200`; `/preview` returned `kind="subtitles"`, `cue_count=2`.
  - Headless Chromium (`/usr/bin/chromium`) loaded `http://127.0.0.1:8767/` (`assets/index-62Ntthch.js`), observed `Complete` UI state, `0` console errors, and `0` page errors.
- **Contract checks:**
  - `asr/recording.contract.json`: `timestamp_provenance.backend="nemotron_ctc"`, `timing_status="full"`, `unit_count=19`, `timed_unit_count=19`, `untimed_unit_count=0`, `unit_statuses=["ctc_aligned"]`, `speaker_id="SPK_00"`.
  - `output/translated.contract.json`: `translation.device_used="cuda:0"`, `raw_source_text == asr_u0["source_text"]` (`True`), `translation_review.status="completed"`, `reviewed=1`, `api_calls=1`, `errors=[]`.
  - `output/subtitles.metadata.json`: `renderable_cues=2`, `ass_displayed_cues=2`, `ass_overflow_cues=0`, `conflict_cues=0`.

## 2. Direct CLI Parity (`fr_fleurs_01_1857.wav`)
- Ran `process-media` (`exit=0`) + `generate-subtitles` (`exit=0`) in `114.547s` total against the same `NEMOVOX_MODEL_RESOURCES` (`riva_local.device="auto"`), `NEMOVOX_LIFECYCLE_CONFIG`, and canonical lock.
- Journal transitions: `nemovox-asr.service` `09:05:35..09:07:08` (`modelopt_mixed`, `fp8` KV, CTC timestamps enabled) -> Riva `device=auto` CUDA (`device_used="cuda:0"`, `peak_vram_mib=8014.75`) -> `nemovox-text-review.service` `09:07:20..09:07:29` (`req#1 done | total 1.2s`).
- Contracts: `timing_status="full"`, `unit_count=23`, `timed_unit_count=23`, `unit_statuses=["ctc_aligned"]`, `speaker_id="SPK_00"`, `raw_source_preserved=True`, `translation_review.status="completed"`, `renderable_cues=2`, `ass_displayed_cues=2`, `conflict_cues=0`.
