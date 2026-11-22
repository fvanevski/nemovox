# Phase C: Live Stage Transitions & Automatic Ownership Proof (`20261122T054544Z`)

All GPU transitions occurred autonomously without any manual `systemctl --user start/stop`, manual model launch, or unrelated service eviction during jobs.

## 1. Read-Only Pre-Run Readiness (`127.0.0.1:8767/api/resource-status`)
```json
{"asr":{"state":"startable","ready":false,"managed":true,"startable":true},"translation_review":{"state":"startable","ready":false,"managed":true,"startable":true}}
```
- Verified both `nemovox-asr.service` and `nemovox-text-review.service` remained `inactive` (`MainPID=0`, `UnitFileState=static`).

## 2. Studio French Subtitles E2E (`job_id=0b4da9e0eb6148c1aefabc9c7f2b846a`, `fr_fleurs_00_1987.wav`)
Options: `source_language=fr`, `target_language=en`, `asr_mode=speaker`, `translator=riva`, `translation_review=balanced`, `style=natural`, `formats=["srt","ass"]`.
Direct `journalctl --user` & `execution.log` sequence (local host time `PST`, `UTC-08`):
1. `21:56:02` — `nemovox-asr.service` started (`Starting Nemovox explicitly managed ASR model...`).
2. `21:56:22` — vLLM V1 engine initialized `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` (`quantization=modelopt_mixed`, `kv_cache_dtype=fp8`, `FP8 audio encoder: 128 Linears`).
3. `21:57:20` — `Application startup complete.` -> CTC ASR completed (`total_wall_time_s=86.1227`, `chunk_request_latencies_s=[7.37]`, `timing_status="full"`, 19 `source_words`, `SPK_00`).
4. `21:57:28` -> `21:57:30` — `nemovox-asr.service` automatically stopped (`Stopped Nemovox explicitly managed ASR model`).
5. Riva CUDA BF16 worker translated `fr -> en` (`device_used="cuda:0"`, `load_s=4.0194`, `infer_s=1.8388`, `elapsed_s=11.361`, `peak_vram_mib=8013.55`) and exited.
6. `21:57:42` — `nemovox-text-review.service` started `ninfer-serve` (`qwen3_8_27b_swift15abl_nvfp4full-dflash2`, `weights 18.0 GiB`).
7. `21:57:49` — Listening `http://127.0.0.1:8001 | model chat`; `21:57:50` — `req#1 done | total 769 ms`.
8. `21:57:50` — `nemovox-text-review.service` automatically stopped (`Stopped Nemovox explicitly managed English translation review text model`, `8.105s` wall time).
9. Subtitles rendered (`srt`, `ass`, `renderable_cues=2`); job finalized `state="complete"`, `progress=100` (`total_wall_time_s=108.613s`).

## 3. Direct CLI Parity (`fr_fleurs_01_1857.wav`, 1.0s / 0.5s Polling)
- `process-media` (`rc=0`, `90.62s` outer wall):
  - `dt=0.06s`: both units `inactive` (`MainPID=0`, `3048 MiB` GPU used).
  - `dt=1.12s` (`22:01:05`): `nemovox-asr.service` `active/running` (`MainPID=3974741`).
  - `22:02:25`: vLLM startup complete (`quantization=modelopt_mixed`, `kv_cache_dtype=fp8`).
  - `dt=88.57s` (`22:02:33` -> `22:02:35`): `nemovox-asr.service` `deactivating/stop-sigterm` -> `inactive/dead` (`MainPID=0`).
- `generate-subtitles` (`rc=0`, `20.6s` outer wall):
  - `dt=0.05s`: both units `inactive`.
  - `dt=5.62s..11.21s`: Riva CUDA worker ran while both systemd units remained `inactive` (`11533..11781 MiB` total GPU used).
  - `dt=11.75s..19.55s` (`22:02:47` -> `22:02:55`): `nemovox-text-review.service` `active/running` (`MainPID=3986007`), `listening on :8001` (`22:02:54`), `req#1 done` (`22:02:55`, `1.2s`).
  - `dt=20.1s` (`22:02:55`): `nemovox-text-review.service` `inactive/dead` (`MainPID=0`), GPU returned to `3048 MiB used / 29079 MiB free`.

## 4. Timeout Cleanup & Actionable Studio Error (`job_id=4d0c3f21f3fc495bab3ea3babcf693af`)
- Initial unconfigured `asr_local` run (missing `api_key_env` while `VLLM_API_KEY` was exported in `systemd --user`) timed out cleanly at 300s (`managed_start_timeout`).
- Confirmed `ModelLifecycle.endpoint()` `finally:` block automatically stopped `nemovox-asr.service` back to `inactive` (`ExecMainStatus=0`), freed VRAM back to `3046 MiB used / 29081 MiB free`, and Studio surfaced: `Transcribing audio and speakers blocked by model resource (managed_start_timeout); see execution.log`.

## 5. Offline / Cached / Same-Language No-Model-Start Verification
Journal window `>= 2026-11-21 22:02:56` recorded `0` lines across `nemovox-asr.service` and `nemovox-text-review.service` during all three runs:
1. Cached CLI `generate-subtitles` resume (`rc=0`): `translation.cache_hits=1` (`elapsed_s=0.0`), `translation_review.cache_hits=1`, `api_calls=0` (`elapsed_s=0.0`).
2. Offline CLI `generate-subtitles --re-render` (`rc=0`).
3. Synthetic English-source contract (`--source-language en --target-language en --translator riva --translation-review balanced`, `rc=0`): `translation.status="bypassed_same_language"`, `translation_review.status="skipped_same_language"`, `reviewed=0`.
