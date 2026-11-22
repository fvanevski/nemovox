# Phase C: Live Stage Transitions & F-01 Credential-Parity Proof (`20261122T063100Z`)

All E2E transitions executed autonomously with **no** `api_key_env` override on `asr_local` (while `VLLM_API_KEY` remained non-empty in both `systemd --user` and CLI/Studio environments) and **shared canonical lock** `~/.local/state/nemovox/model-lifecycle.lock`.

## 1. Read-Only Pre-Run Readiness (`GET http://127.0.0.1:8767/api/resource-status`)
```json
{"asr":{"state":"startable","ready":false,"managed":true,"startable":true},"translation_review":{"state":"startable","ready":false,"managed":true,"startable":true}}
```
- Both `nemovox-asr.service` and `nemovox-text-review.service` remained `inactive` (`MainPID=0`, `UnitFileState=static`).

## 2. Fresh Studio French E2E (`job_id=e184776289884fe2882b9d1d46e0e7d6`, `fr_fleurs_00_1987.wav`)
Options: `source_language=fr`, `target_language=en`, `asr_mode=speaker`, `translator=riva`, `translation_review=balanced`, `style=natural`, `formats=["srt","ass"]`.
Direct `journalctl --user` + Studio timeline sequence (`PST`, `UTC-08`):
1. `22:32:21` — `nemovox-asr.service` started (`Starting Nemovox explicitly managed ASR model...`).
2. `22:32:41` — vLLM V1 engine initialized `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` (`quantization=modelopt_mixed`, `kv_cache_dtype=fp8`).
3. `22:33:43` — `Application startup complete.`; authenticated `/v1/models` passed via `VLLM_API_KEY` fallback (`ready_managed`, `GPU 26875..27339 MiB`), CTC ASR ran (`timing_status="full"`, `19` words, `SPK_00`).
4. `22:33:50` -> `22:33:52` — `nemovox-asr.service` stopped cleanly (`Consumed 1min 51.838s CPU time over 1min 31.094s wall clock time`).
5. Riva CUDA BF16 worker ran (`device_used="cuda:0"`, `load_s=4.0443`, `infer_s=1.8599`, `elapsed_s=11.43`, `peak_vram_mib=8013.55`).
6. `22:34:04` — `nemovox-text-review.service` started `ninfer-serve` -> `22:34:11` listening `:8001` (`auth disabled`, `model chat`, `GPU 30617 MiB used`) -> `22:34:12` `req#1 done | total 771 ms`.
7. `22:34:12` -> `22:34:13` — `nemovox-text-review.service` stopped (`8.489s` wall time).
8. Job completed `state="complete"`, `progress=100` (`total_wall_time_s=112.279s`); both units returned `inactive`, `resource_status` returned `startable`, GPU returned to `3048 MiB used / 29079 MiB free`.

## 3. Direct CLI Parity (`fr_fleurs_01_1857.wav`, Without `api_key_env`)
- `22:34:28` -> `22:35:49` -> `22:35:57`/`22:35:59`: CLI `process-media` automatically started `nemovox-asr.service`, passed authenticated `/v1/models` readiness via `VLLM_API_KEY`, transcribed (`timing_status="full"`, `23` words, `SPK_00`), and stopped `nemovox-asr.service` (`1min 31.203s` wall).
- `22:36:12` -> `22:36:19` -> `22:36:20`: CLI `generate-subtitles` ran Riva CUDA MT, started `nemovox-text-review.service`, served `req#1 done | total 1.2s`, and stopped (`8.614s` wall).

## 4. Offline / Cached / Same-Language No-Model-Start Checks
- Cached resume (`reused_translated_contract`, `total_wall_time_s=0.001`), `--re-render`, and synthetic English-source contract (`translation.status="bypassed_same_language"`, `translation_review.status="skipped_same_language"`, `reviewed=0`) triggered zero `nemovox-asr.service` or `nemovox-text-review.service` journal activity.
