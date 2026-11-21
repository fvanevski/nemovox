# Nemovox Issue #1 Loaded-ASR + Qwen Coexistence Handback

## Consolidated Verdicts

```text
SOURCE_SHA=7b4112cc31784953abbb041fd0b5d7997851841d
GPU_UUID=GPU-c75b2559-b8e1-ebf9-fe82-65cd2b3ca4c0
ASR_SOURCE=AGENT_OWNED
ASR_MODEL_ID=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4
ASR_LOADED_AND_WARMED=PASS
ASR_CTC_PRE_QWEN=PASS
ASR_CTC_POST_QWEN=PASS
LANE_A_TEI_ON=COEXIST_PASS
LANE_B_TEI_OFF=NOT_NEEDED
QWEN_DEVICE_OBSERVED=cuda:0
QWEN_DTYPE_OBSERVED=torch.bfloat16
QWEN_MODEL_LOADS=1
QWEN_ALIGN_CALLS=2
GPU_FREE_MIB_WITH_ASR=4637
GPU_FREE_MIB_MIN_DURING_QWEN=2066
GPU_FREE_MIB_GAIN_FROM_TEI_STOP=1734 (measured in earlier diagnostic TEI stop/restore probe; TEI stop NOT_NEEDED in authoritative Lane A pass)
TEI_CONTAINER_ID_PREFIX=64c86ba8a443
TEI_ORIGINAL_STATE=running (healthy)
TEI_FINAL_STATE=running (healthy)
TEI_RESTORE=NOT_NEEDED (PASS in earlier diagnostic probe)
ASR_FINAL_STATE_AND_OWNER=STOPPED_AND_PORT_8190_CLOSED (test-owned PGID 1224121 terminated cleanly)
STUDIO_AND_OTHER_SERVICES=UNCHANGED
OVERALL_COEXISTENCE=COEXIST_PASS_TEI_ON
LIVE_MANAGED_HANDOFF=NOT_TESTED
ISSUE1_RECOMMENDATION=CLOSE_WITH_DOCUMENTED_OPTIONAL_MODE_LIMITATION
ARCHIVE_PATH=staging/agent-evidence/issue-1-coexistence/nemovox-issue-1-asr-qwen-coexistence-evidence.tar.gz
```

## Timeline (`2026-11-21` UTC, Authoritative Run)

1. **Baseline (`22:56:14Z`):** RTX 5090 `29,045 MiB` free (`3,082 MiB` used by `kwin_wayland` `329 MiB` + `tei-qwen3-embedding` `1,758 MiB`); port `8190` closed; Studio idle; `nemovox-studio.service` and `local_embedding.service` `active`.
2. **ASR Load & Warmup (`22:56:14Z` -> `22:57:48Z`):** Test-owned `scripts/start_vllm_server.sh` (`PID 1224121`, `EngineCore PID 1225862`) ready in `85.56s` (`modelopt_mixed`, `ctc_timestamp_middleware`, `/v1/models` `200 OK`). Warmup `zh_fleurs_00_1906.wav` (`7.5143s`, `23` CTC words, `ctc_error=null`) + OOV `zh_fleurs_01_2006.wav` (`0.7158s`, `ctc_error="alignment_failed"`). Free VRAM with ASR+TEI resident: **`4,637 MiB`**.
3. **Lane A Coexistence (`22:57:48Z` -> `22:58:00Z`):** Real end-to-end `process_media` (`two_span_oov.wav`, `18.56s`, `LIVE_ASR_AND_QWEN_RESIDENT`, `--qwen-gpu-mode available --qwen-policy subtitle --qwen-device cuda:0`) admitted Qwen worker (`PID 1235339`, `2,564 MiB`), loaded `cuda:0` `torch.bfloat16` once (`model_load_count=1, align_count=2`), aligned both spans (`1.9088s` -> `0.0329s`, `42/42` units recovered, `'钾'` `[1.41, 1.65]` & `[11.69, 11.93]`, peak used `30,061 MiB`, min free **`2,066 MiB`**).
4. **Qwen Teardown & Post-Qwen ASR Check (`22:58:00Z`):** Qwen worker exited (`4,635 MiB` free), `/v1/models` `200 OK`, post-Qwen CTC `zh_fleurs_00_1906.wav` passed (`0.2215s`, `23` CTC words, `ctc_error=null`).
5. **Conditional Lane B (`NOT_NEEDED`):** Skipped in authoritative run because Lane A passed (`COEXIST_PASS`). *(Earlier diagnostic probe recorded in [`logs/diagnostic_lane_b_tei_check_sanitized.txt`](logs/diagnostic_lane_b_tei_check_sanitized.txt) also verified `+1,734 MiB` reclaimed from stopping TEI `64c86ba8a443`, `4,411 MiB` min free during Qwen, and clean `docker start` health restoration in `20.36s`.)*
6. **Test-Owned ASR Teardown & Final Check (`22:58:03Z` -> `22:58:10Z`):** Stopped `PID 1224121`/`1225862`, `port 8190` closed, `tei-qwen3-embedding` (`64c86ba8a443`) `running` (`healthy`, `HTTP 200`), `nemovox-studio.service` and `local_embedding.service` `active`, final RTX 5090 `29,047 MiB` free (`3,080 MiB` used).

## Reports & Logs

- [`source-and-runtime.md`](source-and-runtime.md)
- [`asr-and-gpu-samples.md`](asr-and-gpu-samples.md)
- [`qwen-results.md`](qwen-results.md)
- [`tei-lifecycle.md`](tei-lifecycle.md)
- [`coexistence-decision.md`](coexistence-decision.md)
- [`manifest.json`](manifest.json) & [`SHA256SUMS.txt`](SHA256SUMS.txt)
- [`logs/coexistence_results.json`](logs/coexistence_results.json)

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events: []
  omitted_sensitive_values=no
```
