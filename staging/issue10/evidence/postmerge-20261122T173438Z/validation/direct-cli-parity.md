# Phase 3: Direct CLI Parity & No-Model-Start Checks (postmerge-20261122T173438Z)

## 1. Direct CLI Foreign-Language Parity (`fr_fleurs_01_1857.wav`)
- Executed `/workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.cli process-media` (`exit=0`) and `generate-subtitles` (`exit=0`) from `/workspace/nvidia-transcribe` in `114.425s` total.
- Journal transitions:
  - `09:31:39..09:33:11` — `nemovox-asr.service` started (`EngineCore pid=5412`, `quantization=modelopt_mixed`, `kv_cache_dtype=fp8`, `model-ctc-bf16.safetensors` enabled) -> stopped cleanly.
  - Riva worker (`device="cuda:0"`, `dtype="torch.bfloat16"`, `load_s=4.4486`, `infer_s=2.0955`, `peak_vram_mib=8014.75`, `cache_hits=0`).
  - `09:33:24..09:33:33` — `nemovox-text-review.service` (`ninfer-serve` `PID=16491`, `req#1 done | total 1.2s`) -> stopped cleanly.
- Contract verification: `timestamp_provenance.backend="nemotron_ctc"`, `timing_status="full"`, `unit_count=23`, `timed_unit_count=23`, `untimed_unit_count=0`, `speaker_id="SPK_00"`, `raw_source_preserved=True`, `translation_review.status="completed"`, `errors=[]`, `renderable_cues=2`, `ass_displayed_cues=2`, `conflict_cues=0`.

## 2. Offline / Rerender / Same-Language No-Model-Start Checks
- `generate-subtitles --re-render --formats srt,ass`: `exit=0`.
- Synthetic English `generate-subtitles --source-language en --target-language en --translator riva --translation-review balanced`: `exit=0`, `translation.status="bypassed_same_language"`, `translation_review.status="skipped_same_language"`.
- `journalctl --user` during both checks recorded zero `Starting`/`Started` events for `nemovox-asr.service` or `nemovox-text-review.service`.
