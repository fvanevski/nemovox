# Qwen3 ForcedAligner Simultaneous Coexistence Results (Lane A: TEI ON)

Raw telemetry: [`logs/coexistence_results.json`](logs/coexistence_results.json).

## Provenance & Workload

- **Lane:** `lane_a_tei_on` (`COEXIST_PASS`)
- **Provenance:** **`LIVE_ASR_AND_QWEN_RESIDENT`** — Real end-to-end `process_media` (`two_span_oov.wav`, `18.56s`, two 10s chunks) sent live `/v1/chat/completions` requests to the resident Nemotron ASR server on `127.0.0.1:8190`, received real CTC `alignment_failed` (`ctc_vocab_oov` on `'钾'`), wrote `native-alignment.checkpoint.json` (`23,202` bytes), and launched the real `QwenForcedAlignerAdapter` (`--qwen-gpu-mode available --qwen-policy subtitle --qwen-device cuda:0`) **while Nemotron ASR (`24,406 MiB`) and `tei-qwen3-embedding` (`1,758 MiB`) remained GPU-resident**.
- **Admission Threshold:** Pre-Qwen free VRAM with ASR + TEI resident was `4,637 MiB` (`~4.86 GB` decimal / `~4.53 GiB`), exceeding `_qwen_align_worker.py`'s `2,600,000,000` byte threshold (`~2,479.55 MiB`).

## Qwen Worker Metrics & Alignment Output

- **Device & Dtype Observed:** `cuda:0`, `torch.bfloat16` (requested `torch.bfloat16`, zero silent CPU fallback)
- **Session Reuse:** `model_load_count=1`, `align_count=2` (single worker PID `1235339`, resident footprint `2,564 MiB` in `nvidia-smi`)
- **Timings:**
  - Total pipeline wall (`process_media` live ASR + Qwen): `12.2957s`
  - Qwen worker startup wall: `7.8142s` (`model_load_wall_s=5.9827s`)
  - Span #1 (`[0.77, 6.99]`): `1.9088s`, `21/21` valid units, `'钾'` `[1.41, 1.65]`
  - Span #2 (`[11.05, 17.27]`): **`0.0329s`** (58x warm speedup), `21/21` valid units, `'钾'` `[11.69, 11.93]`
- **PyTorch Memory:** `cuda_allocated_bytes=1,841,616,896 B`, `peak_gpu_allocated_bytes=1,900,171,776 B`
- **Validation & Coverage:** `42/42` units recovered (`utt-0000` `[1.17, 6.61]`, `utt-0001` `[11.45, 16.89]`, `external_qwen_recovered`), `0` zero-duration, `0` invalid bounds, `0` non-monotonic units.
- **Post-Qwen Cleanup & ASR Check:** Qwen worker exited cleanly (free VRAM returned from `2,066 MiB` to `4,635 MiB`), `/v1/models` returned `HTTP 200`, and post-Qwen ASR CTC inference succeeded in `0.2215s` (`23/23` CTC words, `ctc_error=null`).
