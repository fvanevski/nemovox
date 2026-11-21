# Test Results & Reproducibility Verification

## 1. CPU Unit & Regression Tests

### A. Existing Production Core Suite (`-m "not gpu"`)
Command: `.venv/bin/pytest -v -m "not gpu"` (logged to `outputs/modular-eval/logs/pytest_prod.txt`)
- Exit code: **`0`**
- Result: **`30 passed, 1 warning in 0.65s`** (confirming production code and existing tests remain unbroken).

### B. Modular Experimental Evaluation Suite
Command: `PYTHONPATH=. outputs/modular-eval/.venv/bin/pytest -v experiments/modular/tests/test_modular_eval.py` (logged to `outputs/modular-eval/logs/pytest_modular.txt`)
- Exit code: **`0`**
- Result: **`8 passed in 0.31s`**:
  1. `test_source_language_normalization_never_defaults_to_french` — PASSED
  2. `test_french_and_chinese_unicode_preservation` — PASSED
  3. `test_alignment_validation_detects_zero_duration_inversions_and_bounds` — PASSED
  4. `test_alignment_validation_detects_repeated_spans_and_missing_chars` — PASSED
  5. `test_global_timestamp_offsets_and_unknown_speaker_preserved` — PASSED
  6. `test_overlap_der_and_attributed_overlap_metrics` — PASSED
  7. `test_non_destructive_wer_cer_and_json_roundtrip` — PASSED
  8. `test_integration_attribution_invariants` — PASSED

## 2. Live GPU Experiments Executed on RTX 5090 (`32607 MiB`)

| Experiment Script / Target | Environment | Exit Code | Key Verified Outputs & Telemetry |
| --- | --- | ---: | --- |
| `experiments/modular/asr/run_asr_eval.py` | `.venv` (`qwen-asr==0.0.6`, `transformers==4.57.6`) | `0` | `30` French & Mandarin cases (`F1..F7`, `F_100s`, `F_300s`, `fr_fleurs_00..03`, `zh_fleurs_00..05`, `O1..O4`, `S1..S3`, `fr/zh_2spk_overlap`, `zh_2spk_seq`, `zh_multichunk`); load `1.9331s` (`4480.79 MiB` peak) |
| `experiments/modular/alignment/run_alignment_eval.py` | `.venv` (`Qwen3-ForcedAligner-0.6B`, `bfloat16`) | `0` | `47` alignment runs (`B1/B2/B3`, `'钾'`, `175s`, `300s` direct, `2x150s` hybrid); load `1.0584s`, `2644` items, `331` zero-durations (`31` runs), `135` repeated spans, `12` past-dur (`zh_2spk_overlap`), `0` errors |
| `Qwen3ASRModel` + `Qwen3ForcedAligner` combined `F_300s` (`<=180s` chunking) | `.venv` (`bfloat16`) | `0` | Load `4.0985s`, run `20.1494s`, peak **`6989.85 MiB`**, `497` items (`426` valid, `70` zero-dur, `8` repeated spans) |
| `experiments/modular/diarization/run_diarization_eval.py` | `.venv-diar` (`transformers==5.23.0`, `float32`) | `0` | Load `0.324s` (`383.16 MiB`), short controls + `F_300s` (`5.97s` stream / `0.52s` offline) + **`F_full` `1379.9s`** (`27.4232s` stream `419.33 MiB` `7` spk / `1.4602s` offline `579.10 MiB` `6` spk) |
| `experiments/modular/integration/run_integration_probe.py` | `.venv` | `0` | Assembled & audited `D1_direct300s`, `D1_chunked180s`, `D2_hybrid` contracts over `F_300s` (`492`/`497`/`525` words) |
| `experiments/modular/build_benchmark.py` | `.venv` | `0` | Recomputed all WER/CER/DER/alignment summaries into `evaluations/modular-bakeoff/benchmark.json` |

*Note:* Initial `run_diarization_eval.py` run hit `TypeError: Object of type dtype is not JSON serializable` on final config serialization after completing GPU inference; fixed via `streaming_config.to_dict()` dtype stringification and re-ran cleanly (`exit 0`). Initial `run_alignment_eval.py` `F_175s` test hit `KeyError: 'text'` and local vs global offset alignment (`<=325` instead of `<=175`) on `ablation_chunk_90s`; fixed and re-ran cleanly (`exit 0`). GPU returned to idle (`3369 MiB / 32607 MiB`) after all runs.
