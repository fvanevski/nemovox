# Forced Alignment Comparison: `Qwen3-ForcedAligner-0.6B` vs Nemotron CTC (Experiment B)

Evaluated via [`run_alignment_eval.py`](file:///workspace/nvidia-transcribe/experiments/modular/alignment/run_alignment_eval.py) in `outputs/modular-eval/.venv` (`qwen-asr==0.0.6`, `bfloat16`). Model load: **`1.0584s`**, **`47` runs total, `2644` lexical items, `0` runtime exceptions**.

## 1. Mandatory Mandarin `'钾'` OOV Test (`zh_fleurs_01_2006`)

Input text (identical across `B1` Nemotron text, `B2` Qwen ASR text, and `B3` FLEURS reference):
> `钙、钾等元素属于金属，银和金等元素当然也是金属。`

- **Nemotron 3.5 CTC outcome:** Aborted entire request (`ctc_timestamps.error = "alignment_failed"`, `0/24` timed units) because `'钾'` is absent from the `5679`-char CTC vocabulary.
- **`Qwen3-ForcedAligner-0.6B` outcome:** **`21/21` aligned characters across `B1`, `B2`, and `B3` (`0` zero-duration, `0` missing chars, `0.030–0.032s` per run)**:
  `('钙', 1.12, 1.36), ('钾', 1.36, 1.60), ('等', 1.60, 1.84), ('元', 1.84, 2.00), ('素', 2.00, 2.24), ... ('属', 6.40, 6.56)`
- **Verdict:** Completely solves single-character CJK OOV alignment without text mutation.

## 2. Quantified Issue #197 Zero-Duration Collapse & Quantization Findings

Observed across all `47` forced-alignment runs (`benchmark.json` -> `alignment_comparison`):
- **`331 / 2644` lexical items (`12.5%`)** received exact **zero-duration spans** (`start == end`), affecting **`31 / 47` (`66%`)** runs.
- **`135` repeated identical spans** where adjacent tokens collapsed to the exact same `(t, t)` bin.
- **`0` negative/inverted spans** and **`0` non-monotonic transitions** (enforced by `fix_timestamp` LIS repair inside `qwen3_forced_aligner.py` — which is *precisely the code path* that clamps misaligned tokens onto identical `80ms` timestamps).
- **Resolution comparison:** `Qwen3-ForcedAligner` timestamps are quantized strictly to **`80ms` (`0.08s`)** intervals (`timestamp_segment_time`), whereas healthy Nemotron CTC alignment runs at **`10–20ms`** resolution (`0` zero-duration units across all `2016/2016` aligned words of `fr_full_episode_90s`).

### Representative Audit Breakdown (`B1` Nemotron Text vs `B2` Qwen ASR vs `B3` Reference)

| Case | Condition | Items | Valid Non-Zero | Zero-Dur | Repeated Spans | Past Dur | Time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `zh_fleurs_00_1906` | `B3` / `B1` / `B2` | 23 | 23 | 0 | 0 | 0 | 0.032 |
| `zh_fleurs_01_2006` (`'钾'`) | `B3` / `B1` / `B2` | 21 | **21** | **0** | 0 | 0 | 0.032 |
| `zh_fleurs_02_1883` (numbers) | `B3` / `B1` | 30 | 28 | 2 | 0 | 0 | 0.036 |
| `zh_fleurs_05_1805` (Latin+ZH) | `B3` (`Scotturb`/`Sintra`) | 23 | 21 | 2 | 0 | 0 | 0.031 |
| `zh_fleurs_05_1805` | `B1` (Nemotron) | 21 | 21 | 0 | 0 | 0 | 0.030 |
| `zh_2spk_seq` (`27.08s`) | `B3` | 67 | 64 | 3 | 0 | 0 | 0.042 |
| `zh_2spk_overlap` (simultaneous) | `B3` combined | 46 | **23** | **19 (41%)** | **11** | **12** | 0.033 |
| `zh_2spk_overlap` (simultaneous) | `B1` Nemotron | 49 | 25 | **24 (49%)** | **17** | 0 | 0.032 |
| `O1` (`9.0s`, real) | `B3` / `B2` | 22 | 21 / 20 | 1 / 2 | 0 | 0 | 0.031 |
| `O2` (`9.0s`, rapid) | `B3` / `B2` | 23 / 21 | 19 / 19 | 4 / 2 | 0 | 0 | 0.032 |
| `S1` (`4.8s` overlap) | `B3` (A-only) / `B3` (A+B) | 10 / 17 | 8 / 12 | 2 / 5 | 0 / 2 | 0 | 0.031 |
| `F1` (`39.0s`, French) | `B1` / `B2` | 114 / 113 | 108 / 107 | 6 / 6 | 0 | 0 | 0.046 |
| `F_100s` (`100s`, French) | `B1` / `B2` | 246 / 239 | 227 / 228 | 19 / 11 | 1 / 1 | 0 | 0.070 |
| `F_175s` (near `180s` limit) | `B1` Nemotron | 366 | 304 | **62 (16.9%)** | 17 | 0 | 0.127 |
| `F_300s` direct (over `180s` limit!) | `B2` (`300s` single call) | 492 | 356 | **136 (27.6%)** | **81** | 0 | 0.354 |
| `F_300s` auto-chunked `<=180s` | `Qwen3ASRModel` + Aligner | 497 | 426 | **70 (14.1%)** | 8 | — | (combined) |

## 3. Code-Level & Empirical Conclusions

1. **Package Limit Discovery:** `qwen_asr.inference.utils.MAX_FORCE_ALIGN_INPUT_SECONDS` is `180` seconds (3 minutes), not 5 minutes. Passing `300s` directly into `Qwen3ForcedAligner.align()` succeeded without raising an error, but doubled zero-duration collapse (`136/492 = 27.6%`, `81` repeated spans) vs `Qwen3ASRModel`'s internal `<=180s` chunking (`70/497 = 14.1%`, `8` repeated spans).
2. **Alignment Failure Modes:** Because the NAR alignment head predicts linear sequence positions, simultaneous speakers (`zh_2spk_overlap`, `S1`) trigger `fix_timestamp` clamping (`41–49%` zero-duration, `12` tokens pushed past the `10.38s` file end). Furthermore, punctuation (`'、'`, `'。'`, `'('`, `')'`) is stripped from output items (`'(', 'Sintra', ')' -> 'Sintra'`), while spoken CJK characters outside Nemotron's CTC vocab align accurately.
3. **Engineering Role:** Do **not** replace healthy Nemotron `10–20ms` CTC French alignment with an `80ms`-quantized aligner that drops `12.5%` of words to zero duration. Instead, deploy `Qwen3-ForcedAligner-0.6B` strictly behind `ExternalForcedAligner` as the **OOV / failed-CTC fallback** on affected sub-windows (`zh_fleurs_01_2006`).
