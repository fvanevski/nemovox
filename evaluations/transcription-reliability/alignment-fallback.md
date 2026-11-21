# Phase B Report: Selective `Qwen3-ForcedAligner-0.6B` Fallback Integration

Phase B integrated optional forced-alignment fallback (`alignment_fallback = "off" | "qwen"`, default `"off"`) in commit `db996420aab2513c324f143586d86b5f86ee0e74`.

---

## 1. Design & Guardrail Guarantees

1. **Non-destructive selectivity:** Native full-chunk CTC timings and subinterval CTC recovered timings are NEVER replaced or modified. `QwenForcedAlignerAdapter` runs ONLY on remaining untimed units after native subinterval recovery fails (e.g. Mandarin CJK characters OOV in Nemotron's CTC vocabulary such as `'钾'`).
2. **Mixed-speaker overlap protection:** `_has_mixed_speaker_overlap` detects overlapping diarization turns within the candidate window and skips Qwen alignment cleanly (`status="skipped_mixed_speaker_overlap"`), preventing a single-stream forced aligner from fabricating alignments across simultaneous voices.
3. **Strict Issue #197 validation (`validate_qwen_items`):** Any item emitted with `start == end` (`zero_duration`), inverted bounds (`negative_or_inverted`), timestamps exceeding audio duration (`past_audio_duration`), or backward starts (`non_monotonic`) is rejected explicitly (`valid_timing=False, start=None, end=None`) — never clamped or interpolated.
4. **Isolated subprocess execution:** `_qwen_align_worker.py` runs inside `/workspace/nvidia-transcribe/outputs/modular-eval/.venv/bin/python` (`qwen-asr==0.0.6, transformers==4.57.6`), selecting `cuda:0` BF16 when `>= 2600 MiB` free VRAM is available or CPU FP32 otherwise, leaving the main vLLM Python 3.13 environment completely isolated.

---

## 2. Live GPU Validation (`off` vs. `qwen`)

Tested against the live `modelopt_mixed` vLLM server with `Qwen3-ForcedAligner-0.6B` co-running in subprocess (`cuda:0`, `torch.bfloat16`):

| Clip / Scenario | Mode | Utterances (`timing_status`) | Unit Alignment Counts | CER / Verbatim Check | Qwen Diagnostics (`external_aligner`) |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`zh_fleurs_01_2006`** (`"钙、钾等元素属于金属，银和金等元素当然也是金属。"`, `8.28s`) | `off` | `1` (`failed: 1`) | `diarization_bounded_only: 21` | `CER = 0.0` (`21/21` chars verbatim incl. `'钾'`) | Not invoked (`0.3829s` total) |
| **`zh_fleurs_01_2006`** | **`qwen`** | **`1` (`full: 1`)** | **`external_qwen_recovered: 21`** | **`CER = 0.0` (`'钾'` verbatim & timed)** | `status="ok"`, `device_used="cuda:0"`, `valid_items=21/21`, `zero_duration_count=0` (`11.5695s` incl. model cold load in subprocess) |
| **`zh_multichunk_synthetic`** (`69.78s`, `30s` chunks) | `off` | `8` (`full: 6, partial: 2`) | `ctc_aligned: 122, subinterval_ctc_recovered: 74, diarization_bounded_only: 2` | Both `'钾'` and `'桥'` untimed, preserved verbatim | Not invoked (`12.9952s`) |
| **`zh_multichunk_synthetic`** | **`qwen`** | `7` (`full: 6, partial: 1`) | `ctc_aligned: 73, subinterval_ctc_recovered: 74, external_qwen_recovered: 1 ('钾' 1.92–2.08s), diarization_bounded_only: 1 ('桥')` | `'钾'` recovered cleanly; `'桥'` flagged zero-duration & kept untimed | **`status="ok"`, `device_used="cuda:0"`, `total_items=2, valid_items=1, zero_duration_count=1`** — live proof that Issue #197 rejection works end-to-end without clamping (`14.6253s`) |
| **`zh_2spk_overlap`** (`10.38s`, synthetic overlap) | `qwen` | `2` (`full: 2`) | `ctc_aligned: 49` | Both speakers preserved verbatim (`0.5694s`) | Not triggered (all units aligned by native CTC) |
| **`fr_full_episode_repaired_90s`** (`1379.9s`, `90s` chunks) | `qwen` | `221` (`full: 221`) | `ctc_aligned: 2064` | `0` failed CTC chunks (`31.4241s`) | Not invoked on healthy French CTC chunks (zero unnecessary latency!) |

---

## 3. Standalone Offline Subprocess Benchmark (`zh_fleurs_01_2006`)

Measured directly with `QwenForcedAlignerAdapter().align(...)` while GPU was idle (`3340 MiB -> 3341 MiB` post-exit teardown):
- Cold subprocess load + alignment of 21 Chinese chars: `~12s`, `status="ok"`, `'钙'` `[1.12, 1.36]`, `'钾'` `[1.36, 1.60]`, all `21` characters valid on an `80ms` frame grid, full VRAM release on subprocess exit.
