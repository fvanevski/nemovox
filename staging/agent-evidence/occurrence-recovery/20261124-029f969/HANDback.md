# Local Qualification Handback — Consolidated Occurrence Recovery (#17 / #18)

`RESULT=PASS` (describes executed qualification and evidence only; does not claim issue closure, semantic accuracy, or policy promotion).

## 1. Identities

- **Repository / Branch**: `fvanevski/nemovox` / `issue/17`
- **Tested Source SHA**: `029f969b43c58d53fd27975ff122c097ead04167` (tree `b4b26b6b8986cd9e26e38418a0de8b2125f8fd7d`)
- **Base SHA**: `9dd863841f72ab5f9aebf5bb4deb322bee99edf1`
- **Prior Qualified Source SHA**: `24f05c71b533ea7eb830ca7215f1690abcf415fc`
- **Prior Evidence Carrier SHA**: `f0d27639cf18a4c3e983c4b3b66d1be9d840541b`
- **Source Edits**: None (`git diff --check` clean; isolated detached worktree)
- **Private Review Archive**: `/workspace/nvidia-transcribe/outputs/handbacks/occurrence-recovery-20261124-029f969/occurrence-recovery-review-029f969.tar.gz` (`678713` bytes, SHA-256 `4d46e32e6f2459cb348389aa4362b6a3fdfd2dddba56391160cc8ef2773d6755`)

## 2. Phase A — CPU-Only Source Regressions, Replay, and Challenger Worker

| Check | Exit | Result |
|---|---|---|
| Core & Studio import-identity binding (`PYTHONPATH=src`) | 0 | Both `.venv` and `.venv-studio` resolve `nvidia_transcribe` to candidate worktree |
| `compileall` (`src/nvidia_transcribe`, `scripts/evaluate_occurrence_recovery.py`) | 0 | Clean |
| Focused regressions (10 test modules, `-m "not gpu"`) | 0 | `153 passed` |
| Full CPU test suite (`-m "not gpu"`) | 0 | `473 passed` |
| Studio test suite (`tests/test_studio*.py`, `-m "not gpu"`) | 0 | `50 passed` |
| `git diff --check` | 0 | Clean |
| Authenticated A3 replay (`805ab0e4...61eefe`) | 0 | 16 runs verified, `4` prior disposition corrections (`2` in `compression/fixed`, `2` in `compression/pause_aware`), full vs `without_inventory` comparison, explicit deferrals |
| Dedicated CPU Whisper challenger setup (`.venv-challenger`) | 0 | Python 3.12.14, `faster-whisper==1.2.1`, `ctranslate2==4.9.1`, `av==18.0.0`, `uv pip check` clean; all 4 `faster-whisper-large-v3` snapshot hashes match prior diagnostic probe |
| Phase A CPU Whisper worker control (`--phase worker`) | 0 (after `av==18.0.0` pin) | `status=observed`, 2 requests, `43.2s` audio, `19.725s` wall, 2 observations, `errors=[]`, `receipt_sha256=885a4afd...9d3c61` |

## 3. Phase B — Live Bounded Recovery & Qwen Control

### GPU & Lifecycle Integrity
- Pre-live GPU: `1835 MiB` used / `30292 MiB` free (RTX 5090). Post-live & post-Qwen GPU: `1691 MiB` used / `30436 MiB` free.
- `nemovox-asr.service`: started and cleanly stopped `5/5` times (one owned scope per region/control, closed before CPU challenger). Journal confirms `5/5` `quantization=modelopt_mixed`, `5/5` `kv_cache_dtype=fp8`, `5/5` `ctc_timestamp_middleware`, `5/5` `trust_request_mm_kwargs=True`, and `VLLM_USE_V2_MODEL_RUNNER=0`.
- `nemovox-text-review.service` remained `inactive/dead`; `nemovox-studio.service` remained `active/running` (unchanged PID `4071576`); `kwin_wayland` unchanged.
- Budget usage: `49 / 72` ASR requests, `930.0828 / 2000.0` ASR audio seconds.

### Bounded Live Recovery Findings (Including Unfavorable / Residual Cases)
- **Original source identity preservation**: 100% across all 5 cases (`30/30`, `16/16`, `14/14`, `14/14`, `2/2`).
- **`compression` (`[260, 370]`, `fr`)**: 30 primary -> 30 canonical utterances (0 untimed). Tasks: `4` observed, `8` deferred. Occurrence resolutions: `4` unresolved. Independent CPU challenger: `4` requests (`27.108s`, `0` errors, `9` segment rows), `0` cross-model events, `2` deferred targets (`status=unresolved`).
- **`boundary` (`[570, 680]`, `fr`)**: 16 primary -> 16 canonical utterances; **`1` residual wholly untimed row remained untimed** (`primary_untimed=1 -> canonical_untimed=1`). Tasks: `4` observed, `3` deferred. Decisions: `1` `unresolved_competing_source`, `1` `already_represented`. Occurrence resolutions: `8` unresolved. Independent CPU challenger: `4` requests (`68.898s`, `0` errors, `14` segment rows), **`1` cross-model event** (`status=partial`), `2` deferred targets.
- **`late_gap` (`[1160, 1270]`, `fr`)**: 14 primary -> 14 canonical utterances (0 untimed). Tasks: `4` observed. Occurrence resolutions: `1` unresolved. Independent CPU challenger: `4` requests (`21.6s`, `0` errors, `4` segment rows), `0` cross-model events, `1` deferred target (`status=unresolved`).
- **`low_flag_control` (`[440, 550]`, `fr`)**: 14 primary -> 14 canonical utterances (0 untimed). Tasks: `4` observed. Occurrence resolutions: `1` unresolved. Independent CPU challenger: `4` requests (`21.672s`, `0` errors, `7` segment rows), `0` cross-model events, `2` deferred targets (`status=unresolved`).
- **`multilingual_control` (`synthetic_zh_2spk_overlap.wav`, `[0, 10.38]`, `zh`)**: Existing FLEURS Mandarin 2-speaker overlap fixture (overlap `2.0s–9.62s`, known reference present). 2 primary -> 2 canonical utterances (`SPK_00`, `SPK_01`, 0 untimed), `0` recovery tasks triggered (`independent.status=no_targets`). Acoustic/semantic truth is not claimed from text matching alone.
- **Summary of unfavorable / residual behavior**: Across the 4 French regions, `14` occurrence resolutions remained explicitly `unresolved`, `11` recovery tasks and `7` independent challenger targets were explicitly `deferred` by policy caps, and `1` untimed row in `boundary` remained unlocalized with `1` `unresolved_competing_source` decision. No silent false additions or silent coercions occurred.

### Qwen Forced-Alignment Control (`--phase qwen`)
- Used cached `Qwen3-ForcedAligner-0.6B` (`cuda:0`, `torch.bfloat16`, `model_load_count=1`, `align_count=1`, peak GPU allocated `1881800192` bytes) on the real `boundary` cross-model event.
- **Result**: `status=completed`, `positive_recovered_units=5`, `negative_recovered_units={"wrong_order": 0, "zero_duration": 0}`.
- Negative controls are labeled synthetic perturbations of the real worker response, not evidence that forced alignment recognizes words.

## 4. Command-Shape / Environment Friction

- **`friction-01` (`recovered_after_retry`)**: Installing `faster-whisper==1.2.1` into the fresh `.venv-challenger` pulled transitive `av==19.0.2`, which dropped the `metadata_errors` kwarg in `av.open(...)` and caused the initial `--phase worker` attempt to fail with `TypeError`. Pinning `av==18.0.0` (matching the read-only evidenced version in `/workspace/pyannote/.venv`) resolved the incompatibility; `uv pip check` passed and the retry succeeded cleanly.

## 5. Stop Conditions Observed

No PR created, no merge, no issue closure, no deployment, no `#19` work, no full-episode campaign, no source edits.
