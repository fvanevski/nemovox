# CUDA Independent Acoustic Challenger Qualification (`20261124-6079094`)

## Source & Repository Identity

- **Repository:** `fvanevski/nemovox` (Issue #19, PR #21, branch `issue/19`)
- **Base commit (`main`):** `4dd4599e51775ca510002783ea650ecd25fd7393` (tree `dfbc53906300996373c34631fe33029b87bbf313`)
- **Prior evidence carrier:** `47c67b1a9bf44849fbccd687b7f2a864e9a4e0fc`
- **Tested candidate commit:** `60790947180385cfce5f32501374f621edb19c26`
- **Tested candidate tree:** `6794424e10c6c804f458b5bccc26d1eb36fd7452`
- **Candidate delta:** 17 files changed (+336 / -62) from prior evidence carrier (see `carrier-to-candidate.diff`)
- **Source edits made:** None (`SOURCE_CHANGES_MADE=no`)
- **Owner checkout / deployment config changed:** None

## Phase A: CPU & Studio Qualification

All imports verified bound to detached candidate worktree (`/tmp/nemovox-issue19-cuda-6079094/src/nvidia_transcribe/`).

- `compileall -q src scripts tests`: **PASS** (exit 0)
- `git diff --check 47c67b1a9bf44849fbccd687b7f2a864e9a4e0fc HEAD`: **PASS** (exit 0)
- Focused regression suite (9 test modules): **PASS** (`168 passed, 1 warning in 0.50s`, exit 0)
- Full CPU suite (`-m "not gpu"`): **PASS** (`526 passed, 1 warning in 11.75s`, exit 0)
- Studio suite (`.venv-studio`, `tests/test_studio*.py`): **PASS** (`50 passed in 3.57s`, exit 0)

## Phase B: CUDA Host Integration & Runtime Library Remediation

- **Host GPU:** NVIDIA GeForce RTX 5090 (32607 MiB VRAM, driver `620.73.04`)
- **Challenger environment (`.venv-challenger`):** Python `3.12.14`, `faster-whisper==1.2.1`, `ctranslate2==4.9.1`, `av==18.0.0` (`uv pip check`: 23 packages compatible, no Python packages added or upgraded)
- **CTranslate2 CUDA probe:** `get_cuda_device_count() == 1`, `get_supported_compute_types("cuda", 0)` includes `int8_float16`
- **Runtime library diagnosis:** Host default CUDA loader path is `/opt/cuda/lib64` (CUDA 13: `libcublas.so.13`). System `/usr/lib/libcudnn*.so.9` loaded cleanly, but baseline inference probe failed with `RuntimeError: Library libcublas.so.12 is not found or cannot be loaded` (`logs/phase-b-cuda-baseline-probe.log`).
- **Isolated remediation:** Reused already-installed host CUDA 12.8 cuBLAS libraries by symlinking into `.venv-challenger/lib/python3.12/site-packages/ctranslate2.libs/` (which `libctranslate2-5cf58fc0.so.4.9.1` searches via `DT_RPATH: [$ORIGIN]`):
  - `libcublas.so.12 -> /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so.12.8.3.14`
  - `libcublasLt.so.12 -> /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so.12.8.3.14`
- **Verification:** Passed CUDA `int8_float16` inference with both `env -u LD_LIBRARY_PATH` (matching Studio/systemd execution) and `LD_LIBRARY_PATH=/opt/cuda/lib64` (`logs/phase-b-cuda-remediation-verify.log`). No system files, drivers, or unrelated environments were modified.

## Phase B: Matched CUDA Challenger Control (`scripts.evaluate_cuda_challenger`)

- **Authenticated inputs:**
  - Prior archive SHA-256: `206ada956e77b9db1263de090fbbeac139c78e1fa26670bc24963cbadcad3498`
  - Original WAV SHA-256: `30234a1c67ae9ed310ed6c76e4b97d4d0c9d54a7220129ac0a05c61eb368f314`
- **Result:** `status=observed`, exit 0 (`logs/matched-summary.json`)
- **Matched crops:** 4 / 4 exact crop bounds and `audio_sha256` matches
- **Checkpoint equivalence:** `true` (all 4 checkpoint files SHA-256 identical)
- **Worker runtime receipt:** `device=cuda`, `device_index=[0]`, `compute_type=int8_float16`, `model_is_loaded=true`, `is_multilingual=true`
- **Wall time (47.7s audio):** CPU baseline `44.8923s` -> CUDA `9.6007s` (**4.68x faster**)
- **GPU telemetry:** Worker PID `419476`, peak worker VRAM `2750 MiB` (37 active samples), returned to `3046 MiB` used / `29081 MiB` free after exit (`logs/matched-cuda-gpu-telemetry.log`)
- **Structural CPU vs CUDA comparison (`logs/matched-cpu-vs-cuda-comparison.json`):** Identical segment counts across all 4 crops (2, 4, 5, 7); exact text match on 2/4 crops and 0.9919 / 0.9878 similarity on the other 2; start/end timing deltas within 0.00s–0.04s; `resolution_decisions=1`, `source_truth_accepted=false`.

## Phase B: Integrated Resource-Transition Control (`scripts.evaluate_recording_context`)

- **Result:** Exit 0, `overlap_violations_count=0` (`logs/integrated-stage-transitions.json`, `logs/integrated-source-quality-summary.json`)
- **Observed sequential stage transitions (UTC):**
  1. `07:39:22.809Z -> 07:39:47.278Z`: `nemovox-text-review.service` (NInfer context, peak `27566 MiB`) -> stopped & released (`3046 MiB`)
  2. `07:39:47.278Z -> 07:41:29.720Z`: `nemovox-asr.service` (Nemotron localized recovery, 10 observations, peak `28398 MiB`) -> stopped & released (`3046 MiB`)
  3. `07:41:30.265Z -> 07:41:39.207Z`: CUDA Whisper worker (`.venv-challenger/bin/python`, PID `442166`, peak `2686 MiB`, 4 crops / 43.164s audio in `9.7592s`, `cuda`/`[0]`/`int8_float16`) -> exited & released (`3046 MiB`)
  4. `07:41:39.207Z -> 07:41:58.026Z`: `nemovox-text-review.service` (later scene reasoning) -> stopped & released (`3046 MiB`)
- **Restoration:** `nemovox-asr.service=inactive`, `nemovox-text-review.service=inactive`, `nemovox-studio.service=active`, no challenger processes remaining, unrelated GPU processes (`kwin_wayland`, `text-embeddings-router`) undisturbed (`logs/restoration-check.txt`).

## Private Review Archive

Full private observations, resolver outputs, and recording artifacts are retained locally (excluded from Git):
- **Path:** `/workspace/nvidia-transcribe/outputs/handbacks/cuda-challenger-20261124-6079094/cuda-challenger-review-6079094.tar.gz`
- **Bytes:** `235320`
- **SHA-256:** `edcc33008e294d0d334fdbfb423c53df500f7b3f2e03469162e455fa42fffdd8`
