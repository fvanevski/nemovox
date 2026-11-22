# Phase B: Real Host Riva `device=auto` CUDA & Contention Verification (20261122T170806Z)

- **Source SHA:** `2bce0d8b3730c7e8784eef9b10cc616c273e86d2`
- **Canonical Lock:** `~/.local/state/nemovox/model-lifecycle.lock` (`NEMOVOX_LIFECYCLE_LOCK` unset)
- **Resource Config:** Isolated candidate `NEMOVOX_MODEL_RESOURCES` copy of `configs/model-resources.example.json` with `resources.riva_local.device="auto"` and `checkpoint_path="/workspace/models/Riva-Translate-4B-Instruct-v2"`.
- **Provenance class:** Real host subprocess/GPU execution against the canonical lifecycle lock (synthetic French recording contract for isolated contention cases A & B; real French FLEURS WAV audio for Studio & CLI E2E).

## 1. Real Canonical-Lock Contention Probes (Exact Child-PID Observation)
Competitor held an exclusive `flock` on `~/.local/state/nemovox/model-lifecycle.lock` via `ModelLifecycle()._lock()`:

- **Case A — Bounded Timeout (`NEMOVOX_LIFECYCLE_LOCK_TIMEOUT_S=0.6`, competitor hold `2.0s`):**
  - Candidate CLI (`generate-subtitles --translator riva --translation-review off`) exited with code `1` in `2.03s` (including Python startup).
  - Structured stderr reason: `status="RESOURCE_BLOCKED"`, `reason="resource_lease_timeout"`.
  - Child `_riva_translate_worker.py` processes observed under CLI PID: `0`.

- **Case B — Wait-Then-Admit Real CUDA (`NEMOVOX_LIFECYCLE_LOCK_TIMEOUT_S=30`, competitor hold `2.5s`):**
  - While competitor held `~/.local/state/nemovox/model-lifecycle.lock`, child `_riva_translate_worker.py` count under CLI PID remained `0` (`workers_before_release=0`).
  - `0.165s` after competitor released the canonical lock, CLI acquired the lease and spawned `_riva_translate_worker.py` (`workers_after_release=1`) with `device="auto"`.
  - Worker completed with CLI exit `0` (`elapsed_s=18.574`):
    - `device_used="cuda:0"`, `dtype="torch.bfloat16"`, `load_s=6.1292`, `infer_s=2.201`, `peak_vram_mib=8009.43`, `cache_hits=0`.
    - `utterance_id="utt-0000"`, `translation_status="translated"`.

## 2. Live Studio & Direct CLI Uncached `device=auto` CUDA Measurements
- **Idle GPU Baseline:** `3049 MiB used / 29078 MiB free` (RTX 5090 `32607 MiB` total).
- **Studio French E2E (`job_id=ef867313f6d14cbdabb1ffedba7d2e3e`, `fr_fleurs_00_1987.wav`):**
  - During Riva `device=auto` stage (`t=93.64s..104.29s`): `nemovox-asr.service` and `nemovox-text-review.service` were both `ActiveState=inactive` (`MainPID=0`).
  - GPU used peaked at `11769 MiB` (`20358 MiB` free); worker reported `device_used="cuda:0"`, `dtype="torch.bfloat16"`, `load_s=4.2585`, `infer_s=1.9862`, `peak_vram_mib=8013.55`, `elapsed_s=13.006`, `cache_hits=0`.
  - Original MT retained alongside completed English review (`review.status="completed"`, `raw_source_preserved=True`).
- **Direct CLI Parity (`fr_fleurs_01_1857.wav`):**
  - Worker reported `device_used="cuda:0"`, `dtype="torch.bfloat16"`, `load_s=3.3436`, `infer_s=2.0087`, `peak_vram_mib=8014.75`, `elapsed_s=11.404`, `cache_hits=0`.
- **Post-Run GPU State:** Restored to `3049 MiB used / 29078 MiB free`.
