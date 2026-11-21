# Nemotron ASR Startup, CTC Checks & RTX 5090 VRAM Samples

Raw logs: [`logs/coexistence_results.json`](logs/coexistence_results.json) and [`logs/asr_startup_and_routes_sanitized.txt`](logs/asr_startup_and_routes_sanitized.txt).

## ASR Startup & Configuration Verification

- **Baseline GPU (TEI ON, ASR absent):** `3,082 MiB` used / **`29,045 MiB` free** (`kwin_wayland` PID `11480`: `329 MiB`; `text-embeddings-router` PID `1216176`: `1,758 MiB`). Port `8190` closed.
- **Agent-Owned ASR Launch (`scripts/start_vllm_server.sh`):**
  - `APIServer` PID `1224121` (`PGID 1224121`), `EngineCore` PID `1225862`.
  - Ready in **`85.56s`** (`2026-11-21T22:56:14Z` -> `22:57:40Z`).
  - Startup log verified: `quantization=modelopt_mixed` (NVFP4/FP8, no BF16 expansion), `middleware=['nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware']`, `Model loading took 19.28 GiB memory and 33.32s`, `Free memory on device (27.87/31.37 GiB) on startup`, `Desired GPU memory utilization is (0.78, 24.47 GiB)`, `Actual usage is 20.13 GiB consumed + 2.2 GiB peak activation + 0.08 GiB CUDAGraph + 2.14 GiB KV cache`.
  - Authenticated `GET /v1/models` returned `HTTP 200` with `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`.

## Pre-Qwen & Post-Qwen ASR CTC Inference

| Stage | Clip | HTTP Status | Wall (s) | CTC Words | CTC Error | Diar Intervals |
| --- | --- | --- | ---: | ---: | --- | ---: |
| Pre-Qwen Warmup | `zh_fleurs_00_1906.wav` | `200 OK` | `7.5143` | `23` | `null` | `2` |
| Pre-Qwen OOV Check | `zh_fleurs_01_2006.wav` | `200 OK` | `0.7158` | `0` | `alignment_failed` | `3` |
| Post-Qwen CTC Health | `zh_fleurs_00_1906.wav` | `200 OK` | `0.2215` | `23` | `null` | `2` |

## RTX 5090 Simultaneous Residency Samples (`nvidia-smi`, `0.25s` polling, `42` samples)

| Phase (Timestamp UTC) | Total (MiB) | Used (MiB) | Free (MiB) | Resident GPU Compute Processes |
| --- | ---: | ---: | ---: | --- |
| Baseline (`22:56:14Z`) | `32,607` | `3,082` | `29,045` | `kwin_wayland` (`329`), `text-embeddings-router` (`1,758`) |
| ASR Loaded & Warmed, TEI ON (`22:57:48Z`) | `32,607` | `27,490` | **`4,637`** | `kwin_wayland` (`327`), `text-embeddings-router` (`1,758`), `VLLM::EngineCore` (`24,404`) |
| **Peak Simultaneous Residency (ASR + Qwen + TEI, `22:57:59Z`)** | `32,607` | **`30,061`** | **`2,066`** | `kwin_wayland` (`327`), `text-embeddings-router` (`1,758`), `VLLM::EngineCore` (`24,406`), Qwen worker PID `1235339` (**`2,564`**) |
| Post-Qwen Teardown, ASR + TEI Still Resident (`22:58:00Z`) | `32,607` | `27,492` | `4,635` | `kwin_wayland` (`327`), `text-embeddings-router` (`1,758`), `VLLM::EngineCore` (`24,406`) |
| Final Post-ASR Teardown (`22:58:03Z`) | `32,607` | `3,080` | `29,047` | `kwin_wayland` (`327`), `text-embeddings-router` (`1,758`) |
