
# Phase A1–A2 Preflight & Pre-Promotion State

- **GitHub `origin/main`:** `8f6c03cbc70e84bf1970ba364207a02c9fe22b44` (tree `21ae1c6d1b682f1061ae36f6896cd0ac87db1580`)
- **Local `/workspace/nvidia-transcribe` `main` before promotion:** `332a99697edcc0b72fcc842168d5b1f57782e203` (tree `6f1b41c26f8439d9eb1c5a4aff11b64e4c40afaa`), clean working tree (`git status --porcelain=v1` empty), confirmed FF ancestor of `8f6c03cbc70e84bf1970ba364207a02c9fe22b44`.
- **Issue #12:** `OPEN`; **PR #13:** `MERGED` (`2026-11-22T20:04:09Z`, mergeCommit `8f6c03cbc70e84bf1970ba364207a02c9fe22b44`).
- **Pre-Promotion `nemovox-studio.service`:**
  - `ActiveState=active`, `MainPID=4180532`, `InvocationID=cd4b825f88744f3b8bd72153aa828aea`, `ExecMainStartTimestamp=Sun 2026-11-22 09:29:06 PST`, `DropInPaths=(none)`
  - `WorkingDirectory=/workspace/nvidia-transcribe`, `127.0.0.1:8765`, active CLI child processes: `0`
  - Retained jobs idle gate: `{'complete': 5, 'failed': 1}` (0 `running` / `queued`)
- **Model units before promotion:** `nemovox-asr.service=inactive (dead)`, `nemovox-text-review.service=inactive (dead)`.
- **Tools & capacity:** `ffmpeg=/usr/local/bin/ffmpeg`, `yt-dlp=/usr/bin/yt-dlp`, `/workspace` 57 GiB free, `/tmp` 15 GiB free.
- **Configured runtime environment keys (values redacted):** `PYTHONPATH`, `NEMOVOX_CLI_PYTHON`, `NEMOVOX_STUDIO_JOBS`, `NEMOVOX_STUDIO_UPLOADS` (hashed preimages recorded privately: `nemovox-studio.service` sha256 `118c256b...`, `configs/vllm_rtx5090.env` sha256 `40cfcd51...`, 6 retained `job.json` hashes in `pre_jobs.sha256`).
