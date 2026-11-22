# Deployment Plan & Private Rollback Ledger (postmerge-20261122T173438Z)

- **Live checkout verified:** `/workspace/nvidia-transcribe` (`HEAD` prior to promotion: `8fee1e177db9742dbdb38c4a22bdd663c20ac80f`, tree `2c733a3036443121cc4c1c7c2b7a09bb0d5680a5`).
- **Pre-promotion cleanliness:** `git status --porcelain=v1` clean; ignored runtime artifacts only (`configs/vllm_rtx5090.env`, `web/dist/`, `web/node_modules/`).
- **Pre-promotion Studio state:** `nemovox-studio.service` active (`MainPID=2922624`, `InvocationID=a863a06b44fe433493aea538ddbcb7f6`, started `Sat 2026-11-21 19:13:11 PST`); `GET /api/jobs` had `0` queued/running jobs (`3` complete, `1` failed).
- **Pre-promotion managed units:** `nemovox-asr.service` and `nemovox-text-review.service` both `inactive` (`MainPID=0`), `3047 MiB used / 29080 MiB free`.
- **Private rollback backup location (kept off GitHub, mode `0700`):** `~/.local/state/nemovox/rollback-issue10-20261122T1728Z/`
  - Contains `rollback-ledger.json`, `nemovox-studio.service.bak` (sha256 `118c256bc94768b52c2f286c8b3cbdc9cf28423718413bdd6e9e3ba2107342d3`), and `web_dist_8fee1e1.tar.gz` (`11` prior `web/dist` files with recorded SHA-256 digests).
  - Model weights, media libraries, and secret values (`VLLM_API_KEY`) were not copied into any repository or evidence path.
- **Promotion plan:** Stop `nemovox-studio.service` -> `git merge --ff-only 332a99697edcc0b72fcc842168d5b1f57782e203` -> verify `.venv` and `.venv-studio` import paths -> `npm run build` in `/workspace/nvidia-transcribe/web` -> start `nemovox-studio.service`.
