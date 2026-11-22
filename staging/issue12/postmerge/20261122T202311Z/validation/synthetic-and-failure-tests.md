
# Phase B4 Isolated Negative & Compatibility Verification

Executed in an isolated same-commit (`8f6c03cbc70e84bf1970ba364207a02c9fe22b44`) instance with isolated `job_root`, `upload_root`, `export_roots`, and stubbed `yt-dlp` / `NEMOVOX_MODEL_RESOURCES` (no network downloads, no production mutation).

- **Remote download stub (`save_remote_media=true` vs `false` + ASR failure):**
  - `save=true`: job failed at ASR (`state=failed`), `remote-media.webm` exported and `remote/source.webm` retained internally.
  - `save=false`: job failed at ASR (`state=failed`), no `remote-media.webm` exported while internal `remote/source.webm` retained.
  - Private/loopback URL rejection: HTTP `[422, 422]`.
- **Path & export safety:**
  - Symlink escape & out-of-root browse: HTTP `403`, out-of-root & `../` traversal submit: HTTP `422` (`[403, 403, 422, 422]`).
  - Idempotent retry: `already_saved`; modified target overwrite refusal: `failed`; missing artifact partial/failed status: `failed`; unwritable destination: `failed`.
- **Backward compatibility, cancel, multi-page logs & >50 history:**
  - Legacy request without `export` field: `state=complete`, `export_report.status=disabled`.
  - Cancel during active `translation` stage: `state=cancelled`, `stage_history=[['asr', 'skipped'], ['translation', 'cancelled']]`, no export directory created.
  - Large UTF-8 log (`>150 KB`, `日本語 café 🎬`): `6` chunk pages + `/log/download` exact match (`True`), history pagination `total=60` (`>50`), artifact download verified.
