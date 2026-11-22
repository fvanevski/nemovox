
# Remote Media Retention, Early Export & Guardrails

Executed on isolated candidate Studio `127.0.0.1:8770` with a stub `yt-dlp` binary validating `--no-playlist` and producing synthetic `.webm` bytes (`33` bytes, sha256 `a650f9c8ff7753722d14e4c559c3c96a47cd0bb5a332c8296cd9e6c91394ec9e`), followed by downstream ASR failure.

## Selected Media Export (`save_remote_media=true`) + Downstream Failure
- URL submitted: `https://example.com/synthetic-test.webm`
- Job terminal state: `failed`
- Recorded `stage_history`: `[['download', 'complete'], ['asr', 'failed']]`
- `export_report.status`: `complete` (`remote_media`: `saved` -> `nemovox-<job_id>/remote-media.webm`, sha256 matched internal `remote/source.webm`)
- `remote/source.webm` retained internally and listed by `/api/jobs/<id>/artifacts` (`remote_artifact_listed=True`).
- `/api/jobs/<id>/log/download` returned HTTP `200` with full execution log.

## Unselected Media Export (`save_remote_media=false`) + Downstream Failure
- URL submitted: `https://example.com/synthetic-test-2.webm`
- Job terminal state: `failed`
- External `remote-media.webm` created: `False`
- Internal `remote/source.webm` retained: `True`

## URL Guardrails Preserved
- Submitting `http://127.0.0.1:8770/clip.webm` returned HTTP `422`.
- Submitting `http://localhost/clip.webm` returned HTTP `422`.
