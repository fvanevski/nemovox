
# Output Root Authorization, Selective Exports, Collisions & Failures

## Host Root Authorization & Escapes
- Configured `NEMOVOX_EXPORT_ROOTS` with two temporary roots (`exports_a:exports_b`), nested `exports_a/subdir/deep`, an outside directory `outside_secret`, and symlink `exports_a/escape_link -> outside_secret`.
- `GET /api/export-directories` returned `200` with `subdir` listed and `escape_link` omitted.
- `GET /api/export-directories?path=<outside_secret>` returned `403`.
- `GET /api/export-directories?path=<exports_a/escape_link>` returned `403`.
- `validate_destination(<exports_a>/../outside_secret)` raised `ValueError` (`traversal_rejected=True`).
- Submitting a job in Chromium with `outside_secret` produced UI `role="alert"`: `Output directory is outside configured writable export roots.`.

## Selective Exports vs Full Internal Archive
- UI job `23dff28c22ea4cbdbba4f267b9e4a6ed` submitted `export={"destination": "/tmp/nemovox-issue12-phaseb/s1/exports_a/manual/nested", "artifacts": ["final_english", "srt"], "save_remote_media": false}` (manual nested destination `exports_a/manual/nested`, selecting only `final_english` and `srt`).
- External directory `nemovox-23dff28c22ea4cbdbba4f267b9e4a6ed/` contained **only**: `['final_english_subtitles.txt', 'synthetic.en.srt']` with exact SHA-256 parity against internal `output/` originals.
- `.ass`, `refined.contract.json`, `translated.contract.json`, `subtitles.metadata.json`, `raw_source_transcript.txt`, `refined_source_transcript.txt` were **not** copied to external destination, and all remained accessible under `Advanced: all retained job artifacts`.

## Collisions, Idempotence, Refusal & Failures
- Second job exporting identical filename (`synthetic.en.srt`) into the same destination folder produced separate `nemovox-<job_id>/synthetic.en.srt` (`status=complete`) with zero collision.
- Idempotent retry of `export_selected` returned `['already_saved', 'already_saved']`.
- Preexisting modified target file (`USER_EDITED`) was preserved untouched and `export_selected` failed closed (`status=failed`).
- Unwritable destination (`chmod 0500`) returned `status=failed` while internal `output/synthetic.en.srt` remained intact.
- Partial export check (requesting `source_transcript` on a `contract`-based `subtitles` job alongside `srt`) kept `job.state=complete` while honestly reporting `export_report.status=partial` and items `[['source_transcript', 'failed'], ['srt', 'saved']]`.
