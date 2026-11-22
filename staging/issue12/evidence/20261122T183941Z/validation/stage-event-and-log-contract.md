
# Typed Stage Events, Single CLI Invocation & Persistent UTF-8 Logs

## Single `generate-subtitles` Subprocess Transitions
Tested on isolated candidate Studio `127.0.0.1:8770` with French synthetic contract, `translator=riva` (CPU stub), `translation_review=balanced` (loopback mock OpenAI server), `condense_high_cps=true`, `formats=[srt, ass]`:
- `STAGE: Generating English outputs` count in `execution.log`: `1` (single CLI subprocess).
- Recorded `stage_history`: `[['asr', 'skipped'], ['translation', 'complete'], ['review', 'complete'], ['condensation', 'complete'], ['render', 'complete'], ['export', 'complete']]`
- Observed Playwright UI stage panel rows: `['skipped\nasr\nUsing an existing recording contract', 'complete\nPrimary translation', 'complete\nEnglish review\ncompleted', 'complete\ncondensation\nok', 'complete\nRender subtitles', 'complete\nexport\ncomplete']`
- Emitted structured events in `execution.log`:
```
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "translation", "status": "running", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "translation", "status": "complete", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "review", "status": "running", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "review", "status": "complete", "detail": "completed"}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "condensation", "status": "running", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "condensation", "status": "complete", "detail": "ok"}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "render", "status": "running", "detail": ""}
NEMOVOX_EVENT {"schema": "nemovox-progress/v1", "stage": "render", "status": "complete", "detail": ""}
```

## Skipped, Same-Language, Cancel & Failure Paths
- Same-language bypass (`en -> en`) on Studio `127.0.0.1:8769`: `[(s['id'], s['status']) for s in stage_history]` = `[('asr', 'skipped'), ('translation', 'skipped'), ('review', 'skipped'), ('condensation', 'skipped'), ('render', 'complete'), ('export', 'complete')]`.
- Cancel during active `translation` stage on Studio `127.0.0.1:8770`: `state=cancelled`, `stage_history=[['asr', 'skipped'], ['translation', 'cancelled']]`, `exported_dir_exists=False`.
- Review error path (invalid JSON from mock text server during initial run): `review` marked `failed` (`detail="completed_with_errors"`), job terminated `failed`, no false success.

## Large UTF-8 Multiline Log Chunking & Download
- Seeded job `000000000000000000000000000000bb` with `166,500`-char (`>180 KB`) multiline UTF-8 log (`日本語 café 🎬`) spanning multiple `65,536`-byte chunk boundaries.
- Chromium `Copy full log` concatenated all `/api/jobs/<id>/log/chunk` pages and wrote exact `166500` chars to clipboard (`exact_match=True`).
- `GET /api/jobs/<id>/log/download` returned HTTP `200` with exact UTF-8 byte parity.
