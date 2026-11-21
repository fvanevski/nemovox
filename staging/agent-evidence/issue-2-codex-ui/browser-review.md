# Browser Polling-Order & Media-Gating Review (`9ee4757ea24bbd68cb190f606b75a7f1af360ea1`)

Verified in headless Chromium (`/usr/bin/chromium --disable-gpu`, `1440x960`) against built `web/dist` served by FastAPI (`uvicorn nvidia_transcribe.studio_api:app --host 127.0.0.1 --port 8765 --workers 1`) using synthetic WebM + SRT/ASS fixtures and Playwright route control.

## Ordered HTTP Request Proof

```text
ORDERED_REQUEST_TRACE:
  REQ GET /api/jobs/11111111111111111111111111111111
  REQ GET /api/jobs
  REQ GET /api/health
  REQ GET /api/media
  REQ GET /api/glossaries
  REQ GET /api/jobs/11111111111111111111111111111111/preview
  REQ GET /api/jobs/11111111111111111111111111111111/artifacts
  REQ GET /api/jobs/11111111111111111111111111111111/logs
  --- PHASE 1 VERIFIED (no video, no /media, no JASSUB) ---
  REQ GET /api/jobs/11111111111111111111111111111111
  REQ GET /api/jobs/11111111111111111111111111111111/media
  --- PHASE 2 VERIFIED (video & /media loaded mid-run without page reload) ---
  REQ GET /api/jobs/11111111111111111111111111111111
  REQ GET /api/jobs/11111111111111111111111111111111/logs
  REQ GET /api/jobs/11111111111111111111111111111111/preview
  REQ GET /api/jobs/11111111111111111111111111111111/artifacts
  REQ GET /api/jobs/11111111111111111111111111111111/artifact/output/synthetic_demo.en.ass
  REQ GET /api/jobs
  --- PHASE 3 VERIFIED (final preview + artifacts + JASSUB active without reload) ---
  REQ GET /api/jobs/22222222222222222222222222222222
  REQ GET /api/jobs
  REQ GET /api/health
  REQ GET /api/media
  REQ GET /api/glossaries
  REQ GET /api/jobs/22222222222222222222222222222222/preview
  REQ GET /api/jobs/22222222222222222222222222222222/artifacts
  REQ GET /api/jobs/22222222222222222222222222222222/logs
  --- SCENARIO B VERIFIED (failed remote download shows unavailable, zero /media request) ---
  REQ GET /api/jobs/33333333333333333333333333333333
  REQ GET /api/jobs
  REQ GET /api/health
  REQ GET /api/media
  REQ GET /api/glossaries
  REQ GET /api/jobs/33333333333333333333333333333333/preview
  REQ GET /api/jobs/33333333333333333333333333333333/artifacts
  REQ GET /api/jobs/33333333333333333333333333333333/logs
  REQ GET /api/jobs/33333333333333333333333333333333/preview
  REQ GET /api/jobs/33333333333333333333333333333333/artifacts
  REQ GET /api/jobs/33333333333333333333333333333333/media
  REQ GET /api/jobs/33333333333333333333333333333333/artifact/output/synthetic_demo.en.ass
  --- SCENARIO C VERIFIED (ordinary local media preview & JASSUB) ---
CONSOLE_ERRORS: []
```

## Synthetic Screenshots Archived

Included inside `issue-2-codex-ui-evidence.tar.gz` (`screenshots/`):
- `scenario_a_pending.png`
- `scenario_a_complete_jassub.png`
- `scenario_b_failed_unavailable.png`
