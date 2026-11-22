# Host Promotion & Live Studio Acceptance Matrix (`20261122T031252Z`)

| Acceptance Criterion | Outcome | Evidence |
|---|---|---|
| Exact installed source (`8fee1e177db9742dbdb38c4a22bdd663c20ac80f`, tree `2c733a3036443121cc4c1c7c2b7a09bb0d5680a5`) & rollback | PASS | `source/installed_identity.json`, `promotion/deploy_receipt.md`, `promotion/rollback_receipt.md` |
| Service promotion & fresh process (`nemovox-studio.service` PID `1006972` -> `2922624`) | PASS | `promotion/deploy_receipt.md`, `validation/production_http.json` |
| Browser serving, navigation, zero console errors, reload persistence | PASS | `validation/browser_gui.md`, `validation/production_http.json` |
| Configuration defaults & interactive `ExplainedSelect` combobox DOM | PASS (DOM) / UNVERIFIED (screen-reader audio) | `validation/browser_gui.md` (`aria-activedescendant`, `aria-live`, keyboard & pointer focus) |
| Real FFprobe 2-track media inspection, source-switch reset, 403/422 validation | PASS | `validation/media_tracks.json` |
| Live `/api/translation-backends` & `/api/glossaries` discovery & UI preview | PASS | `validation/backend_and_glossaries.json` |
| Hosted Studio English-contract subtitle job E2E (`bd26c9d6bb854336ba1e651d2f5fbd3c`) | PASS | `validation/job_e2e.md` |
| Live foreign-language Riva + English text-review E2E | BLOCKED_BY_ENV | `validation/backend_and_glossaries.json` (`127.0.0.1:8001` & `8190` not running; GPU mutations forbidden) |
| Browser `<video>` playback/seek & JASSUB ASS overlay | PASS | `validation/browser_gui.md` (`ASS · libass`, `<canvas>` active, `readyState=4`) |
| Final live readback & unrelated GPU service non-mutation | PASS | `validation/final_live_readback.md` |
