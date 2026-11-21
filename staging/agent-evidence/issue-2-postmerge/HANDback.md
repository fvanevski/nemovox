# Nemovox Studio Merged-Main Fresh-Process Acceptance Handback

**Verdict:** `PASS`

## Source, Worktree & Process Identity

- **Repository:** `https://github.com/fvanevski/nemovox` (`issue #2`, merged PR `#3`)
- **Merged `origin/main` SHA:** `242c418f55cd2afb796ee2c4fa742a10ff04f7a3` (verified via `git fetch origin refs/heads/main`, `FETCH_HEAD` exact match, `git cat-file -t` = `commit`)
- **Merged `origin/main` tree:** `122f8466db22098ba9427373af7e87ceb8421076` (identical to approved pre-squash tree `bc5f067340d8bc84b7129bbdea7589ea401222c8`)
- **Evidence branch (`evidence/issue-2-postmerge-20261121`):** based on `242c418f55cd2afb796ee2c4fa742a10ff04f7a3` (extending handoff commit `8955bbb20999d6fd4f50320bf8cd0f755c628a1a`)
- **Runtime import verified:** `nvidia_transcribe.studio_api.__file__` = `/tmp/nemovox-postmerge-wt-20261121/src/nvidia_transcribe/studio_api.py`, `PROJECT` = `/tmp/nemovox-postmerge-wt-20261121`, Python executable = `/workspace/nvidia-transcribe/.venv-studio/bin/python`
- **Fresh Uvicorn server PIDs:**
  - Primary PID `4003808` (`Sat Nov 21 11:33:18 2026`, `127.0.0.1:18765`, `--workers 1`, exited `-15` on `SIGTERM`)
  - Rejected concurrent owner PID `4003880` (`127.0.0.1:18769`, exited `3` with `"Studio job directory is already owned by another running server."`)
  - Restarted PID `4004565` (`Sat Nov 21 11:33:23 2026`, `127.0.0.1:18765`, `--workers 1`, exited `-15` on `SIGTERM`)

## Criterion-to-Evidence Verdict Matrix

| Criterion | Artifact(s) | Verdict |
|---|---|---|
| Merged-main source | `source-identity.txt` | **PASS** — `main` SHA `242c418f...` & tree `122f8466...` authenticated; runtime imported from isolated worktree; fresh `web/dist` compiled |
| Fresh startup & exclusivity | `fresh-process.json`, `startup.log` | **PASS** — Genuine PIDs `4003808` & `4004565` served `/api/health` (`200`) and `/` (`200`); secondary PID `4003880` rejected (`exit 3`) without disturbing primary |
| Processing & fidelity | `api-job.json`, `api-requests.txt` | **PASS** — Contract job `c46f34300e0f4e1c8334edaf96334344` completed (`queued -> running -> complete`); preserved Mandarin + French Unicode, speakers & `[untimed -> untimed]`; exact HTTP download bytes; negatives returned `403/403/422/422` |
| URL & browser | `browser-review.md`, `postmerge_preview.png` | **PASS** — **Injected fixtures** (`Episode #1? 50% 日本語.en.{srt,ass}`) returned percent-encoded URLs, matched HTTP 200 SHA-256 bytes, loaded JASSUB `ASS · libass` (`1` canvas, `0` console errors) |
| Persistence & cleanup | `restart-cleanup.md` | **PASS** — Job history, cues & 4 artifacts survived PID `4003808 -> 4004565` restart; staged `queued` orphan reconciled to `interrupted`; `0` leftover test-owned processes (`OWNED_REMAINING: 0`) |
| Regression suite | `test-results.md`, `logs/*.log` | **PASS** — Studio API `14 passed` (`exit 0`), `.venv-studio` CPU `85 passed` (`exit 0`), `.venv` CPU `85 passed` (`exit 0`), Vitest `8 passed` + Vite build (`exit 0`) |
| Privacy & provenance | `manifest.json`, `HANDback.md`, archive | **PASS** — Sanitized (`ZERO_HOME_PATH_MATCHES`), `gzip -t` verified, SHA-256 verified |

## Omissions & Limitations

- **Live GPU ASR / SwiftU inference:** `UNVERIFIED` (external GPU services not running and explicitly excluded by post-merge scope).
- **Controlled polling-order & late-upload race browser checks:** Reused from pre-squash approved tree `bc5f067340d8bc84b7129bbdea7589ea401222c8` (`122f8466db22098ba9427373af7e87ceb8421076`, identical tree to merged `main`, archived under `evidence/issue-2-codex-ui-20261121` at `936ded2dd2ff280292795c74cc052c94c992a3d7`).
- **Privacy redactions:** Personal home paths redacted to `<USER_HOME>` in sanitized logs/records; raw scratch lives only temporarily under `/tmp/nemovox-postmerge-scratch-20261121`.

## Command-Shape Friction

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events: []
  omitted_sensitive_values=no
```

REQUIRED OPERATOR ACTION: Attach the verified evidence archive to the next Central ChatGPT message together with this handback.
