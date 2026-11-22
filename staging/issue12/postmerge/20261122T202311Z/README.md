
# Issue #12 / PR #13 — Post-Merge Deployment & Live Production Acceptance (`20261122T202311Z`)

- **Repository:** `fvanevski/nemovox`
- **Issue / PR:** `#12` (`OPEN`) / `#13` (`MERGED` -> `8f6c03cbc70e84bf1970ba364207a02c9fe22b44`)
- **Deployed Branch & Commit:** `main` @ `8f6c03cbc70e84bf1970ba364207a02c9fe22b44` (tree `21ae1c6d1b682f1061ae36f6896cd0ac87db1580`)
- **Prior Live Commit:** `332a99697edcc0b72fcc842168d5b1f57782e203` (tree `6f1b41c26f8439d9eb1c5a4aff11b64e4c40afaa`)
- **Activated Studio Service:** `nemovox-studio.service` (`127.0.0.1:8765`, `MainPID=1121444`, `InvocationID=6cd5cf36a6774b0693d37a196c088464`)
- **Result:** `PASS`

## Summary
- **Phase A (Controlled Promotion):** Authenticated clean `main` (`332a9969`), created private rollback backup (`/workspace/nemovox-issue12-rollback-20261122T201716Z`), verified exact merged `main` (`8f6c03cb`, tree `21ae1c6d`) in an isolated staging worktree (`compileall` exit 0, `257` non-GPU pytest, `49` Studio pytest, `14` Vitest + Vite/TS build), stopped old Studio PID `4180532`, ran `git merge --ff-only 8f6c03cbc70e84bf1970ba364207a02c9fe22b44`, atomically replaced `web/dist` (`index.html` sha256 `05088b0a3e9e6735685f8d72c2d0c24164eb3c0dbb81a8ff3fc3ab365f932f54`), and started fresh `nemovox-studio.service` (new PID `1121444`, new InvocationID `6cd5cf36a6774b0693d37a196c088464`).
- **Phase B (Live & Isolated Acceptance):** Verified live Chromium UI on `http://127.0.0.1:8765`, synthetic compact 2-chunk contract across `subtitles`/`transcription`/`translation` (exact `336`-byte SHA-256 preservation + `290–610s` sidecar chronology), real French FLEURS job `d55d30020bbd448ba36f4dda58f9bc1a` (Nemotron ASR → Riva CUDA → balanced review → SRT/ASS → selected `final_english` + `srt` export), and isolated same-commit negative/compatibility checks (`POSTMERGE_B4_PASS`).
