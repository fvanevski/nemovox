# Issue #10 Post-Merge Production Acceptance Matrix (postmerge-20261122T173438Z)

- **Merged `main` Commit:** `332a99697edcc0b72fcc842168d5b1f57782e203`
- **Merged `main` Tree:** `6f1b41c26f8439d9eb1c5a4aff11b64e4c40afaa`
- **Overall Result:** `PASS`

| Gate / Criterion | Status | Evidence Reference |
|---|---|---|
| Remote `main` SHA/tree authentication & clean prior checkout (`8fee1e1`) | PASS | `source/identity.json`, `deployment/plan-and-backups.md` |
| Private rollback ledger & `web/dist` + unit backup | PASS | `deployment/plan-and-backups.md` |
| Staging preflight (`compileall`, `227 passed` pytest, `5/11` Vitest, Vite build) | PASS | `validation/cpu-frontend.txt` |
| Controlled ff-only promotion, venv/Studio import proof & fresh `nemovox-studio.service` (`MainPID` `2922624 -> 4180532`) | PASS | `deployment/activation-and-rollback.md`, `validation/studio-fresh-process.md` |
| Read-only `/api/resource-status` & `/api/translation-backends` | PASS | `validation/studio-fresh-process.md` |
| Fresh production Studio French E2E (`job_id=96a0b61881844253bc21c72a093bc32f`, Nemotron NVFP4/CTC -> Riva `cuda:0` -> Qwen review -> SRT/ASS/preview/UI) | PASS | `validation/french-studio-e2e.md` |
| Direct CLI parity (`fr_fleurs_01_1857.wav`) & no-start `--re-render` / English bypass | PASS | `validation/direct-cli-parity.md` |
| Final Studio health, idle units, GPU baseline & external processes untouched | PASS | `validation/service-gpu-cleanup.md` |
| F-02 stop-poll timing boundary | NON-BLOCKING (unchanged) | `review/findings.md` |
