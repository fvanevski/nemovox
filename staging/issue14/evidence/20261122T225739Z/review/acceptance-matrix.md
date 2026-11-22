# Integrated Qualification Acceptance Matrix (`81704133`)

| Gate / Phase | Status | Evidence Summary |
|---|---|---|
| Phase A Admission & Deterministic Suites | **PASS** | `compileall=0`; focused `59 passed`; full non-GPU `289 passed`; Studio `50 passed`; Web `14 passed` + Vite build `0`. |
| Phase B Pause Plan (`64.241s` Repair + Episode) | **PASS** | `87` silences, `35` chunks (`22` regular + `1` `early_long` at `64.241s` + `12` forced), `W02c` kept inside single `c07`. |
| Phase C Pilot (`8` Planned Excerpts) | **PASS** | `C1` `29/29`, `C2` `8/8`, `W04a` `7+18` words intact; major CTC drifts eliminated (`POST_GPU=3106 MiB`). |
| Phase C Full 23-Min Episode Regression | **PASS WITH DOCUMENTED EDGE REGRESSIONS** | Utterances `234->169`, `0.19-0.24s` shards `49->12`, micro-utts `90->40`, `W02c`/`W03a`/`W03c`/`W04c` fixed, `6` low-margin TitaNet merges blocked; **new/remaining regressions**: `W05a` ASR omission, `W04b` `c31` CTC swap, `c28` loop -> `1` untimed `u138`, intra-chunk `S05` collapse. |
| Host & Input Invariance + VRAM Release | **PASS** | Original media/job/manifest/ASS SHA256 identical; `nemovox-asr`/`text-review` released (`3104 MiB` baseline); Studio `MainPID=1121444` untouched. |
