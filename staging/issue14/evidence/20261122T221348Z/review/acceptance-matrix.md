# Issue #14 Exact-Head & CTC Timing Probe Acceptance Matrix (`66275bdb`)

| Gate / Scope | Status | Evidence Summary |
|---|---|---|
| Phase A admission & invariance | **PASS** | `main`=`8f6c03cb`, handoff=`b3aa3846`, exact tested source=`66275bdb` (`tree=34307587`); production Studio & inputs untouched. |
| Phase A exact-head deterministic suites | **PASS** | `compileall=0`; focused `32 passed` (`0` failed); full non-GPU `274 passed` (`0` skipped); Studio `49 passed`. |
| Phase A prior 16-chunk replay reuse | **PASS (Valid)** | `git diff 866aa734 66275bdb -- src/` empty; `99dce3f8bfa1a93670b28473209dbf8bd924a54e:staging/issue14/evidence/20261122T215418Z/` metrics reused with legacy recovery accounting corrected (`10-6=4`). |
| Phase B managed ASR lease & VRAM release | **PASS** | `ModelLifecycle` cleanly started & stopped `nemovox-asr.service` (`modelopt_mixed` NVFP4 + CTC); GPU returned `3105 -> 26933 -> 3104 MiB`. |
| Phase B narrow probes (`6` primary + `2` control, `16–24s`) | **MIXED (Informative)** | Fixed `W01b/d` (`43.7–63.7s` & `77.6–82.9s`), fixed `W04a` question timing (`1231.2s`), preserved `C1/C2`; **counterevidence**: `P_W02a` still pulled `-5.3s`, `P_W04a` omitted `18`-word reply, `P_W01a/P_W03b` clipped clauses & collapsed speakers. |
| Phase B matched longer-context (`3` probes, `44–45s`) | **PASS (Decisive)** | `P_W03b_long` & `P_W04a_long` eliminated ALL multi-second drift (`W03b`, `W04a`), restored all `18` omitted reply words, and recovered `3` distinct speaker tags per excerpt. |
| Phase C archived speaker & MT/review attribution | **PASS** | Documented `0.6536`/`0.6637` single-candidate TitaNet false merges, raw diarizer vs `<spk:N>` collapses (`S05` saw `spk=1`), `S02` negation inversion, `S01`/`W04c` cascades, and `5` helpful Qwen repairs. |
| Phase D recovery-counter accounting fix | **PASS** | Fixed `-44`/`-4` -> `4`/`4` via `recovered_boundary_units - extended_recovered_units`. |
