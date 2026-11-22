# Acceptance Matrix: ASR Integrity Gate & Recovery Probes (`4c32acf6`)

| Scope / Gate | Status | Notes |
|---|---|---|
| Phase A Exact Source & Invariance | **PASS** | `main=8f6c03cb`, handoff=`2d6d0624`, source=`4c32acf6` (`tree=a4e866f5`); web diff empty (`81704133` UI PASS reused); originals/production clean. |
| Phase A CPU, Studio & Retry Suites | **PASS** | `compileall=0`; focused `64 passed`; full non-GPU `294 passed`; Studio `50 passed`; explicit 2-chunk fail-closed + partial-resume test `PASS`. |
| Phase B Private Attribution (`c28`, `c31`, `c34`, `W02b/S05`) | **PASS** | Exact loop counts (`32774->14`), unpopulated `c34` diar span `[1352.63..1353.81] spk=1`, `c31` CTC interleave, and `W02b` `c05` diar split verified. |
| Phase C GPU Session (`6` primary + `3` follow-up) | **PASS** | Managed `nemovox-asr.service` (`modelopt_mixed` NVFP4+CTC) started/released cleanly (`3105 -> 3106 MiB` baseline); zero loops (`finish=stop` on all `9`). |
| `c28` Alternative Windows (`P1`, `P2`, `F1`) | **PARTIAL (PROVEN RULE)** | `P2_c28_right` recovered `27` words + `2` spk (`1123.39–1154.48s`); `P1`/`F1` proved sparse-decode counterexamples (`1–2` words over `6–7` diar spans). |
| `W04b` Alternative Windows (`P3`, `P4`, `F2`) | **PASS (`P4_w04b_wide`)** | `P4` (`1242..1274s`) cleanly recovered both `W04b` (`1253.16–1255.55s`) and `W04c` (`1264.44–1266.40s`) with separate speakers. |
| `W05a` & `S05` Recovery (`P5`, `P6`, `F3`) | **PASS (`P6` / `F3`)** | All `3` recovered `W05a` (`1352.00–1353.94s`, distinct spk); `P6` & `F3` also separated `S05` Pichu turn (`1371.7–1374.2s`). |
