# Integrated Qualification Acceptance Matrix (`4c6c7ac5`)

| Phase / Gate | Status | Evidence Summary |
|---|---|---|
| Phase A Source & Host Admission | **PASS** | `main=8f6c03cb`, handoff=`b110657d`, source=`4c6c7ac5` (`tree=a9b08e87`); production & original inputs verified untouched. |
| Phase A `compileall`, Studio, Web | **PASS** | `compileall=0`; Studio `50 passed`; Web `14 passed` + `vite build=0`. |
| Phase A Fixed Compatibility & Resume | **PASS** | Default `chunk_policy="fixed"`, fixed/pause signatures separated (`175113b5 != b79c681e`). |
| Phase A Focused & Full Non-GPU Pytest | **FAIL (STOP)** | **`2 failed, 55 passed`** focused; **`2 failed, 285 passed`** non-GPU (`test_short_media_and_fixed_compatibility:59`, `test_single_near_threshold_embedding_cannot_force_merge:45`). |
| Phase B Read-Only Episode Pause Plan | **PASS (WITH PLANNER FINDINGS)** | `87` pauses, `35` chunks (`23` pause, `11` forced, `max_infer=54.0s`); identified `c01` skipping `64.24s` pause by `0.58s` & `b17` forced cut in `C1`. |
| Phase C Managed Pilot & Full 23-Min Episode | **NOT_RUN (STOPPED AT PHASE A)** | Per contract STOP rule on CPU test failure; `nemovox-asr`/`text-review` remained `inactive`, GPU untouched (`3104 MiB`). |
