# Issue #14 — Structured-Output Meaning-First QC Summary (`20261124T025004Z`)

## Verdict
- **Result**: **PASS** (`6/6` `advisory_only` accepted under unchanged validator + pinned JSON Schema `output_config.format`, `0` contract errors, `0` truncations, `0` source/subtitle mutations).

## Verification Summary
1. **GitHub & Authority**: PR `#15` (`15affc89c8d2eb15e472a609de8afe48b2766f6f`) matches `CENTRAL_SOURCE_SHA=72dbee0c2f2164de31224801f100c564c4570f69` (`tree=f9416e63d62ba41be738c604d3f22e4fc93bb2b5`) and `UNCHANGED_CONFIG_SHA=e5c5a8a6c3443f7bfbefc25119fd392694eca1dc` with zero local changes in `src/`, `tests/`, or `configs/`. Installed NInfer HEAD `a3a3dd5be6c2ecec1479017d5eee219d7a7e5784` verified against `anthropic_messages_request.cpp` and `request_validation.cpp`. Prior archive (`6/6` checksums, `7` unique entries) and prior public packet (`8/8+manifest`) authenticated.
2. **CPU & Studio Gates (`PYTHONPATH=src:.`)**: `compileall` exit `0`; focused 9-file suite `113 passed` (`0.77s`); `.venv` full core `482 passed` (`10.47s`); `.venv-studio` full Studio `482 passed` (`10.11s`); both interpreters resolved candidate `src/` via `__file__`; `git diff --check` clean.
3. **Static Wire & T0 Replay**: Static capture confirmed `response_format_sent=True` and `output_config.format={"type":"json_schema","schema":...}` with direct schema object, exact `request_sha256` & `case_id` enums, 6-key citation enums, top-level `system` instructions, `thinking.enabled` (`1024`, `xhigh`, `1792`) for T2 and `thinking.disabled` (`512`) for T1. Offline T0 replay reconfirmed Run A (`2008` merged / `3` candidates), Run B (`2002` merged / `6` budgeted candidates), Mandarin (`0` candidates), and untouched 6/7-component CLI hashes.
4. **Live Owned NInfer Session (6 calls, `9472` reserved tokens)**:
   - `ModelLifecycle`: `before=startable` (`3112 MiB` used / `29015 MiB` free) -> `during=ready_managed` (`30682 MiB` used / `1446 MiB` free, `qwen3_8_27b_swift15abl_nvfp4full-dflash2`, `18.0 GiB` weights) -> `after=startable` (`3112 MiB` used / `29015 MiB` free).
   - **T2 Calls 0–4 (`xhigh`, `thinking_budget=1024`, `max_tokens=1792`)**: all `5/5` finished `stop` (`1292..1328` completion tokens = `1049` reasoning + `243..279` constrained JSON tokens); decisions `possible_duplicate` (S02 prefix, W03a, W01c second) and `unresolved` (S02 tail, W01c earlier).
   - **T1 Call 5 (`none`, `thinking_budget=0`, `max_tokens=512`)**: finished `stop` (`238` completion tokens, `0` reasoning); decision `unresolved`.
   - All `6/6` passed `validate_semantic_review()` strictly with `source_application_authorized=False`.
