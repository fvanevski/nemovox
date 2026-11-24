# Issue #14 NInfer Reasoning-Budget & Voice-Stream QC Qualification (20261124T020042Z)

- **Result**: `PARTIAL`
- **Admitted PR Head**: `2b9eaeeb096609777987f791ec44d5527c931257`
- **Validated Central Source**: `4c27c5b8d13a173aa8ce2f3c1f9699a37d8a14c0` (`tree=7fb880ff1614f4ce21ab28a87fda5de3e62c62c0`)
- **Validated Central Config**: `a31a7d7d25b083671126f9f63b56151a7526a0d2` (`configs/meaning-first-review.example.json` blob `ba5df2cfbffe00acf0b2d8028d4671f06aa7ec69`)

## Key Verified Findings

1. **CPU Suite Regression (1 Failure in Both `.venv` and `.venv-studio`)**:
   - `.venv`: `458 passed, 1 failed` (`exit=1`); `.venv-studio`: `458 passed, 1 failed` (`exit=1`); focused `6` files: `55 passed, 1 failed`; `compileall` & `git diff --check`: `exit=0`.
   - Failed test: `tests/test_issue14_semantic_review.py::test_injected_tiered_text_client_is_advisory_and_bounded` (`semantic_review.py:296` raises `ValueError` because `reasoned_thinking_budget=512` default violates caller `reasoned_tokens=512` `max - 128` headroom).
2. **T0 Per-Local-Voice Replay (`Run A`, `Run B`, `Mandarin`)**:
   - Canonical invariance preserved (`Run A: 2008`, `Run B: 2002` merged units; accepted 6/7-comp CLI hashes unchanged; `0` false flags on `C1`, `C2`, `W04c`, and `Mandarin` `3269` char units).
   - **S02 (`360s`)**: Per-local-voice matching improved `boundary_ccbf33cf64b16ef94668` from **`2 -> 9`** words on `chunk_003:spk_0` (rank #0) and `boundary_567ba692421cf2d5ca68` from **`4 -> 5`** words (rank #1).
   - **W01c (`90s`)**: `boundary_02e99af11eba174c2a23` records competing tie (`chunk_000:spk_1` vs `spk_2` = `1`); `77.63–80.0s` remains unrecoverable from `chunk_001` (`infer_start_s=80.0s`), `[64.0..67.0]s` confirmed musical transition.
3. **Live Owned NInfer Probe (`run_configured_tiered_text_reviews`, 6 cases, 6912 tokens)**:
   - Lifecycle clean (`startable -> ready_managed -> startable`).
   - `reasoning_effort='xhigh'` (`5` calls) and `'low'` (`1` call) were BOTH accepted without HTTP `400` (`reasoning_effort_fallback=False`).
   - **Root-cause host finding**: Installed NInfer (`a3a3dd5be6c2ecec1479017d5eee219d7a7e5784`) `src/serve/openai_chat_request.cpp` does **not** parse `"thinking_budget"` in `/v1/chat/completions` (`thinking_budget` exists only in `anthropic_messages_request.cpp:875`), silently keeping server `--thinking-budget 16384`. All 6 calls spent 100% of `max_tokens` (`1280`x5 + `512`x1) in `reasoning_tokens` and hit `finish_reason='length'` with `content=''`, cleanly caught as `6/6 unaccepted_model_response` (`Completion budget exhausted by reasoning before final JSON`).
   - **Action for Central**: (a) fix `run_tiered_text_reviews` default `reasoned_thinking_budget` compatibility with caller-supplied `reasoned_tokens=512` (or update the legacy test), and (b) target `/v1/messages` or update installed NInfer `openai_chat_request.cpp` / use `enable_thinking=false` on T1 or rebuild `nemovox-text-review.service` NInfer binary from upstream master supporting `thinking_budget` in Chat Completions!
