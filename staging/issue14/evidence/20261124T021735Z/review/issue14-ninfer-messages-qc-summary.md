# Issue #14 NInfer Messages Bounded QC Qualification (20261124T021735Z)

- **Result**: `PARTIAL`
- **Admitted PR Head**: `9faf8145cb9d368195df1d1e81081c89ddf85c11`
- **Validated Central Source**: `94d96d1bfb84cb0c6c8163511963302b499aeb0c` (`tree=5dd95055ccb57114e579ec3db8e74ea45157ec42`)
- **Validated Central Config**: `e5c5a8a6c3443f7bfbefc25119fd392694eca1dc` (`configs/meaning-first-review.example.json` blob `01755bdb7c759204605b7e2a1f2ff973606fa770`)

## Verified Evidence

1. **CPU Suites (ALL GREEN)**:
   - `compileall` (`exit=0`), `git diff --check` (`exit=0`), focused 8 files (`99 passed`), `.venv` (`468 passed, 0 failed`), `.venv-studio` (`468 passed, 0 failed`). Both interpreters verified importing `ninfer_messages_client`, `resource_lifecycle`, `refinement`, `semantic_review`, and `chunking` from the candidate worktree.
2. **T0 Boundary QC Replay & Canonical Invariance**:
   - `Run A` (`merged=2008`, `cases=3`), `Run B` (`merged=2002`, `cases=6`), `Mandarin zh_ep166` (`15` chunks, `3269` char units, `0` candidates); accepted 6/7-component CLI hashes unchanged.
   - `S02` rank #0/#1 (`matched_words`: `9` and `5`), `W01c` (`80.2-83.254s` + competing tie; `77.63-80s` outside Chunk 1 audio), negative controls (`C1/C2/W04c/Mandarin`) clean (`0` false promotions).
3. **Owned `/v1/messages` Runtime Preflight (Stopped on Call 0 HTTP 400)**:
   - Installed NInfer (`a3a3dd5be6c2ecec1479017d5eee219d7a7e5784`) supports `POST /v1/messages` with `thinking.type="enabled", budget_tokens >= 1024` and `output_config.effort`, parsing top-level `body["system"]`.
   - `ModelLifecycle` cleanly transitioned `nemovox-text-review.service` `startable -> ready_managed -> startable`.
   - Call 0 (`boundary_ccbf33cf64b16ef94668`, `max_tokens=1792`, `thinking_budget=1024`, `effort='xhigh'`) returned `HTTP 400 invalid_request_error`: `"messages: system messages must follow a user message and be final or precede an assistant message"`.
   - **Root Cause for Central**: `NInferMessagesTextClient.complete()` (`src/nvidia_transcribe/ninfer_messages_client.py:56`) forwards `role="system"` inside `body["messages"]` instead of hoisting it into top-level `body["system"]` (and synthetic `test_issue14_ninfer_messages_client.py` did not reject initial `role="system"` inside `messages`). Note also `output_config.format` is available in `anthropic_messages_request.cpp:904-907` if strict JSON formatting is desired alongside prompt guidance.
