# Issue #14 — NInfer Messages System-Wire & Bounded QC Summary (`20261124T023146Z`)

## Verdict
- **System-wire transport & thinking budget control (`NInferMessagesTextClient`)**: **VERIFIED PASS**
- **Overall consolidated campaign outcome**: **PARTIAL** — CPU gates, static wire inspection, T0 replay, `ModelLifecycle`, HTTP 200 `/v1/messages` transport, and `thinking_budget` enforcement (`1024` / `xhigh` for T2, `0` / `none` for T1) all succeeded across 6/6 bounded calls (`9472` reserved tokens), **but `0/6` live model outputs were accepted as `advisory_only` (`6/6` `unaccepted_model_response`) due to client/prompt/schema contract mismatches.**

## Proven Invariants
1. **GitHub & Source Identity**: PR `#15` (`73d5f099805c45fa7061c84d4e16fee83db900b7`) matches `CENTRAL_SOURCE_SHA=9230bde41a418c074140252edb6c46637560adb2` (`tree=5306fce0ad90c271b7bf72a44e178a367ae76fba`) and `UNCHANGED_CONFIG_SHA=e5c5a8a6c3443f7bfbefc25119fd392694eca1dc` with zero source/test/config edits locally. Installed NInfer HEAD is `a3a3dd5be6c2ecec1479017d5eee219d7a7e5784`.
2. **CPU & Studio Gates (`PYTHONPATH=src:.`)**: `compileall` clean; focused 8-file suite `101 passed`; `.venv` full core `470 passed`; `.venv-studio` full Studio `470 passed`; both interpreters confirmed importing candidate `src/` via `__file__`; `git diff --check` clean.
3. **Static Wire Verification (zero model calls)**:
   - Introductory contiguous `role="system"` turns are hoisted cleanly into top-level `system` (concatenated with `"\n\n"` when multiple prefix system messages exist).
   - Non-prefix in-conversation `system` turns remain in their exact message position.
   - System-only requests fail closed locally (`ValueError: NInfer Messages requires at least one non-system message`).
   - T2 sends `thinking={"type": "enabled", "budget_tokens": 1024}`, `output_config={"effort": "xhigh"}`, `max_tokens=1792`. T1 sends `thinking={"type": "disabled"}`, `max_tokens=512`, no `output_config`.
4. **Owned `ModelLifecycle` & Live `/v1/messages` Execution (6 calls total)**:
   - `nemovox-text-review.service` booted lazily (`qwen3_8_27b_swift15abl_nvfp4full-dflash2`, 18.0 GiB weights) and stopped cleanly before context exit (`nvidia-smi` free VRAM returned to `29015 MiB / 32607 MiB`, unrelated services untouched).
   - Zero HTTP 400/5xx errors (`6/6` HTTP 200) and zero reasoning starvation: all 5 T2 calls consumed `1048` reasoning tokens (`1024` thinking + `24` control) leaving `493..744` answer tokens; T1 used `0` reasoning tokens and `371` answer tokens.

## Why 6/6 Live Calls Returned `unaccepted_model_response`
1. **Citation Enum vs Prompt Mismatch (`semantic_review.py:233-239` vs `258-266`)**: `validate_semantic_review()` rejects anything outside 6 field-name enums (`right_context_words`, etc.) in `supporting_evidence`/`contrary_evidence`, whereas `build_semantic_review_messages()` never states those allowed enum keys and asks the model to *"Describe both supporting and contrary evidence"*. Every call emitted helpful prose strings instead.
2. **Missing `"schema"` & Markdown Fence Stripping (`Call 2, 3, 4` and `Call 5`)**: `NInferMessagesTextClient` sets `response_format_sent=False` (does not map `response_format` to NInfer `output_config.format`), `semantic_review.py` runs bare `json.loads(raw)` without stripping ```` ```json ```` fences (Call 5), and Calls 2–4 omitted the `"schema"` key.
3. **T2 Verbose Prose Headroom (`Call 0`)**: With `1048` reasoning tokens inside `1792`, Call 0 truncated at `744` answer tokens (`finish_reason="length"`) while writing long prose arrays. Switching prompts/validation to concise enum keys (as `validate_semantic_review` expects) or structured schema will also dramatically reduce visible answer length.
