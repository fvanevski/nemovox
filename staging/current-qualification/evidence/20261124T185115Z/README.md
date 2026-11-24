# Issue 14 completion qualification — CPU regression gate block on `df82b90`

**Blocking CPU regression failure before live French continuation.** Tested application `df82b900ce1ec859530d1e9858ab7a5f4ee74213`, tree `a75fb56c483a1bea0431f472d182698d3449b826` in the existing isolated detached worktree.

## Test summary

- `git diff --check origin/main...HEAD` (exit 0) and `git diff --check ece833d875b00f754cf2d429fad004c5e70eb14e..HEAD` (exit 0).
- `python -m compileall -q src tests` (exit 0) and module imports in both the CLI and Studio virtual environments (exit 0).
- All 93 tests across `test_issue14_ninfer_messages_client.py`, `test_source_quality.py`, `test_offline_rerender.py`, `test_issue14_semantic_review.py`, `test_issue14_semantic_review_output_contract.py`, `test_issue14_ninfer_thinking_budget.py`, and `test_issue14_asr_integrity.py` passed (`exit 0`), and all new compact comparative-recovery and deferred-rendering tests in `test_translation_review.py` passed.
- **Affected 4-file suite** (`test_translation_review.py`, `test_issue14_ninfer_messages_client.py`, `test_source_quality.py`, `test_offline_rerender.py`): **76 passed, 1 failed** (exit 1).
- **Full CLI `.venv` non-GPU suite:** **427 passed, 2 failed** (exit 1).
- **Full Studio `.venv-studio` non-GPU suite:** **427 passed, 2 failed** (exit 1).

## Failing regressions and exact root cause

1. `tests/test_translation_review.py::test_programmatic_postedit_rejects_legacy_primary_translator`
2. `tests/test_resource_integration.py::test_programmatic_riva_condensation_requires_explicit_text_resource`

In `src/nvidia_transcribe/subtitle_pipeline.py` (`lines 156–160`), the new early canonical-contract check `if not rc_input.is_file(): raise FileNotFoundError(...)` runs **before** the configuration validation checks at `lines 177–184` (`translation_review.profile != "off"` requiring `riva`, and `condense_high_cps` with `riva` requiring explicit `condensation` resource). Both existing tests pass `recording_contract_path=tmp_path / "not-needed"` (or `"not_needed"`) to assert that invalid pipeline configuration raises `ValueError` before reading contract files or creating `output_dir`. Because `FileNotFoundError` is raised at `line 159` instead of `ValueError`, both tests fail deterministically.

Under the contract's CPU-first gate and stop condition for failed required regressions (without editing source or tests locally), live French continuation was not started.

## Preserved state and host health

- The French output directory remains untouched (`asr/manifest.json`, `asr/recording.contract.json`, `asr/source-evidence.v1.json`, `canonical-source.contract.json`, `source-quality.json`, `refined.contract.json`, `translation.checkpoint.json`, `translation-review.checkpoint.json`, `translated.contract.json` SHA-256 digests recorded and verified).
- Prior Mandarin Studio evidence (`eb5bc5243677ede15e7b04c836cfabf84d260955`) remains preserved unchanged.
- `nemovox-asr.service` and `nemovox-text-review.service` remain `inactive`, ports `8190`/`8001` have 0 listeners, `model-lifecycle.lock` is acquirable, and production `nemovox-studio.service` (`InvocationID=6cd5cf36a6774b0693d37a196c088464`, `MainPID=1121444`) reports `/api/health` `"status":"ok"`.

[Original gate matrix](gate-matrix.md) · [Metrics](metrics.json) · [Commands](commands.md) · [Command friction](COMMAND_SHAPE_FRICTION.txt) · [Checksums](SHA256SUMS)
