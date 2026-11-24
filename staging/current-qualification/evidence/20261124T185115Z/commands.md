# Commands and material results

- Authenticated remote `origin`, verified clean isolated detached worktree, checked out exact candidate `df82b900ce1ec859530d1e9858ab7a5f4ee74213` (`tree a75fb56c483a1bea0431f472d182698d3449b826`), and verified PR #15 (`OPEN`, `isDraft: true`).
- Ran `git diff --check origin/main...HEAD` (exit 0) and `git diff --check ece833d875b00f754cf2d429fad004c5e70eb14e..HEAD` (exit 0).
- Ran `python -m compileall -q src tests` (exit 0) and imported `nvidia_transcribe.cli`, `studio_api`, `ninfer_messages_client`, `translation_review`, `subtitle_pipeline` in both the CLI and Studio virtual environments (exit 0).
- Ran affected 4-file suite (`tests/test_translation_review.py`, `tests/test_issue14_ninfer_messages_client.py`, `tests/test_source_quality.py`, `tests/test_offline_rerender.py`): exit 1 (`76 passed, 1 failed` — `test_programmatic_postedit_rejects_legacy_primary_translator`).
- Ran 7-file non-failing check (`tests/test_issue14_ninfer_messages_client.py`, `tests/test_source_quality.py`, `tests/test_offline_rerender.py`, `tests/test_issue14_semantic_review.py`, `tests/test_issue14_semantic_review_output_contract.py`, `tests/test_issue14_ninfer_thinking_budget.py`, `tests/test_issue14_asr_integrity.py`): exit 0 (`93 passed`).
- Ran full non-GPU suite in CLI `.venv` (`-m 'not gpu'`): exit 1 (`427 passed, 2 failed` — `test_programmatic_postedit_rejects_legacy_primary_translator` and `test_programmatic_riva_condensation_requires_explicit_text_resource`).
- Ran full non-GPU suite in Studio `.venv-studio` (`-m 'not gpu'`): exit 1 (`427 passed, 2 failed`, same 2 tests).
- Recorded SHA-256 digests of the 9 preserved French output files and verified host systemd units, zero port listeners (`8190`/`8001`), acquirable `model-lifecycle.lock`, idle GPU, and healthy production Studio `/api/health` (`ok`). No live GPU or NInfer inference ran.
