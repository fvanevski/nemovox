# PR #20 — final CPU regression qualification

Central review of ee0caf027d9df7394407b625ee58608e352db708 identified four blocking defects: remote equal-text relocation, nonexclusive acoustic support, pre-locality unbudgeted subinterval recovery, and unauthenticated original recording. Central implemented corrections in 6a7d4606674cacc05c9d08ce47656f6810139e2a and 6439ecdeae0515e1e3a5508833bc3336f2e2da5c. The final source SHA is supplied by the operator kickoff; do not infer it from this document. No GPU/worker inference or new media campaign is authorized.

## Execution

Use the existing local environment in an isolated detached candidate worktree. Fetch and authenticate the exact full candidate SHA from origin/issue/17; compare base 9dd863841f72ab5f9aebf5bb4deb322bee99edf1. Record native SHA/tree/status and actual import paths for both venvs. Do not modify the owner checkout, models, dependency environments, or service state. Resolve repository-native test commands from prior evidence and run:

- compileall and git diff --check;
- focused tests: test_occurrence_recovery.py, test_source_quality.py, test_source_recovery.py, test_source_occurrences.py, test_recording_and_normalization.py, test_issue14_asr_integrity.py, test_qwen_session_and_policy.py, test_recovery_schedule_replay.py, test_chunking_and_resume.py, test_offline_rerender.py, test_translation.py (use actual tracked file paths; if a named path is absent, inventory and record the equivalent consumer);
- full existing CPU suite and Studio suite under their existing native commands and isolated imports.

No skips, weakened assertions, source edits, or test rewrites. Reuse prior live execution evidence only within its actual scope.

## Sealed evidence replays

Use the authenticated prior archives under outputs/handbacks:
occurrence-recovery-review-029f969.tar.gz (SHA-256 4d46e32e6f2459cb348389aa4362b6a3fdfd2dddba56391160cc8ef2773d6755) and recovery-scheduling-supplement-ebdc06f.tar.gz (SHA-256 a512f262035860a9f3aaa9ac78b523b517c1cc673573af05eacc9eee2b4ffc4d). Preserve exact bytes. In the candidate root with the core interpreter and PYTHONPATH=src:

```sh
python -m scripts.replay_recovery_schedule --archive "$ORIGINAL_ARCHIVE" --output-dir "$NEW_PLAN_REPLAY"
python -m scripts.replay_acoustic_resolution --archive "$ORIGINAL_ARCHIVE" --supplement "$SUPPLEMENT_ARCHIVE" --output-dir "$NEW_RESOLUTION_REPLAY"
```

Use fresh output directories outside the checkout. Do not pass --boundary-wav. The first replay compares counterfactual plans. The second reconstructs event acquisition and resolution using exact prior observed crops, including the targeted boundary pair. It must not call models or substitute new crops. Report actual decisions and receipt changes; no positive correction count is required. Any missing original row, new row, changed timing, or conflicting observation ownership requires Central adjudication. Synthetic regression tests establish the new positive and negative acceptance controls. The prior nine recording alignment diagnostics all report recovery_attempted=false: no legacy subinterval call was exercised by the previous live campaign.

## Return

One evidence-only commit on issue/17, with exact tested source/tree, complete command logs and exit status, import and resource state receipts, sanitized summary and manifest/checksums. Provide a private tar.gz with full replay outputs, logs, and evidence manifest, excluding media, weights, credentials, and duplicate historical archives. Keep private transcript and observations off GitHub. Verify no production changes in evidence carrier and exact remote readback. Stop on material source/test failure or a non-equivalent replay; return concrete evidence, not an alternative runner that silently masks failure. Report command shape friction using the project contract. Central owns all substantive fixes, final review, Codex sequencing, readiness, and merge decisions.
