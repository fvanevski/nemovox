# Final Review-Anchor Qualification Evidence (`20261125T213000Z-e33ff126`)

Sanitized public evidence packet for issue #19 / PR #21 (`issue/19`) evaluating candidate commit `e33ff126f93bb61cb000d7e5335d5ed379616ef9` (tree `bcb501bc418cd4048d7937546729cb5e9c60946b`) against baseline executed source `cabed83dc5685d18acccc354d1519b4e6505cf8e` (tree `99afccd2bc11e1b2036137f1d79aeb94be11535e`) and prior evidence carrier `20cf0e34ebe0bf19086a83eeaaa16f7cb9e3395b`.

**Result:** `FAIL` (replay-runner assertion defect in `scripts/replay_review_anchors.py`; production anchor heuristic verified via read-only diagnostic inspection).

No private dialogue text, raw model responses, private source excerpts, or credentials are included in this public packet. Full private deltas, baseline derived contracts, diffs, and logs are retained in the host-local private archive:
`review-anchors-e33ff126.tar.gz` (`595243` bytes, SHA-256 `05777ad39f9491f92bda999977a6fcb927477a07c11c1337cd59433b5c38598d`).

## Summary

- **CPU & Studio Qualification**: `compileall` PASS (exit 0), `git diff --check` PASS (exit 0), focused `tests/test_translation_review.py` `39 passed` (exit 0), full CPU `-m 'not gpu' tests` `554 passed` (exit 0), Studio `.venv-studio` `51 passed` (exit 0), `web/` tree (`1011e5c23e77821de13e8a1043735f792d9d92c7`) identical to prior qualified commit `3e6f8d8244f818c304bf5d692e4db864a6ae9a1e`.
- **Retained Private Archive**: Authenticated `review-completion-cabed83d.tar.gz` (`3764304` bytes, SHA-256 `6f3f0e308b227e69248e4121f9064c5e7e3e9bb543a6fe554d7e6426866a592a`).
- **Baseline Replay (`cabed83dc5685d18acccc354d1519b4e6505cf8e`)**: `PASS` (exit 0), `249/249` cache hits (`26` bounded + `2` mandarin + `221` french), `0` API calls, `0` changed dispositions, exact metric and checkpoint parity.
- **Candidate Replay (`e33ff126f93bb61cb000d7e5335d5ed379616ef9`)**: `FAIL` (exit 1) in `scripts/replay_review_anchors.py:116` with `ValueError: Unexpected changed disposition: utt_d8d3cc10e050e405afbfb564`.
  - **Root Cause**: On `bounded[24]` (`utt_d8d3cc10e050e405afbfb564`), `_anchors_preserved` properly lifted the false-positive `unresolved_source_name_or_number_change` veto on a function-word contraction. Because its continuation group `[23, 24, 25]` contains comparatively-rejected row `utt_d74ee44171f4dcec5c986c2d`, `translation_review.py` preserved primary MT and reclassified row 24 from `review_rejected_unresolved_source_preserved_machine_translation` to `review_rejected_dependent_passage_preserved_machine_translation` (identical `translated_text`). `scripts/replay_review_anchors.py` requires `new == "corrected"` for every status change and aborted before emitting candidate derived contracts.
  - Per `REVIEW_ANCHOR_QUALIFICATION.md`, replay assertions were not modified (`SEMANTIC_REMEDIATION=forbidden`).
- **Diagnostic Candidate Inspection (Read-Only)**: Across all `249` rows (`249/249` cache hits, `0` API calls, unchanged checkpoints):
  - `bounded`: `changed` `13 -> 16` (+3 joined westward-direction passage rows restored coherently), `unresolved_source_rejections` `3 -> 1` (material lexical guard retained), `dependent_passage_rejections` `3 -> 2` (2 lifted, 1 status-only reclassification with unchanged text).
  - `mandarin`: `0` changed rows (unsupported geographic substitution still prevented).
  - `french`: `changed` `119 -> 120` (+1 sentence-initial preposition capitalization false positive recovered), `unresolved_source_rejections` `10 -> 9`, `rejected_polarity=1`, `comparative_rejections=27`, `dependent_passage_rejections=2` all preserved.
- **Offline Render (`--re-render`)**: `BLOCKED` (official candidate replay runner aborted before emitting candidate derived contracts).
- **Resource State**: CPU-only; `0` GPU inference invocations; GPU memory ~3047–3048 MiB pre/post; `nemovox-studio.service` undisturbed; owner checkout `/workspace/nvidia-transcribe` unchanged.
