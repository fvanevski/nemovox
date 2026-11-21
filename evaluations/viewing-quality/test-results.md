# Verification & Test Results

## 1. Bytecode Compilation & CPU Regression Suite
Executed on `/workspace/nvidia-transcribe/.venv` (Python 3.13.15, pytest 9.2.0) after all source and harness edits:

```bash
.venv/bin/python -m compileall -q src tests experiments && .venv/bin/pytest -v -m "not gpu"
```

- **`compileall`**: Exit `0` (clean across `src/`, `tests/`, `experiments/`).
- **`pytest -v -m "not gpu"`**: **`69 passed, 1 warning in 1.23s`** (`0` failed, `0` skipped — 63 pre-existing CPU unit tests + 6 new viewing-quality regression tests in [test_viewing_quality_regressions.py](file:///workspace/nvidia-transcribe/tests/test_viewing_quality_regressions.py)).

New regression coverage:
1. `test_a1_untimed_cross_chunk_ordering_and_unit_preservation` — anchors untimed items chronologically inside real chunk spans without touching `None` timestamps.
2. `test_a2_existing_contract_reconstruction_and_classification` — validates `audit_and_reconstruct_contract` classifications (`probable_duplicate`, `plausible_independent_missing_timing`, `likely_hallucination_or_loop`, `uncertain`).
3. `test_b_cache_correctness_invalidation_selective_reuse_and_rerender` — verifies SHA-256 batch invalidation on text/neighbor edits while reusing unaffected batches cleanly.
4. `test_c1_a3_translation_echo_dedup_and_mandarin_fragment_cleanup` — verifies trailing-clause merge dedup and single-char Mandarin particle suppression while keeping `'嗯'`.
5. `test_c2_c4_d_positive_duration_slack_extension_third_lane_and_raised_main` — verifies zero negative durations, slack extension, 3rd ASS lane (`TopAlt`), concurrent duplicate collapse, and `raised_main` margin geometry.
6. `test_c3_selective_condensation_keeps_uncondensed_text` — verifies SwiftU condensation preserves `uncondensed_translated_text`.

## 2. Live Regeneration & Zero-Inference CLI Verification

| Check | Outcome | Observed Timing / Output |
| :--- | :--- | :--- |
| Dual-language regeneration (`run_viewing_quality.py --wait-server`) | **PASS** | FR `284` SRT / `284` ASS (`0` omitted, `0` invalid, `0` dups); ZH `281` SRT / `281` ASS (`0` omitted, `0` leaks, `29/32` condensed, `errors: []`) |
| Condensation fix verification (`max_tokens=4096`) | **PASS** | Resolved earlier `utt-0012` `JSONDecodeError`; final ZH pass achieved `api_calls: 2, cache_hits: 30, errors: []` in `16.145s` |
| Pure CPU `--re-render` CLI check (`generate-subtitles --re-render`) | **PASS** | Rendered ZH SRT/ASS with `--ass-layout raised_main --ass-margin-v 150 --ass-max-lanes 3` in `0.059s` (`refinement/translation/condensation` status: `reused_translated_contract`) |
| GPU post-run idle check (`nvidia-smi`) | **PASS** | `ninfer-serve` daemon terminated; zero evaluation processes left on RTX 5090 |
