# Issue #5 Integration & Qualification Findings

## Blocking Findings
- None remaining after local repair.
- Initial `pytest -q -m "not gpu"` failed on `tests/test_qwen_session_and_policy.py::test_qwen_cli_policy_defaults_and_explicit_managed_options` (`assert off.qwen_device == "cuda:0"`, observed `None`) because Central's `cli.py` changed `--qwen-device` parser default from `"cuda:0"` to `None` so `forced_alignment` resource config (`qwen_local.device`) takes precedence when the CLI flag is omitted. Repaired locally in `tests/test_qwen_session_and_policy.py` (see `review/proposed_local_diff.txt`) and re-ran full CPU suite (144 passed).

## Nonblocking Findings
- Live ASR (`127.0.0.1:8190`) and SwiftU/NInfer text (`127.0.0.1:8001`) services were not running during this campaign (`Connection refused`); per contract, live inference was not started/restarted and both remain `UNVERIFIED`.
- In `src/nvidia_transcribe/cli.py`, when `--translator none --condense-high-cps` is passed, `condensation_resource.credential()` is still checked even though `subtitle_pipeline.py` skips condensation when `translation.backend == "none"`. Only triggers if `subtitle_condensation` defines an unset `api_key_env`.
- In `src/nvidia_transcribe/subtitle_pipeline.py`, programmatic callers invoking `generate_subtitles` with `translation=TranslationConfig(backend="riva")`, `condensation=None`, and `condense_high_cps=True` rely on `cfg.condensation or cfg.translation` (CLI always populates `condensation=cond_cfg`).

## Documentation Audit Findings (for Central)
- `README.md`: Still describes only `.env` / `configs/vllm_rtx5090.env.example` and `--base-url` without documenting `configs/model-resources.example.json`, `NEMOVOX_MODEL_RESOURCES`, `TEXT_BASE_URL`/`TEXT_MODEL`, `model_resources.py` in Repository Layout, `--model-resources`, `--*-resource`, `--text-url`/`--text-model`, or `--translator openai_chat`.
- `web/README.md`: Lines 37 and 44 still refer to "vLLM or SwiftU" and "SwiftU refinement"; UI now defaults to `openai_chat` ("Configured text model · contextual") with SwiftU retained as legacy option.
- `docs/gui/ARCHITECTURE.md`: Line 46 still refers to "SwiftU profile/style".
