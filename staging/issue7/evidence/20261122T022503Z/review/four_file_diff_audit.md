# Four-File Diff Audit (`e824321d5680e61044cdbccc9564ca25d2b2f9e9..4d560467fdcf97c0b01b54a9b7cf711e44317eaf`)

1. **`pyproject.toml` (+1 line, line 14)**
   - Adds `"PyYAML>=6.0"` to `[project].dependencies`.
   - Verified via `tomllib` metadata readback and `yaml.__version__` (`6.0.3` in `.venv`, `6.0.4` in `.venv-studio`). Both CLI `load_glossary` and Studio `GET /api/glossaries` load `configs/glossaries/mcog_fr.yaml` and `configs/glossaries/a_will_eternal_zh.yaml` cleanly.

2. **`src/nvidia_transcribe/studio_api.py` (2 lines changed: lines 140, 492)**
   - Line 140: `Options.condense_high_cps: bool = False` (changed from `True`), aligning omitted API payloads with `INITIAL_OPTIONS.condense_high_cps = false` and single-pass English review. Explicit `condense_high_cps=True` continues to validate, persist, and pass `--condense-high-cps`.
   - Line 492: Stage log string updated from `"Refining, translating and rendering"` to `"Translating, reviewing English and rendering"`, accurately reflecting `--refinement off` in Studio.

3. **`tests/test_studio_api.py` (+37 lines: lines 550-583)**
   - Adds `test_condensation_is_opt_in_for_api_and_persisted_job_options` (asserts `Options().condense_high_cps is False`, `Options(condense_high_cps=True).condense_high_cps is True`, omitted `POST /api/jobs` payload persists `condense_high_cps: false` in `job.json`, and explicit `true` survives validation).
   - Adds `test_yaml_glossary_dependency_is_declared_for_cli_and_studio` (asserts `PyYAML` is declared in `pyproject.toml` `project.dependencies`).

4. **`web/src/App.tsx` (1 line changed: line 411)**
   - Updates glossary field `hint` to `"Optional series-specific names and terms for context-aware English translation review. The source-language ASR transcript is never rewritten in Studio. No example glossary is applied automatically."`
   - Accurately reflects Studio's untouched-ASR + English-review-only workflow.

**Audit Verdict:** `PASS` — zero `BLOCKING` or `IMPORTANT` findings.
