# Final Occurrence-Disposition Qualification (Issue #17) — Local CPU Handback

- **RESULT**: `PASS` (bounded final CPU qualification only; not issue #17 closure or PR creation)
- **REPO_ID**: `fvanevski/nemovox`
- **ISSUE_NUMBER**: `17`
- **WORKING_BRANCH**: `issue/17`
- **BASE_SHA**: `9dd863841f72ab5f9aebf5bb4deb322bee99edf1`
- **PRIOR_CPU_QUALIFIED_SHA**: `5df4ace871940cc71e651b04fbdeaa8f3bc1f69b`
- **PRIOR_GPU_TESTED_SHA**: `13c0bb89637f1b451e8a3e7b115fa250a9b8de7d`
- **PRIOR_GPU_EVIDENCE_SHA**: `cc8907e7402e230e53696da770faa20a95d75b23`
- **TESTED_SOURCE_SHA**: `24f05c71b533ea7eb830ca7215f1690abcf415fc`
- **CANDIDATE_WORKTREE**: `/tmp/nemovox-issue17-disp.uonvyW/candidate` (clean, `0` dirty files)
- **UNCOMMITTED_SOURCE_CHANGES**: `none`
- **INDEPENDENT_SOURCE_REVIEW_STILL_REQUIRED**: `yes` (Central owns independent exact-head PR review and subsequent #18 decisions)

## 1. Environment & Import-Path Proof

- Core Python: `/workspace/nvidia-transcribe/.venv/bin/python`
  - `nvidia_transcribe.__file__` -> `/tmp/nemovox-issue17-disp.uonvyW/candidate/src/nvidia_transcribe/__init__.py`
- Studio Python: `/workspace/nvidia-transcribe/.venv-studio/bin/python`
  - `nvidia_transcribe.__file__` -> `/tmp/nemovox-issue17-disp.uonvyW/candidate/src/nvidia_transcribe/__init__.py`
- Execution class: `CPU_ONLY` (no GPU inference, no model service restarts, no A3 archive replay, no full-episode run, no `/workspace/pyannote` modifications).

## 2. Deterministic Validation Matrix

| Check | Exit | Outcome | Artifact |
|---|---:|---|---|
| `compileall` (`src/nvidia_transcribe`, `scripts/evaluate_chunk_geometry.py`) | `0` | PASS | `compileall.log` |
| Focused tests (7 files, `-m "not gpu"`) | `0` | `89 passed, 1 warning` | `focused-tests.log` |
| Full CPU suite (`-m "not gpu"`) | `0` | `454 passed, 1 warning` | `cpu-tests.log` |
| Studio suite (`.venv-studio`, `tests/test_studio*.py -m "not gpu"`) | `0` | `50 passed` | `studio-tests.log` |
| `git diff --check` | `0` | PASS | `git-diff-check.log` |

- New regressions verified in `tests/test_source_occurrences.py` (with all prior overlap, clean-control, genuine-repeat, source-integrity, and resume assertions preserved unchanged):
  - `test_boundary_dispositions_do_not_assert_transitive_or_ownerless_coverage`
  - `test_boundary_matching_two_context_spans_exposes_unselected_counterpart`
  - `test_boundary_disposition_invalid_reference_fails_closed`

## 3. Changed Files (`cc8907e7..24f05c71`)

`docs/source-reconstruction/LOCAL_DISPOSITION_QUALIFICATION.md`, `scripts/evaluate_chunk_geometry.py`, `src/nvidia_transcribe/chunking.py`, `src/nvidia_transcribe/occurrence_dispositions.py`, `src/nvidia_transcribe/recording.py`, `tests/test_source_occurrences.py`.

## 4. Canonical Command-Shape Friction

```yaml
COMMAND_SHAPE_FRICTION:
  encountered: no
  events: []
  omitted_sensitive_values: no
```
