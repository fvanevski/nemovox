# Recognized-Occurrence Locality (Issue #17, A1 Remediation + A2) — Consolidated Local Qualification Handback

- **RESULT**: `PASS` (bounded CPU qualification only; not issue #17 closure)
- **REPO_ID**: `fvanevski/nemovox`
- **ISSUE_NUMBER**: `17`
- **WORKING_BRANCH**: `issue/17`
- **BASE_SHA**: `9dd863841f72ab5f9aebf5bb4deb322bee99edf1`
- **PRIOR_TESTED_SHA**: `a3b59a7898e6aa0509173d751a2c23409d5185a4`
- **PRIOR_EVIDENCE_SHA**: `b97c7f399c057deea4433dd5beb73719e4ae7b6f`
- **TESTED_SOURCE_SHA**: `5df4ace871940cc71e651b04fbdeaa8f3bc1f69b`
- **CANDIDATE_WORKTREE**: `/tmp/nemovox-issue17-a2.xa2g1o/candidate` (clean, `0` dirty files)
- **UNCOMMITTED_SOURCE_CHANGES**: `none`
- **INDEPENDENT_SOURCE_REVIEW_STILL_REQUIRED**: `yes` (Central owns final adjudication and subsequent A3/#18 decisions)

## 1. Environment & Import-Path Proof

- Core Python: `/workspace/nvidia-transcribe/.venv/bin/python`
  - `nvidia_transcribe.__file__` -> `/tmp/nemovox-issue17-a2.xa2g1o/candidate/src/nvidia_transcribe/__init__.py`
- Studio Python: `/workspace/nvidia-transcribe/.venv-studio/bin/python`
  - `nvidia_transcribe.__file__` -> `/tmp/nemovox-issue17-a2.xa2g1o/candidate/src/nvidia_transcribe/__init__.py`
- Execution class: `CPU_ONLY` (no ASR/MT/LLM inference, no model downloads, no service restarts, no GPU workload eviction, no `/workspace/pyannote` modifications).

## 2. Deterministic Validation Matrix

| Check | Exit | Outcome | Artifact |
|---|---:|---|---|
| `compileall` (`src/nvidia_transcribe`, `scripts/replay_source_occurrences.py`) | `0` | PASS | `compileall.log` |
| Focused tests (11 files, `-m "not gpu"`) | `0` | `154 passed, 1 warning` | `focused-tests.log` |
| Full CPU suite (`-m "not gpu"`) | `0` | `445 passed, 1 warning` | `cpu-tests.log` |
| Studio suite (`.venv-studio`, `tests/test_studio*.py -m "not gpu"`) | `0` | `50 passed` | `studio-tests.log` |
| `git diff --check` | `0` | PASS | `git-diff-check.log` |
| 16-chunk matched offline replay (`scripts/replay_source_occurrences.py`) | `0` | PASS (`16/16` chunks) | `raw-replay/summary.json` (+ private `private-details.json`, `private-selection.json`) |

### Verification of Prior Regression Remediation
- `tests/test_recording_and_normalization.py` is **unchanged** from base `9dd86384...`; `test_archived_french_overlap_and_child_voices_regressions` now passes without weakening.
- `tests/test_source_quality.py::test_explicit_canonical_resume_preserves_order_evidence_and_messages_review` asserts `not hasattr(pipeline, "audit_and_reconstruct_contract")` while retaining all canonical-resume assertions, and passes.
- New `tests/test_source_quality.py::test_retained_noncanonical_source_preserves_immutable_ids_order_and_lineage` and expanded `tests/test_source_occurrences.py` (simultaneous speech, occurrence locality, genuine repetition, boundary ownership, alignment admission) all pass.

## 3. Extended 16-Chunk Replay & Matched Selection Ablation

- **Sealed archive**: `/workspace/nvidia-transcribe/outputs/handbacks/source-reconstruction-diagnostic-20261124/source-reconstruction-diagnostic-review.tar.gz`
- **Verified SHA-256**: `307c0a32bd4ae32008be0141d5ac12e066f96a728bfcb179c0b343bf2031fd7b`
- **Lexical & Locality Totals (`16/16` chunks, `exact` projection, `acoustic_locality_certified: false`, `acoustic_truth_accepted: false`)**:
  - `128` recognized turns, `2360` recognized tokens / working units / native CTC observations / provisional associations
  - `261` locality spans: `257` `provisional`, `4` `disputed`
  - `36` locality findings: `30` `large_intra_turn_gap_review_only`, `6` `nonchronological_turn_hypothesis`
  - `45` post-locality untimed units (chunk 3: `14`, chunk 9: `14`, chunk 11: `17`)
- **Matched Offline Selection Comparison (Control vs. Candidate)**:
  - `all_original_observations`: `2360`
  - Control: `selected_units=1993` (`selected_provisional=1912`, `selected_unlocalized=81`), `alternate_not_selected=200`, `context_candidate_unresolved=167`, `utterances=229`, `wholly_untimed_utterances=10`
  - Candidate: `selected_units=1993` (`selected_provisional=1912`, `selected_unlocalized=81`), `alternate_not_selected=199`, `context_candidate_unresolved=168`, `utterances=231`, `wholly_untimed_utterances=12`
  - `selected_only_by_control=33`, `selected_only_by_candidate=33`
  - Disposition transitions (control -> candidate): `selected_provisional -> alternate_not_selected: 33`, `alternate_not_selected -> selected_provisional: 33`, `alternate_not_selected -> context_candidate_unresolved: 1`

### Bounded Inspection of Changed / Disputed Occurrences
1. **Coherent (`provisional`) span partial-selection behavior**:
   - Partial provisional spans dropped from `8` (control) to `3` (candidate): `6` control-partial spans became wholly owned (`0` or `all`) across 3 boundary pairs (`chunk 6 span_346f827e` `13/37 -> 0/37` / `chunk 7 span_413d6035` `25/38 -> 38/38`; `chunk 12 span_4e123262` `20/28 -> 28/28` / `chunk 13 span_ed35ba3f` `8/28 -> 0/28`; `chunk 14 span_cc87e3d5` `9/18 -> 18/18` / `chunk 15 span_79906dd1` `9/18 -> 0/18`).
   - **1 previously whole provisional span became partial**: `chunk 6 span_1e597be159d797c2311853c4` (`[609.4, 621.539]`, `26` units) moved from `26/26` (control) to `23/26` (candidate), with `3` units moving to `alternate_not_selected` (with `related_source_unit_ids` populated) while `3/3` units in `chunk 7 span_7904dcd84db72e34b77cffea` moved `alternate_not_selected -> selected_provisional`.
   - Remaining 2 partial provisional spans were already partial in control and unchanged: `chunk 2 span_79e8b12f37f326e98b785472` (`19/27`, `[265.48, 271.944]`), `chunk 10 span_3ff9381eb1ceb1d4bbe1ebcb` (`3/22`, `[990.28, 996.725]`).
2. **Disputed spans (`4` total; all retained `100%` in selection)**:
   - `chunk 3 span_13e11de8fcfbcd6d525a28d6` (`14/14` selected, untimed, crop `[260.0, 370.0]`)
   - `chunk 9 span_fe55d345e5a845edfd11471e` (`4/4` selected, `[831.88, 832.095]`, crop `[800.0, 910.0]`)
   - `chunk 9 span_f321f370dc1b19bc95e80d2e` (`11/11` selected, untimed, crop `[800.0, 910.0]`)
   - `chunk 11 span_b19a916b2e3dbf0e7612716d` (`17/17` selected, untimed, crop `[980.0, 1090.0]`)
3. **Unresolved context candidates & alternate references**:
   - All `199/199` `alternate_not_selected` dispositions carry non-empty `related_source_unit_ids`.
   - All `168/168` `context_candidate_unresolved` dispositions carry explicit `source_unit_id`, `occurrence_span_id`, `reason: "outside_selected_core_requires_occurrence_adjudication"`, `timing_truth_accepted: false`, `occurrence_truth_accepted: false`, and `related_source_unit_ids: []`.
   - `1` unit (`unit_7ef74038466dec440837ec54`, `chunk 12 span_d9d923d1dca478324c5fab62`, `[1176.12, 1180.0]`) transitioned `alternate_not_selected` (`ctrl_refs=['unit_2b368748491f2533e647011d']`) -> `context_candidate_unresolved` (`cand_refs=[]`).
   - Note: Observed association/selection metrics are policy-ablation outputs only, not independent acoustic ground truth.

## 4. Changed Files (`b97c7f39..5df4ace8`)

`docs/source-reconstruction/LOCAL_QUALIFICATION.md`, `scripts/replay_source_occurrences.py`, `src/nvidia_transcribe/boundary_evidence.py`, `src/nvidia_transcribe/chunking.py`, `src/nvidia_transcribe/contract.py`, `src/nvidia_transcribe/occurrence_locality.py`, `src/nvidia_transcribe/recording.py`, `src/nvidia_transcribe/selective_alignment.py`, `src/nvidia_transcribe/source_events.py`, `src/nvidia_transcribe/source_quality.py`, `src/nvidia_transcribe/utterances.py`, `tests/test_source_occurrences.py`, `tests/test_source_quality.py`.

## 5. Canonical Command-Shape Friction

```yaml
COMMAND_SHAPE_FRICTION:
  encountered: no
  events: []
  omitted_sensitive_values: no
```
