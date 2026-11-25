# Recognized-Turn Reconstruction (Issue #17, A1) — Local Integration & Qualification Handback

- **RESULT**: `FAIL` (A1 only; 2 candidate-introduced CPU test regressions; `SEMANTIC_REMEDIATION=forbidden`)
- **REPO_ID**: `fvanevski/nemovox`
- **ISSUE_NUMBER**: `17`
- **WORKING_BRANCH**: `issue/17`
- **BASE_SHA**: `9dd863841f72ab5f9aebf5bb4deb322bee99edf1`
- **TESTED_SOURCE_SHA**: `a3b59a7898e6aa0509173d751a2c23409d5185a4`
- **CANDIDATE_WORKTREE**: `/tmp/nemovox-issue17-a1.acb5qC/candidate`
- **UNCOMMITTED_SOURCE_CHANGES**: `none` (0 dirty files in candidate worktree; no mechanical source edits needed)
- **INDEPENDENT_SOURCE_REVIEW_STILL_REQUIRED**: `yes` (Central owns semantic review and remediation)

## 1. Environment & Import-Path Proof

- Core Python: `/workspace/nvidia-transcribe/.venv/bin/python`
  - `nvidia_transcribe.__file__` -> `/tmp/nemovox-issue17-a1.acb5qC/candidate/src/nvidia_transcribe/__init__.py`
- Studio Python: `/workspace/nvidia-transcribe/.venv-studio/bin/python`
  - `nvidia_transcribe.__file__` -> `/tmp/nemovox-issue17-a1.acb5qC/candidate/src/nvidia_transcribe/__init__.py`
- Execution class: `CPU_ONLY` (no ASR/MT/LLM inference, no model downloads, no service restarts, no GPU workload eviction, no `/workspace/pyannote` modifications).

## 2. Deterministic Validation Matrix

| Check | Exit | Result | Artifact |
|---|---:|---|---|
| `compileall` (`src/nvidia_transcribe`, `scripts/replay_source_occurrences.py`) | `0` | PASS | `compileall.log` |
| Focused regression suite (9 test files, `-m "not gpu"`) | `1` | FAIL (`1 failed, 100 passed, 1 warning`) | `focused-tests.log` |
| Full CPU test suite (`-m "not gpu"`) | `1` | FAIL (`2 failed, 436 passed, 1 warning`) | `cpu-tests.log` |
| Studio test suite (`.venv-studio`, `tests/test_studio*.py -m "not gpu"`) | `0` | PASS (`50 passed`) | `studio-tests.log` |
| Exact-base isolation of both implicated tests (`9dd86384...`) | `0` | PASS (`2 passed`) | `base-implicated-tests.log` |
| `git diff --check` | `0` | PASS | `git-diff-check.log` |
| Retained 16-chunk raw-response replay (`scripts/replay_source_occurrences.py`) | `0` | PASS (`16/16` authenticated) | `raw-replay/summary.json` (+ private `raw-replay/private-details.json`) |

## 3. Substantive Failure Diagnoses (Candidate Regressions)

Both failures pass on exact base `9dd863841f72ab5f9aebf5bb4deb322bee99edf1` (`2 passed in 0.12s`) and are newly introduced in candidate `a3b59a7898e6aa0509173d751a2c23409d5185a4`. Per `SEMANTIC_REMEDIATION=forbidden`, neither production code nor test assertions were modified.

### Failure 1: `tests/test_source_quality.py::test_explicit_canonical_resume_preserves_order_evidence_and_messages_review`
- **Error**: `AttributeError: <module 'nvidia_transcribe.subtitle_pipeline'> has no attribute 'audit_and_reconstruct_contract'` (`tests/test_source_quality.py:410`).
- **Root cause**: Candidate `src/nvidia_transcribe/subtitle_pipeline.py` intentionally removed the import and invocation of `audit_and_reconstruct_contract` (eliminating the positional pre-quality rewrite on retained-recording entry), but `tests/test_source_quality.py:409` still monkeypatches `"audit_and_reconstruct_contract"` on `nvidia_transcribe.subtitle_pipeline`.
- **Note on missing entry-path regression coverage**: Removing `audit_and_reconstruct_contract` from `subtitle_pipeline.py` leaves no dedicated regression test asserting that a non-canonical retained recording with untimed utterances enters `generate_subtitles` without positional renumbering or heuristic duplicate relabeling.

### Failure 2: `tests/test_recording_and_normalization.py::test_archived_french_overlap_and_child_voices_regressions`
- **Error**: `AssertionError: assert 12 <= 4` (`tests/test_recording_and_normalization.py:83`).
- **Root cause**:
  - The archived fixture `outputs/responses/overlap/fr_2spk_overlap_sot.raw.json` is a 2-speaker overlapping SOT transcript containing `12` interleaved `<spk:0>` / `<spk:1>` tags.
  - On base `9dd86384...`, `build_transcript_contract()` passes `[u.to_dict() for u in parsed_ctc.units]` (no `recognized_turn_index`), and `reconstruct_speaker_utterances()` reconstructs `2` concurrent speaker utterances (`spk=0` `[0.76..7.616, 23 words]`, `spk=1` `[4.16..10.654, 19 words]`).
  - On candidate `a3b59a78...`, `contract.py:build_transcript_contract()` routes through `recover_chunk_alignment()`, stamping each unit with `recognized_turn_index` (`0..11`), and `utterances.py:reconstruct_speaker_utterances()` (lines 354–361) unconditionally sets `should_split = True` whenever `recognized_turn_id` or `recognized_turn_index` changes. This shatters the two concurrent overlapping speaker streams into `12` micro-utterances (1–3 words each, with same-speaker gaps as small as `0.046s`, e.g., `utt[2]` `4.38..4.634` -> `utt[3]` `4.68..5.49`), failing `assert len(c_ov["utterances"]) <= 4`.

## 4. Authenticated 16-Chunk Raw-Response Replay Analysis

- **Sealed archive**: `/workspace/nvidia-transcribe/outputs/handbacks/source-reconstruction-diagnostic-20261124/source-reconstruction-diagnostic-review.tar.gz`
- **Verified SHA-256**: `307c0a32bd4ae32008be0141d5ac12e066f96a728bfcb179c0b343bf2031fd7b`
- **Ledger & raw byte match**: `16/16` accepted observations matched by exact `raw_response_sha256`.
- **Adjudicated vs. actual counts**:
  - Recognized turns: `128` (exact match to adjudicated `128`)
  - Recognized tokens: `2360` (exact match to adjudicated `2360`)
  - Working units: `2360` (`projection_status == "exact"` on `16/16` chunks; `0` punctuation/CJK/cleanup denominator divergence)
  - Native CTC observations: `2360` (exact match to adjudicated `2360`)
  - Provisional associations: `2360` (`complete_per_tag_sequence_provisional`: `2360`; `0` unmatched, `0` ambiguous, `0` untimed working units, `0` lexical differences between `raw_word.word` and `working_unit.text`)
  - Chunks with repeated speaker tags: `13/16` (exact match to adjudicated `13/16`)
  - `acoustic_locality_certified`: `false` on `16/16` chunks (as required; stream matching does not certify timing correctness)
- **Bounded worst association/timing residuals** (non-transcript structural summary; full details in private archive `raw-replay/private-details.json`):
  - **Turn-start backward inversions** between consecutive recognized turns: `23` total across the 16 chunks. Worst examples:
    - Chunk 13 (`repeated_tags: <spk:0>:3, <spk:1>:3`): `2` inversions, max `90.04s` (turn 4 `<spk:0>` `1269.12s` -> turn 5 `<spk:1>` `1179.08s`; turn 2 `<spk:0>` `1203.17s` -> turn 3 `<spk:1>` `1170.70s`, `32.47s`).
    - Chunk 14 (`repeated_tags: <spk:0>:3, <spk:1>:7, <spk:3>:6`): `3` inversions, max `81.35s` (turn 2 `<spk:0>` `1345.80s` -> turn 3 `<spk:1>` `1264.45s`).
    - Chunk 3 (`repeated_tags: <spk:0>:9, <spk:1>:4, <spk:2>:8`): `8` inversions, max `46.80s`.
    - Chunk 0 (`repeated_tags: <spk:0>:2, <spk:1>:3`): `1` inversion, `15.849s` (turn 4 `<spk:2>` `59.56s` -> turn 5 `<spk:3>` `43.711s`).
  - **Large intra-turn timing gaps** (`> 10s` between consecutive tokens in the same recognized turn): `26` total across the 16 chunks. Worst examples:
    - Chunk 14 turn 4 (`<spk:2>`): `72.884s` (`1282.746 -> 1355.63s`).
    - Chunk 0 turn 2 (`<spk:0>`): `54.788s` (`27.955 -> 82.743s`).
    - Chunk 1 turn 3 (`<spk:1>`): `49.479s` (`98.891 -> 148.37s`).
    - Chunk 13 turn 2 (`<spk:0>`): `49.085s` (`1204.095 -> 1253.18s`).
    - Chunk 10 turn 0 (`<spk:0>`, no repeated tags): `34.71s` (`894.29 -> 929.0s`).

## 5. Changed Files in Candidate (`9dd86384..a3b59a78`)

- `A docs/source-reconstruction/LOCAL_INTEGRATION.md`
- `A scripts/replay_source_occurrences.py`
- `M src/nvidia_transcribe/alignment.py`
- `M src/nvidia_transcribe/contract.py`
- `M src/nvidia_transcribe/recording.py`
- `M src/nvidia_transcribe/source_events.py`
- `A src/nvidia_transcribe/source_occurrences.py`
- `M src/nvidia_transcribe/subtitle_pipeline.py`
- `M src/nvidia_transcribe/translation.py`
- `M src/nvidia_transcribe/utterances.py`
- `A tests/test_source_occurrences.py`

## 6. Canonical Command-Shape Friction

```yaml
COMMAND_SHAPE_FRICTION:
  encountered: no
  events: []
  omitted_sensitive_values: no
```
