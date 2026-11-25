# Bounded Matched Acoustic Geometry Evaluation (Issue #17, A3) — Host Qualification Handback

- **RESULT**: `PASS` (bounded matched GPU experiment executed with valid receipts; NOT a claim that shorter geometry is better or a default promotion)
- **REPO_ID**: `fvanevski/nemovox`
- **ISSUE_NUMBER**: `17`
- **WORKING_BRANCH**: `issue/17`
- **BASE_SHA**: `9dd863841f72ab5f9aebf5bb4deb322bee99edf1`
- **QUALIFIED_SOURCE_SHA**: `5df4ace871940cc71e651b04fbdeaa8f3bc1f69b`
- **PRIOR_EVIDENCE_SHA**: `7a57c2aa2e4d1b1c4526ff769a479acb55207ea5`
- **TESTED_SOURCE_SHA**: `13c0bb89637f1b451e8a3e7b115fa250a9b8de7d`
- **CANDIDATE_WORKTREE**: `/tmp/nemovox-issue17-a3.kJh0Ne/candidate` (clean, `0` dirty files)
- **UNCOMMITTED_SOURCE_CHANGES**: `none`
- **INDEPENDENT_SOURCE_REVIEW_STILL_REQUIRED**: `yes` (Central owns adjudication, policy selection, and #18/#19 decisions)

## 1. Preflight & Host Lifecycle Receipts

- Core Python: `/workspace/nvidia-transcribe/.venv/bin/python` -> `/tmp/nemovox-issue17-a3.kJh0Ne/candidate/src/nvidia_transcribe/__init__.py`
- Preflight exits: `compileall=0`, `pytest` (4 files, `-m "not gpu"`)=`0` (`47 passed, 1 warning`), `git diff --check=0`, `--plan-only=0`.
- Authenticated diagnostic archive: `/workspace/nvidia-transcribe/outputs/handbacks/source-reconstruction-diagnostic-20261124/source-reconstruction-diagnostic-review.tar.gz` (SHA-256 `307c0a32bd4ae32008be0141d5ac12e066f96a728bfcb179c0b343bf2031fd7b`); media fingerprint `8d6e97fa0c3b0c674baa4e1864e72ada463dd4f448c7f9e3e6aacfb8249548e9`, stream `0`, duration `1379.901s`, language `fr`.
- Managed ASR lifecycle (`nemovox-asr.service`, resource `asr_local`, checkpoint `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`):
  - Acquired once under `managed_asr_client(ModelLifecycle(), resource, client)` (`Started` `17:15:35`, `Application startup complete` `17:16:58`, `Stopped` `17:17:30`, `ActiveState=inactive`, `Result=success`, `ExecMainStatus=0`).
  - Verified in server journal: `quantization=modelopt_mixed`, `kv_cache_dtype=fp8`, `trust_request_mm_kwargs=True`, `middleware=['nemo.collections.speechlm2.vllm.salm.ctc_serving.ctc_timestamp_middleware']`, `served_model_name=['nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4']`; `start_vllm_server.sh` exports `VLLM_USE_V2_MODEL_RUNNER=0`.
  - GPU memory (RTX 5090): `1973 MiB` used / `30154 MiB` free before and after; no external GPU services stopped or modified; no `/workspace/pyannote` changes.
- **Inference totals**: `28` requests (budget `72`), `1434.0s` audio (budget `2000.0s`), `109.6766s` total request wall time (`compression/fixed` first cold-JIT request took `91.7228s`; all subsequent arms took `1.16–2.85s`). `28/28` `status=returned`, `28/28` `finish_reason=stop`, `28/28` `ctc_present=True`, `0` CTC errors, `0` rejected generations.

## 2. Four-Region Matched Comparison (Favorable, Unfavorable & Insufficient-Evidence Findings)

Both policy arms in every region consumed identical PCM bytes (verified by identical `audio_sha256`), achieved `exact` projection on all chunks, gap-free core coverage, and complete source-unit-to-utterance identity coverage. `acoustic_truth_accepted=false` throughout; two-crop support is same-model positive support only.

| Region (Target) | Ref Events | Fixed (`90/10s`, 2 ch, 130s) | Pause-Aware (`39/46/4s`, 3 ch, 126s) | Key Comparative Observation |
|---|---:|---|---|---|
| `compression` (`298–322s`) | `2` | `0/2` supported (`1` raw lex, `0` sel lex); `351` units, `9` unloc, `54` ctx, `1` partial span, `32/33` strictly timed | `1/2` supported (`[304.8, 307.045]`, onset `-0.0208s`, end `+0.7742s`); `1` pause + `1` forced cut (`78.42s`); `331` units, `18` unloc (`17` unloc spans), `14` ctx, `0` partial, `19/27` strictly timed | Mixed: `pause_aware` gained 1 supported event and reduced `ctx`/`partial`, but increased `unlocalized` spans (`0 -> 17`) and lowered strictly timed rows (`32 -> 19`). |
| `boundary` (`609–643s`, incl. `620–622s`) | `4` | `2/4` supported (`3` raw/sel lex); `191` units, `10` unloc (`6` unloc spans), `11` ctx, `0` partial, `14/18` strictly timed (`1` wholly untimed) | `2/4` supported (`2` raw/sel lex); `1` forced cut (`46.0s`) + `1` pause (`88.19s`); `190` units, `0` unloc, `2` ctx, `2` partial spans, `14/14` strictly timed (`0` untimed) | Trade-off: `pause_aware` supported `[621.502, 623.762]` (`+0.028s`) where `fixed` missed strict match and eliminated untimed/unlocalized rows, **but** lost `[609.432, 621.544]` (which `fixed` supported at `-0.112s`) and added `2` partial provisional spans. On the `1` shared supported event `[639.97, 642.949]`, paired deltas were `0.070s -> 0.044s` (onset) and `0.051s -> 0.023s` (end). |
| `late_gap` (`1212–1252s`) | `0` | `122` units (`115` sel, `7` alt), `10/10` strictly timed | `1` forced cut (`46.0s`) + `1` pause (`80.42s`); `98` units (`98` sel, `0` alt), `9/9` strictly timed | **Insufficient reference evidence** (`0` corroborated reference events between `local_a` 11 units and `local_b` 25 units). `pause_aware` emitted fewer units (`98` vs `122`). |
| `low_flag_control` (`470–505s`) | `0` | `100` units (`99` sel, `1` ctx), `0` partial spans, `14/14` strictly timed | `2` pauses (`37.08s`, `58.49s` early_long), `0` forced cuts; `121` units (`97` sel, `19` ctx, `5` alt), `1` partial span, `15/15` strictly timed | **Insufficient reference evidence** (`0` corroborated events between `local_a` 19 utts and `local_b` 5 utts). `pause_aware` had `1` partial provisional span and `19` `context_candidate_unresolved` (vs `0` and `1` in `fixed`). |

### Planner & Cost Summary
- `pause_aware` ran `3` chunks / `126.0s` audio per 110s region (vs `fixed` `2` chunks / `130.0s` audio — slightly less audio, +1 request per region), requiring forced max-core cuts (`fallback_used=True`) in `3/4` regions (`compression`, `boundary`, `late_gap`), and using purely detected pauses (`0` forced cuts) only in `low_flag_control`.

## 3. Changed Files (`7a57c2aa..13c0bb89`)

`docs/source-reconstruction/LOCAL_GEOMETRY_EVALUATION.md`, `scripts/evaluate_chunk_geometry.py`, `tests/test_chunk_geometry_evaluation.py`.

## 4. Canonical Command-Shape Friction

```yaml
COMMAND_SHAPE_FRICTION:
  encountered: no
  events: []
  omitted_sensitive_values: no
```
