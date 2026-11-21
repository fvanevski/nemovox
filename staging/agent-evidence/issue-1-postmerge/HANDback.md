# Nemovox Issue #1 Merged-Main Host Deployment & Acceptance Handback

## Acceptance Matrix (Criterion -> Source -> Verification -> Result)

| Criterion | Source Authority | Verification & Evidence | Result |
| --- | --- | --- | --- |
| Exact merged `main` identity & clean promotion | `7b4112cc31784953abbb041fd0b5d7997851841d`, tree `3d2cc429f9b49da3079df7d1fc178f6fd09d7987` | Authenticated PR #4 `MERGED`, fast-forwarded `/workspace/nvidia-transcribe` from `242c418f` while Studio idle ([`source-provenance.md`](source-provenance.md), [`host-promotion.md`](host-promotion.md)) | **PASS** |
| Fresh-process Studio acceptance & job persistence | `studio_api.py`, `nemovox-studio.service` | Stopped `4157910` -> started `1006972`, verified `127.0.0.1:8765`, `/api/health`, `/`, hashed JS/CSS, `/api/jobs` + preserved `9be92af3ba144e6eb984ee43af0807de` ([`runtime-health.md`](runtime-health.md)) | **PASS** |
| Merged-main CPU regressions & CLI flags | `src/nvidia_transcribe/*.py`, `tests/` | `compileall` exit 0; focused `57 passed`; `.venv` `120 passed`; `.venv-studio` `120 passed`; `--help` verified ([`cpu-tests.md`](cpu-tests.md)) | **PASS** |
| Bounded CUDA BF16 Qwen alignment replay | `qwen_aligner.py`, `_qwen_align_worker.py`, `selective_alignment.py` | `two_span_oov` (`42/42`, `model_load_count=1`, `align_count=2`, `cuda:0` `torch.bfloat16`, `1.3455s` -> `0.0365s`); `zh_multichunk` (`197/198`); `0` VRAM leak ([`gpu-alignment.md`](gpu-alignment.md)) | **PASS** |
| Read-only managed-ASR feasibility | `gpu_handoff.py`, host systemd/ports/launchers | No `nemovox-asr.service`, no cooperative lease/markers, `8190` closed, `scripts/start_vllm_server.sh` interactive ([`managed-asr-feasibility.md`](managed-asr-feasibility.md)) | **MANAGED_GPU_BLOCKED** |
| Unrelated services & workspace boundaries | `AGENTS.md`, `nemovox-studio.service`, `local_embedding.service` | Both services `active`; `/workspace/pyannote` & `/workspace/models` untouched; zero orphan workers | **PASS** |

## Summary Verdicts

```text
POSTMERGE_SOURCE_IDENTITY=PASS
HOST_PROMOTION=PASS
STUDIO_FRESH_PROCESS=PASS
CPU_REGRESSIONS=PASS
GPU_QWEN_AVAILABLE=PASS
MANAGED_ASR_FEASIBILITY=MANAGED_GPU_BLOCKED: no cooperatively owned ASR unit/lease
LIVE_ASR_RELEASE_RESTORE=NOT_TESTED
ISSUE1_CLOSURE_RECOMMENDATION=REQUIRES_OWNERSHIP_INTEGRATION
GITHUB_MAIN_SHA=7b4112cc31784953abbb041fd0b5d7997851841d
MERGED_MAIN_TREE=3d2cc429f9b49da3079df7d1fc178f6fd09d7987
PRIOR_LIVE_SHA=242c418f55cd2afb796ee2c4fa742a10ff04f7a3
FINAL_LIVE_SHA=7b4112cc31784953abbb041fd0b5d7997851841d
SERVICE_PID_BEFORE=4157910
SERVICE_PID_AFTER=1006972
ARCHIVE_PATH=staging/agent-evidence/issue-1-postmerge/nemovox-issue-1-postmerge-host-evidence.tar.gz
```

## Remaining Risks & Closure Note

- **Managed-ASR ownership gap:** Live `--qwen-gpu-mode managed` ASR release/restore remains `NOT_TESTED` (`MANAGED_GPU_BLOCKED`) because the host has no `nemovox-asr.service` user unit, no cooperative lease file/markers, and no lease participation in `scripts/start_vllm_server.sh` or `studio_api.py`. Central must decide whether to implement unit/lease integration before closing issue #1's live managed-handoff acceptance criterion, or accept CPU/mock lifecycle + `available` CUDA qualification.
- **Conservative multi-diar partial skip (`fine` mode):** As previously noted in PR #4 review, `zh_multichunk_synthetic` (`--qwen-policy fine`) conservatively skips `[44, 66]` (`skipped_partial_span_unanchored_edge`) when speaker `1` has 2 diarization intervals in chunk 0.

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events: []
  omitted_sensitive_values=no
```
