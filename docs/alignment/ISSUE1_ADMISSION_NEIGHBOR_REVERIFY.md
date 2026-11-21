# Issue #1: final admission-neighbor regression qualification (one bounded campaign)

## Why this targeted check exists

Central audited the exact source `28655e0d6fe57d9363e576cd0795a24aae77a4a3` against the accepted original host handback `evidence/issue-1-qwen-20261121@f06cb2a868a701c87c740acbad8c94612d2cd94d`. That run established 107/107 CPU tests in each venv and real RTX 5090 BF16 single-model-load worker reuse. It left managed ASR handoff correctly **blocked** because no cooperatively owned ASR user unit/lease or live 8190 ASR endpoint existed. During independent source review Central found one missed adjacency: a grouped diarization window could cover a trusted CTC-timed neighboring utterance and let a later untimed sentence align against its audio. Central authored correction `0bd775c56d720484e75c1e933628d3b58731a4ef` and two tests. No worker, GPU controller, vLLM launcher, frontend or other source was touched by that correction.

This is a **bounded delta validation** of those revised candidate bounds, plus source-identity reuse of earlier evidence. Do not rerun the entire earlier service campaign, or create another implementation PR.

## Authentication and authority

- `REPO_ID=fvanevski/nemovox`; `ISSUE=1`; `WORKING_BRANCH=issue/1`.
- `BASE_MAIN_SHA=242c418f55cd2afb796ee2c4fa742a10ff04f7a3`.
- `PREVIOUS_QUALIFIED_HEAD=28655e0d6fe57d9363e576cd0795a24aae77a4a3`.
- `CENTRAL_BOUNDS_FIX_COMMIT=0bd775c56d720484e75c1e933628d3b58731a4ef`.
- **Final target**: authenticate the exact current `issue/1` head and ensure it descends from the bounds fix, with changes only in `src/nvidia_transcribe/selective_alignment.py`, `tests/test_qwen_session_and_policy.py`, and this note since the qualified head. If remote source differs, return STALE.
- Read current `AGENTS.md`, issue #1, original `docs/alignment/ISSUE1_QWEN_DESIGN.md`, and this handoff. Live remote SHA and source tree are authoritative; prior test logs bind only to their source SHA.
- Original evidence is a **reference**, not a substitute for newly changed candidate-admission behavior.

## Operator/role and safe setup

Use one OpenCode primary `build` agent on Garuda (this campaign needs fixture/scratch writes, optional narrowly mechanical integration correction, and one GitHub evidence commit). Launch from `/workspace/nemovox-issue1`, the prior dedicated issue-1 checkout, **if it still exists**; otherwise start from `/workspace/nvidia-transcribe` and have the Build agent prepare a clean isolated worktree. Do not use primary Evidence because it cannot perform requisite Git fetch/checkout or archive publication. Never switch, reset, stash, clean or overwrite the live `/workspace/nvidia-transcribe` working tree or interfere with `nemovox-studio.service`.

Fetch `origin/issue/1` **before** local Git object/worktree proof; verify `FETCH_HEAD` SHA equals exact remote branch, commit tree, and full diff against qualified `28655e0...`. Use candidate `PYTHONPATH=<candidate>/src` with existing `.venv`, `.venv-studio`, and isolated `outputs/modular-eval/.venv` Qwen interpreter, verified not to shadow imports. Run tests/benchmarks from a clean candidate source. Prior evidence archive is immutable and must remain intact.

## Mandatory focused regression and admission assertions

1. `python -m compileall -q src/nvidia_transcribe`, then `pytest -v tests/test_qwen_session_and_policy.py tests/test_qwen_fallback.py` on candidate using source-mounted `PYTHONPATH`. Assert the two new tests specifically pass:
   - `test_unique_diar_window_respects_adjacent_native_timed_speech`: one grouped same-speaker diar interval covers three sentences but the sole untimed sentence is clipped between trusted neighbors, never including the first or last timed sentence;
   - `test_neighbor_bounds_fail_closed_if_candidate_overlaps_native_timing`: too-narrow/conflicting neighbor bounds produce `skipped_conflicting_or_long_window`, not a Qwen worker call or invented timestamps.
2. Full `pytest -q -m "not gpu"` in both `.venv` and `.venv-studio`, preserve actual exits. Tests must not be weakened to pass. Target is previous 107 + 2 new = 109 tests, subject to actual collected count.
3. Synthetic adversarial adjacency: inspect/log candidate `(lo, hi, start, end)` with native trusted earlier/later timings; confirm extraction never includes already-timed neighboring speech (aside from the deliberately documented 0.05-second boundary tolerance), across speakers as well as within one speaker. Preserve source text and speaker/timing provenance under candidate application.
4. Recheck default `alignment_fallback=off` no worker and no GPU/service control, mixed overlap skip, and `subtitle` versus `fine` low-value policy.

## Bounded GPU/host qualification

5. Read-only check `nvidia-smi` and service/port occupancy; **never stop or restart** Nemovox Studio, embedding service, vLLM, SwiftU or unrelated consumers. If VRAM remains sufficient, rerun the **same bounded replay harness** from earlier evidence using existing `zh_fleurs_01_2006` and `zh_multichunk_synthetic` Mandarin OOV fixtures and `two_span_oov` if feasible. These replay **recorded** Nemotron raw responses into `process_media` and use the **real CUDA BF16 Qwen worker**, not mocked CUDA. Compare policy `off` against `qwen subtitle`; optionally `fine` on multichunk. Report newly computed candidate ranges and final timed/untimed coverage, zero-duration/invalid statuses, model load and align call counts, actual CUDA/dtype/peak bytes, wall time and pre/post VRAM. Do not claim a multi-span speedup from unrelated fixtures or rerun entire TV episodes.
6. Keep worker and managed controller sources **exactly blob-identical** to previous qualified head. Prior managed lifecycle mock PASS and actual host `MANAGED_GPU_BLOCKED` remain valid evidence only while those relevant sources and tests remain equivalent. Do not invent a new owned ASR unit, service marker or lease; no live managed stop/release/restore in this delta campaign. A real managed result remains UNVERIFIED and cannot be upgraded based on mocks.
7. Any new admission cut clipping the canonical Mandarin OOV anchor incorrectly is a **FAIL** requiring Central semantic remediation; do not widen candidate windows or bypass bounds in the local agent to make the benchmark green.

## Allowed local actions and reporting

- `SOURCE_SEMANTIC_REMEDIATION=forbidden`, `NEW_PR=forbidden`, `MERGE=forbidden`, `ISSUE_STATE_MUTATION=forbidden`.
- Mechanical test/fixture/runtime adaptation permitted only where manifestly value-preserving, bounded, tested, and the full delta is returned; if a source change is unavoidable, commit only with exact branch head readback/no force and rerun invalidated checks on the **new** exact SHA. Preserve central ownership of algorithm and safe service-control decisions.
- Create **one** evidence-only branch `evidence/issue-1-qwen-review-20261121` directly from final implementation SHA, with evidence under `staging/agent-evidence/issue-1-qwen-review/`. One self-contained sanitized `.tar.gz` plus browsable `HANDback.md`, `test-results.md`, `admission-replay.md`, `manifest.json`, bounded raw tests/telemetry and exact diff against previous head; source files must remain unchanged in evidence commit. Include full file hashes, archive SHA-256 and byte count, `gzip -t`, manifest members/digests, privacy scan, GitHub readback and cleanup state. Exclude private media, weights, full episode transcripts, personal file paths, and secrets.
- One consolidated handback: `PASS|FAIL|BLOCKED|STALE`, exact source/evidence SHAs, 2 focused regressions, both CPU-suite totals, measured changed-case replay vs prior accepted measurements, key policy decisions, GPU/cleanup proof, and explicit `MANAGED_GPU_BLOCKED` continuation, plus `COMMAND_SHAPE_FRICTION` canonical ledger (even if empty).
- Stop after evidence publication. Central alone decides PR creation, semantic approval, merge and issue closure; no additional local agent campaign should be initiated on success.

Mandatory ledger:
```text
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events: [<actual bounded events or empty>]
  omitted_sensitive_values=<yes|no>
```
