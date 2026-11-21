# PR #4 Codex remediation: single bounded exact-head qualification

## Source and task

Repository `fvanevski/nemovox`; PR #4 (existing consolidated issue #1 PR); branch `issue/1`.
Base remains `main@242c418f55cd2afb796ee2c4fa742a10ff04f7a3`, unless current remote authority changed.
Previously CUDA-qualified PR head: `d5700b47a4e2f1abb60ebe99f53750c254de2696`, evidence `evidence/issue-1-qwen-review-20261121@a2c5479d5a7f2bbb8420dd6fb1f92adcd8833185`. It **does not establish PASS** for the new source below.

Codex review `6008270229` at the prior head reported three verified blocking issues:

1. P1 [review thread](https://github.com/fvanevski/nemovox/pull/4#discussion_r4655089438): substring cooperative lease marker matching may authorize an unrelated lease. Central corrected *both* ordinary admission and explicit crash recovery with exact parsed/unique assignments.
2. P2 [review thread](https://github.com/fvanevski/nemovox/pull/4#discussion_r4655089439): partially timed utterance crops with untimed prefix/suffix may exclude that speech. Central requires one uniquely grouped/overlapping same-speaker diar interval for missing edge and applies native neighbor/window bounds, otherwise skips without fake timing.
3. P2 [review thread](https://github.com/fvanevski/nemovox/pull/4#discussion_r4655089443): `systemctl_unavailable` after stop can be misclassified as optional deferral. Central introduced `ManagedASRLifecycleError` for any post-stop failure or unverifiable restoration; recording propagates it as fatal; pre-stop admission blocks remain deferrable.

**Controlling implementation SHA:** `5e56304ec848f43807cf9ad91d61350090fd7525` is the Central-authored fix-and-test checkpoint; validate the fresh *current* `issue/1` head, including the documentation commit containing this handoff. Verify that the changes since `d5700b4...` are confined to `gpu_handoff.py`, `selective_alignment.py`, `recording.py`, `tests/test_qwen_session_and_policy.py`, and the two touched docs. If any unrelated remote movement occurs, stop as STALE. Read `AGENTS.md`, issue #1 and `docs/alignment/ISSUE1_QWEN_DESIGN.md` before execution.

## Authority, host and scope

Run OpenCode primary `build` on Garuda, preferably from the already isolated `/workspace/nemovox-issue1` checkout; use the existing `.venv`, `.venv-studio`, and dependency-isolated `outputs/modular-eval/.venv` *via source-bound PYTHONPATH*. The live checkout `/workspace/nvidia-transcribe`, `nemovox-studio.service`, `local_embedding.service`, vLLM, model checkpoints, SwiftU and all unrelated services are out of scope. No service stop/start/restart or ASR ownership marker/lease creation, no full-episode processing, no new PR, merge or issue mutation. CPU tests and bounded optional CUDA are permitted when available VRAM is safe.

This is one **post-Codex remediation qualification**, not a new implementation initiative. Resolve trivial environment or fixture breakage and syntax/import problems if proved and bounded; do not silently rewrite the safety protocol, timing semantics, owner identity, lease policy, source-language text or trusted timing behavior. If a substantive defect remains, return its reproducer and FAIL for Central. Preserve exact provenance after any local source correction; no force writes.

## Required checks

1. Fetch `origin/main` and `origin/issue/1`; authenticate exact SHAs/tree, changed files, and runtime imports before execution. Preserve dirty worktrees by using a new isolated worktree when necessary.
2. Compile package with `python -m compileall -q src/nvidia_transcribe`; run focused Qwen tests `tests/test_qwen_session_and_policy.py tests/test_qwen_fallback.py`. Verify *all newly added* exact-lease/prefix/duplicate/quoted-marker, crash-recovery wrong-lease, partial leading and trailing untimed edge, ambiguous-edge fail-closed, post-stop systemctl transport failure, and recording-level fatal propagation tests. Existing regressions may not be weakened.
3. Run full `pytest -q -m "not gpu"` in the existing `.venv` and `.venv-studio`, report actual counts/exits. The previously qualified head had 109 tests; additional parameterized cases were added. Do not assert an expected count instead of actual pytest output.
4. For ownership controls, use only mocks/fake user-unit inputs: prove exact marker matches, reject `/tmp/a.lock.other` for requested `/tmp/a.lock`, duplicates and malformed values; prove `recover_pending_managed_asr` does not start wrong-lease unit. Test a synthetic post-stop start timeout/`systemctl_unavailable` that raises fatal `ManagedASRLifecycleError`, leaves a pending journal and does not return an accepted completed job. Confirm genuine pre-stop `ManagedASRBlocked` is still deferred without touching a service. No live managed ASR takeover is allowed by this campaign.
5. For selection, inspect candidate spans in adversarial *partially* timed utterances whose native-timed anchor lies at `7..8s` but which have long untimed leading or trailing speech; verify crop contains the uniquely diarized missing edge, respecting max duration, neighboring native timing and overlap protections. With absent/ambiguous diarization, assert reason-coded skip, not invented timing. Recheck existing intra-utterance pause grouping, native-neighbor clipping and `subtitle` vs `fine` admission.
6. If the RTX 5090 has demonstrably spare VRAM, run just the prior bounded real-Qwen BF16 replay fixtures most likely affected by window selection: `zh_multichunk_synthetic` (subtitle and fine) and optionally `two_span_oov`. The Nemotron inputs must be clearly labeled **recorded ASR replay**. Compare exact candidate windows, subtitle-relevant recovered timing, worker `model_load_count`, actual `cuda:0`/observed dtype, peak VRAM and process cleanup against prior qualified `d5700b4` receipts. If GPU unavailable, mark GPU delta BLOCKED, retain earlier worker-source-equivalent evidence, and do not substitute CPU silently. Do not reprocess whole episodes.
7. Confirm PR #4 still points to candidate head, no unrelated service was altered, and `main` remains unchanged. Managed ASR remains `MANAGED_GPU_BLOCKED` due to no owned live ASR unit/lease; mock behavior is not real stop/release/restore acceptance.

## Evidence and stop

Publish **one** self-contained sanitized `.tar.gz` with member SHA-256s, terminal command exits, exact source/head/tree, focused/full CPU results, bounded GPU replay results if performed, fake ownership/lifecycle trace, per-Codex-finding regression mapping and full bounded source delta. No private media/transcripts, credentials, unredacted logs or model weights. Publish browsable `HANDback.md`, `test-results.md`, `codex-remediation.md`, `gpu-replay.md` (or explicit unavailable), `manifest.json`, test logs and archive under `staging/agent-evidence/issue-1-codex/` on one new evidence-only branch `evidence/issue-1-codex-20261121` **based directly on the final exact implementation SHA**. If branch exists, inspect before write; never force-push. Compute archive hash/size, check gzip/member integrity, scan for secrets and verify GitHub ref/file readback.

Return `PASS|FAIL|BLOCKED|STALE`, exact final `issue/1` and evidence commit SHAs, all new regression results, CPU and GPU receipts, managed ownership status, any correction diff, archive hash/bytes, cleanup/service invariants and canonical `COMMAND_SHAPE_FRICTION` ledger. Stop. Central alone adjudicates Codex findings/PR review, consumes the remaining **one** manual Codex request slot (PR #4 default cap 2; one request already created), performs formal approval and decides later merge. No new PR or merge during this handoff.

Canonical ledger:
```text
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events: [<bounded events, or empty>]
  omitted_sensitive_values=<yes|no>
```
