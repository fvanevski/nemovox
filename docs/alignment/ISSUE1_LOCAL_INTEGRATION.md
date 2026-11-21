# Issue #1: cohesive local-host integration and evidence qualification

## Task identity and authority

- Repository: `fvanevski/nemovox`, issue: https://github.com/fvanevski/nemovox/issues/1
- **Implementation branch:** `issue/1`, created from `main@242c418f55cd2afb796ee2c4fa742a10ff04f7a3` (the source tree at issue start).
- **Implementation exact head:** obtain from kickoff; re-read live GitHub and abort if it changes unexpectedly. Do not assume this document's commit is the final head.
- Scope: one cohesive source integration/qualification of centrally authored admission, persistent worker, native two-phase sequencing, explicit device, managed ASR ownership, CLI propagation and CPU regressions.
- Architecture: read `docs/alignment/ISSUE1_QWEN_DESIGN.md` and current `AGENTS.md`. Current GitHub source and issue criteria override historical handoffs.

**Harness:** `LOCAL_AGENT_PRIMARY=build` under OpenCode/other explicitly chosen host agent. Local agent is authorized to perform host-bound tests, dependency verification, bounded fixture instrumentation, and **small evidence-driven in-scope mechanical/integration corrections** (e.g. imports, protocol parsing, test fixture/bindings, backend API adaptation), with focused regression and full diff returned. This is not a transfer of architectural or substantial production development. Never silently alter admission semantics, model-service ownership, GPU safety, native timing, source/spk fidelity or release gates. Escalate an architecture change or material safety invariant breach rather than bypass.

**Git authority:** one `issue/1` working branch and at most one later consolidated PR, created **by Central only after evidence review**. No new PR, merge, issue state change, force push, rebasing origin, source changes on main, or evidence branch merge. If a small local correction must be committed to `issue/1`, use a normal fast-forward push after exact remote-head recheck; send resulting full SHA and full diff. For purely test-run failures from environment, repair the test environment, not production.

## A. Authenticate and perform CPU-first validation

1. Authenticate `origin/main` and `origin/issue/1` fresh; read full current issue #1, AGENTS, this handoff and design. Run `git fetch origin refs/heads/issue/1` before testing newly remote objects, prove full SHA/tree with `git cat-file` and `git rev-parse`. Prefer a clean separate worktree/detached checkout under collision-safe `/tmp` if existing `/workspace/nvidia-transcribe` is dirty or is running the persistent Studio service. Do not reset/clean/stash source. Ensure real `nvidia_transcribe` imports point to the candidate source, not an installed package shadow.
2. Check `python -m compileall -q src/nvidia_transcribe` (use the appropriate interpreter/PYTHONPATH). Run focused `tests/test_qwen_fallback.py`, `tests/test_qwen_session_and_policy.py`, `tests/test_recording_and_normalization.py`, then complete CPU `pytest -q -m "not gpu"` in `.venv` and `.venv-studio` where dependencies permit. Run existing frontend tests only if frontend source changed (none expected), otherwise record tree equivalence and re-use prior UI evidence. Do not weaken or skip broken assertions; retain original traceback and correct within authorized narrow envelope or return FAIL.
3. Specifically verify: default-off no worker/GPU/service action; no invalid character/zero-duration timing assignment; native successful CTC text/times stay identical; mixed speaker and ambiguous spans skipped; `subtitle` vs `fine` criteria; source offsets; process_media finishes all chunk/LID/native retry calls before first worker or managed stop; one process/model load across multiple span requests; protocol ID/timeout/cancel/reaped process group; **real Linux worker parent-death behavior with an owned, harmless fake worker**; CUDA-required never returns CPU success; unit ownership/lease/journal blocked and normal/error/**SIGTERM unwind** restore paths. Do not send test signals to production Studio or ASR.
4. Re-read candidate and assess actual `Qwen3ForcedAligner` library API, isolated `outputs/modular-eval/.venv` availability and correct quantization/dtype, without installing new weights or mixing this venv with Nemotron. If worker backend differs in a narrowly mechanical call signature, correct it, add regression and document. Do not redesign alternate backend, change package architecture or turn off validation simply to pass.

## B. Bounded host/GPU qualification

5. Check GPU and service identities **read-only** first: `nvidia-smi`, `ss -ltnp`, `systemctl --user list-units --type=service --all`, current known ASR 8190 service/launcher, vLLM/SwiftU/embedding workload occupancy, model path identity and user lease/consumer semantics. Never assume a unit exists just because the model endpoint responded. Do not stop/restart any model service, including the deployed `nemovox-studio.service`, merely to create free VRAM.
6. Use a small, already licensed/local **Mandarin OOV** audio fixture and a bounded multi-span contract that preserve the original source transcript; compare `off` to optional `qwen`. Run a GPU-required Qwen session **only when available VRAM makes it safe**; capture actual device, observed parameter dtype (or explicit UNKNOWN), allocated/peak GPU memory, load count, worker PID and lifecycle, warm/cold timings, number admitted/skipped, recovered subtitle-useful utterances/cues, invalid/zero-duration counts, and full wall time. Include at least one real `钾` OOV recovery, preferably multiple spans with model_load_count=1. If safe CUDA allocation unavailable, report `GPU_BLOCKED` rather than invoking CPU automatically.
7. The isolated worker may use a bounded local `qwen-asr` vs compatible native-HF comparison if it is straightforward and safe; otherwise mark `native-HF comparison skipped (optional)`. No full episode reprocessing, model download, local checkpoint mutation, or repeated model benchmarking campaign.
8. **Managed vLLM scenario:** only if the existing ASR instance is demonstrably Nemovox-owned, properly cooperative with an exclusive lease, has the exact declared `nemovox-*.service` user unit, and is safe to quiesce. Verify the actual corresponding launcher and all consumer participation, not only the presence of service markers. Then, with explicit managed flags already authorized by issue #1 and within this safe ownership envelope, run one reversible `ASR native complete/checkpoint -> ASR stop -> VRAM released -> Qwen session -> teardown -> verified ASR restoration/health` smoke with raw process/unit identity. Exercise failure/cancel restoration via CPU/simulated state machine even if no safe live takeover exists. Do **not** modify or patch existing launcher/service ownership markers, restart unrelated consumers, or invent lease authority to force managed readiness. If not safe, report typed `MANAGED_GPU_BLOCKED` with exact reason and leave services untouched. Such a safe block is not proof of real managed-handoff acceptance.
9. If any Qwen attempt fails while ASR is owned and stopped, prioritize restoration using the documented journal/ownership procedure; do not claim restored without `/v1/models` matching the original model. Never restart a unit previously stopped. If recovery cannot be proved, STOP and escalate immediately, retain evidence, do not continue more GPU tests.

## C. Evidence, handback, completion boundary

10. Record a criterion-to-evidence matrix with source/head, test commands and exit codes, relevant changed paths/diff, CPU suite, worker reuse and GPU/perf, native off-vs-on quality and timing, authorization/lease state, release/restore states, cleanup and limitations. Distinguish measured current run, prior historical evidence, controlled mocks, and UNVERIFIED external service. Never fabricate quality coverage or a speedup.
11. Publish one sanitized, self-contained **operator-attachable** `tar.gz` (target ≤20MiB) with manifest/sha256, `HANDback.md`, `test-results.md`, `gpu-benchmark.md`, `ownership-and-restore.md`, full scoped source diff and changed file list, required CPU/GPU/protocol logs and small bounded synthetic/authorized samples. Exclude commercial media, full transcripts, private host paths, secrets, environment dumps, weights, huge logs and credentials; verify member hashes/gzip integrity. Keep original full logs private when sensitive, publish redacted excerpts.
12. Create **one** evidence-only branch `evidence/issue-1-qwen-20261121` directly from the **final exact source implementation SHA** (if branch already exists read/check and reuse, never force-rewrite). Place evidence under `staging/agent-evidence/issue-1-qwen/`, with browsable HANDback/manifest/test/GPU reports and archive; do not merge it. Verify exact parent/source identity and GitHub branch readback.
13. Return `PASS | FAIL | BLOCKED | STALE`, exact source/evidence SHAs, tests and benchmark receipts, which acceptance criteria are supported, explicit managed-mode disposition, original/restored service identities or reason for no handoff, archive size+SHA-256, privacy/integrity checks and any remaining necessary Central code decisions. **Stop after one consolidated handback**, no PR, issue closure, service deployment or repeated local campaign. Central will independently review and handle PR once source/evidence is qualified.

## Stop/escalation and autonomy

Resolve ordinary host-specific compatibility failures and small in-scope integration fixes without serial confirmation. If source code needs a material new algorithm, nontrivial architectural adjustment, modified device/safety gate, unbounded GPU handling, or changes to another live service, STOP and return a minimal reproducer, current head SHA and recommendation. Prioritize completion of the declared issue objective over unrelated modernization.

### Mandatory command-shape friction ledger

```text
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events:
    - scenario=<bounded-key>
      requested_shape=<submitted command>
      effective_shape=<actual executed/rejected command|same-as-requested|unknown>
      rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
      event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
      execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
      correction_or_annotation=<bounded decisive evidence>
      replacement_shape=<successful correction|none>
      recipe_reference=<GHDEV-10 key|none>
      resource_reference=<resource consulted|none>
      contract_conflict=<yes|no|uncertain>
      durable_gap_candidate=<yes|no|uncertain>
  omitted_sensitive_values=<yes|no>
```

If none: `encountered=no; events=[]; omitted_sensitive_values=no`. Report test result and friction separately; a recovered failure is evidence of friction when material.
