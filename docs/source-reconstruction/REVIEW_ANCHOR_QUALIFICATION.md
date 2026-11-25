# Final review-anchor qualification — issue #19 / PR #21

This is one completion campaign, not another ASR or model-quality development
cycle. Central owns the code, tests and replay. The candidate is the exact
commit supplied in the kickoff. Baseline executed source:
cabed83dc5685d18acccc354d1519b4e6505cf8e. Baseline evidence carrier:
20cf0e34ebe0bf19086a83eeaaa16f7cb9e3395b.
Use native Git/GitHub for exact facts. Read AGENTS.md.

## Authority and preparation

Build primary / Hermes or Codex equivalent, CPU-only; no GPU inference
required. Owner checkout /workspace/nvidia-transcribe must remain unchanged.
Fetch origin issue/19 before object proof; prove FETCH_HEAD and current
remote ref equal kickoff SHA. Create isolated detached candidate and baseline
worktrees in private scratch. Authenticate baseline exact object from ancestry
and candidate trees; check cleanliness, native imports and interpreter paths.
Use owner .venv/bin/python and .venv-studio/bin/python with explicit PYTHONPATH
to each worktree/src. No new packages or model downloads. Do not change host
configuration, service ownership, model weights, original media, main or
unrelated services. Read-only nvidia-smi and service readback pre/post; no model
service start/stop. Use Serena for bounded navigation only as needed, Probe
supplemental, RTK only for routine output when evidence is preserved; raw
native Git, test logs and identity proof. OpenViking historical context only.

Use private scratch and evidence under
/workspace/nvidia-transcribe/outputs/handbacks/review-anchors-<UTC>-<candidate-short>.
Set WT, BASE_WT, EVID, OWNER, ARCHIVE variables in a child Bash or equivalent.
ARCHIVE is the prior private review-completion-cabed83d.tar.gz, expected
3764304 bytes, SHA256 6f3f0e308b227e69248e4121f9064c5e7e3e9bb543a6fe554d7e6426866a592a.
Find only in the known prior handback directory if path differs; never edit
it. Authenticate before use. Source delta must be the staged anchor change,
tests, replay and instructions only.

## CPU qualification and exact replay

From candidate worktree, use native commands with complete saved output and
exit codes:
- PYTHONPATH=src "$OWNER/.venv/bin/python" -m compileall -q src scripts tests
- git diff --check 20cf0e34ebe0bf19086a83eeaaa16f7cb9e3395b..HEAD
- PYTHONPATH=src "$OWNER/.venv/bin/python" -m pytest -q tests/test_translation_review.py
- PYTHONPATH=src "$OWNER/.venv/bin/python" -m pytest -q -m 'not gpu' tests
  (use the prior authoritative CPU test invocation if tests includes separate
  Studio collection; authenticate actual test layout and don't omit coverage).
- Run the previous canonical Studio command using .venv-studio and PYTHONPATH=src.
  Reuse frontend tests/build only after proving unchanged web inputs.
- Verify exact imports for both source roots before replay.

The replay program is in candidate scripts; invoke its absolute path with
baseline source imports, then current source imports. Use separate nonexistent
output roots. Run from corresponding worktree (not owner checkout).
```bash
PYTHONPATH="$BASE_WT/src" "$OWNER/.venv/bin/python" "$WT/scripts/replay_review_anchors.py" --archive "$ARCHIVE" --source-root "$BASE_WT" --output-dir "$EVID/baseline" --baseline
PYTHONPATH="$WT/src" "$OWNER/.venv/bin/python" "$WT/scripts/replay_review_anchors.py" --archive "$ARCHIVE" --source-root "$WT" --output-dir "$EVID/candidate"
```
Require 26+2+221 baseline rows exactly reproduce original final/status and
decision identities; same 249 candidate cache hits, zero native inference,
no checkpoint edits/rekeying, unchanged source/primary MT/decisions. On a
cache miss, investigate exact config/import/identity mismatch; do not synthesize
new entries or run live inference. If actual archived config differs from
assumptions, fix only the diagnostic invocation from authenticated evidence
and report. A substantive replay implementation defect returns to Central.

Inspect every private-delta.json, complete joined passages and their source,
primary and approved proposal; distinguish fluency, preserved meaning,
unsupported facts and residual uncertainty. Require the bounded westward
passage restored coherently, Mandarin geographic substitution still rejected,
material uncertainty and negation still guarded, no new unsupported proper
name/number substitutions. Do not claim automatic human-reference truth or
require every ambiguous French idiom fixed. If a newly accepted change is
materially unsupported, return concrete row/passage evidence; do not silently
modify production semantics.

## Offline render and closeout

Use the actual public --re-render CLI shape documented by --help and prior
accepted rerender logs; operate only on candidate derived job output dirs,
with authenticated original media/stem/stream from retained contract and
normal SRT/ASS defaults. If offline render needs additional existing job
metadata, copy only authenticated necessary bytes from sealed archive, never
overwrite derived translated contract or reuse stale render receipts. Verify
all 3 rendered outputs, then repeat rerender on bounded job to prove
byte-identical translated contract/SRT/ASS. No inference. Inspect final
metadata for unrenderable/overflow/conflict counts; report any new material
regression; no need for identical cue counts when target text changes.
If the public rerender requires inaccessible original media, report that
specific blocker and retain all completed CPU/replay evidence.

## Permitted remediation

Within this single campaign resolve paths, imports, test invocation, scratch,
log/packaging and other environment/diagnostic problems autonomously. Safe
formatter/whitespace-only repair if actually necessary: inspect full diff,
run cheap gates, one bounded non-force commit on issue/19 only after live
publication policy, remote exact-head check and readback; re-admit/revalidate
new source identity. No semantics, assertions, cache authorities, model or
dependency changes. Stop only at material semantic defect, identity mismatch
unresolvable without changing evidence, protected surface, or genuine
unavailable prerequisite. Do not create another PR, request Codex, mark ready,
merge, close or deploy.

## Evidence and delivery

Produce self-contained private tar.gz with manifest SHA256, exact source/tree,
baseline/candidate replay stdout and summaries, all private deltas, rendered
contracts and SRT/ASS, before/after hashes, test/import logs, config fingerprints,
GPU/service readback, full diffs and evidence mapping. No full media, weights,
secrets or host-private backups. Return archive path/bytes/SHA256 and attach
archive to Central chat. Publish one sanitized evidence-only commit to existing
issue/19 under staging/agent-evidence/review-anchors/<UTC>-<candidate-short>/
containing summary, sanitized test/replay logs, per-case public metrics,
manifest and friction. Inspect for secrets/private dialogue, explicitly add
only reviewed ignored logs if necessary. Non-force push only with unchanged
remote head; verify exact remote parent/tree and all blob identities; if
publication fails return private evidence without retrying ambiguous writes.
No archive of private dialogue committed. Owner and live service state unchanged.
Remove only disposable worktrees/scratch after archive/readback.

Return direct conversational handback with RESULT=PASS|FAIL|BLOCKED,
TESTED_SOURCE_SHA/TREE, BASELINE_SHA/TREE, EVIDENCE_CARRIER_SHA/TREE,
IMPORT_BINDING, CPU/Studio/frontend-equivalence, archive authentication,
BASELINE_REPLAY_FIDELITY, CACHE_HITS and zero calls, candidate per-case metrics,
full changed-row/passage adjudication, polarity/entity/material controls,
render/rerender and resource restoration, remaining residuals/blockers,
publication and archive receipt. Distinguish original host inference from
current offline replay, do not relabel old live evidence. Central adjudicates
then independently reviews PR #21; merged-main public acceptance separate.

COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events:
    - scenario=<short repository/task scenario key>
      requested_shape=<bounded submitted shape, sanitized>
      effective_shape=<bounded actual shape|same-as-requested|unknown>
      rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
      event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
      execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
      correction_or_annotation=<bounded decisive correction or error>
      replacement_shape=<successful corrected shape|none>
      recipe_reference=<GHDEV-10 recipe key|none>
      resource_reference=<resource|none>
      contract_conflict=<yes|no|uncertain>
      durable_gap_candidate=<yes|no|uncertain>
  omitted_sensitive_values=<yes|no>
If no friction use encountered=no, events=[], omitted_sensitive_values=no.
Record material recovered friction and guard annotations separately from product failures.
