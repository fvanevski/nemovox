# Recognized-occurrence locality: consolidated local qualification

## Governing authority and scope

Issue #17 / roadmap #16; continue on the same issue/17 working branch.
A1 exact source a3b59a7898e6aa0509173d751a2c23409d5185a4 returned FAIL
(two candidate regressions, 436 passed); evidence-only carrier
b97c7f399c057deea4433dd5beb73719e4ae7b6f. Central independently verified
the private archive digest 9d010fb3df7523dbc5d6e73c068890967ec7f24a07faad68d8a49788abf7f991,
12 payload hashes, 128 recognized turns / 2360 exact lexical associations,
23 turn-start inversions and 26 gaps over 10 seconds. Not acoustic truth.

Central has authored the A1 fixes and the A2 pre-selection span/locality
implementation, optional-alignment guard, span ownership, stable dispositions
and matched offline replay. This is one CPU qualification campaign, not a
request for local semantic redesign or a new diagnostic campaign.
A3 matched GPU geometry, final default promotion, #18 recovery and #19
integrated episode qualification remain outside this handoff.

Read AGENTS.md and docs/source-reconstruction/LOCAL_INTEGRATION.md for the
unchanged execution and evidence protocol. This continuation overrides the
previous objective, exact candidate, focused commands, replay deliverables,
and evidence staging details below. The kickoff's full candidate SHA is
authoritative; prior carrier or base SHA is not the execution target.

LOCAL_AGENT_PRIMARY=build
SEMANTIC_REMEDIATION=forbidden
MECHANICAL_REMEDIATION=allowed
SOURCE_REMOTE_UPDATE=forbidden
REMOTE_UPDATE=forbidden
EVIDENCE_REMOTE_UPDATE=allowed
EVIDENCE_ROOT=staging/agent-evidence/source-occurrences
EXECUTION_CLASS=CPU_ONLY
PR_CREATION=forbidden
PRODUCTION_DEPLOYMENT=forbidden

## Execute

1. Authenticate the owner checkout and origin. Fetch exact refs/heads/issue/17
   before proving FETCH_HEAD equals kickoff SHA, then prove object type and
   create a fresh isolated detached worktree. Do not reuse the prior candidate
   as if it were current; do not reset/pull/rebase/switch the owner checkout.
   Use the two existing Python environments and PYTHONPATH=src in candidate
   workdir. Prove actual import paths. Preserve previous handback artifacts.
2. Verify Central remediation and new locality behavior:
   PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe scripts/replay_source_occurrences.py
   PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_source_occurrences.py tests/test_source_quality.py tests/test_recording_and_normalization.py tests/test_issue14_timing_boundaries.py tests/test_issue14_source_evidence.py tests/test_qwen_session_and_policy.py tests/test_chunking_and_resume.py tests/test_issue14_asr_integrity.py tests/test_source_recovery.py tests/test_translation.py tests/test_offline_rerender.py -m "not gpu"
   PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
   PYTHONPATH=src /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q tests/test_studio*.py -m "not gpu"
   git diff --check
   Capture real exits independently. Keep archived overlap test unchanged;
   inspect actual utterance count/voice and lexical coverage in the fixture
   if failure persists. Canonical retained resume and new noncanonical
   retained-entry regression must pass. Re-run previous implicated test
   on exact old tested source only if a new failure needs diagnosis. No
   baseline-wide rerun. Use supported environment binding, not dependencies
   or test weakening; no GPU, service restart, model downloads, media edits,
   /workspace/pyannote changes or frontend campaign.
3. Run the staged replay against the original sealed diagnostic archive:
   PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python scripts/replay_source_occurrences.py --archive /workspace/nvidia-transcribe/outputs/handbacks/source-reconstruction-diagnostic-20261124/source-reconstruction-diagnostic-review.tar.gz --archive-sha256 307c0a32bd4ae32008be0141d5ac12e066f96a728bfcb179c0b343bf2031fd7b --output-dir <new-private-evidence-root>/raw-replay
   If relocated, find within project handbacks only and verify exact digest.
   The runner now performs the original 16-chunk lexical reconciliation,
   locality assessment, and a matched offline selection ablation: identical
   raw units and fixed historical speaker hypotheses, existing unit-midpoint
   control vs new coherent-span ownership. This is not exact historical
   executable replay or acoustic ground truth. Inspect aggregate counts and
   only the most consequential changed/disputed examples, including
   whether a previously selected coherent span became partial, whether a
   known repeat or overlap was lost, and whether unresolved context candidates
   have explicit references. Do not optimize thresholds or infer correctness
   from smaller gap/utterance counts. Record actual selected/untimed and
   context/alternate counts, IDs and candidate windows. No new inference.
4. Run focused tests and actual replay even if an independent semantic
   regression occurs. Bounded routine host/path/log/fixture integration and
   canonical value-preserving mechanical fixes are authorized as before.
   Return semantic failures with minimal reproducer and exact source refs;
   no production edits or assertion changes locally. If the replay script
   fails due to a genuine logic defect rather than archive-layout plumbing,
   report it rather than constructing a competing implementation.

## Evidence and publication

Use original handoff's evidence matrix, private evidence archive and direct
typed handback, with new focused/full CPU/Studio receipts and all three replay
files: raw-replay/summary.json (sanitized), raw-replay/private-details.json
(private), raw-replay/private-selection.json (private). Both private replay
files must be attached in the archive, never committed to this public repo.
Record actual candidate source identity, exact tests/exits, source changes
(if any), input/archive checksums, replay disposition counts and limitations,
resource before/after and canonical COMMAND_SHAPE_FRICTION block. Provide
manifest byte counts/digests, archive SHA-256, byte count and absolute path.
Ensure absence of secrets, proprietary source text, media or weights in
published Git data. Sanitize logs before publishing. Keep private originals
in the operator archive where necessary.

Publish one sanitized evidence-only commit on existing issue/17 after
checking remote ref equals kickoff SHA. Prior friction: .gitignore excludes
*.log, and ordinary directory staging silently omitted six receipts. Stage
normal allowlisted paths, then explicitly stage only allowlisted ignored
receipt files with git add -f -- <exact paths>. Compare the actual index
path inventory (git diff --cached --name-only) against the public manifest
BEFORE commit; fail closed on missing or extra paths. Inspect staged diff.
No recursive force-add, no change to .gitignore, no force-push, source edit
commit, new PR, issue or branch. Prove published remote SHA and public member
inventory after push. If remote changed, report STALE and return private
evidence; do not reconcile or blindly retry.

RESULT=PASS|FAIL|BLOCKED|STALE for this bounded qualification, not issue #17
closure. Return source SHA, evidence carrier SHA, full evidence receipt,
actual test/replay outcomes and any residual or blocker. No extra independent
local semantic review; Central consumes this packet and determines whether
A2 is adequate and what matched A3 geometry work remains. Do not begin A3 or
#18. Reuse accepted earlier evidence only where the original source and
criterion are unchanged. This campaign requires new CPU and replay because
the implementation changed.
