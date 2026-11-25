# Final occurrence-disposition qualification

## Central adjudication

Issue #17: A1/A2 structural qualification accepted at
5df4ace871940cc71e651b04fbdeaa8f3bc1f69b (445 CPU, 50 Studio).
A3 bounded matched acoustic experiment executed at
13c0bb89637f1b451e8a3e7b115fa250a9b8de7d, evidence carrier
cc8907e7402e230e53696da770faa20a95d75b23. Central authenticated
the private 935758-byte archive SHA-256
805ab0e43275171b16c0beebbc6c938d15ed5b055d4a6e5b342180f46d61eefe,
89 payload hashes, 16 recording ledgers/contracts and 28 raw leaf hashes.
The six same-model reference events support fixed 2/6 and pause-aware
3/6; only one shared event. Two regions have no strict reference
events. Results mixed, not proof of accuracy. Keep fixed default and
pause-aware option; defer final policy promotion to #19. No new geometry
campaign. A3 execution criterion accepted under its original tested SHA.

Fresh evidence revealed a final disposition defect: two fixed-arm
alternate links terminate at another alternate with a selected
unlocalized counterpart, and two pause-aware links terminate at
unresolved context. Central correction at e63a155347c3755962770a26208037a55a561055
does not change selection, timing, ASR request, planner, or lifecycle.
It finalizes statuses against the actual selected observation IDs.
An alternate without a directly selected counterpart becomes unresolved;
retain direct links and prior reason. No inferred transitive occurrence
identity. The existing pending-occurrence source-quality mechanism will
now see these unresolved candidates. Current recording signature advanced
to v4, not a request to regenerate historical jobs.

Central executed the exact pure helper (Git blob
4d0ecbda70a88e7d5f49fa87b60ff0e922aea736) over all 16 A3 ledgers:
2155 original working units (1932 selected provisional, 37 selected
unlocalized, 85 alternate, 101 unresolved) -> 1932 / 37 / 81 / 105.
Exactly four changed statuses, original selected IDs and direct
correspondence references unchanged; 81/81 remaining alternates point
to selected observations, all dispositions accounted for, idempotence
and invalid-reference negatives pass. This is an isolated archived
disposition replay, not a fresh execution of the candidate recording path.

## Execute one final CPU-only qualification

Read AGENTS.md and docs/source-reconstruction/LOCAL_QUALIFICATION.md for
the applicable execution/evidence rules. This document supersedes previous
campaign commands and acceptance; kickoff exact candidate SHA controls.
Use a fresh isolated detached worktree after authenticated origin fetch of
refs/heads/issue/17 and proof FETCH_HEAD equals kickoff candidate SHA.
Do not switch/reset the owner checkout. Prove core and Studio imports
resolve to candidate with PYTHONPATH=src.

LOCAL_AGENT_PRIMARY=build
EXECUTION_CLASS=CPU_ONLY
SEMANTIC_REVIEW_AUTHORITY=central-owned
SEMANTIC_REMEDIATION=forbidden
MECHANICAL_REMEDIATION=allowed
SOURCE_REMOTE_UPDATE=forbidden
REMOTE_UPDATE=forbidden
EVIDENCE_REMOTE_UPDATE=allowed
EVIDENCE_ROOT=staging/agent-evidence/occurrence-dispositions
EVIDENCE_PACKET_REQUIRED=yes
PR_CREATION=forbidden
PRODUCTION_DEPLOYMENT=forbidden

From the candidate, use the existing dedicated Python environments:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe scripts/evaluate_chunk_geometry.py
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_source_occurrences.py tests/test_chunk_geometry_evaluation.py tests/test_issue14_timing_boundaries.py tests/test_recording_and_normalization.py tests/test_source_quality.py tests/test_chunking_and_resume.py tests/test_issue14_asr_integrity.py -m "not gpu"
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q tests/test_studio*.py -m "not gpu"
  git diff --check

Capture native exits separately. Exercise the new actual two-context-span
reconciliation regression and direct graph-finalization regressions.
Keep prior overlap, clean-control, genuine-repeat and resume assertions.
Verify no source changes relative to candidate. Report any regression
with native traceback and relevant exact source; do not weaken tests or
locally redesign selection. Complete unaffected independent checks.
Ordinary bounded host/invocation mechanical remediation allowed with
complete diff receipt; no semantic source edits or publishing source edits.

No new GPU inference, A3 campaign, archive reconstruction or full-episode
run. Central already performed the archived finalization replay, and
A3 geometry, acoustic comparison and lifecycle evidence is causally
unchanged by this metadata correction. The candidate production path
and tests still require the above CPU verification. Do not begin #18.

## Evidence

Return identity and import binding, focused/full CPU/Studio logs,
compilation/diff receipts, resource before/after, friction.json,
HANDback.md and manifest with sizes/hashes/exits. Self-contained evidence
archive as previously required; include private material only if needed
to demonstrate an actual new failure, never publish it. No historical
A3 media or raw transcript duplication needed. Publish one sanitized
evidence-only commit under the existing branch after checking exact
remote head. Explicitly force-add only allowlisted ignored log paths,
compare staged inventory with manifest, inspect staged diff and verify
published commit/ref. If stale, return private packet rather than force
push. Preserve tested source identity separately from evidence carrier.
Return canonical COMMAND_SHAPE_FRICTION and archive path/size/digest.

RESULT=PASS|FAIL|BLOCKED|STALE for this bounded qualification only.
Central then performs full exact-head PR review and adjudicates any
remaining blockers. No new PR, merge, close, deployment or #18 work
during this handoff.
