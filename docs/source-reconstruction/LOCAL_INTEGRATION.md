# Recognized-turn reconstruction: local integration and qualification

## Objective

Issue #17, roadmap #16, first coherent implementation campaign (A1). The staged
Central implementation establishes a raw recognized-turn/token inventory,
provisional per-tag native CTC association, explicit unresolved observations,
stable source unit / turn references and recording utterance IDs, and removes
the positional pre-quality rewrite from the normal retained-recording entry.
Direct single-request transcript/challenge lexical projection and translator
grouping use the same recognized-source discipline. It does NOT adjudicate
actual acoustic locality, redesign core selection, optimize chunk geometry or
implement #18 recovery. Preserve the single working branch for A2/A3 and one
eventual PR. No PR now.

Base authority: 9dd863841f72ab5f9aebf5bb4deb322bee99edf1.
The exact candidate SHA is supplied in the kickoff; do not substitute this
base SHA or a local branch name. Follow AGENTS.md and current harness policy.

## Agent execution authority

LOCAL_AGENT_PRIMARY=build
REPOSITORY_VERIFY_AUTHORITY=local-verify (no approved trusted Actions profile established)
SEMANTIC_REVIEW_AUTHORITY=central-owned
SEMANTIC_REMEDIATION=forbidden
MECHANICAL_REMEDIATION=allowed
SOURCE_REMOTE_UPDATE=forbidden
REMOTE_UPDATE=forbidden
EVIDENCE_REMOTE_UPDATE=allowed
EVIDENCE_ROOT=staging/agent-evidence/source-occurrences
EVIDENCE_PACKET_REQUIRED=yes
EVIDENCE_ARCHIVE_REQUIRED=yes
PRIVATE_ARCHIVE_UPLOAD_REQUIRED=yes
INDEPENDENT_PR_REVIEW=central-owned

Build may autonomously prepare a disposable candidate worktree, bind the
existing dedicated environments, resolve ordinary host path/fixture/permission
issues through supported interfaces, run tests, inspect actual failures,
apply canonical formatting or strictly value-preserving mechanical fixes in
the disposable worktree, and collect evidence. Mechanical source edits remain
uncommitted and require a complete portable patch and exact changed-file
inventory for Central to apply/review. Do not use this lane to alter logic,
assertions, validation control, dependencies, schema, or quality thresholds.
Do not bypass a live permission/lease restriction. No PR/merge/issue mutation,
deployment, model download, service restart, unrelated GPU eviction, dependency
upgrade, or changes to /workspace/pyannote. No full episode or GPU campaign.
If a real semantic defect appears, preserve minimal reproducer and continue
independent unaffected checks; return Central the defect rather than inventing
a redesign. A1 qualification is CPU-only.

## 1. Authenticate and bind worktree

From the owner checkout /workspace/nvidia-transcribe, confirm origin is
fvanevski/nemovox and inspect current status. Preserve all unrelated dirty
work. Use native Git for exact identity, not a reducer. Fetch the exact remote
issue/17 selector (git fetch origin refs/heads/issue/17), separately read
git rev-parse FETCH_HEAD and compare to the full kickoff SHA. Only then prove
git cat-file -t <kickoff SHA> equals commit. Do not reset/rebase/pull or switch
the owner's checkout. Use mktemp -d for a collision-safe private scratch root
under the permitted workspace or temporary root. Create an isolated detached
worktree with git worktree add --detach <scratch>/candidate <kickoff SHA>.
Use native git rev-parse HEAD and git status --short in the candidate to prove
exact clean identity; do not assume a cached object proves remote freshness.
Use repository-local instructions from that worktree.

Retain an absolute candidate worktree path and a separate private evidence
root. Run every command from the candidate (explicit workdir). Use
/workspace/nvidia-transcribe/.venv/bin/python for the core Python environment;
for Studio-specific checks use the existing .venv-studio interpreter if needed.
Use PYTHONPATH=src on every candidate Python invocation. Prove actual import
path with PYTHONPATH=src <python> -c 'import nvidia_transcribe; print(nvidia_transcribe.__file__)'.
Never install/edit packages to force this proof. Use native/raw output for
exact SHAs, failures and evidence; RTK for routine verbose output only if it
preserves evidence. Serena for bounded semantic references if helpful; Probe
supplemental only; OpenViking only prior context, not mutable source authority.

## 2. CPU qualification and actual raw replay

Run in the candidate worktree, capture stdout/stderr and real exits separately
(no pipeline status masking):
- PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe scripts/replay_source_occurrences.py
- PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_source_occurrences.py tests/test_parser_and_contract.py tests/test_issue14_source_evidence.py tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_source_quality.py tests/test_source_recovery.py tests/test_translation.py tests/test_offline_rerender.py -m "not gpu"
- PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
- git diff --check
Use the actual Studio environment for Studio-only failures due to missing
installed Studio extras, with the same import-path proof. Do not skip, weaken,
or replace tests to make green. A pre-existing archived fixture absent on the
host is missing evidence, not a passing test. If a substantive failure occurs,
isolate it and run the implicated test against a separate exact-base worktree
where necessary to distinguish inherited defects. No duplicate full baseline
suite unless needed for genuine diagnosis. No frontend rebuild required for
this Python-only campaign unless current repo evidence exposes a consumed
frontend contract requiring it.

Authenticate the operator's sealed private diagnostic archive using SHA-256
307c0a32bd4ae32008be0141d5ac12e066f96a728bfcb179c0b343bf2031fd7b.
Use the private archive path supplied in kickoff; if relocated, search only
the project handbacks directory and require exact digest. Never read or publish
the full source media. Run:
PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python scripts/replay_source_occurrences.py --archive <verified-private-archive> --archive-sha256 307c0a32bd4ae32008be0141d5ac12e066f96a728bfcb179c0b343bf2031fd7b --output-dir <new-private-evidence-root>/raw-replay

The staged runner matches original raw JSON byte digests to all 16 accepted
observations in the sealed source ledger. If actual archive structure differs,
inspect the bounded member inventory, locate the actual authenticated raw
files/ledger, report the difference and preserve receipts. Do not rewrite
archive or substitute approximate data. A mechanical archive-reader adaptation
is permitted ONLY if it preserves exact identity and algorithm; return diff
to Central. Replay is not proof of improved locality. Compare actual counts
against the adjudicated 128 turns / 2360 recognized words / 2360 native CTC
words and 13/16 chunks with repeated tags; explain any unit/tokenization or
projection denominators (punctuation, CJK, cleanup). Report native associations
and ambiguous/unmatched/untimed units and retained evidence. Inspect the
worst association residuals (bounded examples) without a new full diagnostic
campaign. A1 does not claim timing correctness merely because streams match.

## 3. Acceptance evidence and delivery

Required evidence matrix:
| Criterion | Required artifact | Primary observation | Identity/privacy |
|---|---|---|---|
| Exact candidate and source/runtime | identity.txt, diff.patch if any | remote fetch/ref, HEAD, status, import paths, changed files | tested full SHA; public-safe |
| Ordered turns, unmatched and ambiguous observations | focused-tests.log | actual pytest exits and new regression outcomes | exact tested SHA; sanitize |
| Actual raw replay | raw-replay/summary.json and private raw-replay/private-details.json | sealed archive SHA, 16 per-response hashes, exact token/reference coverage, per-chunk association counts | summary sanitized public-safe; detailed transcript-derived evidence private |
| No positional rewrite; current resume/translation/rerender | cpu-tests.log, focused-tests.log, any implicated Studio receipts | existing deterministic tests plus exact source review; explicitly identify missing entry-path regression evidence | public-safe |
| Lifecycle/resource isolation | resources.txt | host process/GPU observations before/after, no model inference, no service mutations | sanitized nonsecret only |
| Failure diagnosis and friction | HANDback.md, friction.json or equivalent, reproducer/patch as applicable | full exits and native errors, exact changed paths | redact private data |

Produce one sanitized review-ready evidence archive .tar.gz (including private
raw-replay detail only via secure operator attachment, never public Git).
Include a manifest of criterion-to-artifact, byte counts and SHA-256, tested
source/runtime identity, actual exits, limitations and omissions. Freeze an
allowlist; verify all members and hashes; reject absolute/traversal paths,
links and extraneous members. Exclude original media, model weights, env,
secrets, full runtime dumps, backups and unnecessary logs. Return archive path,
external SHA-256 and size. Required private evidence must actually be delivered
for independent Central review; a checksum/path alone is insufficient.

On the existing issue/17 branch, publish one sanitized evidence-only commit
under staging/agent-evidence/source-occurrences/<run-id>/ containing public
identity, test receipts, replay aggregate metrics, manifest and HANDback,
without raw transcript text/private-detail archive or secrets. Use a clean
publication worktree/branch based on the exact remote candidate, stage only
the allowlisted evidence paths, inspect complete staged diff. Verify remote
ref still equals kickoff SHA before non-force push and verify post-push
remote full commit and evidence members. Do not publish source edits or
additional branches/PRs. If remote changes or publication denied, return
STALE/BLOCKED and private archive for evidence; do not force/rebase. Separate
tested source SHA from evidence-carrier SHA. If any required artifact cannot
be produced, report its reason and whether criterion is blocked.

Return RESULT=PASS|FAIL|BLOCKED|STALE (A1 only), TESTED_SOURCE_SHA,
candidate worktree path, BASE_SHA, focused and full CPU test counts/exits,
actual replay disposition counts, exact changed paths and uncommitted source
changes, residuals, independent source review still required, and stop.
Return EVIDENCE_PACKET_REQUIRED, GITHUB_EVIDENCE_ROOT,
GITHUB_EVIDENCE_COMMIT, EVIDENCE_REQUIRED_ARTIFACTS,
EVIDENCE_INCLUDED_ARTIFACTS, EVIDENCE_MISSING_ARTIFACTS,
EVIDENCE_PRIVACY_CHECK, EVIDENCE_INTEGRITY_CHECK,
EVIDENCE_ARCHIVE_REQUIRED, PRIVATE_ARCHIVE_ADDS_REQUIRED_EVIDENCE,
PRIVATE_ARCHIVE_UPLOAD_REQUIRED, private archive path/SHA/bytes.
No issue closure or PR creation; Central consumes evidence and advances A2/A3.
Record the canonical friction block even if no friction occurred:

COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events:
    - scenario=<short repository/task scenario key>
      requested_shape=<bounded sanitized requested command>
      effective_shape=<bounded effective shape|same-as-requested|unknown>
      rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
      event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
      execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
      correction_or_annotation=<bounded decisive text>
      replacement_shape=<successful corrected shape|none>
      recipe_reference=<GHDEV-10 recipe key|none>
      resource_reference=<plugin/repository resource used|none>
      contract_conflict=<yes|no|uncertain>
      durable_gap_candidate=<yes|no|uncertain>
  omitted_sensitive_values=<yes|no>

If no events: encountered=no, events=[], omitted_sensitive_values=no.
Record material rejected shapes and recoveries even on PASS. Preserve product
failures in substantive evidence, do not mislabel them as invocation defects.
No bypass or permission weakening.
