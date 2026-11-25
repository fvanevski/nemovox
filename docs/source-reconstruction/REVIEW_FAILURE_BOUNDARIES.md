# Final required-review failure-boundary qualification — #19 / PR #21

One CPU-only completion campaign for the two Central independent review
findings B1/B2. Central has implemented the fixes, tests and docs. No new
PR, model service changes, model inference, dependencies or deployments.
Use existing issue/19, isolated candidate worktree, owner's main unchanged.
Read AGENTS.md and REVIEW_ANCHOR_QUALIFICATION.md for ownership, scratch,
resource, archive, publication and friction boundaries; this document
supersedes its execution steps. Launch /workspace/nvidia-transcribe,
Build primary (OpenCode, Hermes or Codex). Native Git and GitHub authority;
Serena bounded navigation, Probe supplemental, RTK routine only, no filtering
of exact identity, diffs or failures. OpenViking historical context only.

## Inputs and authority

Exact candidate commit/tree supplied by kickoff; previous reviewed head
e9e9aeab6d6e554f258c46ead8ca1b0c76e7c459, tree
494f18f3818bcd6d33ed5070b6ac4b9fbc62a6b5.
Previous successful tested source f45c91d5c92297fe8b93d279a27dadd5ba54c391,
tree a263fdc96fe3a495d88a9f1961a92d88100a0ab6.
Prior completed anchor qualification private archive:
review-anchors-f45c91d5.tar.gz, 1381884 bytes, SHA256
6936aa0de1f80e1ade485e80245998195cb1bc168072871b19977180375e2d6f.
Original completion archive: review-completion-cabed83d.tar.gz, 3764304 bytes,
SHA256 6f3f0e308b227e69248e4121f9064c5e7e3e9bb543a6fe554d7e6426866a592a.
Use paths from kickoff; verify hashes and manifests before reading, no edits
to archived checkpoints. Use prior execution identities, not relabeled tests.

## A. Verify candidate and CPU gates

Fetch origin issue/19, prove exact current remote commit/tree and direct
parent; inspect complete delta. Expected six existing files modified and
this new contract only; no other source edits. Use isolated detached worktree
and owner .venv / .venv-studio explicit PYTHONPATH=src; authenticate imports
for translation_review, subtitle_pipeline, studio_api, studio_exports.
Host GPU/service readback before and after, no changes to running services.
Use private evidence scratch outside tracked tree. Run and save exit codes
and complete native logs:

- python -m compileall -q src scripts tests
- git diff --check e9e9aeab6d6e554f258c46ead8ca1b0c76e7c459..HEAD
- PYTHONPATH=src owner .venv/bin/python -m pytest -q tests/test_translation_review.py tests/test_final_integration_evaluation.py tests/test_review_anchor_replay.py
- PYTHONPATH=src owner .venv/bin/python -m pytest -q -m 'not gpu' tests
- PYTHONPATH=src owner .venv-studio/bin/python -m pytest -q tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py

Use actual absolute owner interpreter paths. Do not ignore Studio tests in
full CPU collection. Verify unchanged web tree and config; prior frontend
14 tests/build evidence reusable. No test-count target.

B1: one-row ordinary max_tokens comparison exhaustion, compact success,
compact exhaustion explicit deferral, transport and non-output stop failure.
Verify actual new regression coverage and all existing multi-row,
malformed, loop, polarity and passage controls pass; no partial cache entry
after exhaustion/failure. B2: real synthetic normal required-review failure
persists contract, invalidates prior metadata and fails; actual public CLI
--re-render with no model calls explicitly salvages, errors retained, exact
contract and SRT/ASS repeatability, invalid/stale receipt rejected. Check
resource blocked reason propagation, existing deferred and normal rerender
regressions. No live endpoint or GPU to exercise synthetic failures.

## B. Exact retained positive replay

Production review miss handling changed, not prompt or cache key or acceptance
guard. Exercise the unchanged sealed replay runner with current source imports
and original completion archive using its documented command:
PYTHONPATH="$WT/src" "$OWNER/.venv/bin/python" "$WT/scripts/replay_review_anchors.py" --archive "$ORIGINAL_ARCHIVE" --source-root "$WT" --output-dir "$EVID/current-replay"
Require all 249 exact hits, zero inference, all original checkpoint bytes
unchanged, 137 accepted edits, four text and one status-only changes versus
original, exact per-row text/status/proposal/comparison/uncertainty
correspondence with previous successful candidate outputs authenticated from
review-anchors-f45c91d5.tar.gz. Prior baseline replay already verified, no
repeat required. Compare rows not volatile elapsed/report timestamps.
Do not rekey or manufacture any checkpoint entries. Preserve previous source
quality qualification as executed; its conservative code fingerprint will
invalidate old stage cache on a new version, not an obligation to repeat ASR.

Use prior successful bounded/ Mandarin/ full French rendered contract bytes
from sealed anchor archive for offline public normal rerender, in private
job dirs with original stem/stream/required metadata and config. Verify all
three SRT/ASS and translated-contract hashes identical to prior qualified
outputs, no model calls. Use genuine original media only if public path
requires it. No repeat full live review or upstream stage. Normal retained
success and explicit salvage are different dispositions. If a prerequisite
is unavailable, return precise evidence and completed checks rather than
starting inference or inventing a receipt.

## C. Delivery

Return one private review-ready tar.gz including manifest SHA256, identity
and input equivalence, tests and static logs, current replay contracts/deltas
and comparison results, normal rerender outputs/hashes and metrics, focused
failure/salvage test receipts, complete diff, host readback and friction.
Use NUL-delimited manifest generation for spaces/apostrophes; exclude media,
weights, secrets and unrelated logs. Publish one privacy-screened sanitized
evidence-only commit on existing issue/19 at
staging/agent-evidence/review-failure-boundaries/<UTC>-<candidate-short>/,
with exact remote head readback, manifest, summary, sanitized logs, and
parent/tree/blob proof. Preserve original private archive; no private
dialogue or full translated contracts in Git. Follow GHDEV-10
COMMAND_SHAPE_FRICTION schema from parent contract.

Routine execution, import, paths, CLI/config discovery, test invocation,
diagnostic and packaging corrections allowed autonomously. Mechanical
formatting-only change if necessary with complete diff and requalification;
no change to production semantics, test expectations, admission, cache
authority, model selection, dependency, or review safety gates. If a
substantive defect surfaces, collect exact reproduction and return rather
than relaxing validation. Do not request Codex, mark ready, merge, close,
deploy or open another PR. Central will adjudicate and re-review exact head.

Report RESULT=PASS|FAIL|BLOCKED, TESTED_SOURCE_SHA/TREE,
EVIDENCE_CARRIER_SHA/TREE, test/static/import/identity results, B1/B2
negative-path outcomes, current replay comparison, normal offline render
hashes, host state, remaining blockers, archive path/size/SHA256,
publication proof, and COMMAND_SHAPE_FRICTION.
