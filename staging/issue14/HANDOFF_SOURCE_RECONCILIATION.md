# Nemovox issue #14 / PR #15: consolidated hybrid source reconciliation host qualification

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
MAIN_PREVIOUS_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CENTRAL_SOURCE_SHA=91011e5b188d4c3e5665f45f9d75b8af54351b01
CENTRAL_SOURCE_TREE=d028068404312155766941ae9bff391790ee427e
CONTROLLING_DESIGN=staging/issue14/SOURCE_RECONCILIATION_DESIGN.md
LATEST_HOST_EVIDENCE=e681203de5eb03bb1fb6a88d34db6e32beb3ee6f:staging/issue14/evidence/20261123T013938Z/
MODEL_EVIDENCE=cbebbc0e13b854bc866f76e96932936699f1dd67:staging/issue14/evidence/20261123T011017Z/
REPO_CHANGE_SCOPE=source_recovery planner, pure reconciliation dossier and focused regression tests
NEW_PR=forbidden; MERGE=forbidden; CODEX_REQUEST=forbidden; PRODUCTION_ACTIVATION=forbidden

## One campaign, objective

Qualify the new Central-authored source-recovery priority/coalescing behavior and source_reconciliation.py against the ALREADY GENERATED private original16/full35/EP166 source ledgers and pinned Qwen 63-window French/60-window Chinese sweeps. Independently adjudicate a small set of source-unit-to-audio-event links and return one inspectable evidence-backed decision packet for the Central-owned final active hybrid source-turn implementation. The new module is an evidence producer, not an active transcript switch.

This is a complete, single host-bound integration campaign. Do not repeat expensive full-episode inference to exercise a pure offline join. Do not create new PRs, redesign source selection, add a new generic model stack or start unrelated quality campaigns.

## Identity and safe preparation

Launch fresh OpenCode Build from the canonical authenticated Nemovox checkout (typically /workspace/nvidia-transcribe). Agent safely fetches the remote branch and authenticates current PR #15 OPEN/DRAFT, issue #14 OPEN, current remote branch/source/main full SHAs, and that CENTRAL_SOURCE_SHA is in ancestry and no unapproved source drift has occurred. Stop STALE if a concurrent source update cannot be reconciled without hidden modifications. Use a fresh private worktree. Do not reset/clean/stash/rebase/force the operator's checkout, original episodes, existing Qwen weights, external GPU services or prior private evidence.

Read COMPLETE controlling design, AGENTS.md, source_recovery.py, source_reconciliation.py, source_events.py, source_event_replay.py, tests, previous evidence packets, and current resource ownership. Refresh /workspace paths before execution; do not assume copied examples are a substitute for fresh host evidence.

## Phase 1. Exact-source CPU gates

Run on the isolated exact source SHA using the current authenticated host Nemovox Python:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu" tests/test_issue14_source_recovery.py tests/test_issue14_source_reconciliation.py tests/test_issue14_source_event_replay.py tests/test_issue14_source_evidence.py tests/test_issue14_source_to_renderer.py tests/test_viewing_quality_regressions.py
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
  git diff --check

Record exact counts/exits/traces. Never weaken a test, change source semantics or edit frozen references to make green. Ordinary test-only fixture/adaptor/import/environment corrections are allowed if small, proven and fully recorded. Semantic failures or source-production architecture changes are Central-owned, with a specific blocker; do not silently redesign the reconciler.

## Phase 2. Authenticate baseline and planner invariants

Reopen private original16 French (16 completed chunks, 2415 lexical units), full35 French (34 completed + c28 generation-limit rejected, 2263 units), and Mandarin EP166 (14 completed + c14 rejected, 3200 units with 1965 unresolved time). Prove the manifest/raw response/media fingerprint hashes match the latest evidence. Re-run strict and forensic replay only to separate new private output files. Do not claim full source recall because source_units_dropped_by_ledger=0.

Adapt the PREVIOUS CPU FireRedVAD-AED/Silero/pyannote activity annotations under the existing activity schema with exact media time, model ID/revision/event ID. Repeat selective and independent-floor plans for all three. For full35 measure budgets 32, 64 and 128, exact floor selected/pending, coalesced activity count, original detector evidence ID union, W04c [1261.5,1263.5] and H2 opening [362.5,364.2] both selected when floor can fit alongside higher-priority authenticated failures. Neither true acoustic overlap nor speaker identity may be lost during coalescing. If max budget cannot cover all floor intervals, status must say partial and list every missing interval. Strict rejected c28 and c14 must never be accepted as complete.

## Phase 3. Offline Qwen observation adaptation and reconciliation

Use already-authenticated private Qwen3-ASR-1.7B @7278e1e70fe206f11671096ffdd38061171dd6e5 pilot and 63-FR/60-ZH source outputs. Their raw files are host-private. Verify actual original media fingerprint, input crop times, crop SHA-256, raw-response SHA-256, exact pinned language and model revision, generation status, and map each response to a unique NEW plan request ID. Do not borrow request IDs from a different media fingerprint or count a previously deferred request as executed. If an earlier stored response lacks the raw bytes/digest necessary for authentication, mark that output UNVERIFIED/UNMAPPED; do not invent an id or raw-response hash. Report mapping completeness and reason of each missing part.

As a test-only mechanical adapter, you may normalize these existing private responses into nemovox-challenger-observations/v1 using the exact contract in SOURCE_RECONCILIATION_DESIGN.md and the source_reconciliation.py validator. No new production model runner implementation, acceptance heuristics, or model retraining is delegated. Run private CLI:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.source_reconciliation --source-evidence <PRIVATE_LEDGER_JSON> --recovery-plan <PRIVATE_PLAN_JSON> --challenger <PRIVATE_CHALLENGER_JSON> --activity <PRIVATE_ACTIVITY_JSON> --audited-events <PRIVATE_REVIEW_JSON> --output <FRESH_PRIVATE_DOSSIER_JSON>

Use a plan that actually admits the independently completed Qwen floor windows; never pretend deferred windows executed. It is fine to produce one 63-FR dossier based on a coverage-only/orig16-derived plan and a separate full35 selective dossier, then prove source/media continuity. All window text stays PRIVATE. Qwen crop text is context-only: a 24s window may contain song, narration, several voices and legitimate repetitions. It may corroborate an event but cannot become a short event's exact transcript without an additional scoped operation.

## Phase 4. Small positive acoustic identity adjudication

Use the authenticated prior human-auditioned region ledger as event time/classification evidence but do NOT invent verbatim source text where it was not recorded. Select a bounded positive sample (2-5 useful regions, NOT all 23 min): W04c/H2/c28 rescue, one real repeat C1/C2, and a SFX-to-dialogue false correspondence are strong candidates. If direct host audio audition is available, separately record a handful of exact source words/characters and positive mapping of explicit source unit IDs to independently heard audio event IDs. Store method, event ID, bounds, corresponding original response unit IDs, review-evidence SHA and whether source_text_verified=true. Any mapping without actual listening or known direct audition provenance remains unresolved. Do not claim the agent listened if it only inspected model outputs.

Exercise both outcomes: same acoustic event (cross-request repeats can share one event identity without deleting either original observation) and different speech events (similar source text must stay distinct). Include a false Qwen song/SFX control, a real W04c positive speech event, Mandarin overlap at 525.16, and incomplete c28/c14. Treat forced alignment as provisional: last packet proved false Mandarin text on SFX can receive positive monotonic timestamps. You need not rerun Qwen or aligner GPU inference unless archived evidence is genuinely incomplete and a tightly bounded owned-model diagnostic is essential. No production v1 recording/translation/renderer edits.

## Phase 5. Exact-head evidence and one handback

Publish ONE sanitized evidence commit under staging/issue14/evidence/<new-UTC-run-id>/ on SAME draft PR #15, after fresh source and remote PR SHA proof. No rebase/force/merge, no new PR. Required files:
  README.md
  source-identity.json
  validation/cpu-gates.json
  validation/planner-budget-comparison.json
  validation/challenger-input-provenance.json
  validation/audio-event-reconciliation.json
  validation/source-fidelity-cases.md
  review/architecture-readiness.md
  COMMAND_SHAPE_FRICTION.txt
  manifest.json

Record source/media/checkpoint SHA, source vs host verification, source/event identity mapping, before/after coalesced activity counts, independent floor admission under 32/64/128 budgets, coverage of W04c/H2/c28, Qwen 63/60 response adapters and exact missingness, manual reviewed event count and unverified word references, model lyric/SFX false output classifications, zero source unit deletions, and exact remaining canonical source-turn selection inputs. A host-private bounded self-contained tar.gz with small source transcript excerpts, manifests/hashes, scorecard, diffs, errors/logs, and limited annotated audio-event samples is appropriate for Central's next audit; NEVER publish full copyrighted transcript, WAV, raw episode media, weights or secrets to public GitHub. If making a private archive, return host path plus digest and send to operator for private upload only if useful.

The public packet must be self-describing, all members SHA256/byte counted, and committed in one non-force exact-head operation. Inspect ignored/staged file membership and verify all files through GitHub readback. If a source repair is needed, first record exact source changes and full SHA and rerun affected tests, with no weakening of checks. The operator's canonical checkout and external running services remain untouched.

Return:
  ISSUE14_SOURCE_RECONCILIATION:
    RESULT=PASS|PARTIAL|BLOCKED
    ADMITTED_PR_HEAD=<full SHA>
    VALIDATED_SOURCE_SHA=<full SHA and tree>
    MAIN_SHA=<full SHA>
    CURRENT_PR_HEAD=<full SHA>
    SOURCE_DIFF=<none|full paths/hashes/diff>
    CPU_FOCUSED_AND_FULL=<commands/exits/counts>
    ORIGINAL_REPLAY_IDENTITIES=<FR16/FR35/ZH, raw hashes and rejected chunks>
    FLOOR_BUDGET_BEFORE_AFTER=<budgets 32/64/128, activity coalescing, W04c/H2>
    QWEN_CHALLENGER_ADAPTER=<63/60 mapped, raw hashes/coverage/status, unavailable cases>
    AUDITED_EVENT_DECISIONS=<same/different/unknown unit links, reviewed source texts>
    MODEL_HALLUCINATION_CONTROLS=<theme song/SFX/false wording and no auto adoption>
    SOURCE_LINEAGE_INVARIANTS=<stable IDs, reviewed vs provisional, 0 drops>
    HOST_AND_GPU_NONMUTATION=<actual evidence>
    WER_CER=<UNVERIFIED unless independently reviewed verbatim source>
    PUBLIC_EVIDENCE_ROOT=<full SHA:path>
    MANIFEST_SHA256=<digest>
    PUBLIC_MEMBER_COMPLETENESS=<yes/no>
    PRIVATE_ARCHIVE=<path/sha256 or none>
    NEXT_CENTRAL_IMPLEMENTATION=<one source-event promotion and end-to-end stage>
    MATERIAL_BLOCKERS=<list/none>
    ISSUE14=OPEN PR15=DRAFT
  COMMAND_SHAPE_FRICTION:
    encountered=<yes|no>
    events=<[]|complete canonical event fields: scenario, requested_shape, effective_shape, rewrite_provenance, event_kind, execution_effect, correction_or_annotation, replacement_shape, recipe_reference, resource_reference, contract_conflict, durable_gap_candidate>
    omitted_sensitive_values=<yes|no>

Conclude at the PASS/PARTIAL/BLOCKED handback. This campaign qualifies the evidence join and boundary policy but does not certify that all model source words are correct or release an active hybrid production transcript.
