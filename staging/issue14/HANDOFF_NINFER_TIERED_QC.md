# Issue #14 / PR #15: NInfer reasoning-budget and voice-stream QC qualification

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=4c27c5b8d13a173aa8ce2f3c1f9699a37d8a14c0
CENTRAL_SOURCE_TREE=7fb880ff1614f4ce21ab28a87fda5de3e62c62c0
CENTRAL_CONFIG_SHA=a31a7d7d25b083671126f9f63b56151a7526a0d2
PREVIOUS_HANDOFF_HEAD=e757c3161c7dac1e5a2228aa44774f30722586c0
PRIVATE_PREVIOUS_ARCHIVE=/workspace/nemovox-issue14-meaning-qc-private/issue14-meaning-first-qc-20261124T013230Z.tar.gz
PRIVATE_PREVIOUS_ARCHIVE_SHA256=1e4e932d76fd9c8ce6329a104b7e23b468e13b5a915a7bc224
CONTROLLING_AUDIT=staging/issue14/AUDIT_MEANING_FIRST_NINFER_REASONING_20261124.md
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_FULL_EPISODE_ASR_RERUN=yes
NO_NEW_ASR_RIVA_PYANNOTE_VLM=yes
NO_MODEL_DOWNLOAD=yes
NO_UNOWNED_MODEL_SERVICE_CHANGE=yes
NO_LOCAL_PRODUCTION_SOURCE_EDIT=yes
NO_AUTO_SOURCE_PROMOTION=yes

## Outcome

ONE consolidated CPU-first and bounded owned-model validation for the Central-authored correction of NInfer thinking-budget starvation, independent model-effort levels and interleaved-speaker lexical evidence. Confirm meaning-first reasoning can produce FINAL, structured and appropriately cautious responses for W01c and S02, not simply internal analysis exhausted by max_tokens. Leave ordinary source/English/ASS untouched; do not pursue exact word-for-word fidelity or launch a new complete episode.

## A. Exact source/evidence authority

Fresh authenticated GH PR #15 head and main, Issue #14 OPEN/PR DRAFT. Prove the current head is a descendant of CENTRAL_SOURCE_SHA, that source/tests match this exact source revision and config matches CENTRAL_CONFIG_SHA. A later evidence-only commit is acceptable; any source/test/config drift requires stop/escalation to Central. Fresh collision-safe isolated worktree, no operator checkout mutation. Read AGENTS.md, audit above, source refinement.py OpenAIChatTextClient, semantic_review.py, chunking.py, recording.py, model_resources.py, resource_lifecycle.py, configs/meaning-first-review.example.json and focused tests. Preflight client request parameter support from installed NInfer version rather than assuming upstream master.

Authenticate previous private archive SHA (6/6 internal checksums, 7 unique tar entries), same forensic archive SHA 52568c845ab3e7c2541e3711b8950cccde93a11164cb7eadf3c706d4bf923b8e (21/21) and public packet at e757c3161c7dac1e5a2228aa44774f30722586c0:staging/issue14/evidence/20261124T013230Z (8/8 members with content identities and byte sizes). Check original Run A/B media/chunk hashes, original source ledger and prior reviewed subtitle output before doing new interpretation. Do not overwrite or regenerate originals.

## B. CPU, local Qwen compatibility and no-regression tests

From candidate worktree CWD with PYTHONPATH=src:. in both interpreters:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_ninfer_thinking_budget.py tests/test_issue14_semantic_review.py tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_refinement.py tests/test_translation_review.py
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check

Verify candidate imports for refinement/semantic_review/chunking/recording and tested source SHA/tree in BOTH interpreters. If tests fail, preserve exact traceback and continue only safe read-only work; do not weaken tests, redesign algorithms or write production source. New tests must prove strict unsupported-effort refusal, low remaining usable after a separate bad effort, explicit final-answer headroom, structured empty-content failure reporting, per-local-voice matching across interleaved streams, stable policy-digest execution and original source nonmutation.

## C. Reproduce real Run B QC and preserve controls

Offline replay original Run A/B raw lexical chunk observations and diarization_activity_hypotheses, using the existing source planner at Central candidate SHA. Run plan_boundary_semantic_reviews under the INTERNAL default and the REVISED example policy, then compare to previous handback:
- W01c boundary 90s, chunk1 speech [80.2,83.254] in left context, original chunk0 missing earlier dialogue. No fabricated 64-67 speech over music. Report source ID, chronology, continuing missing [77.63,80] span and source/neighbor coverage.
- S02 boundary 360s, chunk4 [350.0,359.812] including material negation; left chunk3 dispersed -19 to -58s and mixed with other speakers. Check if per-LOCAL-VOICE correspondence captures a longer coherent sequence than prior 4-word global left stream, without asserting global speaker identity or acoustically verified timestamps.
- C1 clean, C2 true repetition, W04c music/SFX, W04a exclamation, Mandarin character-grain. No false promotion/merger/deletion.
- Same original Run A 2008 / Run B 2002 merged units; accepted 7-component CLI source/English/SRT/ASS hashes unchanged. This candidate only affects diagnostic review candidates, not source acceptance.

Report candidate IDs, policy SHA changes (new example model_review changes digest intentionally), tiers, root observations, ambiguous left voices and whether the new structure adds genuinely usable meaning/speaker evidence without inducing false matches. Do not change thresholds to force a particular exact word count.

## D. Actual bounded NInfer/Qwen3.8 reasoning recovery

With verified existing Nemovox-owned text resource and ModelLifecycle only, one managed session. No server restart/configuration change merely to get the test green; existing --thinking-budget 16384 is a server default, and the new client should override it PER REQUEST using typed thinking_budget.

Use the NEW exact configs/meaning-first-review.example.json as the versioned source of both planner policy and model_review kwargs. Invoke:
  run_configured_tiered_text_reviews(request, client=managed_client, policy=example_policy)

Maximum SIX total model requests, 6912 total reserved completion tokens, 5 reasoned up to 1280 tokens each (thinking_budget=512, effort=xhigh), 1 light up to 512 (thinking_budget=128, effort=low). If actual candidate order/count changes, obey same six-call and total budget caps; DO NOT add another six-call retry campaign. The first xhigh request is the compatibility preflight; if server rejects thinking_budget or xhigh, STOP model phase, capture raw 400 error/status/server version and precise request shape. Never silently retry by removing the per-request thinking cap or disabling effort for all cases.

Check actual request_diagnostics.reasoning_effort_sent and thinking_budget_sent for each, decoded content JSON, finish_reason, usage.reasoning_tokens and completion_tokens, and enforce that final content is nonempty and schema-validated when treating a review as accepted advisory. Inspect responses for semantic plausibility, polarity, contextual narrative coherence, counterevidence and appropriate abstention where the provided observation does not establish acoustic truth. A model's words/structured JSON alone NEVER authorize source promotion. Capture each prompt/response digest, source identity, effort and caps, and sanitized model/backend/version receipts. No forced word-for-word translation, no automatic source revision. Successful completion of all six is desirable but not worth bypassing integrity when a case cannot be decided.

Optional audiovisual evidence NOT required in this campaign; W01c/S02 have adequate archived audio/ASR context for the focused client/QC test. Do not incur new vision model calls. Do not run any ASR, pyannote or Riva model.

## E. Evidence, exit boundary and handback

ONE complete sanitized public evidence packet/commit on existing PR #15, one SHA-256-manifested unique-entry private .tar.gz with full bounded logs, raw request/response (including reasoning_content only private), source lineage and test evidence; no secrets, model weights or copyrighted full episode publicly. Verify remote Git blob identity, manifests and current head after commit, no src/tests/config drift and service post-scope resource restoration. Report command-shape friction accurately (especially request compatibility and client fallback/endpoint options). Local Build may resolve host path/interpreter/shell mechanics without altering source/guardrails.

Return:
 ISSUE14_NINFER_TIERED_QC:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<SHA40>
   VALIDATED_SOURCE_SHA=4c27c5b8d13a173aa8ce2f3c1f9699a37d8a14c0
   VALIDATED_SOURCE_TREE=7fb880ff1614f4ce21ab28a87fda5de3e62c62c0
   VALIDATED_CONFIG_SHA=a31a7d7d25b083671126f9f63b56151a7526a0d2
   CURRENT_PR_HEAD=<SHA40>
   MAIN_SHA=<SHA40>
   SOURCE_TEST_CONFIG_DIFF=<none|paths>
   CPU_FOCUSED_CORE_STUDIO=<counts/exits/imports>
   PRIOR_ARCHIVE_AND_PUBLIC_AUTH=<6/6,21/21,8/8; original media/input digests>
   W01C_QC=<candidate, chronology, semantic risk, missing audio context>
   S02_QC=<correct source sequence, negation, new per-voice alternatives>
   NEGATIVE_CONTROLS=<C1,C2,SFX,Mandarin, no mutation>
   VERSIONED_POLICY=<digest,review_model_kwargs,case order>
   MODEL_COMPATIBILITY=<NInfer version, xhigh/low,thinking_budget request/response checks>
   MODEL_T1_T2=<up to six, nonempty JSON accepted/abstained and model-use telemetry>
   MEANING_FIRST_OUTPUT=<context/polarity/readability and traceable limitations>
   APPLICATION_BOUNDARY=<no auto source mutation, remaining general adapter if any>
   RESOURCE_LIFECYCLE=<owned lease, exact post-release status>
   PUBLIC_EVIDENCE_ROOT=<head:path>
   MANIFEST_SHA256=<sha256>
   PUBLIC_MEMBERS_VERIFIED=<count>
   PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
   HUMAN_WER_CER_DER=UNVERIFIED
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|actual failures/recovery>
   omitted_sensitive_values=<yes|no>

Do not create another PR, merge, deploy, close issues or run full-episode GPU. Following this single result, Central should independently audit source/evidence, decide whether source-review decisions can be connected safely to the already verified opt-in revision path, and then perform formal exact-head review when the whole Issue #14 acceptance criterion is met. Meaning-first output is the goal; WER/CER perfection is not.
