# Issue #14 / PR #15: one bounded NInfer Messages meaning-QC qualification

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=94d96d1bfb84cb0c6c8163511963302b499aeb0c
CENTRAL_SOURCE_TREE=5dd95055ccb57114e579ec3db8e74ea45157ec42
CENTRAL_CONFIG_SHA=e5c5a8a6c3443f7bfbefc25119fd392694eca1dc
PREVIOUS_HOST_EVIDENCE_HEAD=81011f54ab068bc495331dae6b4ba8372713c708
PREVIOUS_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-ninfer-qc-private/issue14-ninfer-tiered-qc-20261124T020042Z.tar.gz
PREVIOUS_PRIVATE_SHA256=cbbf57d0196e376a504aed905fe121a2579a0d25c7d85f8c2069991bfcd330c9
CONTROLLING_AUDIT=staging/issue14/AUDIT_NINFER_MESSAGES_20261124.md
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_FULL_EPISODE_RERUN=yes
NO_ASR_RIVA_PYANNOTE_VLM=yes
NO_MODEL_DOWNLOAD=yes
NO_HOST_NINFER_BUILD_OR_SERVICE_CONFIG_CHANGE=yes
NO_UNOWNED_GPU_EVICTION=yes
NO_LOCAL_PRODUCTION_CODE_EDIT=yes
NO_AUTOMATIC_SOURCE_PROMOTION=yes

## Objective

Complete one CPU-first plus bounded real-model qualification of the general NInfer Qwen3.8 meaning-first QC transport. Correct the previous **protocol-level** thinking-starvation issue, not the model's allegedly insufficient ability to reason. Allow fluent nonliteral English output and utterance-level timing while preserving consequential source meaning and negative controls. The first T2 call serves as the **in-batch** protocol preflight, not another extra inference campaign. Prefer ONE full handback to repeated tiny iterations.

## A. Authenticated source, exact worktree and independent CPU validation

Refresh PR #15 and Issue #14. Confirm OPEN/DRAFT, current main/base, CENTRAL_SOURCE_SHA/TREE as latest source/test state and CENTRAL_CONFIG_SHA as latest config; subsequent doc/evidence-only commits are fine, any source/test/config drift stops write qualification. Use fresh collision-safe isolated worktree; preserve operator checkout. Read AGENTS.md and audit above, ninfer_messages_client.py, resource_lifecycle.py, refinement.py, semantic_review.py, chunking.py, recording.py, config and focused tests. Inspect actual installed NInfer SHA and its Anthropic Messages parser at src/serve/anthropic_messages_request.cpp; verify original SHA a3a3dd5be6c2ecec1479017d5eee219d7a7e5784 or record differences, not historical assertions.

Authenticate previous private archive SHA cbbf57d0... (6/6 CHECKSUMS entries; seven unique tar members) and prior published packet at 81011f54ab068bc495331dae6b4ba8372713c708:staging/issue14/evidence/20261124T020042Z/ (7/7 member IDs and bytes; manifest). The original source evidence, French media, W01c/S02 chunk identities and accepted seven-component output are immutable. Do not re-run full ASR or overwrite prior scratch.

From isolated worktree CWD, PYTHONPATH=src:. both original host interpreters:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_ninfer_messages_client.py tests/test_issue14_ninfer_thinking_budget.py tests/test_issue14_semantic_review.py tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_resource_lifecycle.py tests/test_refinement.py tests/test_translation_review.py
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check

If a named test file differs, discover the actual existing path and report the change. Capture actual exits and counts; no skipping/xfail/rewriting production logic. Check candidate __file__ for NInferMessagesTextClient, resource_lifecycle, refinement, semantic_review, chunking under BOTH interpreters. A source regression returns the exact traceback and a minimal repro; do not retry GPU to avoid addressing CPU failure.

## B. Host static protocol check and conservative replay

Under read-only host access, examine installed NInfer a3a3dd... source (or fresh SHA) for POST /v1/messages support, thinking.type enabled with budget_tokens >=1024 and disabled, response content blocks and output_tokens_details.thinking_tokens. Confirm the actual service's http://127.0.0.1:8001/v1 is managed/owned and serves both /v1/messages and /v1/chat/completions. The latter has no enforceable request thinking_budget on this version. Do not restart or rebuild host NInfer to pass qualification.

Recompute bounded Run A/B T0 QC candidate order and per-local-voice lexical correspondence from archived raw chunk 0,1,3,4 and source ledger. Confirm W01c 80.2-83.254s context, S02 350-359.812s including negation, per-local-voice 2->9 and 4->5 evidence, and musical/SFX/C2/genuine repetition/Mandarin negative controls. Require the original merged unit counts and canonical seven-component reviewed subtitles remain unchanged, with exact input/output hashes. Don't demand every French word or attempt to invent the missing W01c [77.63,80] part.

## C. ONE owned text-model session, at most six calls TOTAL

Use existing, verified Nemovox-owned text resource and ModelLifecycle, with:
  from nvidia_transcribe.resource_lifecycle import managed_text_client
  from nvidia_transcribe.semantic_review import run_configured_tiered_text_reviews
  with managed_text_client(manager, resource, wire_protocol="ninfer_messages") as client:
      result = run_configured_tiered_text_reviews(
          original_digest_pinned_request, client=client,
          policy=original_digest_pinned_versioned_policy)

Use new configs/meaning-first-review.example.json from THIS branch, not the old version. The plan and model execution MUST use the same exact policy SHA. No manual overrides to weaken answer headroom or evidence citations.

Budget and limits:
  case_limit=6
  token_budget=9472 reserved total
  T1=512 completion, thinking.type=disabled, reasoning_effort=none
  T2=1792 completion, thinking.type=enabled, budget_tokens=1024, output_config.effort=xhigh
  abort_on_reasoning_starvation=true
Each case counts once; first T2 case is the runtime protocol compatibility check within these six. If HTTP 4xx/5xx, bad budget, unsupported effort, missing typing/usage or invalid response occurs, preserve exact trace and do NOT fall back to /v1/chat/completions or change service default. A systemic empty content + finish=length + 100% reasoning must stop further model calls (source batch guard), not waste another five. Record source-bound case SHA, actual payload controls, stop_reason, response content SHA and thinking/output token usage; do not disclose secrets in public packet.

Require multiple independently valid final JSON or justified refusals/abstentions, rather than a scripted verdict. Check S02 consequential negation/time displacement and W01c mixed generation-context problem; evaluate intended meaning, idiomatic clarity, dialogue sequence, local speaker uncertainty and contrary evidence. A model-reasoned hypothesis is advisory_only and NEVER substitutes for positive audio evidence to establish new source unit ownership. No automatic source or translated text edits; no global speaker-name guessing.

If the model service cannot safely be admitted, run all CPU evidence and return MODEL_PHASE=BLOCKED with exact reason. The host may fix ordinary path/env/test-running mechanics but may not patch production code, weaken validation or rebuild NInfer without Central approval.

## D. One final evidence commit, complete handback

Publish one sanitized public evidence-only root/commit on EXISTING draft PR #15 and one private self-contained tar.gz with unique paths + SHA256 manifest, source SHA/tree, config identity, full CPU logs, import paths, original evidence SHA, actual /v1/messages payload controls and provider response/usage (private if sensitive), policy SHA, case-level meaning-first decisions, negative controls, GPU/ModelLifecycle before-during-after and command-shape friction. No full media or complete copyrighted source transcript on public GitHub. Re-verify remote manifest/files and that no source/test/config drift occurred after pinned authority.

Return:
 ISSUE14_NINFER_MESSAGES_QC:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<full SHA>
   VALIDATED_SOURCE_SHA=94d96d1bfb84cb0c6c8163511963302b499aeb0c
   VALIDATED_SOURCE_TREE=5dd95055ccb57114e579ec3db8e74ea45157ec42
   VALIDATED_CONFIG_SHA=e5c5a8a6c3443f7bfbefc25119fd392694eca1dc
   CURRENT_PR_HEAD=<full SHA>
   MAIN_SHA=<full SHA>
   SOURCE_TEST_CONFIG_DIFF=<none|paths>
   CPU_FOCUSED_CORE_STUDIO=<exits/counts/true imports>
   PRIOR_ARCHIVE_AND_PUBLIC_AUTH=<6/6,7/7 and exact SHA/bytes>
   INSTALLED_NINFER_CONTRACT=<full SHA, parser controls, service identity>
   T0_BOUNDARY_QC=<W01c, S02, C1/C2/music/Mandarin controls, unchanged source>
   VERSIONED_REVIEW_POLICY=<source policy digest and exact model_review controls>
   MESSAGES_TRANSPORT=<request wire, T1 disabled/T2 enabled 1024, response usage>
   MODEL_T1_T2=<up to 6, accepted/advisory/abstained/failed, token audit>
   MEANING_FIRST_QUALITY=<polarity/continuity/English readability/uncertainty>
   MODEL_LIFECYCLE=<before/during/after owned resource>
   APPLICATION_BOUNDARY=<why advisory does not mutate accepted source>
   PUBLIC_EVIDENCE_ROOT=<full head:path>
   MANIFEST_SHA256=<sha>
   PUBLIC_MEMBERS_VERIFIED=<count>
   PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
   HUMAN_WER_CER_DER=UNVERIFIED
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|actual status>
   omitted_sensitive_values=<yes|no>

No second model batch, no new PR, no release, deployment, merger or closure. Following this single result Central will independently review code/evidence, evaluate whether a generic review-to-revised-output adapter still needs implementation and proceed toward exact-head PR review under the meaning-first objective, not per-word perfection.
