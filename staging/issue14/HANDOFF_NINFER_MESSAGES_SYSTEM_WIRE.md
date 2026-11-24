# Issue #14: Consolidated NInfer Messages system-wire and meaning-QC qualification

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=9230bde41a418c074140252edb6c46637560adb2
CENTRAL_SOURCE_TREE=5306fce0ad90c271b7bf72a44e178a367ae76fba
UNCHANGED_CONFIG_SHA=e5c5a8a6c3443f7bfbefc25119fd392694eca1dc
PREVIOUS_EVIDENCE_HEAD=083e5ba1bb627a20538a502ee8b44799b45b2de5
PRIVATE_INPUT_ARCHIVE=/workspace/nemovox-issue14-messages-qc-private/issue14-ninfer-messages-qc-20261124T021735Z.tar.gz
PRIVATE_INPUT_SHA256=7996f724d5fbf0ab7a3b9a1a988a8288cbad7369c6dac75d579910f968444303
PREVIOUS_PUBLIC_ROOT=staging/issue14/evidence/20261124T021735Z/
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_ASR_RIVA_PYANNOTE_VLM=yes
NO_FULL_EPISODE_RERUN=yes
NO_MODEL_DOWNLOAD=yes
NO_NINFER_REBUILD_OR_SERVICE_CONFIG_CHANGE=yes
NO_UNOWNED_GPU_EVICTION=yes
NO_LOCAL_PRODUCTION_SOURCE_EDIT=yes
NO_AUTO_SOURCE_PROMOTION=yes

## Goal and exact fix

ONE consolidated CPU and optional owned-model qualification of the Central-authored Anthropic Messages wire correction. The prior full CPU/Studio suites passed (focused 99, core 468, Studio 468), but the only live-model request returned HTTP 400 because a leading OpenAI role=system was forwarded inside Anthropic messages instead of top-level system. New NInferMessagesTextClient hoists ONLY contiguous introductory system instructions; keeps in-conversation system turns in place, and rejects system-only requests. Synthetic tests verify T1/T2 and real semantic review request shape. The source, translation, meaning-QC and config algorithms are otherwise unchanged.

Product quality target: fluent, grammatical, intelligible English reflecting intended meaning, polarity, narrative continuity and relevant speaker distinctions, not word-for-word French/English mapping. Event-level timing can suffice. Evidence-driven source correction must remain separate from optional nonliteral translation.

## A. Authority and CPU gates

Refresh authenticated PR #15, Issue #14, branch and main. Require PR OPEN/DRAFT, issue OPEN, latest source/test tree pinned above and unchanged committed config from UNCHANGED_CONFIG_SHA. Later docs/evidence-only commits are acceptable. Fresh isolated collision-safe worktree; preserve operator checkout. Read AGENTS.md, new handoff, ninfer_messages_client.py, resource_lifecycle.py, semantic_review.py, versioned config and focused tests.

Authenticate prior private archive SHA256 7996f724d5fbf0ab7a3b9a1a988a8288cbad7369c6dac75d579910f968444303, 7 unique members, 6/6 member checksums. Authenticate seven public evidence members at PREVIOUS_EVIDENCE_HEAD against GitHub blobs and byte counts. Preserve original source WAVs, recording, Run A/B source ledgers and accepted seven-component subtitle outputs. Verify installed NInfer source SHA a3a3dd5be6c2ecec1479017d5eee219d7a7e5784 or report any drift; inspect its Anthropic parse_system and parse_messages contracts, not a historical assumption.

With candidate worktree CWD and PYTHONPATH=src:. on each relevant command:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_ninfer_messages_client.py tests/test_issue14_ninfer_thinking_budget.py tests/test_issue14_semantic_review.py tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_resource_lifecycle.py tests/test_refinement.py tests/test_translation_review.py
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check

Capture exact exits/counts; both interpreters must import actual candidate source via __file__. Do not skip, xfail, weaken tests, edit production, or run GPU if any CPU regression fails. Preserve complete traceback/minimal repro and return PARTIAL if source fails.

## B. Static wire and meaning-first QC replay

Verify generated Messages request from actual versioned policy/prompt, WITHOUT model execution:
- Initial full reviewer instructions appear at top-level system (string). No leading role=system in body.messages; first user turn remains unchanged.
- Introductory multi-system text is retained in order; valid in-conversation system turns stay at their actual position.
- T2 has thinking.type=enabled, budget_tokens=1024, output_config.effort=xhigh, max_tokens=1792. T1 has thinking.type=disabled, max_tokens=512.
- No silent fallback to OpenAI Chat Completions, no ignoring thinking budget.

Use authenticated prior Run A/B source evidence to reconfirm W01c left context [80.2,83.254], S02 [350,359.812] including negation and per-voice 9/5-word displaced matches, and C1/C2, music/SFX, Mandarin negative controls. Existing merged source counts A 2008 / B 2002 and accepted 7-component source/English/SRT/ASS digests MUST stay unchanged. Do not rerun ASR/CTC, fabricate missing W01c [77.63,80] speech, or perform literal word-perfection audits.

## C. ONE ModelLifecycle session, maximum six calls TOTAL

Only if CPU and wire-shape gates pass, use verified owned nemovox-text-review.service and the existing ModelLifecycle:
  with managed_text_client(manager, resource, wire_protocol="ninfer_messages") as client:
      result = run_configured_tiered_text_reviews(
          pinned_request, client=client, policy=same_pinned_policy)

Versioned policy is configs/meaning-first-review.example.json at this branch: six-case limit, 9472 reserved completion tokens total; up to five T2 calls (1792 max output/1024 thinking/xhigh) and one T1 (512 output, thinking disabled, effort none), abort_on_reasoning_starvation=true. The first T2 call is the IN-BATCH runtime protocol preflight, never an extra seventh call. No server rebuild, alternative model, external service eviction or repeated batch.

On HTTP 4xx/5xx or invalid response transport shape, STOP the model phase immediately, preserve sanitized exception and actual request/response identities, no fallback to unbounded Chat Completions. Require actual nonempty final JSON, source/case/policy IDs, supported and contrary evidence, plus appropriate abstentions. Inspect stop reasons, actual thinking/output tokens, full protected raw receipts, source and speaker/timing uncertainty. If a response is truncated but transport works, MAY use ONE private versioned policy+plan variant with more answer headroom WITHIN the SAME six-total-call ceiling, without changing committed config or weakening claim acceptance; record exact SHA and rationale.

Assess W01c and S02 meaning, S02 negation and timing, narrative coherence, natural readable English and genuinely distinguishable speaker turns. Nonliteral English is permitted if meaning-preserving. Model judgment is advisory and does NOT establish acoustic occurrence, mutate source, install translations or auto-promote raw units. Report whether a generic advisory-to-reviewed-correction bridge remains necessary; do not perform substantive bridge implementation locally.

## D. Single evidence publication and return

One sanitized evidence-only commit on existing draft PR #15 plus one self-contained private tar.gz with unique members/checksums. Report exact source/tree/config, CPU command results/imports, authenticated old archive and public SHA/byte identities, static live wire shape, original QC controls, provider version, actual model requests/responses/usage and per-case meaning-first findings, ModelLifecycle before/during/after, and command-shape friction. Keep full audio/media/transcripts, model weights, secrets and private reasoning text out of public GitHub. Verify the newly published public blobs/bytes and current head; no source/test/config drift. No merge, deployment, new PR or issue closure.

Required handback:
 ISSUE14_NINFER_MESSAGES_WIRE:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<full SHA>
   VALIDATED_SOURCE_SHA=9230bde41a418c074140252edb6c46637560adb2
   VALIDATED_SOURCE_TREE=5306fce0ad90c271b7bf72a44e178a367ae76fba
   VALIDATED_CONFIG_SHA=e5c5a8a6c3443f7bfbefc25119fd392694eca1dc
   CURRENT_PR_HEAD=<full SHA>
   MAIN_SHA=<full SHA>
   SOURCE_TEST_CONFIG_DIFF=<none|paths>
   CPU_FOCUSED_CORE_STUDIO=<counts exits candidate imports>
   PRIOR_PRIVATE_PUBLIC_AUTH=<6/6 and 7/7, SHA/byte verification>
   INSTALLED_NINFER_SHA=<full SHA and source contract>
   SYSTEM_WIRE_SHAPE=<top-level system, message order, exact controls>
   T0_CASES=<W01c S02 and negative controls, original outputs unchanged>
   MODEL_T1_T2=<up to six calls, final JSON and usage, accepted/abstaining/errors>
   MEANING_FIRST_QUALITY=<negation, continuity, natural English, event timing>
   LIFECYCLE=<owned resource before/during/after>
   APPLICATION_BOUNDARY=<advisory vs source revision, remaining integration needs>
   PUBLIC_EVIDENCE_ROOT=<head:path>
   MANIFEST_SHA256=<SHA>
   PUBLIC_MEMBERS_VERIFIED=<count>
   PRIVATE_REVIEW_ARCHIVE=<host path SHA size>
   HUMAN_WER_CER_DER=UNVERIFIED
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|actual events>
   omitted_sensitive_values=<yes|no>

Return ONE complete handback. Central will audit it, determine any remaining generalized code to connect evidence-grounded judgments to the opt-in revision path, and proceed to exact-head review when acceptance is justified. No serial micro-reports.
