# Issue #14 / PR #15: qualify versioned JSON Schema reasoning and final advisory contract

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=72dbee0c2f2164de31224801f100c564c4570f69
CENTRAL_SOURCE_TREE=f9416e63d62ba41be738c604d3f22e4fc93bb2b5
UNCHANGED_CONFIG_SHA=e5c5a8a6c3443f7bfbefc25119fd392694eca1dc
PRIOR_HOST_EVIDENCE_HEAD=94b1c6b6382481510409277b672b9747f514b25f
PRIOR_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-messages-wire-private/issue14-ninfer-messages-wire-20261124T023146Z.tar.gz
PRIOR_PRIVATE_SHA256=bd294a2d705e4658e5b40b88b251df24eab9fd80b22620bc5f950c4e955cc29f
PRIOR_PUBLIC_ROOT=staging/issue14/evidence/20261124T023146Z/
CONTROLLING_AUDIT=staging/issue14/AUDIT_MEANING_QC_STRUCTURED_OUTPUT_20261124.md
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_FULL_EPISODE_RERUN=yes
NO_ASR_RIVA_PYANNOTE_VLM=yes
NO_MODEL_DOWNLOAD=yes
NO_HOST_MODEL_REBUILD_OR_CONFIG_CHANGE=yes
NO_UNOWNED_GPU_EVICTION=yes
NO_LOCAL_PRODUCTION_SOURCE_EDIT=yes
NO_AUTO_SOURCE_PROMOTION=yes

## Goal

Complete ONE consolidated CPU and bounded owned-model campaign to qualify the Central-authored structured-output repair. The previous run verified six real HTTP 200 NInfer Messages calls, enforceable thinking budgets and proper ModelLifecycle restoration, but all six were rejected at the schema/citation/format contract. The new response schema, prompt and adapter MUST produce completed JSON with reference-key evidence arrays. Do not pursue perfect lexical WER or another full ASR run. Evaluate intended meaning, negation, narrative continuity, grammatical/idiomatic English, important speaker distinctions and event-level timing.

## A. Authority and CPU gates

Fetch LIVE GitHub PR #15 head and Issue #14. Require PR OPEN/DRAFT, Issue OPEN, CENTRAL_SOURCE_SHA and CENTRAL_SOURCE_TREE the latest source/test state, versioned config unchanged since UNCHANGED_CONFIG_SHA, and no intervening changes under src/, tests/, configs/. Later evidence/docs-only commits allowed. Fresh isolated worktree at actual latest PR head; no operator checkout modification. Read AGENTS.md and controlling audit, semantic_review.py, ninfer_messages_client.py, resource_lifecycle.py, meaning-first-review.example.json and all relevant focused tests.

Authenticate prior private tar SHA256 bd294a2d705e4658e5b40b88b251df24eab9fd80b22620bc5f950c4e955cc29f with 6/6 member checksums, seven unique members. Revalidate eight prior GitHub public evidence member blobs/bytes plus manifest at prior head; preserve media/WAV/source, accepted six/seven-component reviewed source and English/SRT/ASS hashes. Installed NInfer source must be observed fresh; expected commit a3a3dd5be6c2ecec1479017d5eee219d7a7e5784, whose request_validation.cpp handles Anthropic output_config.format (json_schema with direct schema object). Do not blindly assume local binary matches historical source.

CWD candidate and PYTHONPATH=src:. on every Python command, both host interpreters:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_semantic_review_output_contract.py tests/test_issue14_ninfer_messages_client.py tests/test_issue14_semantic_review.py tests/test_issue14_ninfer_thinking_budget.py tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_resource_lifecycle.py tests/test_refinement.py tests/test_translation_review.py
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check

Report exact imports/exits/counts, both core and Studio candidate __file__ paths, all regressions including missing schema, prose citations, bad identity, Markdown fence, truncation, stale request and Anthropic wire format. No disabling or weakening tests. If any focused/full CPU gate fails, collect traceback and deterministic repro, skip GPU phase, return PARTIAL for Central correction.

## B. Protocol and deterministic review controls

Inspect the REAL installed NInfer parser for Anthropic output_config.format=type json_schema with direct schema, not an OpenAI nested json_schema/name wrapper. Replay actual Run A/B immutable chunk evidence and versioned policy, retaining previous W01c ~80.2-83.254s and S02 ~350-359.812s negation/displaced correspondence plus music/SFX, genuine repetition and Mandarin character controls. Do NOT rerun ASR/CTC, fabricate dialogue, auto-delete raw observations or modify original corrected subtitle outputs.

Without invoking model, capture the static outgoing T2 and T1 request shapes:
  - Anthropic top-level system carries the concise schema+six citation-key instructions.
  - Body messages starts with user, preserving source case and SHA.
  - output_config.format contains {"type":"json_schema","schema":{...}} with exact required fields, allowed enum keys and current case/request IDs; this schema is supported by the inspected backend source.
  - T2 thinking.enabled, 1024 budget, xhigh effort, 1792 max_tokens; T1 thinking.disabled and 512 max_tokens.
  - No fallback to unsupported Chat Completions, no removal of output constraints on HTTP errors. Response_format_sent is true for constrained requests.

## C. ONE owned NInfer session, six-call ceiling

Only if CPU, wire shape, source authority, media provenance and model ownership pass, use the existing Nemovox-owned service under ModelLifecycle, selecting wire_protocol=ninfer_messages. Run run_configured_tiered_text_reviews with a plan computed from the EXACT same configs/meaning-first-review.example.json. Do not alter its committed token/effort settings. Up to SIX total calls: 9472 reserved completion tokens, five reasoned T2 slots (1792 total /1024 thinking) and one T1 (512 total, thinking disabled). FIRST T2 case is the protocol and JSON Schema backend compatibility check WITHIN the six, not an additional seventh call. Do not launch a second batch. On initial HTTP 4xx/5xx, unsupported JSON Schema, schema compilation failure, resource/lease conflict or other fatal transport issue, STOP that campaign and return exact sanitized failure, not a model/server rebuild or downgraded output format.

For actual accepted responses require finish_reason=stop, nonempty final JSON, proper schema/request/case identity, allowed decision, supporting and contrary evidence keys, meaningful short rationale or justified uncertainty, with source_application_authorized false. Raw LLM reasoning can be retained privately but NOT public. Record output_config.format telemetry, token usage, response/output SHA, exact candidate ID, review tier and evidence references. Missing/malformed fields are NOT a pass. If T2 answer length still fails while format works, preserve exact reason, token counts and whether a small private policy variant could improve headroom; do not change production source/config or automatically rerun the full batch. Genuinely uncertain audio may lead to a VALID unresolved advisory, which is success for the judgment contract, not failure.

Evaluate particularly S02 polarity and timing displacement, W01c omitted scene continuity, and T1 non-speech/interjection ambiguity. Favor natural readable English rather than a word-for-word output. Do not let the model claim unheard words, new speaker IDs, or acoustic truth without independent evidence. Maintain immutable original and accepted seven-component revised source/translation/SRT/ASS digests and host resource restoration.

## D. ONE consolidated evidence commit and handback

Publish ONE sanitized evidence-only commit on the SAME DRAFT PR #15 and one self-contained unique-member/checksum private archive. Include exact code/source/config SHA/tree, tests/imports, authenticated previous archive/public members, observed NInfer source, static JSON-schema wire shape, actual model call/request/response/usage identities, accepted/rejected/abstaining outcome counts, concise meaning-first quality analysis and negative controls, ModelLifecycle before/during/after, immutable canonical artifact digests, and COMMAND_SHAPE_FRICTION. Do not publish full copyrighted transcript, private reasoning logs, credentials, weights or media to GitHub. Verify current PR head and public Git blobs/bytes after commit, no source/test/config drift. No new PR, deployment, merge or closure.

Return:
 ISSUE14_MEANING_QC_STRUCTURED_OUTPUT:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<full SHA>
   VALIDATED_SOURCE_SHA=72dbee0c2f2164de31224801f100c564c4570f69
   VALIDATED_SOURCE_TREE=f9416e63d62ba41be738c604d3f22e4fc93bb2b5
   VALIDATED_CONFIG_SHA=e5c5a8a6c3443f7bfbefc25119fd392694eca1dc
   CURRENT_PR_HEAD=<full SHA>
   MAIN_SHA=<full SHA>
   SOURCE_TEST_CONFIG_DIFF=<none or exact paths>
   CPU_FOCUSED_CORE_STUDIO=<exit/counts/import paths>
   PRIOR_ARCHIVE_PUBLIC_AUTH=<6/6,8/8, exact SHA/bytes>
   INSTALLED_NINFER_CONTRACT=<SHA and schema parser evidence>
   STATIC_JSON_SCHEMA_WIRE=<T1/T2 format keys, case/request identity>
   T0_BOUNDARY_QC=<W01c/S02 and negative controls, unchanged originals>
   MODEL_T1_T2=<number called, accepted/rejected/abstained, finish and token use>
   STRUCTURED_OUTPUT_ERRORS=<missing fields, invalid citations, fences, truncation if any>
   MEANING_FIRST_QUALITY=<polarity/continuity/readable English/uncertainties>
   RESOURCE_LIFECYCLE=<owned service before/during/after>
   APPLICATION_BOUNDARY=<advisory only; next safe connection if needed>
   PUBLIC_EVIDENCE_ROOT=<head:path>
   MANIFEST_SHA256=<SHA>
   PUBLIC_MEMBERS_VERIFIED=<count>
   PRIVATE_REVIEW_ARCHIVE=<host path SHA size>
   HUMAN_WER_CER_DER=UNVERIFIED
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|actual records>
   omitted_sensitive_values=<yes|no>

ONE complete handback, not serial mini-reports. After this proof Central should independently evaluate minimal advisory-to-meaning-preserving-English application, then perform an exact-head PR review when the Issue #14 feature criteria are met. Never change tests/source acceptance to manufacture a green result.
