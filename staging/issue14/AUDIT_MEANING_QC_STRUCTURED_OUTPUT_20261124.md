# Issue #14: Structured-output QC contract repair, 2026-11-24

**Repository:** fvanevski/nemovox, existing draft PR #15; Issue #14 OPEN.
**Central source/test authority:** 72dbee0c2f2164de31224801f100c564c4570f69
**Source tree:** f9416e63d62ba41be738c604d3f22e4fc93bb2b5
**Unchanged versioned review config authority:** e5c5a8a6c3443f7bfbefc25119fd392694eca1dc
**Status:** Central code and tests are staged, NOT yet candidate-bound host qualified.

## Evidence admitted, failure classified

Uploaded private handback archive issue14-ninfer-messages-wire-20261124T023146Z.tar.gz, SHA256 bd294a2d705e4658e5b40b88b251df24eab9fd80b22620bc5f950c4e955cc29f, 76,644 bytes; independently checked all 6/6 member SHA and seven unique regular archive entries. Previous host proof at evidence-only commit 94b1c6b6382481510409277b672b9747f514b25f:staging/issue14/evidence/20261124T023146Z/: all eight public member file Git blobs and byte counts independently read back, plus manifest. Host core evidence: focused 101/101, core 470/470, Studio 470/470, no source/test/config drift since its admitted source 9230bde41a418c074140252edb6c46637560adb2.

The NInfer Messages wire now succeeds (six HTTP 200 calls, real per-request thinking budget, no model ownership leak). Its previous leading system placement defect is resolved. The 6/6 *unaccepted* reviews are a structured-output CONTRACT defect rather than a model transport or source-fidelity failure. The private response records corroborate:
- Call 0 produced incomplete JSON and reached finish_reason=length after 1,048 reasoning tokens and visible completion use. It omitted the required schema key while writing verbose prose citations.
- Call 1 included schema, but both evidence arrays contained verbose narrative strings instead of six allowed citation keys. Host validator rejected contrary evidence.
- Calls 2, 3, and 4 omitted the required schema and used prose evidence descriptions. They nonetheless identified meaningful ambiguity, including S02 negation and W01c uncertain context, but were not valid machine-readable advisories.
- T1 Call 5 wrapped its JSON in a Markdown json fence and supplied prose evidence citations.
Every rejected response remained nonapplying. Host ModelLifecycle stopped and restored the owned service. There is no basis to merge, auto-rewrite source or claim that fluent model reasoning independently established speech.

Product objective: readable, grammatical, clear English faithful to intended meaning and narrative sequence. Preserve consequential negation, causation and relevant speaker transitions with usable event timing; literal 100% word-level source accuracy is NOT a gate. Source raw evidence integrity and independent claims of hearing still require source-bound verification.

## Verified backend contract

NInfer exact commit a3a3dd5be6c2ecec1479017d5eee219d7a7e5784:
- src/serve/anthropic_messages_request.cpp parse_effort consumes output_config.format for /v1/messages using JsonFormatProtocol::Anthropic.
- src/serve/request_validation.cpp parse_json_output_format accepts Anthropic type=json_schema plus a DIRECT schema object, without OpenAI's nested json_schema/name wrapper. The server's structured-output constraint is distinct from simple JSON prompt text.
- The installed runtime previously confirmed top-level system, thinking.type enabled/disabled, output_config.effort, and actual thinking/output token telemetry.

## Central implementation

1. semantic_review.py defines ONE exact six-key evidence citation vocabulary used for schema and validator. semantic_review_json_schema builds a compact JSON Schema for each case; required keys and enum constants bind schema version, current candidate ID and request SHA. Supporting/contrary fields are arrays of allowed reference KEYS. Explanatory paragraphs belong in rationale, not those arrays.
2. build_semantic_review_messages explicitly states the required fields, exact citation keys, short rationale, contextual meaning objective, no Markdown fences and no unsupported audio/visual provenance. It asks for a bounded reply rather than multi-paragraph explanations. The validator remains authoritative and still refuses evidence misattribution.
3. run_tiered_text_reviews sends response_format type=json_schema and pins its schema to the case; it rejects an altered request digest BEFORE spending model tokens, and still checks complete finish, verified source identity, evidence and abstention.
4. ninfer_messages_client.py sends Anthropic output_config.format with type=json_schema and direct schema object. It does not silently downgrade to unconstrained Chat Completions or prose on server rejection; it reports the actual output-format flag in transport telemetry.
5. _decode_semantic_review_json narrowly permits a single exact enclosing JSON Markdown fence when a provider disregards requested format. Any surrounding prose, wrong/missing schema, untrusted citation text, altered request ID, or truncated response remains unaccepted.
6. Regression tests cover constrained T1/T2 wire shape, exact case/request schema and six reference keys, real observed malformed-output classes, clean fence handling, explicit stale-request preflight, and no source/translation/subtitle mutation.

This repair changes only semantic-review advisory presentation and its dedicated model transport. No new ASR/CTC, translation, speaker, model selection, source revisions or renderer behavior is introduced. Existing seven-component reviewed revision remains an independent accepted-output control.

## Next qualification boundary

One CPU-first, then at most ONE managed NInfer text session of six versioned cases. First actual request checks output_config.format compatibility within the normal six-call budget. Strict success requires structured FINAL JSON that passes the unchanged semantic evidence validator, including supporting and contrary evidence or justified abstention. At least one T2 S02/W01c and one T1 advisory should be evaluated when present; no arbitrary 100% case acceptance target if evidence is genuinely insufficient. Truncated outputs, unsupported citation/ID claims and genuine backend JSON grammar failures are real failures, not reasons to weaken validation.

The model answers remain advisory. After host proof Central should independently determine the smallest optional bridge from valid, evidence-bound reasoning to the existing reviewed-revision or English postedit path and perform a current exact-head PR review. Do not create a new PR, deploy, merge, close Issue #14, rerun full episode or silently alter NInfer service configuration.
