# Bounded matched acoustic geometry evaluation

Issue #17, roadmap #16. Central qualification of the A1 remediation and A2
structure at 5df4ace871940cc71e651b04fbdeaa8f3bc1f69b is accepted, with unresolved
acoustic locality. Evidence carrier 7a57c2aa2e4d1b1c4526ff769a479acb55207ea5.
Prior packet: private archive SHA-256
7a92656f012df214f443f1292ee71cf8f171b2148b58bf8ee39b63e158f12faa.
445 CPU and 50 Studio tests passed; 2360/2360 dispositions, 199 alternate
references all resolve to selected observations, 168 unresolved context
candidates, 4 disputed spans retained, partial provisional spans 8 -> 3.
No acoustic truth or issue closure established. Do not repeat the A1/A2
replay campaign; production code is unchanged in this handoff.

## Authority and execution

Read AGENTS.md, docs/model-resources/LIFECYCLE.md and the applicable evidence
and worktree rules in docs/source-reconstruction/LOCAL_QUALIFICATION.md.
The kickoff exact candidate SHA controls. Central authored the evaluation
runner and CPU tests; local Build executes host-bound validation, not
strategy/model selection/semantic remediation. Do not create a new PR.
Keep functional names, not issue/slice identifiers, in normal user-facing code.

LOCAL_AGENT_PRIMARY=build
EXECUTION_CLASS=GPU_INTENSIVE_ASR
SEMANTIC_REVIEW_AUTHORITY=central-owned
SEMANTIC_REMEDIATION=forbidden
MECHANICAL_REMEDIATION=allowed
REMOTE_UPDATE=forbidden
SOURCE_REMOTE_UPDATE=forbidden
EVIDENCE_REMOTE_UPDATE=allowed
EVIDENCE_ROOT=staging/agent-evidence/chunk-geometry
EVIDENCE_PACKET_REQUIRED=yes
PR_CREATION=forbidden
PRODUCTION_DEPLOYMENT=forbidden

## CPU preflight (before ASR acquisition)

1. Authenticate the owner checkout and origin, fetch exact refs/heads/issue/17
   and prove FETCH_HEAD equals kickoff candidate SHA. Prepare a fresh isolated
   detached worktree without resetting or switching the owner checkout.
   Bind /workspace/nvidia-transcribe/.venv/bin/python with PYTHONPATH=src
   from the candidate. Prove actual import paths and candidate clean state.
   Use the existing dedicated environments; no dependency upgrade or downloads.
2. Run compileall -q on src/nvidia_transcribe and scripts/evaluate_chunk_geometry.py;
   run python -m pytest -q tests/test_chunk_geometry_evaluation.py
   tests/test_source_occurrences.py tests/test_issue14_timing_boundaries.py
   tests/test_chunking_and_resume.py -m "not gpu", followed by git diff --check.
   Capture exact exits. No repeat full A2 qualification merely for evidence
   files and this evaluation-only source; if any production source changes,
   Central must adjudicate and requalify it.
3. Authenticate the original diagnostic archive with SHA-256
   307c0a32bd4ae32008be0141d5ac12e066f96a728bfcb179c0b343bf2031fd7b:
   /workspace/nvidia-transcribe/outputs/handbacks/source-reconstruction-diagnostic-20261124/source-reconstruction-diagnostic-review.tar.gz
   If relocated, discover within known handbacks, not arbitrary system trees.
   Use the actual original media path or relocated same media. The runner
   resolves stream/duration/fingerprint from authenticated source metadata.
   If archive layout needs bounded mechanical adaptation, verify the same
   source identity from sealed metadata and return full patch; do not substitute
   a different episode, language, stream, or unauthenticated input.
4. Invoke the staged runner first with --plan-only and a new private output
   directory (commands below). Verify exact media identity and preregistered
   matched regions. No ASR acquisition in plan-only.
5. Verify current actual ASR resource mapping, local checkpoint, executable,
   service markers, endpoint, installed quantization and CTC configuration;
   inspect nvidia-smi and current relevant service state without exposing
   credentials. Use the actual existing model-resources and lifecycle local
   configuration (NEMOVOX_MODEL_RESOURCES / NEMOVOX_LIFECYCLE_CONFIG, or
   explicit --model-resources with actual file). Do not overwrite config.
   Validate expected Nemotron NVFP4 checkpoint and modelopt_mixed with actual
   service receipts when acquired, plus CTC middleware/capture flags and
   VLLM_USE_V2_MODEL_RUNNER=0. No arbitrary alternate model or CPU downgrade.
   If capacity unavailable because the operator's local LLM or other service
   is active, report RESOURCE_BLOCKED; do not stop external services. Operator
   can release their own resource before a fresh GPU attempt. Use one existing
   managed ASR acquisition scope across the entire bounded campaign. No manual
   service restart or unowned service eviction, no /workspace/pyannote changes.

## Matched GPU campaign

The runner invokes the actual current process_media path, not a separate
transcription implementation. All arms use French, speaker mode, same local
model, no MT, text review, Qwen or TitaNet. Native bounded generation retry,
subdivision and subinterval recovery remain active; actual requests counted.
Four independent 110-second comparison regions (440s unique source coverage,
not full episode):

| Region | Full comparison coverage (absolute seconds) | Reference target | Distinct local crops |
|---|---|---|---|
| Compression | 260–370 | 298–322 | 286–329 / 291–335 |
| Boundary | 570–680 | 609–643 | 599–650 / 605–658 |
| Late gap | 1160–1270 | 1212–1252 | 1201–1259 / 1207–1265 |
| Low-flag control | 440–550 | 470–505 | 460–511 / 465–517 |

Compare existing fixed 90s core/10s context to existing pause-aware planner
target 39s, max requested core 46s, max context 4s. Actual pause decisions,
tail slack, effective leaf sizes, retries and wall/audio cost must be reported.
Both arms consume identical PCM region bytes. Distinct local crops cover the
same predeclared target; two-crop positive support is only a same-model
reference. No issue-specific values enter production. Reference target
intervals are from the prior evidence and may not contain the actual
displaced speech; absence at a hypothesized interval is not disproof.

Example execution from the candidate, using actual host paths:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python scripts/evaluate_chunk_geometry.py --diagnostic-archive <authenticated-archive> --output-dir <new-private-plan-dir> --plan-only
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python scripts/evaluate_chunk_geometry.py --diagnostic-archive <authenticated-archive> --output-dir <new-private-run-dir> [--media <relocated-identical-media>] [--model-resources <actual-host-resource-file>] [--asr-resource <actual-named-asr-resource>]

Bracketed arguments are optional only when host authority supplies them.
Use actual paths, not literal placeholders. Capture native exits/logs and
resource state before, during and after. Model startup health and ownership
evidence are host tasks. No secret environment dumps. The runner caps actual
inference at 72 requests and 2000 audio seconds, including normal recovery;
do not bypass the cap or extend regions if a run fails. Preserve completed
arms/receipts and report the incomplete region and cause. Do not restart
the whole campaign for independent evidence failures.

Inspect all four matched region summaries and bounded consequential examples
in private-comparison.json and the fresh recording/ledger/raw leaves:
- Verify both arms' identical input SHA, accepted raw leaf hashes, gap-free
  effective core coverage, complete source-unit dispositions and exact
  selected-to-utterance identity coverage. Include actual planner geometry
  and forced cuts.
- Compare matched reference-event positive support, timing deltas, exact
  source/selected lexical sequence evidence, unresolved and ambiguous
  matches. Inspect differing decisions, not every transcript word.
  Strict event-match absence is not a proven omitted event; a long gap is
  not alone a timing defect. Same-model consensus is not human verification.
- Specifically assess early compression, the previously substituted
  boundary span around 620–622s and the entire 1212–1252s target region.
  Assess low-flag control degradation and meaningful simultaneous speech or
  repeats where actually present. Treat speaker identity as hypothesis;
  do not equate label strings across crops.
- If no reference events are obtained for a target, report it as insufficient
  reference evidence, not a successful repair or proof of silence.
  Preserve competing hypotheses and observed crop-relative/absolute offsets.
- Report actual requests, inferred audio seconds, service acquisition and
  cleanup, measured wall time and resource receipts. No final policy
  promotion locally. Do not run a full episode, additional model campaign,
  translation/render campaign, or #18/#19 implementation.

## Evidence delivery

One sanitized evidence-only commit on existing issue/17, with exact source
SHA separate from carrier. Include identity, compile/test/diff receipts,
plan.json, summary.json (inspect/sanitize first), resource identity and
bounded lifecycle receipts, manifest hashes, HANDback.md and friction.json.
Use explicit allowlisted git add -f for ignored receipt files and verify
staged index against public manifest before committing. Fetch/authenticate
current remote head before non-force publication, verify full SHA and paths
after push. No source edits, force-push, PR creation or closure.

Attach one self-contained private tar.gz containing required public receipts
plus the complete private-comparison.json, private-requests.json, per-run
recording.contract.json, source-evidence.v1.json, manifest.json, actual
accepted raw response JSON and any relevant rejected-generation JSON with
checksums. Exclude WAVs, original media, weights, secrets and unrelated logs.
Do not publish source transcripts or raw ASR/CTC in Git. If any public
summary fields contain private text or paths beyond safe identity facts,
sanitize them and describe omission. Include private originals where needed.
Manifest binds archive artifacts to exact run names, crop offsets, SHA-256
and byte sizes. Return absolute archive path, bytes, SHA-256 and public
evidence root/carrier SHA. If authorized mechanical fixes were required,
include full patch including untracked files privately and public-safe
where possible; no local semantic source edits. Record resource and
COMMAND_SHAPE_FRICTION blocks as before.

RESULT=PASS|FAIL|BLOCKED|STALE means this bounded experiment executed with
valid receipts, NOT that the short geometry is better. Report evidence
limitations and both favorable and unfavorable results, no test-count
targets or unsupported default recommendation. Central determines whether
to promote geometry now or defer final defaults to #19, and whether the
unified recovery work requires combining #17 and #18. Stop after one
campaign. Do not close #17 or deploy.
