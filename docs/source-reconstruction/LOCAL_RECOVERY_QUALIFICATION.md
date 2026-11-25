# Consolidated occurrence recovery qualification

Issue #18 and remaining #17 consumer corrections, one existing issue/17
branch. The kickoff's exact candidate SHA controls. The previous A1/A2,
A3 GPU and final disposition evidence is accepted at its original SHAs.
This campaign qualifies the changed recovery/consumer path, not a new
geometry comparison or full-episode workflow. No PR creation locally.

## Authority and preparation

Read AGENTS.md, docs/model-resources/LIFECYCLE.md, CONFIGURATION.md, and
applicable execution/evidence provisions in
docs/source-reconstruction/LOCAL_DISPOSITION_QUALIFICATION.md.
Central owns architecture, implementation, regressions and adjudication.
Local Build owns host integration, bounded dependency setup and execution.
Autonomously resolve ordinary host path/environment/invocation friction;
no semantic implementation or threshold edits, no test weakening.

Authenticate origin, fetch refs/heads/issue/17 and verify FETCH_HEAD equals
kickoff candidate. Fresh detached worktree; never reset/switch owner checkout.
Use PYTHONPATH=src with the actual core and Studio host environments;
verify imports bind candidate. Capture native commands, exits and logs.

## Phase A: CPU-only, including independent worker

From candidate:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe scripts/evaluate_occurrence_recovery.py
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_occurrence_recovery.py tests/test_source_occurrences.py tests/test_source_quality.py tests/test_source_recovery.py tests/test_qwen_session_and_policy.py tests/test_translation.py tests/test_chunking_and_resume.py tests/test_recording_and_normalization.py tests/test_issue14_timing_boundaries.py tests/test_chunk_geometry_evaluation.py -m "not gpu"
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q tests/test_studio*.py -m "not gpu"
  git diff --check

Run the staged replay against the existing sealed A3 archive:
  /workspace/nvidia-transcribe/outputs/handbacks/chunk-geometry-20261124-13c0bb8/chunk-geometry-a3-evidence-13c0bb8.tar.gz
  SHA-256 805ab0e43275171b16c0beebbc6c938d15ed5b055d4a6e5b342180f46d61eefe

  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m scripts.evaluate_occurrence_recovery --phase replay --a3-archive <actual-archive> --output-dir <new-private-replay-dir>

Verify 16 runs, four prior disposition corrections, all stable task IDs,
full and inventory-ablated scheduler comparison, explicit deferrals. This
is not a replay of the historical executable. Inspect bounded severity/
late-case coverage, no arbitrary gap-as-speech inference.

Host challenger integration is expressly authorized by #18. Locate the
existing faster-whisper-large-v3 multilingual CTranslate2 cached snapshot
used in the retained diagnostic probe; authenticate actual model identity
from the prior diagnostic metadata and current files. Check required local
model.bin, config.json, tokenizer.json, preprocessor_config.json. Do not
download models or modify /workspace/pyannote. Prepare/reuse dedicated
/workspace/nvidia-transcribe/.venv-challenger (or an existing equivalent
Nemovox-owned isolated environment) using a supported installed Python.
Install only needed faster-whisper/CTranslate2 dependencies, preferably
matching the prior diagnostic version if evidenced; package downloads
permitted for this dedicated environment only. Record exact interpreter,
versions, pip check, cached checkpoint hashes. Do not upgrade host packages
or other environments. Never guess model identity or replace it with a
different model. No need to install when already qualified.

Use a private copy of the actual existing resource configuration, with
only the explicit source_challenger role and whisper_cpu binding added.
Do not change production service configuration. Preserve credential
indirection, lifecycle configuration and other roles. Keep local config
private; return sanitized resource fingerprints and versions. The actual
cached checkpoint and worker interpreter paths are host-specific inputs.

Diagnostic archive:
  /workspace/nvidia-transcribe/outputs/handbacks/source-reconstruction-diagnostic-20261124/source-reconstruction-diagnostic-review.tar.gz
  SHA-256 307c0a32bd4ae32008be0141d5ac12e066f96a728bfcb179c0b343bf2031fd7b

  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m scripts.evaluate_occurrence_recovery --phase worker --diagnostic-archive <actual-diagnostic-archive> --model-resources <private-resource-config> --output-dir <new-private-worker-dir>

If original media relocated use --media with authenticated identical
fingerprint and stream. This CPU control uses two bounded crops, actual
worker/receipt admission, no semantic correctness claim. Require real
CPU int8 worker observations and matching input/checkpoint receipts.
If resource setup fails, preserve deterministic test/replay evidence;
do not bypass worker admission or start GPU campaign on broken source.

## Phase B: GPU-intensive ASR and subsequent Qwen control

Operator must release their own GPU-resident coding LLM if required.
Use remote-capable local harness backend for GPU phase. Inspect nvidia-smi,
exact owned resource/unit markers and live identity. Never stop unrelated
services. Validate cached Nemotron checkpoint, modelopt_mixed, FP8 KV,
CTC middleware, VLLM_USE_V2_MODEL_RUNNER=0 and capture flags. Use existing
managed resource lifecycle. Record resource state, startup, cleanup and
unchanged external services. No implicit download, CPU downgrade or
production deploy.

Run staged actual acoustic path on four existing registered 110s regions,
one fixed-geometry primary each, followed by bounded source-quality
recovery. This is not another A3 geometry sweep. Use one owned ASR scope per
region; runner closes it before independent CPU evidence. Text/vision
review explicitly excluded in this evaluation, not in normal CLI/Studio.
No MT, rendering, full episode or #19 campaign. The existing budgeted
client caps total ASR at 72 requests / 2000 audio seconds; per-region
recovery cap is four primary window pairs, two extra contexts, 240 audio
seconds, two independent targets / 100 audio seconds. Wall admission and
worker timeout are explicit.

Include one existing bounded multilingual control (prefer known Mandarin
fixture/clip from previous qualified evaluations, including positive
repeats/overlap if available). Discover only within known Nemovox fixtures,
prior handbacks, local evaluation inputs, or known existing original
media. Use actual evidence-backed language and a short interval, no
download. Record its source/stream/crop/fingerprint and known reference
if available. Absence is explicit qualification gap, not a reason to
substitute an invented or unrelated reference.

  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m scripts.evaluate_occurrence_recovery --phase live --diagnostic-archive <actual-archive> --model-resources <private-resource-config> --output-dir <new-private-live-dir> --control-media <existing-control> --control-language zh --control-start <actual-seconds> --control-seconds <duration-at-most-60>

Optional --media for relocated original; use actual paths/args. Omit
control arguments only if no authentic control exists, report that gap.
Inspect actual requests, acquired/deferred tasks, origin identity coverage,
residual unlocalized rows, cross-model positive support, false-addition/
repeat/overlap risk, and concrete favorable AND unfavorable decisions.
No claimed true accuracy from matched text alone. No threshold changes.

After verified ASR cleanup, verify cached Qwen checkpoint and its actual
worker environment. No model download; inspect all needed cached assets.
Run staged Qwen control using actual credible ordered text from the live
positive observations. The runner uses one real GPU worker session and
replays reversed/zero-duration candidate responses as labeled synthetic
negative controls (not evidence that forced alignment recognizes words).
Require actual positive timing evidence and zero recovered units for
negative replays. If no credible positive input or GPU admission, report
explicit BLOCKED; do not fabricate a reference or change thresholds.

  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m scripts.evaluate_occurrence_recovery --phase qwen --live-output-dir <actual-live-dir> --model-resources <private-resource-config> --output-dir <new-private-qwen-dir>

Complete unaffected independent checks if a check fails. If CPU source
regressions fail, do not expend GPU resources on unqualified candidate.
Do not expand beyond registered campaign or re-run entire campaign for
nonsemantic evidence friction. Preserve completed partial results.

## Evidence and stop

One sanitized evidence-only commit on issue/17, root
staging/agent-evidence/occurrence-recovery/<date>-<tested-short-sha>.
No source commits/pushes locally. Compare exact staged allowlist with
manifest; force-add only specific ignored receipt paths. Fresh remote
head readback before non-force push, exact readback afterward.
If stale, do not rebase/force; return private packet.

Public: exact identities, test/compile/diff logs, sanitized replay,
worker, live and Qwen summaries, dependency/resource/lifecycle receipts,
bounded discrepancy counts, HANDback.md, friction.json and manifest.
Private self-contained archive adds private task/worker/control data,
per-region recording contract, source ledger, manifest, accepted raw
response JSON, source-quality report (including raw challenge observations),
canonical-source contract, actual request receipts and relevant failure
receipts. Exclude original media, WAVs, weights, venv, secrets, private
configuration and unrelated logs. No raw source text or media paths in
public summaries. Include SHA-256, byte sizes, exact tested source and
separate carrier identity; archive absolute path, bytes, SHA-256.
Do not duplicate previous sealed archives themselves. Attach private
archive to Central. Report canonical COMMAND_SHAPE_FRICTION.

RESULT=PASS|FAIL|BLOCKED|STALE describes executed qualification and
evidence, not issue closure, accuracy, or policy promotion. Distinguish
unavailable evidence from failure. Stop after one coherent campaign;
no PR, merge, closure, #19 or deployment. Central adjudicates residual
semantic uncertainty, regression results and final combined PR.
