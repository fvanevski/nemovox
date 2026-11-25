# Recovery scheduling and evidence completion

Central adjudicated the original consolidated campaign at tested source
029f969b43c58d53fd27975ff122c097ead04167, carrier
25fd9d15c24bd5623a3ba8cf93d132b713fa4bd4. Private archive:
 /workspace/nvidia-transcribe/outputs/handbacks/occurrence-recovery-20261124-029f969/occurrence-recovery-review-029f969.tar.gz
 SHA-256 4d46e32e6f2459cb348389aa4362b6a3fdfd2dddba56391160cc8ef2773d6755
 678713 bytes; Central verified all 51 payload hashes and 15 public blob
 identities, 5 ledger hashes and all 40 localized raw-response hashes.

Host execution/worker qualification accepted at original exact source.
All 76 canonical rows remained byte-equivalent as parsed JSON; 1 wholly
untimed row remains, 14 unresolved spans. One exact cross-model event
was already represented, with one missing word timing. Actual Qwen
positive 5 and synthetic negatives 0/0 accepted. No model-quality claim.

Central found actionable scheduling defects: broad parent fallback tasks
starved a specific original untimed target; independent admission applied
priority only within each task; numeric unassociated observations scanned
whole chunks; low-priority scans could enlarge focused windows. Central
implemented global independent priority, bracket-based primary scheduling,
localized native-observation planning, protected focused-window geometry,
and explicit whole-event support versus unresolved individual word timing.
The new requirements-challenger.txt records the actual compatible worker
dependency combination. No source acceptance thresholds or budgets raised.

## Scope

This is one CPU-only qualification and evidence completion task. Reuse
previous GPU and Qwen evidence; no model service restart, new Nemotron
or Qwen inference, full episode, geometry sweep, PR, #19, deployment,
merge or issue closure. Local coding LLM can remain loaded. No new
dependency installation if existing environment already matches.

Read AGENTS.md and applicable scope/authority/evidence rules from
LOCAL_RECOVERY_QUALIFICATION.md. Kickoff exact SHA controls; fetch the
branch and verify, isolated detached worktree, owner checkout untouched.
Central owns source semantics; mechanical environment/path fixes allowed.

## Execute

Bind core and Studio Python imports to fresh candidate (PYTHONPATH=src).
Record actual native commands and separate exits:

 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe scripts/replay_recovery_schedule.py scripts/evaluate_occurrence_recovery.py
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_occurrence_recovery.py tests/test_source_quality.py tests/test_source_recovery.py tests/test_source_occurrences.py tests/test_chunk_geometry_evaluation.py -m "not gpu"
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q tests/test_studio*.py -m "not gpu"
 git diff --check

Use qualified challenger env Python, exact dependencies in
requirements-challenger.txt, verify versions and uv pip check. Do not
modify /workspace/pyannote. Reuse private resource config.

Run staged deterministic replay of all 5 prior live cases (no inference):

 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m scripts.replay_recovery_schedule --archive <sealed-prior-review-archive> --output-dir <fresh-private-replay-dir>

Require preserved discovery evidence IDs, no duplicate task IDs, earlier
deferred boundary target admitted globally on frozen observed tasks, and
candidate primary plan admits a useful focused bracket. Compare actual
before and counterfactual after queues and explicit deferrals. Do not
claim candidate primary acoustic outputs are executed. Inspect event
support receipts and pending word-timing distinction.

After CPU tests/replay PASS, one additional CPU Whisper crop pair only.
Use the exact retained prior boundary input.wav, authenticated against
the prior live summary input_audio_sha256
d53c26af1f8dc1e0cfeb3b19735b69973cf9a5512bce0664e8445fe27139f066.
Original run path is
 /tmp/nemovox-issue18-qual.4BKJfH/phase-b-live/boundary/input.wav
or locate identical retained bytes. If absent, reconstruct the registered
570-680 second crop using the authenticated original source from prior
diagnostic evidence and exact prior extraction function; require exact
WAV hash, not merely similar duration. If unavailable, report worker
BLOCKED, complete unaffected replay/evidence. Do not invent or substitute.
The runner validates exact audio and performs only the one newly admitted
target, with previous localized Nemotron observations retained:

 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m scripts.replay_recovery_schedule --archive <sealed-prior-review-archive> --output-dir <fresh-private-targeted-dir> --boundary-wav <exact-boundary-wav> --model-resources <private-existing-resource-config>

No demand for a correction. The short recognized phrase may be a genuine
repeat or duplicated hypothesis; neither silence nor lexical identity
alone can retire the original. Preserve favorable and unfavorable results
and all unresolved observations. No threshold tuning.

## Complete omitted primary raw response evidence

The prior private archive had empty recording/asr/chunks directories.
Actual manifests identify recording/chunks/chunk_*.raw.json under the
retained original live run directories, not that empty archive path.
Read each of the five authenticated archived recording manifests. Locate
the exact actual raw_response_path for each of the nine completed primary
chunks (and any required rejected-generation receipts if present).
Check bytes SHA-256 against manifest raw_response_sha256; retain exact
bytes and a path-independent mapping with case, chunk_index, expected and
actual hash, size, archived-relative path. Include these in the new
private evidence archive, not in public Git. If a file is unavailable,
report explicit missing case/index/hash; do not regenerate inference
or manufacture response JSON from a ledger. Preserve prior archive
identity and do not duplicate the entire old archive.

## Evidence and stop

One sanitized evidence-only commit under
staging/agent-evidence/recovery-scheduling/<date>-<tested-short-sha>.
No source changes/pushes, no PR. Exact candidate and separate carrier,
test logs, identities, resource/dependency receipts, replay/targeted
summary, omitted-response mapping with hashes (no private paths/text),
HANDback.md, friction.json, manifest. Private self-contained supplement
archive includes private replay and target observations/decisions,
actual nine raw responses and mapping, relevant failure evidence.
Exclude audio/media, weights, env files, credentials and raw transcript
content from public Git. Do not repeat unrelated historical private data.
Hash/size/archive path, manifest checks, staged allowlist and exact
nonforce readback required; attach archive to Central.

Prior dependency friction is now a repo-specific documented constraint,
not a new GHDEV command-shape recipe. For new friction provide canonical
COMMAND_SHAPE_FRICTION block, with effective command and event_kind,
execution_effect, correction, provenance, and contract conflict fields
as applicable. Do not restate resolved old friction as a new incident.

RESULT=PASS|FAIL|BLOCKED|STALE applies to qualification/evidence only.
A negative acoustic result is not a test failure; missing or invalid
evidence is explicit. Complete unaffected checks, preserve partial
results, and stop. Central adjudicates combined #17/#18 readiness.
