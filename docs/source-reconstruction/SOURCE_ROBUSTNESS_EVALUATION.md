# Source robustness: targeted evidence comparison

Use this driver only for private, bounded experiments following the completed
Studio intake for issue #24. It is not a new user-facing mode or a ground-truth
annotation tool. The current production scene path retains rejected responses,
identifies invalid decision predicates and permits at most one corrective
request per case; all source-edit admissibility remains unchanged.

CPU: run the focused scene/occurrence tests and the replay phase on the
authenticated archive. The sequence detector only nominates source-order
correspondence with disjoint provisional CTC times. Evaluate all nominations
at the three screening lengths, including false positives and existing task
coverage; no chosen threshold or speech retirement is authorized by this run.

GPU: use the acoustic phase with the Central-provided private JSON crop plan.
The primary matched arm recreates the archived WAV exactly and compares against
the same right edge and core with left context removed. Local crops use the
same primary generation settings and reconstruct absolute source times.
The independent CUDA phase follows release of the owned native stage. Compare
raw recognition, CTC, bounded independent hypotheses, and retained canonical
rows. Missing recognition is not proof of silence. Record differences in
model/runtime identities, actual costs, unresolved speech and speaker truth.
The scene phase uses reconstructed failed inputs and successful initial
controls, the real native Messages client, and the same source-edit safeguards.
A corrected structure is not evidence that a source change is warranted.

Use the normal resource lifecycle and existing cached models; do not start a
concurrent local GPU coding model, evict another workload, download weights,
or change production configuration. Use a fresh private output directory for
each phase. Keep archive, probe plans, media, raw responses and derived text
off GitHub. Return a single private handback with immutable source and input
identities, observed positive/negative results, costs, and resource restoration.
Remove this diagnostic driver and obsolete controls when successor mechanisms
have been selected and qualified, unless a current useful purpose remains.


## Local execution contract

The kickoff supplies the exact candidate commit, original archive identity,
and private crop plan. Use an authorized Build or equivalent host supervisor
with normal permissions. GPU-heavy execution requires a remote-backed coding
agent or an operator-established free-capacity run; a GPU-backed local coding
agent may do only CPU preparation until the operator frees capacity through
the normal admission path. Do not change global harness settings or stop an
unrelated service.

From the owner workspace, authenticate the origin and current source; use
native Git or the documented RTK raw proxy if the actual harness requires it.
Fetch only refs/heads/issue/24; prove FETCH_HEAD equals the supplied full
candidate SHA, then prove the exact object is a commit and create a fresh
detached worktree at it. Do not switch or modify the owner checkout.
Check the complete base-to-candidate diff inventory and cleanliness.
Use a fresh private root with mode 0700 under outputs/handbacks, with the
private plan and archive outside the worktree.

Run from the candidate root, with the owner interpreter and candidate imports:

    PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe scripts/evaluate_source_robustness.py
    PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_source_robustness_evaluation.py tests/test_scene_evidence.py tests/test_source_quality.py tests/test_occurrence_recovery.py tests/test_source_recovery.py -m "not gpu"

Verify nvidia_transcribe and scripts imports resolve under the candidate root.
Save actual commands, exits and unfiltered failure evidence. Use git diff
--check on the full base-to-candidate delta, not just an empty worktree diff.
Do not change expected results or weaken validation. Resolve path, interpreter,
resource-config and invocation problems using the actual existing interfaces.
Substantive defects return to Central with native evidence.

Bind ARCHIVE, ARCHIVE_SHA256, PRIVATE_ROOT, PLAN_PATH and RESOURCE_CONFIG to
the kickoff/actual existing host paths. PRIVATE_ROOT must exist; each phase
output subdirectory must not yet exist.

    PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m scripts.evaluate_source_robustness --phase replay --archive "$ARCHIVE" --archive-sha256 "$ARCHIVE_SHA256" --output-dir "$PRIVATE_ROOT/replay"

Record candidate nominations and existing task coverage at each screening
length; compare against the sealed observed raw sequence, not a desired
subtitle string. No required positive correction count.

For GPU work, record GPU/process/owned-resource identities and effective
resource/lifecycle config fingerprints before/after. Keep the actual
historical resource binding when available. Verify Nemotron modelopt_mixed
and native CTC, existing cached CUDA Whisper, and native Messages text route.
Report mismatches explicitly; do not silently substitute model or device.

    PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m scripts.evaluate_source_robustness --phase acoustic --archive "$ARCHIVE" --archive-sha256 "$ARCHIVE_SHA256" --plan "$PLAN_PATH" --model-resources "$RESOURCE_CONFIG" --output-dir "$PRIVATE_ROOT/acoustic"
    PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m scripts.evaluate_source_robustness --phase scene --archive "$ARCHIVE" --archive-sha256 "$ARCHIVE_SHA256" --model-resources "$RESOURCE_CONFIG" --output-dir "$PRIVATE_ROOT/scene"

If the authenticated original media has moved, use --media with the exact
matching file. A fresh private result root is mandatory for any retry. Retain
prior partial receipts. Do not replay successful model operations solely to
repair a reporting error. If acoustic inference is blocked, finish CPU
evidence and any independently admissible scene phase; report the blocked
scope. No new model downloads, no unrelated service changes, no full episode,
no broad geometry sweep. Keep actual GPU work separate from local-model
coding inference. No deployment, source semantic edits, GitHub writes or PR.

Return one private tar.gz containing the plan, tested commit/tree and actual
imports, focused test output, replay nominations, native raw responses and
reconstructions, independent observations, scene decisions/rejected receipts,
comparison report, resource/time/VRAM/cleanup evidence, missing or failed
steps, and command-shape friction. Exclude whole WAV collections (retain
hashes/crop bounds), original media, weights, credentials and unrelated
logs. Verify every included member against a manifest, verify the final
archive hash/size and use a direct conversational handback. Keep the original
Studio job, archive, owner source and service configuration unchanged. Attach
the verified private archive to Central. The branch is not review-ready and
does not authorize promotion, merge or closure.


## Successor production qualification

The sealed targeted comparison is reusable as a private counterfactual
evaluation; it must not be described as a fresh production acquisition.
Run the successor CPU suite, including tests/test_targeted_occurrences.py,
and the full CPU repository suite. Run the current production projection
and selectors over the authenticated comparison archive:

    PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m scripts.replay_source_robustness --archive "$COMPARISON_ARCHIVE" --sha256 "$COMPARISON_SHA256" --output "$PRIVATE_ROOT/production-replay.json"

Report exact retired source IDs, remaining unresolved fragments, recovered
event text/bounds, origin lineage and unrelated changes. The known case
should retire the nine displaced fragments only with positive earlier,
later and context-removed support; the meaningful exposition should become
renderable at approximate event bounds. All six historical first scene
replies should validate after lossless normalization of known duplicate
IDs. These are hypotheses to check, not required fabricated outputs.
Do not change thresholds, tests or evidence association to manufacture a pass.

Then, if the CPU/replay gate passes, qualify one fresh normal candidate
source-quality run on the authenticated original recording/media with the
current registry and prior job policy. Use the supported CLI/Studio
continuation entry, a new private output directory, candidate-bound source
and fresh source-quality identity. Copy the exact immutable recording
artifacts when required by that entry, not old canonical or downstream
checkpoints. Do not rerun primary full-episode ASR merely to regenerate
retained input. The candidate normal scheduler must perform its own real
acquisitions. Preserve fresh source-quality and canonical output, stage
costs, Riva/English review and SRT/ASS behavior. Report any real failure
without substituting the counterfactual replay for the normal route.

Use one independently selected meaningful multilingual clip beyond these
discovery cases, preferably an already available local Mandarin recording
with verified identity, containing a genuine repeated phrase or previously
unrenderable material. Use the existing public pipeline with the same
normal policy, bounded media (approximately 60–120 s, enough for relevant
context); compare original and canonical source, real observed evidence,
rendering and negative controls. Do not require a false occurrence to
exist in the independent clip, or invent human acoustic truth. If the
clip fails to exercise a relevant mechanism, record the evidence gap
rather than claim generalized benefit. Do not launch a broad sweep.

Run applicable repository Studio/API/frontend checks from their documented
current instructions. Record actual commands, results, candidate identity,
GPU acquisition/release and total per-stage costs. No deployment, PR or
GitHub writes. Return one private review archive; if a substantive defect
blocks qualification, stop affected inference, preserve usable results and
traceback, and complete unrelated CPU checks. Central owns fixes and final
review. Do not run the superseded registered GPU comparison again.
