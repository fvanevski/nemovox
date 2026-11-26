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


The successor qualification supersedes the earlier registered acoustic and
scene comparison execution instructions on this branch. Preserve both sealed
input archives; do not rerun that earlier campaign. The context-removed
request is a raw-generation contrast. It is not used as a timed-row
authority if its CTC projection is rejected. Local positive competing
speech and independent observations are still required for retirement.

For the full French downstream continuation, the currently inspected CLI
supports the following invocation (bind all paths to verified host values;
run from the candidate root, using the actual existing lifecycle environment).
Use a unique output path and do not supply a sidecar/export destination:

    PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.cli generate-subtitles "$MEDIA" --recording-contract "$RECORDING_CONTRACT" --source-quality on --output-dir "$PRIVATE_ROOT/french" --source-language fr --target-language en --audio-stream 0 --chunk-policy fixed --chunk-seconds 90 --context-seconds 10 --speaker-embeddings none --alignment-fallback off --refinement off --translator riva --translation-review balanced --style natural --formats srt,ass --ass-layout standard --ass-max-lanes 3 --model-resources "$RESOURCE_CONFIG"

Use the original job's immutable authenticated recording artifacts; no
copy of the historical source-quality checkpoint or downstream checkpoint.
Omit --glossary (baseline empty); do not enable condensation. Verify the
actual loaded candidate source and exact source/media resource bindings.
Use native CLI help/AGENTS.md to resolve entrypoint issues if necessary.
If the normal continuation cannot consume the authenticated recording
without a source change, report the precise blocker before rerunning
full primary ASR. The full Studio wall-time comparison is not valid for
this continuation; compare source-quality/subtitle stages separately.


## Live-failure remediation qualification

The live candidate d0d4af0dc95edcc823b654a25dba89f35738a739 failed
both languages. Its successor a010a7ebbb8c57d4bdd2d6ed0862df8646663dbf
passed the focused and full CPU suites, applicable Studio/API and frontend
checks, the original sealed comparison and the French retained-live
selector replay. The Mandarin retained-live selector recovered six events
(65/402 source units, 14.16 s) without losing source units or collapsing
the unresolved repeat hypotheses. This is a real insufficiency of output
coverage for that retained evidence, not a fresh test of the changed
acquisition scheduler.

The earlier independent targets covered only 28.3105 unique target
seconds; their actual crop union ended at 50.8454 s. The new dry structural
plan admits six failed-core windows covering 0–104.5 s of the same 119 s
clip. Its independent crop projection costs 236.5 s against the existing
240 s limit. These are planning measurements, not recovered speech or
actual acquisition receipts. The 104.5–119 s core remains deferred in
that dry plan.

### Continuation and evidence-specific gates

This section supersedes earlier instructions that make sufficient
retained-evidence Mandarin dialogue coverage a prerequisite for any live
qualification. Do not relax source acceptance predicates, increase
budgets, or accept an unusable output to pass a gate.

1. Admit the exact current candidate. The documentation-only successor
   may reuse the authenticated CPU, integration and replay receipts of
   a010a7ebbb8c57d4bdd2d6ed0862df8646663dbf if complete delta and loaded
   source/configuration identity prove unchanged functional inputs.
   Preserve their original tested identity. Run diff/identity/import
   checks. Repeat executable tests or replay only if the equivalence
   check fails or an applicable exact-head verification requirement
   independently mandates it.

2. Retain the Mandarin replay result as **partial selector evidence /
   insufficient dialogue coverage**. It passed preservation and
   evidence-admission controls; it did not pass useful whole-clip
   coverage. Do not relabel the 16.17% result as successful live
   qualification. The changed acquisition path has not been exercised.
   A selector safety regression, identity mismatch, failed CPU gate or
   common resource-integrity problem remains blocking. Missing historic
   acoustic coverage is not a reason to prevent acquiring the evidence
   necessary to test that acquisition path.

3. Execute the French normal continuation and Mandarin retained
   failed-recording continuation as independent qualification arms in one
   handback, sequentially under the existing GPU lifecycle. A semantic
   failure in one language does not block the other. The existing
   French command above and previous live handback's Mandarin options
   govern. Use the same authenticated media/recording/manifest/ledger
   bytes and fresh private outputs with no old source-quality or
   downstream checkpoints. Do not inject replay observations. For
   Mandarin, retain the actual whole-primary CTC failure input rather
   than rerunning primary recognition first.

4. For Mandarin, verify the dry structural plan and actual independent
   admission against current policy and task/target/crop identities.
   The dry plan excludes recording-context nominations and is not an
   assertion that the fresh schedule must be identical. Report actual
   native/independent crops, union of acquired target coverage, costs,
   and all deferrals including the clip tail. Diagnose residual omissions
   by separating unacquired material, recognition/segmentation
   disagreement, ambiguous source association, selector conflict,
   unresolved timing and rendering/review failures. Compare original
   source-unit coverage and actual rendered dialogue, distributed
   throughout the clip; do not claim adequate coverage from a handful of
   recovered clauses or an empty output. Report actual missing narrative
   and meaningful residuals. No numerical pass threshold is inferred
   from the prior 16.17% result. Useful output remains a substantive
   acceptance criterion requiring Central review.

5. If the retained failed-recording Mandarin continuation yields
   meaningfully usable dialogue and completes required review, execute
   one fresh primary-to-render run on the same clip. If the continuation
   is inadequate, preserve evidence and skip only that dependent
   fresh-primary run. Complete the independent French arm regardless,
   subject to common identity, CPU and resource safety gates. Do not
   select another clip or start a broad benchmark.

6. Required English review must actually complete. Capture actual
   malformed native proposals and bounded keyed-object correction
   receipts when encountered. Transport failures and rejected alternate
   replies remain failures. Empty recognized-dialogue outputs must fail
   explicitly; legitimate no-dialogue recordings may have no cues.
   Verify fresh-process continuation and offline rerender for each
   successful live output. Record actual stage costs, resource peaks and
   restoration, and all unresolved limitations. Do not deploy, merge,
   create a PR or close the issue. Return one private authenticated
   evidence archive covering both independent arms.
