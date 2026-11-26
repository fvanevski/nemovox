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


## Ordered event recovery qualification

This section supersedes the previous live-failure remediation qualification
instructions. The fresh live candidate fdf6e655fce882cc1ae39d63c1e1781f94cfda0b
retired all nine French displaced fragments, but failed exposition recovery.
Mandarin acquired independent crops covering 106.5/119 seconds, costing
236.5/240 audio seconds, yet localized only 84/402 original source units.
Both arms completed required review and rendered; Mandarin review incorrectly
preserved a calendar-month MT error despite a correct comparator-approved
relative-duration proposal. Neither arm qualified.

The successor replaces exact-subspan failed-CTC selection with ordered
observed lexical-anchor correspondence, retains timing-only wording
alternatives with exact source-unit lineage, and separately localizes
corroborated events with ambiguous primary origins. The French prefix
correction uses already-acquired local native observations. Short fully
failed recordings can consolidate failed-core windows to fit the existing
budget, including the tail. No model or resource substitution is authorized.

1. Authenticate the exact Central candidate commit/tree, origin and the
   immutable inputs. Verify candidate imports and diff checks. Production
   code changed: run focused recovery, planner, source quality,
   targeted occurrence, scene, uncertainty, translation review and replay
   tests, then the complete non-GPU CPU suite and applicable Studio/API
   checks. Unchanged frontend-only evidence may be reused only after
   verified functional equivalence. Preserve actual execution identities.
2. Replay the authenticated original comparison and latest live evidence
   through the successor production selectors. The preceding live archive
   supplies the immutable Mandarin primary recording and ledger, while
   the latest live archive supplies fresh observations from both arms.
   Use the following commands from the candidate root:

       PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m scripts.replay_source_robustness --archive "$COMPARISON_ARCHIVE" --sha256 "$COMPARISON_SHA256" --output "$PRIVATE_ROOT/original-comparison-replay.json"

       PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m scripts.replay_live_source_quality --intake "$INTAKE_ARCHIVE" --intake-sha256 "$INTAKE_SHA256" --archive "$PRIOR_LIVE_ARCHIVE" --sha256 "$PRIOR_LIVE_SHA256" --observations-archive "$LATEST_LIVE_ARCHIVE" --observations-sha256 "$LATEST_LIVE_SHA256" --output "$PRIVATE_ROOT/latest-live-replay.json"

   Check actual source lineage, event bounds, rejected alternatives,
   genuine-repeat controls and all six original scene decisions. French:
   preserve the nine justified retirements and recover the corroborated
   exposition without claiming exact entity spelling or speaker identity.
   Mandarin: require exact original-unit partition, no incompatible
   event localization or invented word times, and material improvement
   over 84/402 localized units on the same sealed observations. Central's
   isolated helper experiment recovered 241 units and two independently
   localized events with ambiguous primary origins; this is an exploratory
   reference, not a mandatory output count or full-candidate receipt.
   Do not count unbound events toward original source-unit recovery.
   Record wording disagreement, residual omissions and actual temporal
   coverage. No arbitrary whole-clip usability threshold is inferred
   from retained evidence. An observed selector safety regression,
   identity failure or CPU failure blocks affected inference. Failure
   to improve on the same evidence is a substantive selector failure,
   not an acquisition-coverage limitation.
3. Qualify French and Mandarin as independently admissible live arms,
   sequentially under the existing GPU lifecycle. A language-specific
   replay or live failure blocks only its affected/dependent work;
   common CPU, identity, resource-integrity or lifecycle failures block
   both. Use the same authenticated French original recording and media
   and the retained Mandarin whole-primary CTC-failure recording and
   119-second clip; see the existing French CLI command above and the
   prior Mandarin handback for exact job options. Use fresh output
   roots with no source-quality/downstream checkpoints or replay
   injection. Do not rerun full French primary ASR.
4. For Mandarin, verify the actual planned/admitted windows and crops
   (including the 104.5–119 s region), union coverage and actual
   acquisition costs against existing 240-second independent and
   configured native limits. The compact full-coverage plan is a dry
   expectation, not evidence of acquisition or recovered speech.
   Compare old and new localized original units and unbound events
   separately. Audit source wording, origin ambiguity, omitted material
   and meaningful English subtitle coverage throughout the clip. Verify
   timing-only recovery does not freeze the erroneous calendar-month
   MT when both model families support the relative-duration phrase;
   retain material-source, entity and number guards for unsupported
   changes. Check actual translation/review dispositions, not just the
   final text. A handful of timed clauses is not a useful subtitle result.
5. Only after meaningfully usable retained failed-recording Mandarin
   continuation and required review, run one fresh primary-to-render
   job on the same clip. If inadequate, skip that dependent job and
   finish the independent French arm. No new model downloads, broad
   benchmarks, alternative clips, budget increases or threshold changes.
6. For successful outputs, verify fresh-process continuation and offline
   rerender, actual SRT/ASS content, resource restoration and unchanged
   original job, owner checkout, Studio and unrelated GPU workload.
   Report actual phase times, resource peaks, failure modes and unresolved
   limitations. Complete required English review; capture malformed
   proposals and bounded recovery receipts when encountered.
   Do not deploy, merge, create a PR or close the issue.

Return one private authenticated review archive containing test results,
replay reports, normal acquisition/output receipts, meaningful output and
residual-omission comparisons, resource costs/restoration, exact identity
and input authentication, and canonical GHDEV-10 command-shape friction.
Reference prior archives instead of copying them wholesale. Exclude
credentials, media, model weights and bulk WAV files. Central owns
substantive implementation corrections and the acceptance decision.
