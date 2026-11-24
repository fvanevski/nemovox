# Issue #14 — current-workflow qualification

Status: Central-authored consolidation candidate, not accepted or merge-ready. Existing PR #15 remains draft. This is the sole active Issue #14 qualification campaign. Do not resurrect prior staging trees, offline revision commands, deleted fixtures, or legacy output hashes.

## Scope and authority

Use the exact branch and full head supplied by Central in the kickoff; authenticate the remote ref, check the PR is draft, and verify the checkout. Use an isolated worktree if the primary checkout contains changes. Preserve any user edits. Read AGENTS.md. No merge, ready transition, new PR, issue closure, production deployment, service redesign, or unrelated mutation. Local Build may resolve ordinary host wiring, invocation, permissions, and narrow deterministic mechanical integration problems within scope. Semantic source-selection, speaker, translation, lifecycle, or test-invariant changes return to Central with a minimal repro, full diff and evidence, not an improvised algorithm. Never weaken tests or bypass resource controls.

## CPU-only verification (first)

Continuation after the first CPU handback: Central corrected the source-event
timing correspondence and missing-contract rerender defects on the same branch.
The previous evidence remains a failed qualification at its original source SHA.
Use `rtk proxy` where the local harness requires it; preserve native exit status
and unfiltered failure evidence. Before other suites, run:

- PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_source_quality.py tests/test_offline_rerender.py
- PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python staging/current-qualification/evidence/20261124T043313Z/reproduce.py

The published reproduction must exit 0 with zero managed ASR acquisitions for
the missing-contract CLI probe and an explicit unresolved decision for the
displaced dialogue. The new tests additionally require ordinary timing jitter
to remain accepted, ambiguity to reach the quality report, distinct repeated
turns to survive, and valid SRT/ASS rerender to remain inference-free.
If these and the remaining CPU checks pass, proceed directly to the existing
GPU phase in this handoff; no separate intermediate handback or approval.
Do not rerun or reuse the failed application's historical source qualification.

From the candidate root, record full commit/tree identity, dirty status, Python import location, installed runtime identity and effective nonsecret config fingerprints. Use existing host environments, not a replacement dependency stack:
- PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests
- PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_source_quality.py tests/test_issue14_semantic_review.py tests/test_translation_review.py tests/test_chunking_and_resume.py tests/test_resource_integration.py
- PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
- PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
- In web: npm test; npm run build using the existing dependency lock/environment. Report missing dependencies honestly.
- git diff --check; check live consumers for references to removed modules/commands/configs, including scripts, docs, tests and imports. Historical external Git history is not an active consumer.
- Verify parser defaults Riva and balanced review (explicit alternative primary translator defaults to off); current-media Studio jobs forward original media and existing recording; imported contracts without media skip acoustic review; current-version rerender is inference-free. Verify no stale checkpoint accepted across the source pipeline identity change.
- Do not enter GPU qualification while a material CPU/consumer failure remains. Narrow mechanical fixes must preserve the functional invariant, be reviewed as a complete diff, and be separately identified. If source changes, final evidence must bind the actual final code head.

## Live-quality remediation qualification

The previous correction campaign passed CPU and live execution/lifecycle but failed quality.
Source changes now support split/join same-voice correspondence, a bounded third distinct
context for competing wording and short insertions, and coarse anchored timing of noisy
untimed source text. They do not certify disputed speech. The English reviewer adds a
bounded comparative Messages judgment for proposed changes. Short real overlap and
media-end scheduling are corrected. Optional TitaNet is off by default; explicit selection
remains available, no download or fallback.

Run the original CPU reproduction and the additional independent renderer reproduction:
- PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python staging/current-qualification/evidence/20261124T045622Z/reproduce_overlap.py
- PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q tests/test_source_quality.py tests/test_subtitles.py tests/test_translation_review.py tests/test_offline_rerender.py

Check CLI and Studio defaults and explicit speaker embeddings/audio stream forwarding.
Use the supported single-command CLI, explicit --speaker-embeddings none (or default),
--audio-stream 0, and --chunk-policy pause_aware for the French episode. No TitaNet
acquisition necessary. Do not repeat the two-step workaround. Do not rerun the old
application. New CPU checks must pass before GPU work; if so proceed without handback.

First use bounded fresh source clips covering the previously observed French named-question
and Mandarin gap, plus a normal control. Examine actual original audio with an admitted
listening capability or request a brief operator listening judgment if necessary; if
unavailable, report auditory verification as unverified, not PASS. The previous
same-model agreement alone is not ground truth. Do not hardcode episode text or accept
a phrase because it matches the old hypothesis. If the bounded source/translation checks
still reveal a material product defect, return the decisive evidence instead of consuming
another full-episode run. If viable, proceed with one fresh French episode and one
independent Studio control using the current actual head. Measure escalation coverage
and cost (up to 18 windows, duration-scaled, and 8 extra acoustic contexts). Report
the effect of comparative review on grammar, idioms, preserved meaning and model calls.
A residual uncertain word is not itself a failed quality gate if dialogue meaning and
useful timing are preserved; a material omission or mistranslation remains a blocker.

## GPU-intensive qualification (only after CPU gates)

One fresh real-media French episode and one small independent different-language control (prefer a Mandarin scene/episode with dialogue, repeated words or overlapping speakers). Use the original local media identified by the operator's existing host context or existing source path metadata; do not import historical outputs, hand-reviewed dossiers, ASR caches, revised contracts, or golden cue hashes. Use fresh collision-free output roots under the project. No output to the library. No model downloads. Do not stop or evict external GPU services. Inspect nvidia-smi and actual model-resource/owned-unit configuration; run only through the existing lease/lifecycle admission and sequential ownership. If the local harness itself blocks required inference, prepare evidence and exit; return one exact operator continuation command using the existing lifecycle, not a bypass.

Exercise normal public entrypoints: use CLI generate-subtitles with original media, --source-quality on --translator riva --translation-review balanced --refinement off --fresh, and the appropriate source language or verified auto-language. Use the existing model-resource configuration and current local checkpoints. Record selected audio stream, source SHA256, chunk policy, model/resource versions and fingerprints, owned service start/release evidence, peak VRAM, wall-time breakdown and request counts. Prefer pause_aware for the French challenge if supported. Exercise the same source-quality path through a new Studio job on the control or a bounded independent clip, with SRT/ASS and no live export. Do not run unnecessary duplicate full-episode jobs just for evidence.

Inspect original ASR, source-quality.json, canonical-source.contract.json, translated.contract.json, and actual rendered subtitles. Examine consequential meaning, polarity, speaker turns, temporal ordering, readable English, real repeated speech and overlapping dialogue, rather than a literal WER target. Listen to bounded original-media windows where practicable and distinguish model evidence from operator auditory judgments. Sample current detected candidates, accepted changes, unresolved cases, deferrals, and plausible false positives. Report whether a source-quality case was repaired, correctly retained, or unresolved; never treat same-ASR corroboration as human or independent-model verification. Current candidate supports insertion/extension/timing recovery with positive corroboration; it deliberately leaves ambiguous displaced/repeated alternatives unresolved. A consequential unresolved issue is not a clean quality pass: provide exact original and selected time/text hypotheses, observation IDs, candidate case, and sufficient bounded evidence for Central to design the fix. Report whether the case/window budgets are effective and runtime feasible on one RTX 5090. Do not silently increase budgets or weaken evidence gates.

Prove current canonical source is what Riva receives, original MT is retained, English postedit is bounded and does not introduce meaning/polarity regressions, and real cues survive to SRT/ASS with useful event timing. Confirm source-quality stage metadata/event reaches Studio and the prior ASR is unmodified. Demonstrate inference-free rerender from the current translated contract; no repeat ASR required. Check final owned service/lifecycle restoration and no unrelated service effects.

## Handback (single consolidated packet)

Report PASS, FAIL or BLOCKED separately for CPU, CLI, Studio, source quality, translation/review, renderer, runtime/resource lifecycle. Preserve native exits and counts, commands, source/config/runtime identity, full changed-file inventory and diff, representative bounded case samples and failures, timing/VRAM/coverage and explicit uncertainty. Include manifest/checksums. Publish only sanitized compact reports and synthetic or bounded nonprivate examples on the same branch under staging/current-qualification/evidence/ (or attach a review-ready private tar.gz when required). Do not publish original media, private subtitle/transcript corpora, credentials, model weights, or raw sensitive logs. Keep large/private raw evidence host-local, return exact paths/digests and secure delivery details if needed. Return full final SHA/tree and evidence-carrier SHA separately. Do not claim host PASS if required evidence missing. No historical replay or artificial equality to old output. Stop at one consolidated handback.

COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events:
    - scenario=<short repository/task scenario key>
      requested_shape=<sanitized bounded requested invocation>
      effective_shape=<effective command|same-as-requested|unknown>
      rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
      event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
      execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
      correction_or_annotation=<bounded decisive correction/error>
      replacement_shape=<successful corrected shape|none>
      recipe_reference=<GHDEV-10 key|none>
      resource_reference=<resource|none>
      contract_conflict=<yes|no|uncertain>
      durable_gap_candidate=<yes|no|uncertain>
  omitted_sensitive_values=<yes|no>

If no friction: encountered=no, events=[], omitted_sensitive_values=no.
