# Nemovox Issue #14 / PR #15 — transcript-first source-evidence integration and acoustic adjudication

## Authority and execution role

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE=14
EXISTING_PR=15
BRANCH=issue/14-longform-timing-boundaries
BASE_MAIN=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CENTRAL_SOURCE_SHA=84713a13fd521b3d99b91f09863aff95153c8fff
CENTRAL_SOURCE_TREE=6580e4fc96372df4dc4e2ee217d8034c4770c187
GOVERNING_DESIGN=staging/issue14/ARCHITECTURE_REDIRECTION.md
SLICE_DESIGN=staging/issue14/SOURCE_EVENT_SLICE.md
PROVEN_ACOUSTIC_EVIDENCE=36fbf98021094b58efac5de76c7a3432f0a3a7f8:staging/issue14/evidence/20261123T000342Z/
PRIOR_FULL_RUN_EVIDENCE=a1812212f6e5c7a82bcd0a27769130fda420a3d3:staging/issue14/evidence/20261122T225739Z/
PRIOR_BASELINE=5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z/
TARGET=qualify one coherent Central-authored source-evidence architecture slice, not certify the entire Issue #14 fidelity objective
NO_NEW_PR=yes; NO_MERGE=yes; KEEP_DRAFT=yes; NO_CODEX_REQUEST=yes; NO_PRODUCTION_ACTIVATION=yes

Use a genuinely fresh OpenCode Build task launched from the existing Nemovox checkout (/workspace/nvidia-transcribe is the host convention). You own safe read/fetch/worktree preparation in a fresh private source worktree; do not reset, rebase, stash, clean, overwrite or silently synchronize the operator's existing checkout. Respect the nearest repository AGENTS.md. If that directory is unavailable, locate the authenticated local Nemovox checkout and record the substitution. Do not change /workspace/pyannote or any live model/service/config.

## Objective (one consolidated campaign)

Integrate and qualify the Central-authored immutable evidence producer and CPU offline replay:

- `src/nvidia_transcribe/source_events.py`
- `src/nvidia_transcribe/source_event_replay.py`
- `src/nvidia_transcribe/recording.py` (sidecar/manifest integration)
- `tests/test_issue14_source_evidence.py`
- `tests/test_issue14_source_event_replay.py`

The new `source-evidence.v1.json` must capture raw-sha-linked ASR observations and all emitted lexical units before current boundary deletion, with per-request speaker-tag hypotheses, CTC-status separation, unverified sequence correspondences, and native diarization activity hypotheses. It is deliberately shadow-only; normal `nemotron-recording-contract/v1`, checkpoint behavior, Riva/Qwen translation and ASS must remain byte/semantic compatible when identical inputs and configs are used. Never infer source recall from fewer fragments, accepted acoustic timing from monotonic CTC, speech from model diarization, or a unique acoustic occurrence from similar text.

Use *one* Build campaign: exact source validation, CPU + existing cache replay, bounded source-event/audio correspondence adjudication, one sanitized evidence publication. Do not create one handoff or PR per diagnosed adapter/fixture discrepancy.

## Phase 0. Identity and source admission

1. Authenticate the GitHub repo/PR/issue via native authenticated gh or equivalent permitted GitHub read; require issue OPEN and PR #15 OPEN/DRAFT, with branch `issue/14-longform-timing-boundaries`. Establish live `main` SHA and branch SHA. Abort as STALE if current branch head no longer includes the Central code unchanged and an unexpected source movement is not explicitly reconciled by Central.
2. From the existing checkout, `git fetch origin refs/heads/issue/14-longform-timing-boundaries`; prove the fetched ref equals the remote PR head (full 40-char SHA); prove that source commit `84713a13fd521b3d99b91f09863aff95153c8fff` is its ancestor and that `git diff 84713a13fd521b3d99b91f09863aff95153c8fff <fetched-head> -- src/nvidia_transcribe tests` is initially empty. Prove commit/tree and remote identities using Git, not branch-name assumptions. Use `git worktree add --detach <fresh-collision-safe-worktree> <fetched-head>` under a private working root; never operate on the operator's dirty checkout.
3. Read current `AGENTS.md`, this handoff, both design documents and the published acoustic evidence summary. Use current source as authority for command flags; report any interface mismatch rather than guessing. The canonical environment is /workspace/nvidia-transcribe/.venv when authenticated; do not install packages or borrow /workspace/pyannote/.venv.

## Phase 1. Deterministic qualification and bounded repair

Execute, recording complete commands, workdir, interpreter identity, exit codes, stdout/stderr in private logs:

```bash
# Run from the isolated fetched-head source worktree. Use the authenticated
# Nemovox venv Python (normally /workspace/nvidia-transcribe/.venv/bin/python).
PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu" tests/test_issue14_source_evidence.py tests/test_issue14_source_event_replay.py
PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
git diff --check
```

Do not merely quote historical counts. Inspect actual results. The focused integration regression must prove `source-evidence.v1.json` is emitted, manifest SHA matches exact bytes, stable unit IDs survive identical-source resume and v1 recording contract remains unchanged. Exact evidence + test-proven correction takes precedence over a stale snapshot.

**Narrow integration authority:** You may fix source-contract/serialization, actual cache manifest field differences, deterministic import/interface errors, safe fixtures and directly implicated minor wiring defects in the five listed files, plus the smallest adjacent test necessary to establish unchanged behavior. First inspect the complete diff, then apply a bounded correction on the *same existing issue branch*, using a new non-force source commit only when needed and only after proving the remote head remains unchanged. Rerun relevant and full CPU gates on the new source head. No independent design of ASR consensus, new VAD/aligner algorithms, canonical source-event acceptance, source text surgery, speaker inference, translation, renderer, GPU lifecycle or changed test oracles to force green. If the required fix changes architecture or a major semantic decision, return a concrete BLOCKED finding with a proposed regression and evidence. Keep repairs within the same handoff; do not stop at the first safe within-scope environment/fixture issue.

## Phase 2. Offline authenticated cached-data replay (no ASR/model restart)

Use published `source-identity.json` and host manifests to locate the actual original 16-chunk and pause-aware 35-chunk private ASR roots, using immutable raw SHA, media fingerprint, source offsets and plan rather than guessing a directory from file names. Historic Studio job reference: `/workspace/nvidia-transcribe/outputs/studio/jobs/980af31f014245c4b200e0af8b33ebca/`. It is a historical reference, NOT permission to modify its files or to assert its original source SHA. Do not touch original media, jobs, output or production services. If evidence roots are absent, mark missing, don't fabricate.

For each authentic full cached job, run source-event replay in a fresh chmod-700 private output root using the actually authenticated Python and exact worktree `PYTHONPATH=src`:

```bash
PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.source_event_replay --asr-dir "<AUTHENTICATED_CACHED_ASR_DIR>" --output "<FRESH_PRIVATE_OUTPUT>/source-evidence.json"
```

For the historical full35 job containing `c28 finish=length`, use `--forensic-incomplete` on the same command *with a different new output filename* only if ordinary replay rejects that completion. It must explicitly list rejected chunk indices/digests, preserve a source-coverage gap and never silently include their words as completed source. If other legacy cache shape incompatibilities are encountered, inspect real manifest/raw shape and make only the narrow authorized adapter fix, preserving fail-closed SHA/source proof.

Read-only compare source observations against legacy `recording.contract.json` and the baseline ledger: observations per real inference window, complete vs incomplete, displaced CTC/rejected timing, stable IDs, exact match candidates, missing speech not present in any source observation, and genuine repeats/overlaps that must survive. Compare original 16 and 35 conditions by original media time, not chunk number or `utt-XXXX`. This is a correspondence *candidate* evaluation: do not delete words or automatically accept a final source turn. Explicitly identify that the offline replay cannot reproduce prior Qwen/subinterval timing and TitaNet global speaker reconciliation.

Minimum priority comparison regions: c28 1099–1156s; W04a/b/c 1230–1274s; W05a/S05 1338–1379.901s; W02b 209–232s; W03a 985–1002s. Keep controls C1 686–700s, C2 1022–1036s, theme 25–40s, SFX 1200–1216s, H1 120–140s, H2 360–380s, H3 900–920s and H4 920–940s. If private EP166 source is authenticated, include its two already-labeled Mandarin windows (~96–114s and 518–536s). Reuse independently heard labels from the prior packet, clearly distinguishing host-perceived acoustic observations from model output and Central-independent verification.

## Phase 3. Limited acoustic/model-method readiness and architectural decision evidence

Inspect read-only whether Silero VAD, Qwen3 ForcedAligner and pyannote Community-1 are *already locally installed/licensed and compatible with the owned runtime*. Record exact package/checkpoint identities and availability without downloading weights, changing dependencies, evicting other GPU services or assuming an old model path is still valid.

Where an **already approved** read-only CPU acoustic activity detector can be run directly on short privately retained excerpts without altering a service, optionally evaluate its false speech-positive/negative triggers against known H2 omitted words, c28 voices versus SFX, theme/song and clean controls. This does not certify speech content. A resident GPU forced aligner or diarization pilot is optional only if an existing Nemovox-owned admission/lifecycle method is positively authenticated and can run without shared-service disruption; otherwise mark not executed and return the needed setup/decision, not a speculative score. No 23-minute ASR, no new model selection/promotion, no new code for a detector/aligner and no arbitrary model downloads. Use a few existing labeled clips, not a new exploratory benchmark campaign.

Provide a **go/no-go design comparison** for future Central acceptance: (a) observed unit/event coverage including missing-source gaps, (b) lexical candidate match false-positive risk for actual repetition/overlap, (c) CTC chronology contradictions, (d) which missing acoustic/speaker/forced-alignment signals are necessary before promoting candidates into accepted `SourceTurn` and `WordAlignment`, and (e) estimated resource/runtime cost if measured. Do not call shadow evidence a successful transcript-first replacement.

## Publication and handback

One sanitized packet under `staging/issue14/evidence/<new-UTC-run-id>/`, preferably with these members:

- `README.md`: source-bound scope, source commit and real limitations, host environment summary, tested commands + exit status.
- `source-identity.json`: actual PR/branch/base SHA and tree, runtime Python identity, media fingerprint and prior raw manifest digest identities, no secret or private absolute path disclosure.
- `validation/cpu-gates.json`: focused/full CPU test counts, exact exit status, diff cleanliness and any repair commit provenance.
- `validation/replay-comparison.json`: original16/full35 forensic status, raw digests, source unit/turn candidate/correspondence counts by time-region, rejected chunk gaps and prior metrics with SAME denominator labels.
- `validation/case-adjudication.jsonl`: W/C/S, H1–H4 and optional Mandarin event-by-event decisions, independent acoustic vs ASR/CTC origin, ambiguities and known omissions.
- `review/architecture-readiness.md`: evidence-backed decision about the source-event layer and what must be implemented centrally next; no model upgrade recommendation unsupported by trials.
- `COMMAND_SHAPE_FRICTION.txt`: schema below, including recovered issues.
- `manifest.json`: SHA-256+byte length for every public member except itself and the exact source-evidence scope, readable through GitHub without a separate archive.

Do not put source WAVs/media, full episode transcript, raw model responses, actual names/path fragments from proprietary media, credentials, model weights, large logs or complete private annotation text on GitHub. Private detailed ledger and raw hashes live in chmod-700 scratch; publish bounded source identity, numeric metadata and short sanitized descriptors only. Do not publish the complete raw `source-evidence.v1.json` if it includes episode text. Keep private audit data for possible secure attachment only when indispensable, and flag its availability.

If a qualifying source adapter repair was needed, record the new exact source SHA/tree, requalify that exact source, and publish the evidence against that new SHA; never reuse previous PASS unqualified. Publish only one evidence commit after reviewing complete public-safe diff, only on the existing PR branch, only if GitHub head still equals the admitted current branch and with a non-force CAS/fast-forward route. On race, STOP STALE and return an attachable private evidence packet rather than reconcile by rebase/force push. Never create another PR or modify Issue #9.

### PASS / PARTIAL / BLOCKED

`PASS`: focused/full CPU gates pass at exact source, source-evidence integration and old-v1 compatibility verified, authenticated available cached replay evaluated and privacy-safe manifest verified. No architecture-completion claim.
`PARTIAL`: deterministic code passes but private data/model-specific evidence is missing, inaccessible or requires a new non-authorized resource, or one defect remains unresolvable without Central substantive design.
`BLOCKED`: source identity/material provenance mismatch, invalid SHA/incomplete artifact, failing non-repairable regression or protected host/mutation boundary. Report actual failure and stop the affected work rather than bypassing it.

Return direct conversation handback with:
```text
ISSUE14_SOURCE_EVENT_HANDOFF:
  RESULT=PASS|PARTIAL|BLOCKED
  REPO=... PR=15 ISSUE=14
  ADMITTED_HEAD_SHA=<full>
  FINAL_SOURCE_SHA=<full>
  SOURCE_TREE=<full>
  CURRENT_PR_HEAD_SHA=<full>
  MAIN_SHA=<full>
  TESTS_FOCUSED=<exact command, counts, exit>
  TESTS_NON_GPU=<exact command, counts, exit>
  INTEGRATION_SIDECAR=<manifest hash, path, coverage semantics>
  ORIGINAL16_REPLAY=<source identities, result and limits>
  FULL35_REPLAY=<source identities, rejected chunks and result>
  ACOUSTIC_CORRESPONDENCE=<bounded case decisions>
  MANDARIN_CONTROLS=<verified or unavailable>
  OPTIONAL_MODEL_READINESS=<identities, read-only measurement or NOT_RUN>
  SOURCE_REMEDIATION=<none|exact complete diff and commit>
  PRODUCTION_OR_ORIGINAL_INPUT_MUTATION=no|details
  SOURCE_V1_COMPATIBILITY=PASS|FAIL|UNVERIFIED
  EVIDENCE_GITHUB_ROOT=<full SHA:path|none>
  EVIDENCE_COMPLETE=<yes|no and missing items>
  EVIDENCE_MANIFEST_SHA256=<digest|none>
  PRIVATE_MATERIAL_WITHHELD_FROM_GITHUB=yes|no
  NEW_ARCHITECTURAL_BLOCKERS=<material concrete defects only>
  NEXT_CENTRAL_IMPLEMENTATION=<one concise cohesive objective>
```

### Canonical command-shape friction block (always return separately)

```text
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events:
    - scenario=<bounded scenario>
      requested_shape=<sanitized requested invocation>
      effective_shape=<actual execution or rejection|same-as-requested|unknown>
      rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
      event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
      execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
      correction_or_annotation=<bounded decisive fact>
      replacement_shape=<corrected shape|none>
      recipe_reference=<GHDEV-10 key|none>
      resource_reference=<source used|none>
      contract_conflict=<yes|no|uncertain>
      durable_gap_candidate=<yes|no|uncertain>
  omitted_sensitive_values=<yes|no>
```
When none occurred, `encountered=no`, `events=[]`, `omitted_sensitive_values=no`. Command friction is distinct from the substantive PASS/FAIL.

**Stop boundary:** complete the integrated campaign once acceptance/evidence is sufficient; do not start a new corrective search for downstream MT, ASS lane rendering, dependency upgrades or unrelated local service issues.
