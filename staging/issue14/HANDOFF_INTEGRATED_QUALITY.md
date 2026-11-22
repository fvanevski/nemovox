# Issue #14 / Draft PR #15: integrated adaptive ASR, speaker and review qualification

## Boundaries and immutable source

This is ONE substantial host qualification campaign. Central owns and has committed the substantive implementation. Local OpenCode Build owns isolated execution, bounded diagnostic script adaptation, model lifecycle admission and return of a self-contained evidence packet. No redesign of source, test semantics or model strategy is delegated.

Repository: fvanevski/nemovox
Issue: #14 OPEN
Draft PR: #15 OPEN/DRAFT
Branch: issue/14-longform-timing-boundaries
EXACT_TESTED_SOURCE_SHA: 4c6c7ac5719fc70b8db3a671e2970409a9a11e29
EXACT_TESTED_TREE: a9b08e87ea5d5ef3e57e41381b56b0388565e126
BASE_MAIN_SHA: 8f6c03cbc70e84bf1970ba364207a02c9fe22b44
SOURCE_EVIDENCE: d50a81f6f74b70fef52bbd7b9f39b470019a4e8e:staging/issue14/evidence/20261122T221348Z/
BASELINE_EVIDENCE: 5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z/
BOUNDARY_REPLAY_EVIDENCE: 99dce3f8bfa1a93670b28473209dbf8bd924a54e:staging/issue14/evidence/20261122T215418Z/
HARNESS: OpenCode build, CPU first, managed RTX 5090 only after CPU and preflight
EVIDENCE_ROOT: staging/issue14/evidence/<new UTC run>/
PRODUCTION_CHECKOUT: /workspace/nvidia-transcribe (READ ONLY)
ORIGINAL_JOB: /workspace/nvidia-transcribe/outputs/studio/jobs/980af31f014245c4b200e0af8b33ebca/ (READ ONLY)
ORIGINAL_MEDIA: /opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm (READ ONLY)
PRIVATE_ACOUSTIC_BASELINE: /workspace/nemovox-mcog-private (READ ONLY)
REMOTE_SOURCE_WRITES: forbidden
LOCAL_SOURCE_OR_TEST_EDITS: forbidden (except private disposable diagnostic harness code outside repository)
REMOTE_EVIDENCE_WRITE: one sanitized evidence-only commit on existing PR branch
NEW_PR, CODEX_REQUEST, READY_TRANSITION, MERGE, DEPLOY, ISSUE_CLOSURE: forbidden
UNRELATED_GPU_SERVICE_EVICTION, MODEL_WEIGHT_DOWNLOADS, GLOBAL_CONFIG_CHANGES: forbidden

Original historical job execution Git SHA is UNVERIFIED. Exact source requalifies a candidate, not the historic recording.

## Central-authored production changes to verify

A) New speech_chunks.py: FFmpeg silencedetect on the SELECTED audio stream proposes mid-pause boundaries. Core partition is exact and gap-free. Policy pause_aware is an EXPLICIT opt-in for this candidate, favoring ~39s cores up to 46s with context capped at 4s and bounded tail slack. The default remains fixed to preserve older 90s jobs and legacy checkpoint signatures until qualification and an explicit rollout decision. Use --chunk-policy pause_aware in all candidate pilot/full-run commands. Forced cuts without eligible pauses are EXPLICITLY logged. Detector failures are recorded, never represented as confirmed pauses.

B) recording.py / subtitle_pipeline.py / cli.py: policy wired into both CLI and Studio-launched generation, exact chunk plan plus detector status are included in checkpoint config signatures, and plan metadata is persisted. New plan may NOT resume old 90s raw chunk responses under the same manifest signature. Do not claim a silence boundary guarantees true word/speaker boundary. The defaults are experimental until this host report qualifies them.

C) speakers.py: a single TitaNet candidate barely above configured threshold is no longer unconditionally merged (0.6536 and 0.6637 examples). Record explicit ambiguity instead of conflating distant characters. Keep confident positive merges and same-chunk cannot-links.

D) translation_review.py: if source French AND independent primary Riva English support negation, reject a Qwen candidate that loses lexical negation, preserving original English MT and recording review-rejection reason. Equivalent negative rephrasings and positive corrections must remain eligible. This is a narrow polarity guard, not a semantic universal verifier or replacement for the source-time repair.

E) Studio API, web App and config: advanced ASR chunk strategy with explicit fixed default and pause_aware opt-in; the real JobRunner passes --chunk-policy to process-media. Frontend tests plus Studio schema regression enforce compatibility. Fixed policy remains the default in Studio and CLI until explicit rollout.
F) CPU tests/test_issue14_pause_chunks.py, tests/test_issue14_speaker_review_guards.py and Studio tests: model-free unit tests. Existing W/C/S boundary and Mandarin controls must remain intact. Source/MT and raw CTC provenance stay inspectable. Renderer Issue #9 unchanged.

## Phase A: authenticate immutable source and run CPU gates

1. Authenticate fresh GitHub main, PR head, draft state, Issue #14; require main BASE_MAIN_SHA and branch descendant of EXACT_TESTED_SOURCE_SHA with no substantive unapproved edits. Create a NEW collision-safe isolated worktree from exact SHA (do NOT reset/clean/stash production). Verify commit tree, AGENTS.md, changed-file inventory, source import paths, and effective CLI --chunk-policy parse.
2. Verify original video and job hashes against BASELINE_EVIDENCE; check 16 raw chunk identities, production git and Studio PID/InvocationID, active Studio jobs, Nemovox owned model units, other GPU usage and lease ownership. Never modify originals, venvs, model configs or running jobs.
3. Run python -m compileall -q src tests scripts, focused pytest of new pause/speaker-review tests plus prior timing, chunking, review, speaker, recording, Studio smoke; full .venv pytest -q -m 'not gpu'; .venv-studio pytest for Studio APIs; web npm test and npm run build (TypeScript and Vite); record exact counts, warnings, skips, exits and source imports. Test with PYTHONPATH bound to isolated candidate src. A failure from production semantics is a STOP and exact reproducer, not license to modify tests or source.
4. Verify fixed-policy DEFAULT backward compatibility and resume: identical fixed config does not reuse a pause-aware signature, and two runs with the same adaptive plan may resume matching completed chunks. Confirm the manifest includes actual plan metadata and forced-boundary flags. Verify Studio options accept pause_aware, reject unsupported values, and generate the exact --chunk-policy argument. Default Studio and CLI remain fixed. Never delete or overwrite baseline outputs to test resume; use disposable fixture outputs.
5. Source-only boundary reconciliation evidence from previous host replay can be REUSED for unchanged boundary_evidence.py, but changed speaker merge and chunk planning must be freshly tested. Do NOT call reused evidence a new inference run.

## Phase B: inspect full-episode pause plan BEFORE GPU

1. Run silencedetect + plan_pause_aware_chunks read-only on verified 1379.901s episode. Report FFmpeg version, audio stream selection, silence thresholds, actual interval count, plan core/infer ranges, longest inference, count of detected-pause vs forced boundaries, selected core durations and plan signature.
2. Correlate every chosen cut within the previous acoustic W01..W05/C1/C2 sample ranges and their source word timestamps. For problematic boundaries, inspect limited PRIVATE source audio to distinguish natural silence, SFX-only gap, mid-utterance/phrase and unknown. Do not promote a signal-only pause to proven speech boundary.
3. Critical gate: W01a long song/verse, W02a surname/silence, W03b speaker changes, W04a question/reply and C1/C2 must not be hard-cut without overlap coverage. Check both sides' infer windows. If a cut clearly destroys source coverage before running a model, mark BLOCKING and avoid the full-episode run; return exact boundaries to Central.
4. If detector fails, report its failure and the candidate forced plan; do not silently call it acoustic success. If the user has intentionally selected the legacy fixed mode, verify that path separately; do not pretend it exercises pause-aware segmentation.

## Phase C: one managed model pilot, then optional ONE full episode

1. Only after CPU PASS and planner/admission pass, acquire existing Nemovox-owned ASR/Riva/review resources through ModelLifecycle as appropriate, with exact resource fingerprint, checkpoint/CTC, language fr, speaker mode, quantization and request parity. Never start/stop unrelated models. Reuse PRIVATE verified clips where useful; no checkpoint download or global service edits.
2. First pilot up to 8 candidate inference excerpts from the new episode plan covering W01a/b/d, W02a, W03b, W04a and clean/repetition C1/C2. Use actual planned infer bounds rather than manually recutting to known-good 45s probes. Record raw CTC privately and publish hash-bound per-case source coverage, alignment error to VERIFIED acoustic anchors, diarization speakers, omitted words, duplicate/source replacements, and pilot latency. Inspect forced-cut cases carefully. Previously shown 16–24s omission of W04a's entire 18-word reply MUST NOT recur unnoticed; C1 29 clean words and C2 8-token genuine repeat/simultaneous voice remain controls. W01a singing may remain acoustically unsupported: do not invent it.
3. Pilot stop if: source is lost in a clean control, a correct W04a reply vanishes, major new false speaker merges occur, unowned GPU resources are affected, or model admission is unsafe. If a pilot reveals an unanticipated failure in the Central algorithm, return that reproducer; do not optimize cut thresholds locally.
4. If pilot demonstrates improvement and no blocking source/control loss, run exactly ONE full 23-minute source episode end-to-end via the exact candidate CLI with EXPLICIT --chunk-policy pause_aware in a NEW isolated output root. Allow actual Nemotron ASR+CTC, TitaNet, Riva fr-en, Qwen balanced post-review and SRT/ASS render; preserve original media and old job/ASS. No Studio production installation or restart. Use current host-managed model resources and existing safe lease choreography. This is a private qualification run, not a replacement of the user's existing subtitles.
5. Evaluate complete new output relative to original baselines: word coverage on W01..W05/S01/S02/S05 and controls, source timing vs acoustic anchors, duplicate fragments, recovery counts, uncertain speaker links, 0.21s micro-utterances, total true/unknown speakers (do not target a fixed count), source->Riva->review lineages, S02 polarity, beneficial Qwen repairs, number of cues, and 18/22 CPS (diagnostic only; Issue #9 owns renderer). Report whether W02c/W03a 10/7 residual duplicate fragments persist and whether W05a's 6 recovered words remain. Record the effective chunk policy and exact plan digest in the full run.
6. Verify original files/hashes again and ASR/Riva/Qwen services released with GPU returning to operator baseline. Preserve a private rollback/no-mutation ledger. A blocked model service or pilot failure is a partial handback, not a manufactured PASS. A full episode is not mandatory if an earlier gate fails.

## Phase D: evidence, authorization boundary

Publish one compact sanitized packet with:
- README.md, source-identity.json and planner/source contract digest;
- validation/tests.md (actual commands, suites/exits/warnings/skips, import origin);
- validation/chunk-plan.json (core/infer times, forced/pause basis, raw-input digests, no dialogue);
- validation/pilot-cases.jsonl (W/C/S cases, source indexes, acoustic method/uncertainty, coverage/CTC/diar outcomes);
- validation/full-episode-metrics.json (ONLY if genuinely run; denominator-consistent baseline vs candidate, timing, duplication, speaker, review, CPS; mark others null);
- validation/speaker-review.md (low-margin links, positive speaker controls; S05 raw diarization vs true emitted words);
- validation/polarity-review.md (S02 source/Riva/Qwen and helpful review counterexamples; rejected decisions & cached review status);
- validation/causal-assessment.md (per-case PASS/PARTIAL/FAIL and blocker priority; no unsourced victory claims);
- review/acceptance-matrix.md, review/findings.md, COMMAND_SHAPE_FRICTION.txt and manifest.json with each member SHA256/byte count.
- Optional sanitized read-only diagnostic harness if it materially aids Central; never publish raw copyrighted transcript, whole ASR/CTC JSON, media/audio, private host paths, tokens, weights or full journals.

Keep original and newly generated media/English/French subtitles and raw model responses PRIVATE under authorized host scratch. Do not push test output with raw dialogue to GitHub. Publish evidence-only commit on the existing issue branch after fresh remote-head CAS and full privacy/diff audit, then re-fetch and verify manifest parity. If remote head moved, do not force/rebase; return an offline sanitized archive and classify STALE. Leave PR #15 draft, Issue #14 open, main untouched.

## Handback

LOCAL_AGENT_HANDBACK:
 RESULT=PASS|PARTIAL|FAIL|BLOCKED
 EXACT_TESTED_SOURCE_SHA=4c6c7ac5719fc70b8db3a671e2970409a9a11e29
 EXACT_TESTED_TREE=a9b08e87ea5d5ef3e57e41381b56b0388565e126
 CURRENT_MAIN_SHA=<full SHA>
 PR_HEAD_CARRIER=<full evidence SHA>
 SOURCE_AND_PRODUCTION_ISOLATION=PASS|FAIL
 COMPILEALL_FOCUSED_NON_GPU_STUDIO_FRONTEND=<actual tests, exits, skips/warnings>
 FIXED_MODE_AND_RESUME_ISOLATION=PASS|FAIL
 STUDIO_CHUNK_POLICY_UI_API_CLI=PASS|FAIL
 PLAN_SILENCES_PAUSE_CUTS_FORCED_CUTS=<counts, max infer, plan digest, detector status>
 PILOT_ASR_CTC=<case IDs, lexical coverage, errors, speaker/repeat controls>
 FULL_EPISODE_RUN=<performed|blocked/not needed; exact reason and source-bound job ID>
 FULL_EPISODE_METRICS=<actual denominator and baseline comparison or null>
 SPEAKER_MERGE_SAFETY=<0.6536/0.6637 and clean positive pairs>
 S02_NEGATION_AND_HELPFUL_QWEN=<actual decisions and review statuses>
 ORIGINAL_INPUT_AND_MODEL_RESOURCE_INVARIANCE=PASS|FAIL
 REMAINING_BLOCKING_CASES=<identifiers/earliest layer>
 EVIDENCE_GITHUB_ROOT=<full carrier SHA:path>
 MANIFEST_HASHES=PASS|FAIL|BLOCKED
 PRIVATE_SUPPORT_REQUIRED=<none|bounded artifacts and location>
 COMMAND_SHAPE_FRICTION=<encountered=no/events=[] or precise typed incidents>
 PR15_ISSUE14_STATE=DRAFT/OPEN

Central will independently audit exact-head evidence before requesting Codex review, changing PR readiness, authorizing merge or production deployment. The new adaptive planner is NOT yet qualified by a source-code commit alone. Issue #9 stays separate.
