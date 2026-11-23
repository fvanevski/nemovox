# Issue #14: one consolidated default ASR recovery acceptance handoff

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=af9bd948c3eec613a4b43ab11af0b0abac781864
CENTRAL_SOURCE_TREE=abb3f8c76c99fb975deb054966963c9077bf080e
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CONTROLLING_DESIGN=staging/issue14/DEFAULT_SOURCE_RECOVERY_DESIGN.md
PREVIOUS_EVIDENCE=79bb78363a843b53e69cef799feb02c9de0d8172:staging/issue14/evidence/20261123T162337Z/
HOST_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-final-chronology-private/issue14-final-chronology-evidence-20261123T162337Z.tar.gz
HOST_PRIVATE_SHA256=55ceb2c7ad477d8f6d3b15813f4846c3d9154c1d287728641032f82cfa9dca53
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
CHALLENGER_EXPERIMENT=whisperx_pyannote_bounded_shadow_comparison
CHALLENGER_SOURCE_AUTHORITY=none_read_only_evidence_only
CHALLENGER_ENVIRONMENT=host_cached_models_and_installed_packages_only
CHALLENGER_PYANNOTE_ROOT=/workspace/pyannote_READ_ONLY

## Mission

Complete ONE consolidated CPU+GPU validation of the Central source correction that addresses all three actionable default-pipeline root causes: single generation-limit failure, ownerless S02 boundary speech, and silent confident CTC displacement/compression. Also independently inspect the residual speaker/turn failures using the existing private W/C/S audio evidence. Include ONE tightly bounded WhisperX/pyannote multilingual challenger comparison, prior to the full default Nemovox run, to determine whether an independent ASR/forced-alignment/diarization stack explains current source omissions and turn failures. This is an evidence experiment, NOT permission to change production ASR, speakers, tests, model configuration or CLI behavior. Return one review-ready packet, not serial tiny handoffs. No open-ended model survey/download, production service eviction, unrelated Studio or Issue #9 renderer changes, experimental architecture rewrite, skips or weakened guards.

## Phase A: exact source, static and CPU proof (CPU-only)

Fresh isolated OpenCode Build session/worktree, not reused stale object. Authenticate current main/issue/pr state and SHA ancestry; fetch latest PR head, confirm CENTRAL_SOURCE_SHA and its tree and no src/tests drift after it. Read whole AGENTS.md, DEFAULT_SOURCE_RECOVERY_DESIGN.md, recording.py, timing_evidence.py, chunking.py, boundary_evidence.py, source_events.py, source_unit_lineage.py, source_promotion.py, subtitles, and all associated tests. Confirm host original worktree/media/jobs/models intact; check archive SHA+45/45 member hashes and prior public manifest. Use current discovered Python environment (prior /workspace/nvidia-transcribe/.venv/bin/python Python3.13.15) and run:
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_asr_integrity.py tests/test_issue14_timing_boundaries.py tests/test_issue14_source_event_replay.py tests/test_issue14_source_recovery.py tests/test_issue14_source_reconciliation.py tests/test_issue14_source_promotion.py tests/test_issue14_source_unit_lineage.py tests/test_issue14_speaker_review_guards.py tests/test_issue14_source_to_renderer.py
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
 /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu' tests/test_studio_api.py
 git diff --check
Use actual test path after repository discovery; do not silently skip non-existing tests, replace commands only with the actual matching host test package and report precise reason. Prior 400/400, 112/112 and 50/50 were for 8c0244bc..., NOT validation of this commit. If any new test fails, diagnose exact root cause; no semantically changed source policy locally. Minor environment/test-fixture mechanical corrections only if unambiguously safe; otherwise report BLOCKED/PARTIAL with reproducible evidence.

Also perform a non-mutating comparator inventory now, before any GPU work. Inspect /workspace/pyannote and any previously installed Whisper/WhisperX/Faster-Whisper or pyannote venvs, installed package versions, cached model snapshot IDs/revisions, torch/CUDA support and available French/Chinese aligners. AGENTS.md PROHIBITS changes to /workspace/pyannote, its .venv and existing model checkpoints. Do not pip install, download a model, call an online inference API or accept gated model terms for the user. Only use already-authorized, cached assets; any required secret stays in its existing environment and is NEVER printed or written to a report. An unavailable optional model is NOT a blocker for the existing Nemovox campaign. Consult the actual installed CLI/module --help and package API instead of guessing invocation flags.

## Phase B: offline full-source replay and truthful recovery

Use actual archived original16 and new default_fr_job 16-chunk raw response files from original host private output (do not cross-run source IDs); validate media fingerprint, raw SHA, inferred audio window and source ordering. Re-run the current recording's timing audit, speaker normalization, boundary reconciliation and utterance builder from exact accepted raw responses WITHOUT model inference, using a private replay adapter if necessary. Do not overwrite historical source outputs. Record same denominators and raw unit IDs.

Target S02 shared 360s boundary: left chunk3 indices 280,282,283 (C'est/pas/notre) CTC ~364.87-365.255 (>core end360), right chunk4 indices14..22 longer negation CTC ~357.32-359.81 (<core start360). Verify new straddled_overlap_source_unlocalized conservatively retains ONE longer source hypothesis, never emits a false timed/speaker-certified subtitle, never deletes original ledger alternatives, and does NOT contaminate previously independent W05a 6/6 recovered, C1 29 clean, C2 repeated and 1327/1329 genuinely separate shouts. Check all 22 indexed cases. The original S02 English polarity should be proven from independently reviewed source and compared with old Riva baseline; unlocalized transcript itself is NOT a corrected English cue.

CTC: verify prior matching long displaced phrase now retains text/identity but loses invalid trusted timestamp; no false deletion due shift. Compare W01a/b, W02a, W03b (specific 11-word 1.365s region), S04, W04c, 66 large-gap warnings. Count truly quarantined CTC units, type/locations and any false positives or missing flagged examples. Real compressed multiword rate is only a REVIEW screen; do not assume all 22 can be timed automatically without audio evidence. Validate clean controls and Mandarin OOV/untimed preserved. If the ownerless recovery over-promotes false duplicate or loses lexical source, return exact offending unit evidence.

ASR generation integrity: prove synthetic first length->second stop safely completes, first length->second length raises incomplete, legacy completed-but-length checkpoint still fails closed, retry cap=1, both accepted/rejected SHA receipts, no hidden model text cleaning. Existing raw chunk12 (initial 16384-token length) is immutable; if actual new primary inference hits length use same bounded retry and preserve all raw attempts. No retry on unrelated network failure/pathological generation unless separate explicit design.

## Phase C: bounded acoustic adjudication (host audio only)

Independently inspect the smallest positive/negative source clips needed: S02 phrase just before/after 360s, W01a/b, W02a, W03b fast/compressed excerpt, S04, W02b/W02d/S05 mixed voice tags, W02c/W03a cross-speaker duplicates, W04c/H2 already reanchored, C1/C2/1327/1329 controls. Use existing native audio review, FireRedAED/diar evidence or owned Qwen inference only where it helps a specific case; no global model discovery, no WAV/model download. Record direct-audition statements as AGENT audio review, not human WER/CER truth. For S02, if enough evidence positively certifies source text/time, prepare a new separate v1 direct-reviewed source-turn revision using source_reconciliation/source_promotion with original/replayed raw fingerprints, not fake CTC time. Do not force speaker identities from one reused tag. If not acoustically supported, report explicit unrenderable unknown timing and semantic limitation.

## Phase C2: bounded WhisperX/pyannote independent challenger (isolated CPU + sequential GPU)

**Purpose and ownership:** Compare external ASR text, forced alignment and diarization on the SAME authenticated audio windows that Nemotron/Qwen and independent agent audio reviews used. Keep the experiment entirely in a fresh private scratch/evaluation root, outside src/, tests/, tracked model config and /workspace/pyannote. Reading /workspace/pyannote programs, checkpoints and existing outputs is allowed; changing its code, files, .venv or model assets is forbidden. This phase produces evidence, not a replacement default pipeline or a new PR. Execute after CPU/source replay, before the complete default Nemotron episode run.

**Input selection:** Build a frozen manifest of roughly 12 French + 4 Mandarin windows, about 8-30 seconds each with sufficient pre/post speech context, normally no more than ~8 minutes of total sampled audio. Select the smallest useful intervals from the original 22 W/C/S catalog and exact archival media fingerprint: French S02 around 360s negation; W04c reanchored at ~1264.3-1266.3; H2 at ~364.7-366.2; W05a six-word omission ~1352-1354; W02b child voices (including prior ~196-235s separation failure); W02c and W03a cross-chunk duplicate/voice conflicts; W03b compressed/merged turn; W01a/b and W02a drift; S04 severe CPS; C1 clean and C2 repeated rapid shouts/1327-1329 genuine separate events. These are priorities, not an instruction to fabricate missing case times: derive ALL exact extract windows from validated media, source ledger and case catalog. For Mandarin include EP166 multi-speaker 525s handoff, an overlap control, and 1-2 transcript/timing-challenged regions selected from actual chunk/audio coverage, NOT guessed timing of the 41 currently untimed utterances. Choose up to ~16 nonredundant windows; add a few seconds or split a dense voice transition where needed to avoid misleading clipped syllables. Preserve actual media stream, sample rate, window boundaries and WAV SHA for every case; never mutate original assets.

**Models, if already available:** Use installed multilingual Whisper large-v3 (cached large-v3-turbo may be an OPTIONAL speed comparator, not a replacement for large-v3), language explicitly French (fr) or Mandarin Chinese (zh), task=transcribe/source language (NEVER task=translate). Run Whisper or WhisperX recognition first as an independent raw-text hypothesis. Separately run WhisperX language-specific forced alignment for French and Chinese only when the required aligner assets are actually installed/cached and compatible. Separately run existing pyannote speaker diarization, preferably the previously used baseline and cached speaker-diarization-community-1 if actually available; record exact model/version and which outputs are normal overlapping versus optional exclusive assignment. Community-1 gated model availability must be established without disclosing any Hugging Face token; if unavailable, skip that comparator rather than downloading or changing authorizations. Avoid converting a 3.1 checkpoint into a Community-1 claim. WhisperX alignment is conditional on correctly transcribed lexical input and cannot establish word truth, and exclusive diarization must NOT be used to prove overlapping speakers absent.

**Execution:** Use separate processes in the existing compatible offline environments. First test one FR + one ZH clip at selected sample rate/language and explicit source transcription to establish memory/adapters. For the rest, sequentially execute ASR, alignment and diarization within Nemovox-owned and other-resource-safe GPU admission: verify nvidia-smi and ModelLifecycle/lease constraints, keep kwin/embeddings/Studio and non-Nemovox services untouched, do not concurrently hold the Nemotron or Riva GPU lease, release models before the next GPU phase and verify VRAM restoration. Never force another process to release GPU resources, switch to a cloud endpoint or silently downgrade CPU/CUDA; report an explicitly elected CPU-only comparator as such. If versions are incompatible, assets absent, out-of-memory or runtime cannot be safely admitted, report exactly what was NOT EXECUTED and why, and CONTINUE the existing CPU/default-pipeline campaign. Do not modify dependency locks or host-global Python state to rescue the experiment.

**Comparisons, kept separate by component:** (1) source text: Nemotron raw/retained, cached Qwen challenger, Whisper raw, and only independently audio-reviewed sample annotations as bounded references; false speech during music/SFX is a negative control. (2) timing: Nemotron CTC hypothesis vs WhisperX forced-aligner estimates vs independently auditioned event interval, preserving clip offsets and distinguishing missing versus suspect word timing. (3) diarization: pyannote overlapping segments/RTTM vs Nemotron/TitaNet speaker regions, voice changes, child/adult boundaries, actual simultaneous speech and true C2 repetitions; use exclusive speaker labels only as a separate presentation output. (4) English consequence: show whether corrected source words would remove documented wrong English meaning/polarity, but do NOT automatically insert challenger text into Nemovox source/translation contracts.

Record one row per frozen case with source audio fingerprint, clip SHA/absolute interval, transcript variants, alignment status, speaker turns/overlaps, audio-reviewed witness (agent-attested vs user/human reference), conclusion and uncertainty. Compute sample WER/CER/DER only on precisely matched reference spans with truly independent annotated transcripts/speaker labels and declared normalization/collar rules; otherwise mark those measures UNVERIFIED and use clearly labeled qualitative event-level error comparisons. Do not use Whisper/Nemotron agreement as acoustic proof or credit longer raw transcript text as true coverage.

**Decision boundary:** On the first small batch, determine whether the challenger's incremental evidence changes at least one material missing-dialogue, CTC-drift or speaker-turn decision without violating negative controls. If informative and existing cached model/runtime permits, run up to ONE additional complete French WhisperX shadow pass, and optionally one complete Mandarin shadow pass only if the bounded Chinese overlap/untimed tests show a compelling, explicitly documented reason. Full challenger episodes remain completely separate from the mandatory ONE Nemotron default run and may be omitted for speed/safety. No additional research campaigns, model upgrades or production integration. Recommend future component adoption (raw ASR challenger, evidence-only timing, pyannote diarization, or no change) with exact positive and regression cases; Central adjudicates architecture after this handback.

## Phase D: one complete default recording and English comparison (GPU)

Phase C2 is NOT a prerequisite for Phase D success: missing/weak WhisperX assets or inconclusive findings must not postpone normal Nemovox acceptance. After gates/focused offline proof, ONE new private-root full ~23min French primary ASR process_media run through actual default fixed 90/10 chunk plan and existing managed Nemotron endpoint/model. Use new job root and identical source media fingerprint; do not delete or mutate previous jobs, raw WAVs, media, Studio, embeddings, desktop or models. Record each raw finish status/generation attempts and rejected receipt, timing audit, boundary sequence/straddled/extended diagnostics, lexical source count and utterance word/speaker/timing status. The new single auto retry should not require a manual resume if the 1st attempt gets the same random generation-length failure and 2nd finishes; if both incomplete, fail closed and report PARTIAL. If no length failure occurs, synthetic tests cover retry and raw provenance.

Translate REAL new source to English via already installed Riva under ModelLifecycle, render SRT/ASS using same natural profile, refinement=off/review=off and comparable lane config. Compare legacy baseline 234/309, last default 217/290, new default, and existing opt-in reviewed FR overlay without mixing distinct source run IDs. Metrics: preserved raw words, timed/untimed, correct-versus-ambiguous source acoustic events, duplicate episodes, missed W05a, S02 negation, S01/S03/S04 and W02b mixed voices, 18/22 CPS and cue survival. Do not portray a change from 290 to more or fewer cues as guaranteed improvement; do not silently render unverified timestamps. Source-bound Studio smoke/read-only rerender, cleanup and current NVIDIA runtime/service nonmutation proof.

Mandarin EP166 is the cross-language control: existing 41 untimed utterances and incomplete c14 remain, unchanged unless the same CPU boundary policy legitimately changes only evidence labels. No unrelated full Mandarin run.

## Phase E: one evidence commit and exact handback

Publish exactly ONE new sanitized public packet to staging/issue14/evidence/<new-UTC-id>/ in THIS PR, via non-force exact-head verified commit; public README, source-identity, CPU/STUDIO gates, retry raw-provenance/attempt counts, S02 straddle source-unit proof, CTC quarantine report with false positives, 22-case full chronology, English before/after, language controls, GPU/service-state, residual blockers, COMMAND_SHAPE_FRICTION and SHA256/bytes manifest. Keep a self-contained small private tar.gz with complete source log/test traces and bounded event/audio samples/diagnostic hashes but no giant media/weights or credentials. No WAV or full copyrighted episode transcript in GitHub. Add validation/whisperx-pyannote-challenger.json (or a similarly named sanitized summary) describing inventory, cached model IDs/revisions, selected FR/ZH case IDs and clip hashes, executed/skipped stages, separate text/timing/diarization comparisons, candidate improvements, regressions, GPU/resource nonmutation, and whether a conditional full shadow run was justified. Include granular raw/hypothesis artifacts, bounded exact sampled texts, RTTM/JSON and executed command traces in the PRIVATE archive only. No Hugging Face tokens, full transcript/audio, model weights or service credentials in either public artifact. Readback SHA and bytes for EVERY public manifest member, current exact PR head, and confirm no src/tests drift. PR stays draft and Issue #14 open. Do not merge, deploy, request Codex yet, or close issue.

Required handback:
 ISSUE14_DEFAULT_SOURCE_RECOVERY:
  RESULT=PASS|PARTIAL|BLOCKED
  ADMITTED_PR_HEAD=<sha>
  VALIDATED_SOURCE_SHA=<sha>
  VALIDATED_SOURCE_TREE=<sha>
  CURRENT_PR_HEAD=<sha>
  MAIN_SHA=<sha>
  SOURCE_DIFF=<none|paths+complete diff>
  CPU_FOCUSED_FULL_STUDIO=<exits, counts, warnings>
  GENERATION_RETRY=<synthetic and live exact raw SHA, attempts and limit stop>
  S02_OWNERLESS_SOURCE=<chunk3/chunk4 unit IDs, 1 accepted hypothesis, untimed no fabricated truth, source-to-English status>
  CTC_MULTIWORD_AND_CROSSCHUNK=<before/after, exact quarantine and false-positive review, C1 control>
  SPEAKERS_AND_22_CASES=<W/C/S case matrix: pass/partial/fail and genuine voice versus unknown>
  WHISPERX_PYANNOTE_CHALLENGER=<FR/ZH windows, exact installed models/revisions, ASR text, forced alignment, diar/overlap, reference authority, matched/false speech, any full shadow run, omitted reasons, adoption recommendation>
  FULL_FR_DEFAULT=<fresh job identity, raw chunks, source integrity/reconstruction, diagnostics>
  FR_ENGLISH_COMPARISON=<baseline, new default, reviewed overlay, timing/cues/CPS/polarity>
  MANDARIN_CONTROL=<existing 41 untimed/69 unmatched and report>
  HOST_GPU_NONMUTATION=<before/after>
  PUBLIC_EVIDENCE_ROOT=<sha:path>
  MANIFEST_SHA256=<sha>
  PUBLIC_FILES_VERIFIED=<count>
  PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
  WER_CER=<UNVERIFIED if no human full reference>
  ISSUE14_ACCEPTANCE=<PASS|PARTIAL with original acceptance criteria>
  REMAINING_CENTRAL_BLOCKERS=<exact root cause list>
  ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events=<[]|canonical records>
  omitted_sensitive_values=<yes|no>

One complete host campaign. If a semantic production defect is found, return concise diagnostic/trace and stop that unsafe subpath; do not weaken tests, quietly remove lines, or invent a different acoustic policy locally.
