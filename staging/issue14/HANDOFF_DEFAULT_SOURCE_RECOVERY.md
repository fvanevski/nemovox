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

## Mission

Complete ONE consolidated CPU+GPU validation of the Central source correction that addresses all three actionable default-pipeline root causes: single generation-limit failure, ownerless S02 boundary speech, and silent confident CTC displacement/compression. Also independently inspect the residual speaker/turn failures using the existing private W/C/S audio evidence. Return one review-ready packet, not serial tiny handoffs. No model survey/download, production service eviction, unrelated Studio or Issue #9 renderer changes, experimental architecture rewrite, skips or weakened guards.

## Phase A: exact source, static and CPU proof (CPU-only)

Fresh isolated OpenCode Build session/worktree, not reused stale object. Authenticate current main/issue/pr state and SHA ancestry; fetch latest PR head, confirm CENTRAL_SOURCE_SHA and its tree and no src/tests drift after it. Read whole AGENTS.md, DEFAULT_SOURCE_RECOVERY_DESIGN.md, recording.py, timing_evidence.py, chunking.py, boundary_evidence.py, source_events.py, source_unit_lineage.py, source_promotion.py, subtitles, and all associated tests. Confirm host original worktree/media/jobs/models intact; check archive SHA+45/45 member hashes and prior public manifest. Use current discovered Python environment (prior /workspace/nvidia-transcribe/.venv/bin/python Python3.13.15) and run:
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_asr_integrity.py tests/test_issue14_timing_boundaries.py tests/test_issue14_source_event_replay.py tests/test_issue14_source_recovery.py tests/test_issue14_source_reconciliation.py tests/test_issue14_source_promotion.py tests/test_issue14_source_unit_lineage.py tests/test_issue14_speaker_review_guards.py tests/test_issue14_source_to_renderer.py
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
 /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu' tests/test_studio_api.py
 git diff --check
Use actual test path after repository discovery; do not silently skip non-existing tests, replace commands only with the actual matching host test package and report precise reason. Prior 400/400, 112/112 and 50/50 were for 8c0244bc..., NOT validation of this commit. If any new test fails, diagnose exact root cause; no semantically changed source policy locally. Minor environment/test-fixture mechanical corrections only if unambiguously safe; otherwise report BLOCKED/PARTIAL with reproducible evidence.

## Phase B: offline full-source replay and truthful recovery

Use actual archived original16 and new default_fr_job 16-chunk raw response files from original host private output (do not cross-run source IDs); validate media fingerprint, raw SHA, inferred audio window and source ordering. Re-run the current recording's timing audit, speaker normalization, boundary reconciliation and utterance builder from exact accepted raw responses WITHOUT model inference, using a private replay adapter if necessary. Do not overwrite historical source outputs. Record same denominators and raw unit IDs.

Target S02 shared 360s boundary: left chunk3 indices 280,282,283 (C'est/pas/notre) CTC ~364.87-365.255 (>core end360), right chunk4 indices14..22 longer negation CTC ~357.32-359.81 (<core start360). Verify new straddled_overlap_source_unlocalized conservatively retains ONE longer source hypothesis, never emits a false timed/speaker-certified subtitle, never deletes original ledger alternatives, and does NOT contaminate previously independent W05a 6/6 recovered, C1 29 clean, C2 repeated and 1327/1329 genuinely separate shouts. Check all 22 indexed cases. The original S02 English polarity should be proven from independently reviewed source and compared with old Riva baseline; unlocalized transcript itself is NOT a corrected English cue.

CTC: verify prior matching long displaced phrase now retains text/identity but loses invalid trusted timestamp; no false deletion due shift. Compare W01a/b, W02a, W03b (specific 11-word 1.365s region), S04, W04c, 66 large-gap warnings. Count truly quarantined CTC units, type/locations and any false positives or missing flagged examples. Real compressed multiword rate is only a REVIEW screen; do not assume all 22 can be timed automatically without audio evidence. Validate clean controls and Mandarin OOV/untimed preserved. If the ownerless recovery over-promotes false duplicate or loses lexical source, return exact offending unit evidence.

ASR generation integrity: prove synthetic first length->second stop safely completes, first length->second length raises incomplete, legacy completed-but-length checkpoint still fails closed, retry cap=1, both accepted/rejected SHA receipts, no hidden model text cleaning. Existing raw chunk12 (initial 16384-token length) is immutable; if actual new primary inference hits length use same bounded retry and preserve all raw attempts. No retry on unrelated network failure/pathological generation unless separate explicit design.

## Phase C: bounded acoustic adjudication (host audio only)

Independently inspect the smallest positive/negative source clips needed: S02 phrase just before/after 360s, W01a/b, W02a, W03b fast/compressed excerpt, S04, W02b/W02d/S05 mixed voice tags, W02c/W03a cross-speaker duplicates, W04c/H2 already reanchored, C1/C2/1327/1329 controls. Use existing native audio review, FireRedAED/diar evidence or owned Qwen inference only where it helps a specific case; no global model discovery, no WAV/model download. Record direct-audition statements as AGENT audio review, not human WER/CER truth. For S02, if enough evidence positively certifies source text/time, prepare a new separate v1 direct-reviewed source-turn revision using source_reconciliation/source_promotion with original/replayed raw fingerprints, not fake CTC time. Do not force speaker identities from one reused tag. If not acoustically supported, report explicit unrenderable unknown timing and semantic limitation.

## Phase D: one complete default recording and English comparison (GPU)

After gates/focused offline proof, ONE new private-root full ~23min French primary ASR process_media run through actual default fixed 90/10 chunk plan and existing managed Nemotron endpoint/model. Use new job root and identical source media fingerprint; do not delete or mutate previous jobs, raw WAVs, media, Studio, embeddings, desktop or models. Record each raw finish status/generation attempts and rejected receipt, timing audit, boundary sequence/straddled/extended diagnostics, lexical source count and utterance word/speaker/timing status. The new single auto retry should not require a manual resume if the 1st attempt gets the same random generation-length failure and 2nd finishes; if both incomplete, fail closed and report PARTIAL. If no length failure occurs, synthetic tests cover retry and raw provenance.

Translate REAL new source to English via already installed Riva under ModelLifecycle, render SRT/ASS using same natural profile, refinement=off/review=off and comparable lane config. Compare legacy baseline 234/309, last default 217/290, new default, and existing opt-in reviewed FR overlay without mixing distinct source run IDs. Metrics: preserved raw words, timed/untimed, correct-versus-ambiguous source acoustic events, duplicate episodes, missed W05a, S02 negation, S01/S03/S04 and W02b mixed voices, 18/22 CPS and cue survival. Do not portray a change from 290 to more or fewer cues as guaranteed improvement; do not silently render unverified timestamps. Source-bound Studio smoke/read-only rerender, cleanup and current NVIDIA runtime/service nonmutation proof.

Mandarin EP166 is the cross-language control: existing 41 untimed utterances and incomplete c14 remain, unchanged unless the same CPU boundary policy legitimately changes only evidence labels. No unrelated full Mandarin run.

## Phase E: one evidence commit and exact handback

Publish exactly ONE new sanitized public packet to staging/issue14/evidence/<new-UTC-id>/ in THIS PR, via non-force exact-head verified commit; public README, source-identity, CPU/STUDIO gates, retry raw-provenance/attempt counts, S02 straddle source-unit proof, CTC quarantine report with false positives, 22-case full chronology, English before/after, language controls, GPU/service-state, residual blockers, COMMAND_SHAPE_FRICTION and SHA256/bytes manifest. Keep a self-contained small private tar.gz with complete source log/test traces and bounded event/audio samples/diagnostic hashes but no giant media/weights or credentials. No WAV or full copyrighted episode transcript in GitHub. Readback SHA and bytes for EVERY public manifest member, current exact PR head, and confirm no src/tests drift. PR stays draft and Issue #14 open. Do not merge, deploy, request Codex yet, or close issue.

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
