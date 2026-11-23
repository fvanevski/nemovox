# Issue #14 PR #15: final source chronology, audio revision and full episode gate

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=8c0244bc22176b8e5f1ce2d2e6e3480505d9ce4e
CENTRAL_SOURCE_TREE=cc3cb0c1721fcc416d308d8392b789cdc7b63801
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CONTROLLING_DESIGN=staging/issue14/FINAL_CHRONOLOGY_ACCEPTANCE_DESIGN.md
PREVIOUS_PUBLIC_PACKET=5b0d055c5cc8b409f4fa9edb3013b27e41490808:staging/issue14/evidence/20261123T050209Z/
PREVIOUS_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-fidelity-private/issue14-integrated-source-fidelity-evidence-20261123T050209Z.tar.gz
PREVIOUS_PRIVATE_SHA256=c740218141e3c370d94f864daa54079e5e9ee25395bfd82a003ccde4a4da5e3c
NO_NEW_PR=yes
NO_MERGE=yes
NO_PRODUCTION_DEPLOY=yes
NO_ISSUE_CLOSE=yes

## Mission

ONE consolidated host source-fidelity acceptance campaign. Verify corrected Central code, redo independent direct audio reviews at correct W04c/H2 timing, prevent duplicate source events, exercise actual default primary long-form French pipeline, produce English subtitles and complete source-bound evidence for Issue #14. No serial micro-handoffs, generic ASR/model survey, new download, renderer issue #9 refactor or local-agent redesign of Central source policies.

## A. Safe admission and CPU

Fresh OpenCode Build. Authenticate live GitHub issue/PR/main and exact source commit. Fetch newly remotely advanced branch BEFORE local object/worktree proof. Prove Central source in PR ancestry, no newer src/tests changes. Make fresh isolated non-destructive detached worktree and private scratch/output paths. Never reset/clean/rebase/stash/force operator checkout or modify /workspace/models, old media, old ASR jobs or services.

Read full repo AGENTS.md, design, source_promotion, source_reconciliation, source_unit_lineage, timing_evidence, boundary_evidence, chunking, speakers, recording, tests and original 22 indexed W/C/S case catalog. Authenticate prior public/private packets. Record actual Python interpreter path from current host. Execute:
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_source_promotion.py tests/test_issue14_source_unit_lineage.py tests/test_issue14_source_reconciliation.py tests/test_issue14_timing_boundaries.py tests/test_issue14_speaker_review_guards.py tests/test_issue14_source_recovery.py tests/test_issue14_source_event_replay.py tests/test_issue14_source_to_renderer.py tests/test_viewing_quality_regressions.py
 PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
 git diff --check

Previous 393 PASS/1 FAIL does NOT validate this SHA. Require both exact-source suites green and previous French-in-Mandarin fixture failure fixed without weakening foreign-review production gate. If code fails semantically, return concise root-cause traceback and minimal reproducer to Central; no gate weakening. Mechanical test/env fixes within limited authority only.

## B. Re-anchor W04c and H2 against NEW auditioned audio

Old positive review windows were wrong:
 W04c old [1261.5,1263.5], old WAV [1261.3,1263.7], actual speech reported near [1264.3,1266.5].
 H2 old [362.5,364.2], old WAV [362.3,364.4], actual speech reported near [364.8,366.0].
These are approximate centers, not prevalidated exact new timestamps. Extract NEW bounded 16k mono WAV clips containing each complete event with context under existing host-approved extraction, compute SHA256 on actual file, listen independently using native audio review, document exact speech/audio-event start-end, voice, and uncertainty. The old audio crop SHA MUST be rejected if recycled as support for reanchored events. Do not copy old reviewed text or revision hash as proof. Bind the new review record digest and review_audio_crop_start_s/end_s/sha256 fields to each audited event. Direct audio listening is agent-attested only, not human WER/CER.

Construct new private audited-events/v1 input from authenticated source ledgers and real reviewed changes; preserve other previous events only when valid. Run source_reconciliation.py with original16 ledger, authenticated plan and challenger; generate NEW private dossier, never overwrite previous artifacts. Same event ID only if same acoustic occurrence; canonical source_revision_id must change as dictated by reviewed bounds/text/digest; source_turn_id stable. Run negative prior-crop failure, wrong-media, source-language, and stale-revision tests.

Use raw source-unit IDs to determine acoustic ownership, not English text similarity/CTC placement. H2 may be same event as original utt-0074 at [364.86,366.38] even though ASR text differs; link that unit to new heard event ONLY after positive direct audio verification. W04c overlaps partly mapped utt-0208 and displaced utt-0207; keep all raw observations, do not append double phrase or delete partial rows. Independent overlapping speakers stay unresolved unless positively differentiated.

## C. Reconcile source, duplicate safety and all 22 cases

Run exact source_unit_lineage.py and source_promotion.py --source-evidence on FR original16 and ZH EP166 with refreshed source/dossier identity. FR all-unit examples utt-0193 9/9 and utt-0206 5/5 can be explicitly superseded by same-event verified source (not appended). ZH utt-0063/64/65 fully mapped, utt-0062 partial, all 41 untimed remain. Revisit known H1 provisional duplicate of utt-0027, W04c partial, H2 overlap, 4 context-bloated Qwen provisional segments, true C2 rapid repetition and 1327/1329 shouts, source VAD misses, song/SFX false words. Record accepted/held/rejected and source units remaining. Never remove original raw text or claim word-level timing/identity without audio evidence.

Run original indexed 22 case replay under current timing_evidence/boundary_evidence/normalization/speaker/utterances, reporting for each: original vs candidate source words, hypothesis vs credible acoustic time, owning chunk and overlap, source unit IDs, speaker purity/unknown, matched/duplicate risk, final utterance and English cue disposition. Check W05a SIX source words from c14 right context when c15 missed them; W02c/W03a/W03c single genuine passages; W04b/S03 more-complete neighbor; W01-W04/S04 drift, H2/W04c; S02 negation and Qwen review positive controls; C1 29 clean tokens and C2 legitimate repetitions. Keep distinct voices and low-margin embedding nonmerges. If unresolved, label accurately and preserve hypothesis.

## D. ONE fresh 23-minute French default pipeline run and English render

After CPU/focused audio admission, execute ONE full French primary Nemotron recording process_media through the normal current CLI/recording path on original-fingerprint 23-minute episode in a NEW output/job root. Inspect current --help and existing model resource/admission commands before running, do not invent runtime flags. Use existing configured model checkpoint, original16-equivalent plan and source language fr unless explicit documented reason to differ. Respect Nemovox ModelLifecycle ownership; do not disrupt kwin, embedding server, Studio service or unrelated models. Record completion of all chunks, finish_reason and generation_limit rejects, 2415 source raw-token comparison where valid, CTC evidence audit quarantines, boundary sequence matches/context rescue, speaker/reconstruction diagnostics and transcript quality for all indexed cases. This default-path run is REQUIRED to distinguish genuine upstream correctness from overlay-only success. If safe owned model admission is unavailable, return PARTIAL with exact blocker rather than false PASS.

Produce actual Riva fr->en source/MT SRT+ASS from freshly produced default recording with same known translation settings (natural, refinement/review off, layout standard); validate CLI/Studio source-bound smoke and resume/offline rerender. Compare old baseline 234/309, unsafe 241/320, prior vetted 237/312, corrected orig16 opt-in overlay, and NEW default pipeline using identical denominator definitions. If fresh primary raw chunk SHA differs from orig16, DO NOT cross-join orig16 lexical IDs to it; build new exact raw-unit ledger/lineage or separately label the older corrected overlay as a different run. Report acoustic coverage, duplicate speech, missing source, timing error/unknown, fragmentation, speaker ambiguities, source/MT/review lineage, final cues, S02 polarity, 18/22 CPS without changing Issue #9 renderer.

## E. Mandarin cross-language check, evidence and closure readiness

Read-only/CPU replay ZH EP166 exact 2832/2901 unit match, 69 unmatched incomplete c14, 41 untimed retained unrenderable. Prior Riva English baseline 304 and vetted 305 cues were genuine English renders, not new full-episode validation on this commit. The latest partial-source overlap may HOLD ZH reviewed turn1: report new count and rationale. A bounded zh->en Riva sample may be run under safe resource admission if it would resolve a substantive new issue; no broad Mandarin inference campaign or invented timestamps.

Publish ONE sanitized evidence commit to existing PR staging/issue14/evidence/<new-run-id>/, README, source identity, CPU results, corrected audio bounds/digests, 22-case table, raw-unit/source promotion counts, default French full run provenance, actual English SRT/ASS quality, Mandarin controls, runtime restoration, residual blockers and COMMAND_SHAPE_FRICTION, plus complete SHA256/bytes manifest. No raw WAV, media, full copyrighted scripts, weights or secrets publicly. Preserve self-contained private review archive (small bounded excerpts, actual clip/provenance hashes, test logs, source IDs, source/MT/review comparisons and full diff if any). Verify new public commit/readback/manifest and PR exact-head. No other PR, merge, Issue close, Codex review request, Studio deployment or architecture refactor.

HANDOFF RETURN:
ISSUE14_FINAL_CHRONOLOGY:
 RESULT=PASS|PARTIAL|BLOCKED
 ADMITTED_PR_HEAD=<40hex>
 VALIDATED_SOURCE_SHA=<40hex>
 VALIDATED_SOURCE_TREE=<40hex>
 CURRENT_PR_HEAD=<40hex>
 MAIN_SHA=<40hex>
 SOURCE_DIFF=<none|path+full-diff>
 CPU_FOCUSED_FULL_COMPILE=<counts, exits, warnings>
 W04C_H2_NEW_AUDIO_REVIEW=<old/new intervals, new WAV SHA, separate review SHA, revision IDs, linked source IDs, no fake proof>
 FR_ZH_IMMUTABLE_LINEAGE=<counts, partials, displaced, retained>
 SOURCE_PROMOTION_AND_DUPLICATE_POLICY=<before/after, H1/W04c/H2/C2, controls>
 CASES_22=<all case IDs/decisions, genuine missing versus evidence-only>
 FULL_FRENCH_DEFAULT_ASR=<new job identity, plan, chunk finish/timing/boundary/speaker diagnostics>
 REAL_ENGLISH_FR_SUBTITLES=<translation model and fair comparison, cues, source/Mt/review, CPS>
 MANDARIN_CROSS_LANGUAGE=<41 untimed, c14 incomplete, source+English distinction>
 GPU_SERVICE_NONMUTATION=<GPU before/after and services>
 PUBLIC_EVIDENCE_ROOT=<exact SHA:path>
 MANIFEST_SHA256=<hash>
 PUBLIC_MEMBER_VERIFIED=<count>
 PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
 ISSUE14_ACCEPTANCE=<PASS|PARTIAL with criteria>
 WER_CER=<UNVERIFIED unless independent reference>
 REMAINING_CENTRAL_BLOCKERS=<list|none>
 ISSUE14=OPEN PR15=DRAFT
COMMAND_SHAPE_FRICTION:
 encountered=<yes|no>
 events=<[]|canonical records>
 omitted_sensitive_values=<yes|no>

Return one complete handback. A material source correctness failure is a blocker to diagnose, not authority for local semantic redesign or bypassing tests.
