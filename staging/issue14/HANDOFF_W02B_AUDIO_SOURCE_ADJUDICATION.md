# Issue #14 / PR #15: decisive W02b audio split and displaced-source turn adjudication

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=7bb957ed829649d965445c0aee91bf062b698849
CENTRAL_SOURCE_TREE=f42962cfe3986b2af19bd2e23cfb8ea189a09370
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
PREVIOUS_HOST_EVIDENCE_HEAD=7c875d6f4a142c6f0f352188d52d54cd0f761277
PRIOR_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-independent-anchors-private/issue14-independent-anchors-evidence-20261123T183454Z.tar.gz
PRIOR_PRIVATE_SHA256=86f26714223e56b5c1a30f6412a8e06cb5871c976c346eebd48c8da3d16f5e
CONTROLLING_AUDIT=staging/issue14/AUDIT_INDEPENDENT_ANCHORS_20261123.md
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_PRODUCTION_SOURCE_EDIT=yes
NO_MODEL_DOWNLOAD=yes
NO_NEW_LARGE_GPU_WORKLOAD=yes
NO_NEW_FULL_EPISODE_RUN=yes
NO_NEW_AV_MODEL_PASS=yes

## Objective: one decisive, bounded acoustic qualification

Resolve *one actual source-invariant question* left by the prior W02b six-turn report: whether `Toi?` (source ordinal 100, unit_be0c12232c62c487035804c4, CTC [220.28,220.937]) and `Parfaitement!` (ordinal 101, unit_893da33a8142fd3a01c053e5, CTC [221.0,222.158]) belong to the same audible voice, or two distinct rapid turns. Also re-check the voice of the final W02b utterance [231.9,235.85] (source ordinals 128..139) and establish exact event-to-original-unit ownership for W04a/W03b sustained displaced-CTC cases where audio evidence permits. Avoid redoing previous 411-test, full-language, Whisper/pyannote or 8-scene VLM campaigns. This is host-bound primary media/audio evidence gathering and *dry-run source-event reconciliation*, not delegated source architecture.

## A. Authority, old evidence and host-safe preflight

Use one fresh isolated OpenCode Build worktree of current PR #15. Verify PR OPEN/DRAFT, Issue #14 OPEN, current main, exact full PR head; ensure `CENTRAL_SOURCE_SHA` remains ancestor and `src/` and `tests/` have no intervening changes. Read AGENTS.md, CONTROLLING_AUDIT, source_reconciliation.py, source_unit_lineage.py, source_promotion.py, source_events.py and existing Issue #14 regressions. Verify prior archive SHA and all 252 unique member checksums; duplicate tar paths are identical, NOT separate evidence. Verify previous public GitHub manifest and audio WAV/event source fingerprints. Keep original media, original job outputs, /workspace/pyannote and services unchanged. No need to repeat 411/411 host suite if source/test SHA exactly same and the prior candidate-bound test receipts remain authenticated. If source changed, STOP and report STALE instead of assuming prior validation.

## B. W02b split: audio first, imagery second, no identity guessing

Use authenticated FR video/audio source, archived FR_W02b_child_voices.wav [212,236] and source ledger. Make only two additional local waveform/short listening excerpts, approximately [218.7,222.6] and [231.4,236.0], with original dialogue mix and (if channels meaningfully differ) original channels intact; sample-accurate source-media offsets and SHA, no destructive filters. First audition without showing case text, Whisper transcript or screenshot. Record audible change point(s), voice timbre/pitch/cadence and whether the consecutive short phrases clearly come from the SAME or DIFFERENT child voice. Repeat with the two suspected phrases revealed. Distinguish true acoustic turn, pause/breath, non-speech, and uncertain voice. For final W02b turn compare [231.9,235.85] to reference regions of separately audio-attested Voice A and Voice B, not scripted character names.

Compare source ordinals 100,101 and 128..139 to actual raw parent chunk, CTC hypothesis and Whisper segment. Treat the merged [220.22,221.66] Whisper segment and single Nemotron local speaker tag as INSUFFICIENT to prove same voice. Confirm all candidate event boundaries fall within original input WAV inference windows; retain shared speaker ID as evidence of collapse, not as truth. Produce explicit per-token/phrase acoustic decisions:
- `ordinal_100_voice`, `ordinal_101_voice`: `same|different|UNRESOLVED`, event subinterval(s), independently audible words and status
- `W02b_final_turn_voice`: `Voice_A|Voice_B|different_unknown|UNRESOLVED` relative to independently reviewed nearby turn references
- Number of distinct acoustic turns/voices actually supported, with uncertainty where needed.

Only AFTER locking a hash of the new audio-only review, consult archived visual frames. Existing video contact-sheet timestamps were *synthetic sample centers*, not true decoder PTS, and case IDs in visual-only prompts leaked labels. The original images can support visual appearance but not subsecond timing or blinded accuracy. If an exact frame cut would resolve a dispute, use a bounded ffprobe/ffmpeg read of [219.8,222.5] with actual source decoder `best_effort_timestamp_time` or `showinfo pts_time` and `-copyts`, preserving selected frame-to-decoder PTS mapping; cross-check against the correct media stream/timebase. For any observed A/V offset, report it rather than assuming audio/video start_time=0 establishes lip sync. Do NOT run new vision-model inference or describe frame-derived PTS as acoustic event truth. Character identities `Tao`/`Zia` remain unverified unless direct visual identity evidence and acoustic speaker correspondence both independently pass.

## C. W04a / W03b: bound displacement, prevent false word/time edits

Read-only recheck W04a [1201-1208] (SFX/grunts), [1229-1238] (actual speech), [1247-1256] (panting and speech) using prior three short WAVs, original source unit IDs/parent request bounds, and previously attested turns; separate event `utt-0187` and `utt-0188` word ownership. For W03b [~985-1014], inspect the four immutable Whisper-v3 lexical correspondence blocks (median signed shifts -5.925,+6.44,+11.43,+10.1s) and original audio, focusing only on uniquely associated source phrases with verifiable positive acoustic events. If an ASR phrase has multiple possible audio matches, preserve alternative and unknown timing rather than assigning convenient word timestamps.

Use existing `source_reconciliation` and `source_unit_lineage` OFFLINE in a private scratch root to build one candidate audited-event dossier of only positively *direct-audio-attested* source turns with source text, precise bounded event interval, immutable original word-unit IDs, review WAV hash, and explicit conflict/unknown status. Use verified complete reviewer provenance; the previous agent-attested listening is NOT a human gold transcript. Do NOT mark any event `independently_auditioned`, `source_text_verified` or `same_audio_event_verified` unless the new review truly establishes those facts. When lexical units from one legacy utterance map across distinct acoustic events, its whole-utterance supersession MUST remain blocked until a Central-authored compliant remedy exists. Do not run `source_promotion` as production write and do not alter default source, ASR model, Studio, translation, or SRT/ASS. Report explicitly which event proposals are eligible for further Central promotion, which remain provisional and why.

## D. One concise evidence commit and archive

Publish ONE sanitized new evidence root on the SAME draft PR #15, no new PR. Public content: exact source/PR identity, evidence table W02b ordinal100/101/final, W04a displaced source-unit coverage, W03b four candidate matching/audio status, true AV frame PTS verification IF executed, case-specific residual blocks, CPU validation reused with source SHA binding, command-shape friction, SHA256/bytes manifest. **No screenshots, WAVs, private endpoint/credentials, full copyrighted scripts or large media publicly.** Private archive: focused WAV hashes and bounded clips, source-unit table, listening notes and review hash, original vs candidate event ledger and provenance, optional timestamped small selected frame sequence with source PTS and hashes, command output and logs; avoid duplicating tar entries, exclude huge media/weights. Verify exact remote GitHub readback SHA/bytes, current PR head, source diff none, Issue OPEN, PR DRAFT.

Required handback:
 ISSUE14_TURN_ADJUDICATION:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<fullSHA>
   VALIDATED_SOURCE_SHA=<fullSHA>
   VALIDATED_SOURCE_TREE=<fullSHA>
   MAIN_SHA=<fullSHA>
   CURRENT_PR_HEAD=<fullSHA>
   SOURCE_DIFF=<none|paths>
   PREVIOUS_EVIDENCE_CHECKSUMS=<252/252 + prior public packet>
   CPU_GATE_AUTHORITY=<unchanged same-source 411/411 or exact reason rerun>
   W02B_ORDINAL_100_101=<same|different|UNRESOLVED, exact audio bounds, crop SHA, unit linkage>
   W02B_FINAL_TURN=<Voice_A|Voice_B|different_unknown|UNRESOLVED, direct audio criteria>
   W02B_TURN_COUNT=<supported count, not assumed 6>
   W04A_UTT_0187_0188=<acoustic event bounds, source unit ownership, SFX control, uncertain spans>
   W03B_SHIFTED_UNITS=<positive audio event/unit correspondences vs model-only conflicts>
   AV_PTS_CHECK=<executed exact original PTS vs omitted with rationale, no new model calls>
   RECONCILIATION_DOSSIER=<private artifact SHA, audited vs provisional, all-units-one-event gate>
   HUMAN_WER_CER_DER=UNVERIFIED
   PUBLIC_EVIDENCE_ROOT=<sha:path>
   PUBLIC_MANIFEST_SHA256=<sha>
   VERIFIED_PUBLIC_MEMBERS=<count>
   PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
   CENTRAL_INTEGRATION_DECISION_INPUT=<precise ready vs blocked source-turn candidates>
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|records>
   omitted_sensitive_values=<yes|no>

Return one complete result. If audio truly cannot distinguish ordinal 100 vs 101 or final speaker, leave unknown and complete W04a/W03b bounded evidence rather than inventing a positive resolution.