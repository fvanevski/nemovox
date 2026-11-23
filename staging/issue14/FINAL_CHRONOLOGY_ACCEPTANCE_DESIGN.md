# Issue #14: corrected acoustic source revisions and full French acceptance

Status: Central changes staged, HOST TESTS PENDING. Existing Issue #14 OPEN and PR #15 DRAFT. This is the last coordinated host qualification before Central independent review if gates pass.

## Previous evidence

User handback PARTIAL at 5b0d055c5cc8b409f4fa9edb3013b27e41490808. Public evidence: staging/issue14/evidence/20261123T050209Z/, manifest SHA256 ff04756c15a1ba4fecc832ed66dbbabfb6a60bc6e767976a4aef6301b1e9b624. Central independently verified 11 member hashes/sizes + manifest. Private archive issue14-integrated-source-fidelity-evidence-20261123T050209Z.tar.gz SHA256 c740218141e3c370d94f864daa54079e5e9ee25395bfd82a003ccde4a4da5e3c, size 1848600; Central inspected 65 members and verified all 57 private CHECKSUMS entries. CPU focused 89 pass/1 fail; full 393 pass/1 fail, sole failure was Mandarin model-only fixture retaining an unrelated French audio review. No source diff in host campaign.

Host source evidence: FR orig16 2051/2051 original lexical units matched, 2 complete same-event mappings utt-0193 (9/9) and utt-0206 (5/5), 1 partial utt-0208 (5/7), 1 displaced utt-0207. ZH EP166 2832/2901 matched, 69 unresolved from incomplete c14, 3 fully mapped utt-0063/0064/0065, partial utt-0062 (18/28), 41 untimed remain unrenderable. FR Riva baseline 234 utterances/309 cues; old unsafe 241/320, vetted 237/312, agent-attested corrected 239/314. Full ZH Riva English 285/304 baseline, vetted 285/305. Source WER/CER UNVERIFIED, no full independent human reference.

Critical timing correction: prior event W04c [1261.5,1263.5] is music-before-speech; actual shout reported near [1264.3,1266.5]. Prior H2 [362.5,364.2] is SFX-before-speech; actual phrase reported near [364.8,366.0]. Old WAV crops [1261.3,1263.7] SHA 9f8218629c0bcd0edf550c16e0a157b89d7104f48ce7376bbe4768c1ee805071 and [362.3,364.4] SHA f7f52e83d7996d52675958c392c0236fa8214dcb1a50ffa34c3495898aa84c86 DO NOT cover the corrected spoken intervals. Reuse neither as positive evidence for reanchored words. The corrected precise bounds/wording must be established by new listening.

## Central source changes

- tests/test_issue14_source_promotion.py now removes unrelated FR-reviewed evidence from the synthetic Mandarin model-only test. Production cross-language fail-closed validation remains intact.
- src/nvidia_transcribe/source_promotion.py now holds unreviewed model-only text that exactly normalizes to a nearby legacy source phrase (2.5s; H1 Qu'est-ce que c'est versus utt-0027). This is a provisional-acceptance quarantine, NOT a dedup/delete of original speech, and distinct reviewed real repetition survives.
- A reviewed event already partially attributed to legacy source units (W04c/utt-0208; potentially ZH utt-0062) now stays in source evidence but is not ALSO appended to rendered output until acoustic unit ownership can be fully resolved. Existing originals survive. No forced merge, guessed time, or blanket speaker assignment.
- An audio-reviewed source event overlapping legacy with no authenticated source-unit linkage is quarantined pending distinct voice or same-event disambiguation.
- Reviewed revision IDs MUST match canonical hash(media,event_id,text,review_sha,source_start,source_end). Same real event id stays stable under corrected bounds; new revision identity changes. Tampered stale reviewed text/time/digest fails.
- src/nvidia_transcribe/source_reconciliation.py accepts optional BUT COMPLETE trio review_audio_crop_start_s/end_s/sha256. If supplied, the audited event MUST fall entirely inside the auditioned WAV. Corrected W04c/H2 MUST provide NEW WAV bounds/SHA and a separately verifiable new listening record. The tool cannot inspect WAV bytes; host must verify the SHA of actual audio.
- Regression tests cover changed language fixture, stale review SHA, corrected bounds crop validity, H1 echo, W04c partial legacy, H2 overlap and distinct real speech preservation.

Upstream default process_media already invokes timing_evidence.py and boundary_evidence.py via chunking.py, with source/speaker review guards. Issue #14 requires proof via fresh complete 23-minute French default recording, not just an opt-in old-contract overlay.

## Required acceptance criteria

1. Full focused/non-GPU/static gates PASS at exact Central source SHA, no skips/bypass.
2. Fresh positive W04c/H2 audition evidence at correct bounds; no stale positive source revisions or audio crop provenance, CTC hypotheses and raw source remain intact.
3. FR and ZH raw-unit lineage, no deletion; explicit all-unit same-event supersession only. Partial W04c/ZH turn1 held if it would duplicate already-rendered speech. Preserve C2 legitimate repetitions, nearby distinct shouts, real simultaneous voices.
4. All original 22 indexed W/C/S French cases replayed and adjudicated. W05a six authentic source words from c14 vs missing c15; W02c/W03a/W03c duplicated spans, W04b/S03, H2, W04c, drift, S01/S02 source/translation polarity. C1 clean 29 tokens, negative song/SFX and low-margin speaker false merges.
5. One fresh full French primary-model source recording in isolated job root exercising default timing, boundary and speaker processing, followed by actual English Riva translation/SRT/ASS and same-denominator quality comparisons. If incomplete raw generation or unknown timing persists, report honestly.
6. Mandarin EP166 is a cross-language generalization check. Do not invent times for 41 untimed source turns or claim recovery of incomplete c14.
7. Preserve user media, historical ASR jobs, current host services, model resources and existing main checkout. No Issue #9 renderer rewrite, no unrelated model changes.
8. Central independent exact-head PR review, appropriately timed Codex review, operator-authorized merge, and merged-main/Studio acceptance remain later distinct gates. Unresolved residual cases must be explicitly assessed; no full WER/CER without independent reference.
