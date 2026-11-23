# Issue #14: Generalized raw source-unit amendment and long-form acceptance

**Status:** Central source added to the existing draft PR #15 at 1f086156b4626b7bbd61d76b982cd9e90e42bcf6, tree be6b1d4c88baee2fa2019c8bdcc751c75de4e2aa. This NEW source has not passed host qualification. Issue #14 remains OPEN.

## Prior verified acceptance and authority

The uploaded issue14-reviewed-revision-cli-evidence-20261123T204610Z.tar.gz is SHA256 1d388add68b128b9dded43ea9b11d0432d17f47c2802e8d4d11ff1b363a96ff5, size 723861 bytes. All 43/43 checksum entries passed and 44 regular entries were unique. Ten public evidence files matched ten authenticated GitHub blob IDs on c08412d8ea7f4d7a677c040a27a8e5730872e325. Prior candidate source 32bf172d44f2f195b518edc2abd9cbba278c98f4 passed focused 73/73, core 441/441 and Studio 441/441, with no source/test difference in evidence-only commit.

A live first-class, explicit opt-in, model-free reviewed-revision CLI produced 218 reviewed recording/translated turns and 289 renderable subtitle cues, identical to the earlier accepted canonical source/English/SRT/ASS outputs. It selected six complete reviewed connected components, superseding eight legacy utterances with nine reviewed events. It preserved all 2,008 original legacy lexical units and kept SPK_UNK as global identity. Ten real CLI negative controls failed closed. This is valid source-application qualification, NOT all Issue #14 source-acoustic acceptance.

## Residual structural evidence gaps

1. W02b utt-0034: 20/26 source units positively assigned across audio events w02b_ev_1 [212.05,215.55] and w02b_ev_2 [215.70,218.25]; six source units require direct acoustic review.
2. W03a utt-0147 + utt-0150: those legacy units are covered, but four unowned raw c11 ordinals 0..3 are linked to the same reviewed event as candidate c10:75..78 words. Prior reviewed WAV starts at 984s, not early competing 980-984s. Same-occasion identity cannot yet be accepted despite a provisional echo association recorded in the old dossier.
3. W03b utt-0154 + utt-0156: 18/18 legacy units in the first and 1/3 in the second assigned. Two unassigned units remain amid different acoustic turns near 999-1010s.

The other 204 historical singletons are simply unreviewed; the CLI reports them together with the three partial components as 207 non-ready IDs. That is not 207 discovered new errors.

## Central implementation

- NEW src/nvidia_transcribe/reviewed_unit_amendment.py: pure CPU function apply_reviewed_unit_amendments(original_recording, raw_ledger, original_dossier, review_request) returns NEW dossier plus report. Input schema nemovox-reviewed-unit-amendment/v1 binds the exact three canonical object digests, source media SHA and language, original ASR raw-response SHA and inference window, legacy-unit ownership, existing direct-audio-reviewed event and source turn.
- Two general types of directly reviewed evidence:
  (a) assign_owned_unit connects a previously unlinked legacy-owned source unit to one positively auditioned event; never moves a previously assigned unit.
  (b) reviewed_alternate_observation moves an *unowned* raw observation from output-unit participation to explicit same-occurrence alternate evidence, only if an independently listened acoustic crop covers BOTH the reviewed speech event and competing occurrence window and explicitly rules out a separate repeat. Its preferred owner must already have been positively linked in the original dossier; one action cannot promote another proposed assignment to bypass this.
- All raw observations, response identities, original contracts and dossiers remain unchanged. Each accepted action carries independently listened crop SHA/bounds, review evidence SHA, original parent inference SHA and source unit ID. The derived dossier gets a new exact SHA. The existing legacy source-lineage verifier is applied BEFORE and AFTER amendment. The module does not run inference, assign speaker characters, invent word timings, mutate production, discard raw words or automatically select any correction.
- src/nvidia_transcribe/source_fusion_application.py conditionally embeds reviewed alternate raw-observation metadata in accepted event turns while never rendering those raw alternatives as second subtitles. The original six-component dossier (no alternates) retains its previous shape and is subject to replay hash verification.
- NEW/updated tests: tests/test_issue14_reviewed_unit_amendment.py and tests/test_issue14_source_fusion_application.py cover safe assignment and alternate review, stale/repeat/false-review/incorrect-owner and ASR window refusals, and preservation of raw lineage. Host validation is still required.

## Evidence-vs-judgment architecture

An acoustic observation, ASR lexical hypothesis, diarization, speaker voice evidence, visual figure evidence, VLM assessment and reviewed source correction remain distinct claim types. Context-sensitive model/heuristic relevance is tunable; strict source-unit ownership, original inference input coverage and real audio-review claims are factual acceptance preconditions. There is NO fixed global ranking of audio versus visual cues, and NO policy allowing a model's fluent statement or lexical similarity alone to remove a genuine repeated spoken event. If evidence remains insufficient the appropriate output is unresolved/abstain.

## Final acceptance gate

The governing live Issue #14 explicitly requires full nonGPU tests, complete evidence for the 22 indexed cases, negative controls, and ONE final approximately 23-minute episode regression in a NEW isolated job/output root after focused source improvement. It also separates a late independent exact-head PR review, operator merge approval and post-merge/live Studio acceptance.

The next host campaign combines the three residual acoustic reviews and potential derived dossier with one complete full-episode regression. Do not repeat unnecessary Riva/ASR/VLM work outside the specifically required full episode and tiny newly reviewed turns. The six already accepted components are baseline controls. If any missing unit or alternate repeat remains uncertain, preserve that component as blocked rather than working around source-integrity guards.

Issue #9 owns ASS lane scheduling/readability; no global renderer changes, unrelated Studio feature work, model retraining or model surveys belong here. Do not merge PR #15 or close Issue #14 based solely on an offline CLI proof.
