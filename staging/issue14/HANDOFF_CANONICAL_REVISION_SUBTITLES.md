# Issue #14 / PR #15: verify stable source identity and canonical revised SRT/ASS

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=59a1b6a07613c1d7901f549af9c61db1d5db8696
CENTRAL_SOURCE_TREE=14aabba2a241fdec636970ffb6a2a8bc07fff82e
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CONTROLLING_AUDIT=staging/issue14/AUDIT_FUSION_DOWNSTREAM_20261123.md
PRIOR_EVIDENCE_COMMIT=a6eadf27e83e3e50df7b5be574cb10199d859c18
PRIOR_PUBLIC_ROOT=staging/issue14/evidence/20261123T195952Z/
PRIOR_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-downstream-subtitles-private/issue14-downstream-subtitles-evidence-20261123T195952Z.tar.gz
PRIOR_PRIVATE_SHA256=d171504e8cfd361351348406b84f596973b3971e7692e15e4ad7f554e650e467
ORIGINAL_DEFAULT_ARCHIVE=/workspace/nemovox-issue14-default-recovery-private/issue14-default-source-recovery-evidence-20261123T174459Z.tar.gz
NO_NEW_PR=yes
NO_MERGE=yes
NO_ISSUE_CLOSE=yes
NO_DEPLOY=yes
NO_NEW_ASR=yes
NO_NEW_RIVA=yes
NO_NEW_VISION_RUN=yes
NO_LIVE_DATA_MUTATION=yes

## Objective and scope

One consolidated, focused CPU-only validation that the CENTRAL root-cause fix in source_fusion_downstream.py safely rejoins historically renumbered translated rows by immutable original_utterance_id, retains nine existing Riva translation receipts, and creates a CANONICAL isolated 218-utterance English contract and SRT/ASS without manually rekeying upstream data. Do not implement per-case fixes, relax lineage, silently delete unknown CTC observations, or redo earlier GPU exploration. A green diagnostic rekeyed contract is not enough; call the actual fixed production function on the two original untouched contracts.

## A. Rebind exact authority and verify CPU tests

Use a fresh isolated OpenCode Build worktree from authenticated current PR #15 HEAD. Confirm PR OPEN/DRAFT, Issue #14 OPEN, current main SHA; verify CENTRAL_SOURCE_SHA is an ancestor and latest source/test change and its tree matches, and no source/tests changes occurred after it. Fetch remote ref before local object proof, and do not disturb any operator checkout. Read AGENTS.md, CONTROLLING_AUDIT, utterances.py, source_fusion_downstream.py and tests. Authenticate previous archive SHA with 34/34 unique member checksums; previous public GitHub manifest SHA256 7c8cbde661d9a1914d9ab44ebcf57353e8aeed22422d36478e9dbd7c0417009b and all 10 remote public members. Verify the prior default archive/checksums and only extract required exact files into NEW private scratch if host file paths have moved. Preserve all original artifacts.

Use exact candidate imports with PYTHONPATH=src:.:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_source_fusion_downstream.py tests/test_issue14_source_fusion_application.py tests/test_issue14_evidence_fusion.py tests/test_issue14_source_unit_lineage.py tests/test_issue14_source_promotion.py tests/test_issue14_source_reconciliation.py
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check
Check that both interpreters resolve nvidia_transcribe.source_fusion_downstream to this worktree. Do not weaken tests/validation or mark any previously failed test xfail. Actual counts come from results, not expected numbers. If an existing command/path is unavailable, inspect and use equivalent repo-approved invocation and report exact friction.

## B. Prove historical bijection using EXACT original contracts

Authenticate original recording.contract.json SHA256 a67056da45130ae59bed594857372023c11e3c18a5f5b9bc85e167457b3b550e and translated.contract.json SHA256 9422b3861743be135bf91baca40b43ef6de39b662f7032d78a1322b744df94fd. Confirm media fingerprint and 217 original source IDs; verify each translated row's original_utterance_id is unique, exists in original source and its immutable source text/start/end/speaker fields agree. Expected prior historical renumbering: five display/source ID mismatches in 0148..0152; if actual evidence differs, diagnose exact discrepancy. No source text mutation, no edited input JSON or external id-rekeying during canonical validation.

Authenticate private_revised_recording.json SHA256 dd1a497dad70d4b8bd047adc3f0306f5eed45b69fc972e2e6849130915b02900, plus the prior local-acoustic review and nine riva_9turn_receipts.json from PRIOR_PRIVATE_ARCHIVE, with their internal SHA/metadata provenance. Read section logs/review notes only to recover exact file identity, not as sole proof. Rebuild verified local-voice overlay if needed using Central attach_local_acoustic_voices; keep voice_0 / voice_1 scoped to direct audio reviews and all nine new global speaker_id=SPK_UNK; unknown breath remains unlabeled.

## C. Canonical assembly with zero new model inference

On ORIGINAL untouched original recording and original translated contracts, and the authenticated voiced private revised source, use Central assemble_revised_translation(original_recording, previous_translated, revised_with_local_voices, nine_receipts, enable=False), then enable=True. **Do not** run diagnostic rekey script or replace translated row IDs in caller input.

Expected: preview exactly nine new source-revision IDs; actual canonical output 218 turns: 209 original retained and nine reviewed new turns. All 209 retained original source text/start/end/speaker and English translation must equal exactly the corresponding *stable* original_utterance_id English row, not the old renumbered display ID. Expect five prior translated display/source ID differences, with two among retained source turns utt-0150/0151. Verify no active grouped_utterance_ids, merged_into, or probable_duplicate_of points to a superseded legacy source; any historical duplicate reference invalidated by revision must be explicitly preserved in historical metadata only. Verify historical ID map SHA receipt, unchanged original input JSON hashes, unique output source IDs, no dropped/unattributed original lexical units, and new turns retain correct event time, unknown global voice and unverified word alignment.

Run negative controls in existing tests plus one private replay: duplicate original_utterance_id, swapped alias lacking positive source equality, group reference to superseded original, stale fresh Riva revision ID/translation receipt, and wrong original recording SHA must fail closed. Do not change policies to make a failing example green.

## D. Private canonical render and user-relevant comparison

Write canonical translated contract to NEW private scratch root; use existing subtitles.render_subtitles for one bounded private SRT/ASS. Run existing render logic, NOT generate_subtitles re-render_only (which may load prior translated cache), and do not write in the user's media library. Compare to original baseline and archived non-canonical diagnostic, examining W02b fast child exchange and long turn, W03b reviewed acoustic events, W04a SFX removal and actual speech at 1231-1234s, S02 polarity, W05a source survival, C2 distinct repeats, S05 and offscreen/music controls. Report actual cue counts/18 and 22 CPS/ASS lane and overflow/unrenderables with matching denominators, no grouping across superseded IDs, local voice evidence not mistaken for global speaker tags, and verified bounds. If the result differs from the diagnostic, explain from exact source IDs and render behavior rather than adjusting output post hoc. Report unnatural English loanword choices as glossary/review tuning candidates; do not quietly replace Riva text or claim post-edit was performed. No need for new Riva/vision/ASR.

The c10:75..78/c11:0..3 raw overlap remains proposed/unresolved; do not silently dispose c11 observations or mark the three incomplete components ready. A new bounded acoustic review is NOT required for this identity regression unless previous evidence is contradictory. Preserve human WER/CER/DER=UNVERIFIED.

## E. Single evidence packet and finish

Publish ONE sanitized evidence-only commit to existing draft PR #15 with source SHA/tree, CPU gates, 217-row prior identity map diagnostics, exact canonical assembly receipt, 209/209 retained English comparison, nine existing receipt checks, private render comparison and case controls, outstanding blocked source components and translation quality notes, runtime commands/friction and SHA256+bytes manifest. Keep full 218-turn contract, translated contract, private bounded SRT/ASS, detailed hash/case logs and provenance in one NEW self-contained tar.gz with unique files/checksums, no full video/media/model weights or credentials. Verify all public members against remote GitHub exact head and source diff none. PR stays DRAFT, issue OPEN; do not merge, deploy, close or request release.

Return:
 ISSUE14_CANONICAL_REVISION_SUBTITLES:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<fullSHA>
   VALIDATED_SOURCE_SHA=59a1b6a07613c1d7901f549af9c61db1d5db8696
   VALIDATED_SOURCE_TREE=14aabba2a241fdec636970ffb6a2a8bc07fff82e
   CURRENT_PR_HEAD=<fullSHA>
   MAIN_SHA=<fullSHA>
   SOURCE_DIFF=<none|exact paths>
   CPU_FOCUSED_FULL_STUDIO=<counts and worktree __file__>
   INPUT_AUTH=<archive/translated/recording/Riva source-revision SHA>
   PRIOR_IDENTITY_REBIND=<217 bijection, five renumbered, 0 stable source mismatches>
   CANONICAL_ASSEMBLY=<actual call, 209 retained English/source exact, 9 fresh reviewed, output digest>
   STALE_ID_GATES=<group, merged, probable_duplicate, duplicate alias, false source match>
   PRIVATE_CANONICAL_SRT_ASS=<counts, 18/22 CPS, ASS lanes, case-level comparisons>
   LOCAL_VOICES=<scoped W02b metadata, global unknown, uncertain breath>
   REMAINING_BLOCKS=<3 incomplete components, c11 raw observation, word timing and MT review>
   PUBLIC_EVIDENCE_ROOT=<exact head:path>
   PUBLIC_MANIFEST_SHA256=<sha256>
   PUBLIC_MEMBERS_VERIFIED=<count>
   PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
   HUMAN_WER_CER_DER=UNVERIFIED
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|canonical>
   omitted_sensitive_values=<yes|no>

Return one complete handback, not sequential micro-reports. If the real canonical assembly still fails, preserve the exact repro and return PARTIAL, with no local semantic source edits. Ordinary local env/command mechanics within this scope may be fixed by Build without Central reauthorization.