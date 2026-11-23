# Issue #14 / PR #15: qualify configurable evidence fusion and general source-event partitions

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=420173025dd28837adc91dfdb970227f3795124e
CENTRAL_SOURCE_TREE=3c98b4f72ab7d68924abcb3a5d1fe8fec2fc42b8
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CONTROLLING_DESIGN=staging/issue14/EVIDENCE_FUSION_ARCHITECTURE.md
PREVIOUS_PUBLIC_EVIDENCE=1f441c1171d6b0a0e566fcba3deecba3f074b845:staging/issue14/evidence/20261123T185939Z/
PREVIOUS_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-w02b-adjudication-private/issue14-w02b-adjudication-evidence-20261123T185939Z.tar.gz
PREVIOUS_ARCHIVE_SHA256=6c5602c1dcce81603f701f8ff3a82d5dc21b619d20f308947ecac35d94e11d73
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_CHANGE_TO_EXISTING_SOURCE_OR_TEST_MEANING=yes
NO_AUTOMATIC_SOURCE_PROMOTION=yes
NO_MODEL_DOWNLOAD_OR_FULL_EPISODE_RUN=yes

## Goal

Qualify ONE Central-authored, production-disabled CPU evidence-fusion scaffold as an extensible mechanism for combining original ASR, CTC, speaker diarization, direct-audio observation and limited visual evidence. Reproduce exact lineage with the new general event partition plan on the real French W02b/W03b/W04a dossier, then exercise the Central-authored **opt-in in-memory reviewed-unit application adapter** on only complete connected components using a new private root. Verify source and speaker fidelity without W02b/W03b/W04a case-specific patches. Do not silently patch specific phrases or bypass original all-units-one-event production gate.

## A. Source authority and complete CPU gates (isolated worktree)

Start a fresh OpenCode Build harness and isolated worktree of the EXACT current PR branch. Verify current main, PR draft/head, Issue OPEN and that CENTRAL_SOURCE_SHA/tree are the latest source/test change; no intervening source/test diff. Read AGENTS.md, design, source_evidence / source_unit_lineage / source_reconciliation / source_promotion, and both new evidence_fusion.py and source_fusion_application.py + all associated tests/config. Previous archive exact SHA and 94/94 checksums, public manifest SHA and all 8/8 members; earlier 411/411 host tests are historical only.

With candidate worktree as CWD and PYTHONPATH=src:. (the known correct worktree + evaluations import path), use the existing host interpreters:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_evidence_fusion.py tests/test_issue14_source_fusion_application.py tests/test_issue14_source_challenger_timing.py tests/test_issue14_source_unit_lineage.py tests/test_issue14_source_promotion.py tests/test_issue14_source_reconciliation.py
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check
Use only actual discovered test paths, report any unavailable test rather than silently skipping. Show import __file__ for new module in BOTH interpreters, not main checkout. The Central standalone 7/7 tests are a local authoring receipt, not these host gates. No changed test skip, weakening or semantic source policy. Diagnose and return exact failure if present; mechanical environment repair is allowed only when it leaves production source, tests, dependency and acceptance meaning unchanged.

## B. Reconstruct source event partitions (private CPU)

Use the existing matching original French 16-chunk default job from /workspace/nemovox-issue14-default-recovery-private/default_fr_job/asr/{recording.contract.json,source-evidence.v1.json}, and the NEW /workspace/nemovox-issue14-w02b-adjudication-private/dossier/candidate_reconciliation_dossier.json. Match identical media fingerprint/language/chunk raw SHA and original inference-window bounds; do not join distinct run IDs. Verify the original private source file identity against prior manifest before calculating new partitions.

Invoke nvidia_transcribe.evidence_fusion.plan_reviewed_source_partitions(recording, ledger, dossier) as an OFFLINE Python call, write to a NEW private scratch file (do not overwrite host jobs). Verify 2008/2008 original units and no duplicate source IDs, classify all 13 review-attested turns and compare with prior 5 eligible / 8 blocked legacy review. Do not force these exact counts as targets if actual lineage reports differ, explain with exact IDs. Validate **entire reviewed-event connected components**: if an event spans two legacy utterances and one is only partly accounted for, neither is eligible for a replacement. Reject a fused-model support claim whose dependency ancestry includes a policy-excluded observation. W02b utt-0036, W04a utt-0187/0188 and W03b utt-0154/0156 must receive general one-to-many coverage/unknown status from their raw-unit memberships, never use case-name selection logic. Report connected/partially connected event groups, unsupported units and whether any initial plan falsely claims completeness, especially source units shared by different ASR chunks. Run source_fusion_application.build_reviewed_source_revision against the authenticated identical recording/ledger/dossier with expected_plan_sha256 computed from the freshly returned partition plan. First run enable=False and require new_recording=None. Then run enable=True **ONLY for complete connected components** and keep the resulting new v1 recording entirely in a private isolated scratch output, NOT any active default job/Studio path; if zero components qualify, report that rather than bypass. Check exact original recording digest and deep equality, no raw lexical observations dropped, no duplicated selected unit IDs, reviewed-event count and source text/revision identities, source event bounds inside parent inference WAV, original unselected turns preserved, SPK_UNK for new turns and no word-level time invented. Repeat a blocked incomplete component selection to verify fail-closed. Preserve existing source_promotion's legacy all-units-one-event and ±3s policy unchanged.

Read code closely: source_unit_lineage's existing all-units-one-reviewed-event and ±3s guard remains valid as a gate for its OLD whole-utterance strategy. The new planner is a different dry-run representation, not authority to disable those checks in source_promotion. Do NOT modify original source contract, translate, render or apply new speaker IDs. If an input artifact lacks a required field, return a minimal schema repro to Central; do not fabricate missing lexical IDs.

## C. Evidence-fusion cases and policy tuning (CPU first)

Build a small private input-case matrix from *already admitted evidence*:
- French W02b [212-236]: two acoustically distinct child voices, ordinals100/101 rapid exchange and ambiguous breath, one Nemotron and one Whisper merged speaker result, visual shot/mouth observations with exact post-review decoder PTS.
- W04a: original SFX/rockfall, independently heard actual speech and misleading CTC estimates, two old utterance/source event cross-boundaries.
- W03b: four displaced lexical correspondences, correctly located c10 overlaps, genuine duplicate vs independent acoustic event alternatives.
- Negative controls: genuine C2 rapid repetition/1327-1329s separate shouts, reaction/offscreen silent person, sound-only instrumental/music false caption and a ZH character-level control. Keep each actual observed claim, uncertainty and SHA.
- Context control: one synthetic live-action scene with verified synchronization compared to the same abstract observations under animated or unverified-sync policy. Do not pretend to have watched real live-action reference material.

Construct nemovox-evidence-case/v1 per case with exact media/window, claim types, root source families and derived_from edges. Acoustic model observations all ultimately share original_audio and visual-only original_video; a fused result depends on both and is NOT a third independent witness. Native audio audits may be distinct observation methods, not genuinely statistically independent raw acoustic evidence. Assign claim-specific applicability using configs/evidence-fusion.example.json and a versioned variant; avoid tuning to a fixed case answer. Verify generic decisions/counterevidence/abstain and policy hashes change when rules/prompts change. A 'support' label still has correction_application=not_authorized_evidence_review_only and must not alter record.

OPTIONAL: If the already configured Qwen vision/text resource safely fits within explicit ModelLifecycle ownership, use at most 2–4 short model-review calls with the serialized review-request and structured returned response to test prompt suitability. Use opaque scene IDs, no source label leakage or fresh image extraction; reuse prior observations/frames privately as needed. Pin exact prompt/profile/model, respect existing host model service and GPU leases. If model unavailable, mark NOT_EXECUTED without compromising CPU campaign. No new full episode, model survey or pyannote/Whisper inference.

## D. One publication and ready-to-implement Central handback

Publish ONE sanitized evidence-only root on the existing PR #15 with CPU/static outcomes, source import identity, complete partition and **reversible application preview/new-contract** diagnostics (including failures and blocked components), model/policy assessment examples and negative controls, recommendations for generic unit-partition-to-source-revision adapter, command-shape friction and SHA256/byte manifest. No proprietary full media, transcripts, screenshot contact sheets or token/endpoint details in public GitHub. Keep full case JSON/model replies, selected event-unit groups/ledger hashes, policy variants and execution logs in ONE self-contained private tar with unique filenames and checksums, no model weights/large media. Verify GitHub public member readback and exact branch HEAD, source file hashes, PR draft/Issue OPEN.

Required handback:
 ISSUE14_EVIDENCE_FUSION:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<fullSHA>
   VALIDATED_SOURCE_SHA=420173025dd28837adc91dfdb970227f3795124e
   VALIDATED_SOURCE_TREE=3c98b4f72ab7d68924abcb3a5d1fe8fec2fc42b8
   CURRENT_PR_HEAD=<fullSHA>
   MAIN_SHA=<fullSHA>
   SOURCE_DIFF=<none|paths/diffs>
   CPU_FOCUSED_FULL_STUDIO=<exit/count/actual candidate import>
   PRIOR_EVIDENCE_AUTH=<archive, public manifest, source raw SHA>
   FUSION_CASES=<supported, conflicted, excluded, abstain, root-family dependency, policy digests>
   SCENE_CONTEXT_TUNING=<animation vs live action synthetic, sync status, policy diffs and calibration limits>
   FUSION_MODEL_PROBE=<executed/not_executed, hosted model revision/prompt/correct parsing/gpu nonmutation>
   SOURCE_UNIT_PARTITIONS=<2008 source units coverage, reviewed events, complete/partial/unknown, multi-event groups>
   OPT_IN_SOURCE_REVISION=<preview/apply on complete private components, source/ledger/dossier SHA, new event vs original unit accounting, reversible contract SHA, SPK_UNK/event bounds and blocked control>
   APPLICATION_GATE_GAPS=<what generic connector must validate before acceptance; no automatic promotion>
   NEGATIVE_CONTROLS=<C2, SFX, speech hallucination, language-character and unknown voice>
   PUBLIC_EVIDENCE_ROOT=<sha:path>
   MANIFEST_SHA256=<sha>
   PUBLIC_MEMBERS_VERIFIED=<count>
   PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
   HUMAN_WER_CER_DER=UNVERIFIED unless actual independent reference
   NEXT_CENTRAL_SOURCE_ADAPTER=<actual adapter qualification outcome, remaining production opt-in integration/renderer acceptance and sample ready/blocked components>
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|canonical>
   omitted_sensitive_values=<yes|no>

If a real source semantic bug is found, preserve exact repro and report it to Central; do not locally invent replacement architecture, weaken validators or edit production source under a mechanical label. Complete this campaign in one handback rather than serial micro-handoffs.