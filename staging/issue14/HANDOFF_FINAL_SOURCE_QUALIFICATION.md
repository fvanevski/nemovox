# Issue #14 / PR #15: residual source-event recovery and complete long-form qualification

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=c53f2c3ea4a35c013ed06d7c5d4d72c0e87d9b82
CENTRAL_SOURCE_TREE=69d80d1dc6dbe9fb7c072225932bd112828142cf
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
PRIOR_CLI_EVIDENCE_HEAD=c08412d8ea7f4d7a677c040a27a8e5730872e325
PRIOR_CLI_PRIVATE_SHA256=1d388add68b128b9dded43ea9b11d0432d17f47c2802e8d4d11ff1b363a96ff5
PRIOR_CLI_PUBLIC_ROOT=staging/issue14/evidence/20261123T204610Z/
CONTROLLING_DESIGN=staging/issue14/REVIEWED_RAW_UNIT_AMENDMENT_AND_FINAL_ACCEPTANCE.md
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_MODEL_DOWNLOAD=yes
NO_UNOWNED_GPU_EVICTION=yes
NO_PRODUCTION_SOURCE_EDIT_BY_LOCAL_AGENT=yes
NO_RAW_UNIT_DELETION=yes
NO_RENDERER_LANE_ARCHITECTURE_CHANGE=yes

## Goal and scope

ONE consolidated campaign: qualify the new Central review-gated source-unit amendment and alternate lineage, independently audit the remaining six plus two unassigned legacy units and four unowned c11 raw units in the three incomplete reviewed connected components, and perform the Issue #14-required full approximately 23-minute episode regression if the installed Nemovox-owned services safely admit it. Do not split this into serial tiny handoffs, create another PR, independently redesign ASR/diarization or chase unrelated models.

Use OpenCode Build as primary for an isolated worktree, local fixture execution, native media evidence, bounded owned GPU execution and one evidence publication. Central retains source architecture and substantive implementation ownership. Local may recover command shapes and ordinary environment mechanics that preserve behavior, test meaning, security and source authority. No semantic production-source changes or weakening guards.

## A. Fresh authority, prior proof and complete CPU gates

Fetch current authenticated PR HEAD before materializing a fresh collision-safe worktree. Verify PR OPEN/DRAFT, Issue OPEN, main, exact CENTRAL_SOURCE_SHA/TREE, ancestor and absence of later source/test change. Read AGENTS.md, controlling design, source_unit_lineage.py, source_reconciliation.py, evidence_fusion.py, reviewed_unit_amendment.py, source_fusion_application.py, existing CLI and tests. Previous private CLI archive SHA256 1d388add68b128b9dded43ea9b11d0432d17f47c2802e8d4d11ff1b363a96ff5 must pass 43/43 checksums and its public GitHub evidence root 10/10 blobs. Verify original default French archive/ledger, prior Riva receipts, W02b dossier and previous reviewed correction source by exact checksum, never path assumption. Correct source ledger member is default_fr_job/asr/source-evidence.v1.json, NOT phase_b/.../source_evidence.json.

With candidate worktree CWD, PYTHONPATH=src:. (test source and evaluations imports), both actual host interpreters:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_reviewed_unit_amendment.py tests/test_issue14_source_fusion_application.py tests/test_issue14_reviewed_revision_runner.py tests/test_issue14_evidence_fusion.py tests/test_issue14_source_unit_lineage.py tests/test_issue14_source_reconciliation.py tests/test_issue14_source_fusion_downstream.py tests/test_subtitles.py tests/test_translation.py
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.cli apply-reviewed-revision --help
Verify BOTH interpreters import candidate reviewed_unit_amendment, application and runner __file__ in isolated worktree. When tee is used set -o pipefail. No skipped or xfailed tests. Prior 441/441 host gates do NOT verify the new implementation. If source/tests fail, return the exact traceback and execute only further safe read-only steps; do not locally invent source fixes.

First replay unchanged original recording, original raw ledger and original dossier through plan_reviewed_source_partitions: 2008/2008 source legacy units, same hash 48ea4dfeea98815c20af587227aca415b3832a49a26490026425f11b8ded9ba1, six complete/three partially reviewed/204 unreviewed singleton components. Preserve the original 218-turn, 289-cue SRT/ASS accepted hashes; do not change old evidence.

## B. One bounded acoustic session for all three incomplete components

Use authenticated original episode audio and existing archived WAVs. Make minimal additional original-mix WAV crops with exact absolute media offsets and SHA. First direct-listen without transcript or screenshot hypotheses, then consult source IDs and CTC evidence; record both passes. No new ASR, pyannote, Whisper, VLM or long translation during this phase.

B1 W02b utt-0034: six missing raw owned unit IDs from 26. Distinguish w02b_ev_1 [212.05,215.55] from w02b_ev_2 [215.70,218.25] and contextual region ~209.8-219s. For EACH missing unit ID/index/text identify positively auditioned same event vs separate occurrence, SFX, or UNRESOLVED. Do not override audited later Toi?/Parfaitement! voice separation without new contradictory acoustic proof.

B2 W03b utt-0154 + utt-0156: two missing units from utt-0156. Re-audition [998.5,1012s] to distinguish w03b_interroges_eglises_repartant [999.25,1005.25] from w03b_savaient_viendrions [1009,1010.3], source unit word ownership and true rapid repeats.

B3 W03a utt-0147 + utt-0150: c11 ordinals 0..3 are UNOWNED raw observations with possible c10:75..78 equivalents. Prior positive WAV starts at 984s and DOES NOT cover ~980-984. Extract a NEW original audio review crop approximately 979-991s explicitly covering both the early competing CTC hypothesis and true ~985.75-989 event. Independently establish one or separate spoken occurrences. If unable to exclude a distinct earlier occurrence, leave all four ALTERNATE_UNVERIFIED, block the component and report why; do not promote the old echo hypothesis alone.

Bind every positive result to original raw unit ID/chunk/ordinal/parent raw-response SHA/ASR inference window, exact review crop media/time/SHA, event ID/source text, first-hand direct listening record/SHA, and explicit contrary evidence/unknowns. No deletion of original raw units.

## C. Newly staged generic amendment, source and translation replay

For ONLY positively justified actions create one private nemovox-reviewed-unit-amendment/v1 request with freshly computed exact full recording_content_sha256, ledger_content_sha256, dossier_content_sha256, media SHA/language and action objects:
  kind=assign_owned_unit OR kind=reviewed_alternate_observation
  source_unit_id=<exact immutable raw unit ID>
  audio_event_id=<existing independently reviewed acoustic event>
  review_method=direct_audio_listening
  same_audio_event_verified=true
  source_unit_identity_verified=true
  review_evidence_sha256=<actual new review evidence SHA>
  review_audio_crop_sha256=<actual review crop SHA>
  review_audio_crop_interval_s=[verified absolute start,end]
For reviewed_alternate_observation add preferred_owned_source_unit_id, competing_occurrence_window_s and separate_occurrence_excluded_by_listening=true only when true. The preferred unit must have been positively linked already in the ORIGINAL dossier, not introduced in the same request. Do not label any uncertain or unreviewed item as true.

Call nvidia_transcribe.reviewed_unit_amendment.apply_reviewed_unit_amendments directly as a CPU function, writing derived_dossier and report ONLY in new private scratch, and re-run full source_unit_lineage and plan_reviewed_source_partitions using that derived dossier. Require raw observation count unchanged, zero source text deletion, stable original input SHA, new dossier/plan SHA and truthful ready/block reasons. Test fail-closed cases: stale or cross-media input, duplicated unit, false direct review, unowned unit assigned as owned, borrowed reviewer evidence, insufficient crop/competing interval, wrong event, owner laundering, separate occurrence not excluded. If no valid action, retain original dossier and report NOT_ADMITTED without fabricating a derived review.

If any formerly blocked component is now genuinely complete, use existing source_fusion_application and reviewed-revision CLI with new exact plan/selection to generate new private reviewed turns, preserving reviewed alternate source observation lineage and unknown word-level timing. If the new turns need English translation, generate only their revision-bound receipts via one bounded admitted existing Riva worker pass. If unavailable, keep source-only qualification, no fake English or stale translation reuse. The SIX originally accepted components are a frozen exact-SHA regression control.

## D. One whole-episode post-source-change qualification and 22-case matrix

Issue #14 requires one complete approximately 23-minute episode regression at the candidate source in a NEW isolated job/output root after focused acceptance. Acquire only verified Nemovox-owned ModelLifecycle services and do not stop/evict unrelated GPU processes. If safely available, execute ONE complete episode source/subtitle run using authenticated original media and comparable settings, with full source response/manifest, timed/untimed source, English, SRT/ASS and GPU cleanup evidence. No checkpoint downloads, general model survey or repeated long-form reruns. If GPU/lease unavailable, complete CPU replays and mark FULL_EPISODE_GPU_REGRESSION=BLOCKED, not PASS.

Compare both ordinary default pipeline and opt-in reviewed correction explicitly against the original 20261122 234-utterance/308-ASS baseline and current accepted 218/289 revised track. For all 22 original indexed W/C/S cases report source coverage, displaced CTC vs positive audio, duplicates vs true repetitions, clean C1 29/29, C2 rapid repetition, W01 drift and W01c/W05a recovered six source words, W02c/W03a/W03c overlap, W04b/S03 core masking, W02b/W03b/S05 speaker turns, W04a SFX mismatch, S01/S02/W04c translation fidelity and polarity, Mandarin character-grain control if available. Report 18/22 CPS consistently with exact chosen duration denominator and 0 unverified human WER/CER/DER claims. Do NOT collapse source output regression and explicit evidence-based revisions into one success claim.

## E. One public/private packet, stop boundary

Publish ONE sanitized evidence-only root/commit on EXISTING draft PR #15, no source/test edits by local: candidate SHA/tree, gates, three residual ID-level decisions, new review crop SHA, derived dossier/input/plan SHA and preserved raw-unit ledger, corrected source/revision translation receipts, 22-case source and SRT metrics, full-episode execution/status and host cleanup, negatives and COMMAND_SHAPE_FRICTION. Private tar.gz self-contained manifest/sha with bounded WAVs, complete small contracts/decision records/test logs, no duplicate entries, secrets, full video, model weights or full copyrighted transcript published. Verify public remote blob SHA and exact PR head after commit.

Return:
 ISSUE14_FINAL_SOURCE_QUALIFICATION:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<fullsha>
   VALIDATED_SOURCE_SHA=c53f2c3ea4a35c013ed06d7c5d4d72c0e87d9b82
   VALIDATED_SOURCE_TREE=69d80d1dc6dbe9fb7c072225932bd112828142cf
   CURRENT_PR_HEAD=<fullsha>
   MAIN_SHA=<fullsha>
   SOURCE_TEST_DIFF=<none|exact paths>
   CPU_FOCUSED_CORE_STUDIO=<counts, exact imports>
   PRIOR_EVIDENCE_AUTH=<43/43 and 10/10, original hashes>
   THREE_RESIDUAL_COMPONENTS=<unit IDs, decisions, unresolved>
   RAW_C11_ALT=<early competing occurrence auditioned, one/two/unknown, crop/review SHA>
   REVIEWED_UNIT_AMENDMENT=<actions, original/derived dossier/plan SHA, no raw deletion>
   ELIGIBLE_COMPONENTS=<six original plus any newly supported>
   SOURCE_REVISION=<private SHA, unit lineage, speaker/time uncertainty>
   NEW_TRANSLATIONS=<only newly reviewed bound receipts or NOT_EXECUTED>
   ALL_22_CASES=<before/after and evidence matrix>
   FULL_EPISODE_GPU_REGRESSION=<EXECUTED|BLOCKED, run ID, same-denominator metrics, resource receipts>
   NEGATIVE_CONTROLS=<repeat, SFX, chronology, Mandarin, stale source>
   PUBLIC_EVIDENCE_ROOT=<head:path>
   MANIFEST_SHA256=<sha>
   PUBLIC_MEMBERS_VERIFIED=<count>
   PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
   HUMAN_WER_CER_DER=UNVERIFIED
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|records>
   omitted_sensitive_values=<yes|no>

Return ONE full handback rather than serial mini-reports. If evidence cannot support a previously blocked component, preserve its blocked status and complete all other safe work. Central will perform final independent exact-head issue-gate review; no merge/deployment/issue closure without operator authorization.
