# Issue #14 / PR #15: qualify explicit reviewed-revision CLI on real accepted French evidence

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=efc5016790c6e3ae99ac7c58aeb32e52a7a1307f
CENTRAL_SOURCE_TREE=08629e6f6f988bc46d710b5c20dcaaf0bba52f71
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
PRIOR_PUBLIC_EVIDENCE=40a9c97e85ee5f74f76f51ef1bd57aafe339198d:staging/issue14/evidence/20261123T202749Z/
PRIOR_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-canonical-revision-private/issue14-canonical-revision-evidence-20261123T202749Z.tar.gz
PRIOR_PRIVATE_SHA256=a698df447d63c7bf56c55db5000f5d95ef19947993e143548a7b36c5dfff8f7a
PRIOR_DEFAULT_ARCHIVE=/workspace/nemovox-issue14-default-recovery-private/issue14-default-source-recovery-evidence-20261123T174459Z.tar.gz
PRIOR_FUSION_ARCHIVE=/workspace/nemovox-issue14-evidence-fusion-private/issue14-evidence-fusion-evidence-20261123T194055Z.tar.gz
PRIOR_DOWNSTREAM_ARCHIVE=/workspace/nemovox-issue14-downstream-subtitles-private/issue14-downstream-subtitles-evidence-20261123T195952Z.tar.gz
CONTROLLING_AUDIT=staging/issue14/AUDIT_CANONICAL_TO_REVIEWED_CLI.md
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_NEW_ASR_RIVA_VLM_PYANNOTE=yes
NO_MODEL_DOWNLOAD=yes
NO_MAIN_JOB_OR_SIDECAR_WRITES=yes
NO_PRODUCTION_SOURCE_CHANGE_BY_LOCAL_AGENT=yes

## Goal

Qualify the CENTRAL-authored new **nvidia-transcribe apply-reviewed-revision** CLI on the exact original French 217-row recording/English contract, raw source-evidence ledger, 13 reviewed acoustic events, nine real previously captured Riva receipts and the six previously eligible source-unit connected components. Produce a new private 218-turn translated contract and SRT/ASS that agree with already accepted canonical outputs. This is ONE bounded CPU-only pipeline-integration acceptance; no further exploratory model campaigns, case-specific repairs or micro-handoffs.

## A. Authority, files and tests

Fresh isolated OpenCode Build worktree at live authenticated PR #15 head, verify issue OPEN, PR DRAFT, main SHA, source commit/tree and no src/tests change after source SHA. Read AGENTS.md, controlling audit, runner + CLI + tests, existing source_fusion_application, source_fusion_downstream and source_unit_lineage. Verify uploaded canonical private archive SHA and 25/25 checksums, 8/8 prior public members against GitHub; authenticate default archive SHA 1b98ad231c3a15e98da41c76a9056d636f49a07819e298cbd1b55c46663aac4f (56/56), fusion archive SHA 16ef680e419cc2bbe71fa9607e17a6c6323a104b7e23b468e13b5a915a7bc224 (28/28), and downstream archive SHA d171504e8cfd361351348406b84f596973b3971e7692e15e4ad7f554e650e467 (34/34) as needed. Extract only required original recording / translated / ledger, dossier, approved components, local voice review, prior Riva receipt list and canonical output hashes into one new isolated private root, with exact SHA comparisons; never assume a file path still exists if an authenticated archive can supply it. Do not modify /workspace/pyannote, source jobs, models, active user services or operator checkout.

Run candidate-bound commands under BOTH host interpreters, CWD fresh worktree with PYTHONPATH=src:.:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_reviewed_revision_runner.py tests/test_issue14_source_fusion_downstream.py tests/test_issue14_source_fusion_application.py tests/test_issue14_evidence_fusion.py tests/test_issue14_source_unit_lineage.py tests/test_issue14_source_reconciliation.py tests/test_subtitles.py tests/test_translation.py
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.cli apply-reviewed-revision --help
Prove candidate __file__ import paths for runner/application/downstream in both interpreters. Run command with Bash pipefail when tee is used; record actual test counts and failures, no skipping/xfail/weakening.

## B. Explicit six-component selection (not automatic acceptance)

Recompute plan_reviewed_source_partitions on identical 217-row original recording, source-evidence ledger, and candidate_reconciliation_dossier (2008/2008 original units), and compare SHA to prior plan_sha256 48ea4dfeea98815c20af587227aca415b3832a49a26490026425f11b8ded9ba1. Confirm 6 complete components and 3 blocked; list all component IDs with old-unit and event memberships, confirm 8 original legacy IDs superseded and 9 reviewed events. If any identity differs, STOP application and report PARTIAL with trace; do not select new events or tune logic to force the expected count.

Construct NEW private nemovox-reviewed-component-selection/v1 JSON from only those six approved ready IDs, expected_plan_sha256 and freshly computed canonical _digest of original recording, ledger, dossier:
  {"schema":"nemovox-reviewed-component-selection/v1",
   "selected_component_ids":[... six actual approved ready IDs ...],
   "expected_plan_sha256":"48ea4dfe...",
   "recording_content_sha256":"48c0e1cb...",
   "ledger_content_sha256":"8c0edd52...",
   "dossier_content_sha256":"e6371203..."}
The ellipses in this handoff are EXPLANATORY, not literal JSON or invented values. The actual source SHA must match the complete 64-char canonical hash, not the prefix; take it from the already authenticated plan and calculate locally. Never mark the three incomplete components ready, auto-delete c11:0..3, assert global speaker IDs, or invent word-level timestamps.

## C. End-to-end offline CLI validation

First invoke **read-only** preview:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.cli apply-reviewed-revision --original-recording <AUTH_REC> --source-evidence <AUTH_LEDGER> --review-dossier <AUTH_DOSSIER> --selection <NEW_APPROVED_SELECTION>
Check CLI exit 0, no new output directory, selection digests, six eligible IDs, explicit blocked components, no models loaded.

Then invoke **opt-in private application** using the SAME four inputs plus:
  --previous-translated <AUTH_ORIGINAL_TRANSLATED>
  --translation-receipts <AUTH_9TURN_RIVA_RECEIPTS_JSON>
  --local-voice-review <AUTH_LOCAL_VOICE_REVIEW>
  --output-dir <NEW_NONEXISTENT_PRIVATE_OUTPUT>
  --episode-stem fr_canonical --apply
Output MUST contain reviewed.recording.contract.json, reviewed.translated.contract.json, reviewed.revision.report.json, fr_canonical.en.srt, fr_canonical.en.ass with authenticated input/output digests. No output under user media/library or historic jobs. Receipt JSON from archived contracts/riva_9turn_receipts.json has {"receipts":[...]} and nine rows. Verify original input file SHA unchanged before/after; service ownership/GPU occupancy unchanged (no endpoint access or GPU lease).

Expected exact accepted corpus outcome if all identical inputs/config: reviewed recording and translated contract contain 218 utterances = 209 retained + 9 new, source-unit lineage unaffected, SPK_UNK new global speakers, 9/9 Riva receipts reused, prior translation aliases normalized, no active superseded refs, 289 renderable cues, SRT SHA256 b0fb66ed937c1fd0168a8a33cef8432be8e0830972b24752f565d6e767edb007 and ASS SHA256 2ac26cb94315ee33e452971be310056e76b87a8af0f618e5e9d23097d5d3c779 when stem fr_canonical and default config are unchanged. Compare *actual content* W02b (rapid child voices), W03b (collapsed source overlap), W04a (no SFX text replacing real speech), S02, W05a, C2, S05; preserve prior diagnostic and canonical human WER/CER/DER=UNVERIFIED.

## D. Fail-closed CLI negatives and remaining boundaries

Test in NEW scratch names only: stale plan SHA, one blocked component, duplicate selection ID, stale dossier/recording digest, missing one of nine translation receipts, intentionally invalid local voice event identity, output_dir already existing, unsafe filename stem and malformed media identity. Each must fail and leave original inputs plus prior outputs untouched; for input failure no new published output directory. If staging/render fails, preserve failure log but no accepted partial result. No new GPU inference. If a true invariant failure arises, preserve exact traceback/repro and return PARTIAL to Central rather than modifying source or weakening gates.

Produce read-only residual dossier summary for three blocked components (utt-0034; utt-0147+0150; utt-0154+0156), focusing on what raw-unit evidence is missing, not new ASR or extra acoustic campaigns. Distinguish raw c11:0..3 from a verified duplicate; no source deletion. Report local voice metadata and whether new ASS preserves usable local alternation (the renderer still uses unknown global identities). Riva term choices 'nacale' and 'langage mu' remain translation quality tuning candidates; no hard-coded source substitutions.

## E. One evidence commit and handback

Publish ONE sanitized evidence-only commit under SAME draft PR #15 with source/test SHA binding, CPU/core/Studio exit/count receipts, accepted plan identity, CLI preview/apply and output digests, 209/9 translation accounting, canonical SRT/ASS SHA comparison, blocked-component report, safety negatives, host GPU/service nonmutation, command-shape friction, and SHA256/byte public manifest. Put complete private source/translation/render outputs, selection, exact commands and logs in ONE self-contained private tar.gz with checksums and unique members; no full copyrighted media, weights or credentials. Verify GitHub exact-head and all public member digests after push. No new PR, merge, deployment, issue close, broad research or additional handoffs.

Return:
 ISSUE14_REVIEWED_REVISION_CLI:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<full SHA>
   VALIDATED_SOURCE_SHA=efc5016790c6e3ae99ac7c58aeb32e52a7a1307f
   VALIDATED_SOURCE_TREE=08629e6f6f988bc46d710b5c20dcaaf0bba52f71
   CURRENT_PR_HEAD=<full SHA>
   MAIN_SHA=<full SHA>
   SOURCE_DIFF=<none|paths>
   CPU_FOCUSED_FULL_STUDIO=<exit counts and import paths>
   SOURCE_AND_PRIOR_EVIDENCE_AUTH=<25/25 and 8/8, original file digests>
   SELECTION_AND_PLAN=<full SHA digests, six complete/three blocked, 2008 unit identity>
   CLI_PREVIEW=<exit and no-write proof>
   CLI_APPLY=<exit, private output path, complete report/digests>
   CANONICAL_COMPARISON=<218 source, 209 retained, nine new, SRT/ASS byte sha, 289 cues>
   LOCAL_ACOUSTIC_VOICE_STATUS=<scoped labels, SPK_UNK global>
   FAIL_CLOSED_NEGATIVES=<cases, exits, no published partial>
   RESIDUAL_COMPONENTS=<3 blocked, acoustic and raw-unit evidence gaps>
   GPU_AND_HOST_NONMUTATION=<service, VRAM, originals>
   PUBLIC_EVIDENCE_ROOT=<head:path>
   MANIFEST_SHA256=<sha256>
   PUBLIC_MEMBERS_VERIFIED=<count>
   PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
   HUMAN_WER_CER_DER=UNVERIFIED
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|canonical>
   omitted_sensitive_values=<yes|no>

Return one complete evidence packet, no repeated model work or micro-handoffs. A passing isolated CLI is a feature qualification, NOT authorization to enable it by default, write sidecars or merge the PR.
