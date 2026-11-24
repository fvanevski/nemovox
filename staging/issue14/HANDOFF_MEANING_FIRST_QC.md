# Issue #14 / PR #15: consolidated meaning-first QC and reasoning qualification

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=622715afc00dd6d215fe27c9b5844f615d762971
CENTRAL_SOURCE_TREE=1317b67868baec61852e9d84d087b0e0e3994d5d
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CONTROLLING_DESIGN=staging/issue14/MEANING_FIRST_TIERED_QUALITY_DESIGN.md
FORENSIC_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-omission-diag-private/issue14-omission-diagnostic-20261124T010009Z.tar.gz
FORENSIC_SHA256=52568c845ab3e7c2541e3711b8950cccde93a11164cb7eadf3c706d4bf923b8e
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_FULL_EPISODE_RERUN=yes
NO_NEW_ASR_RIVA_PYANNOTE=yes
NO_MODEL_DOWNLOAD=yes
NO_UNOWNED_GPU_EVICTION=yes
NO_PRODUCTION_SOURCE_EDIT_BY_LOCAL_AGENT=yes
NO_AUTOMATIC_SOURCE_PROMOTION=yes

## Goal

One CPU-first host qualification of Central-owned meaning-first boundary review detection plus optional bounded low/high-effort LLM adjudication on archived W01c/S02 evidence. The objective is meaningful, fluent, intelligible English with correct polarity, narrative continuity, practical utterance timing and important speaker boundaries, NOT perfect word-for-word recovery. No new full-length ASR or independent model surveys.

## A. Source/evidence authority and CPU gates

Fetch and authenticate live PR #15 head, base/main SHA, draft/open and Issue #14 OPEN. Verify CENTRAL_SOURCE_SHA/tree are latest src/tests ancestor with no intervening source/test edits. Fresh isolated worktree, no modification to operator checkout. Read AGENTS.md, controlling design, semantic_review.py, chunking.py, recording.py, evidence_fusion.py, translation_review.py, config and tests. Verify forensic private tar SHA (21/21 members, 27 unique entries), exact Run A and B raw chunk 0,1,3,4 plus original media and WAV IDs. No remote/local writes beyond one final sanitized evidence commit.

CWD worktree and PYTHONPATH=src:. for BOTH interpreters:
  /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests
  /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_semantic_review.py tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_issue14_evidence_fusion.py tests/test_issue14_source_fusion_downstream.py tests/test_translation_review.py
  /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check
Run with PYTHONPATH=src:. prefix on each Python command. Verify candidate __file__ imports for semantic_review/chunking/recording in both interpreters. Use actual repo test paths, report substitutions. Do not weaken tests or make semantic source edits locally; a production bug is reported with a focused regression and traceback for Central.

## B. Reproduce high-value context QC without modifying original output

Construct per-chunk normalized lexical units from authenticated raw ASR/CTC/source-evidence observations, preserving exact chunk/source-unit IDs and original timestamps. Use actual source-evidence diarization_activity_hypotheses (model activity only) as corroborative input, NOT acoustic truth. Call staged plan_boundary_semantic_reviews and replay reconcile_chunk_boundaries in private CPU scratch for both run A and B.

- W01c: boundary 90s, chunk1 80.2-83.254s left-context dialogue, chunk0 incomplete around 77.63-83s; foreground versus overlapping distant shout and music transition distinguishable. Must detect coherent candidate and report 77.63-80s cannot be recovered from chunk1 audio which begins 80s.
- S02: boundary 360s, chunk4 351.24-359.812s aligned left-context speech including meaningful negation; chunk3 duplicate text CTC displaced ~19-58s early and speaker-interleaved; identify competing source evidence without blindly choosing nominal core or inventing timing.
- Controls: C1 clean dialogue, C2 genuine repeated interjection, W04a/W04c non-speech SFX/music and Mandarin character-level. Do not promote char-grain words, silence, fake music captions or short ambiguous repeats.
- Compare default internal policy with configs/meaning-first-review.example.json using policy and request SHA. Report number of candidate cases, review tiers, deferred budget, source-unit IDs, and any missed genuine semantic defects. Original source/7-component accepted CLI output and render must remain unchanged.

## C. Actual bounded reasoning, conditional on owned text service

If verified Nemovox-owned text-review service is available, pass injected OpenAIChatTextClient through run_tiered_text_reviews, at most SIX cases with max ~7000 total reserved output tokens. Use T1 low effort for one ordinary grammar/readability case and T2 higher effort for W01c/S02 and one negative ambiguity. Store model/resource/prompt identity, request SHA, raw response SHA, actual effort fallback, budgets, accepted/abstaining decisions, contrary evidence and possible natural-English paraphrase quality; no model conclusion is automatically an acoustic fact or production rewrite. If text service unsafe/unavailable, mark MODEL_PROBE=NOT_EXECUTED; proceed with CPU path.

Optional T3 only if an unresolved speaker/scene inference actually benefits from vision and existing preverified hosted model and resource admission, at most one bounded probe. Animated mouth motion is weaker than physically synchronized live action; no blanket VLM inference and no production vision integration.

## D. Evaluation and one consolidated evidence packet

Evaluate W01c/S02 primarily for intended meaning, negation/polarity, coherence across adjacent scenes, intelligible grammatical English, absence of fabricated dialogue and correct event-level cue placement. Human word-level WER/CER/DER is UNVERIFIED and not a required perfect-score target. Record remaining source uncertainty, acoustic/cross-chunk provenance and which decisions are review-only vs eligible for existing opt-in source-revision gate. If substantive loop closure remains, describe one generic integration contract, not scene-specific patches.

Publish ONE sanitized evidence-only commit to existing draft PR #15 and one self-contained private tar.gz with SHA256 manifest, unique entries and bounded source/model receipts. Public packet: source exact SHA/tree, tests/counts/import identities, W01c/S02 candidate proof, 22-case controls relevant to QC, model budget/reasoning outcomes, output-quality matrix, blockers and command-shape friction. Keep copyrighted raw responses and audio in the private packet, not public GitHub. Recheck exact PR head, no src/tests drift, draft/issue status.

Return:
 ISSUE14_MEANING_FIRST_QC:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<40char SHA>
   VALIDATED_SOURCE_SHA=622715afc00dd6d215fe27c9b5844f615d762971
   VALIDATED_SOURCE_TREE=1317b67868baec61852e9d84d087b0e0e3994d5d
   CURRENT_PR_HEAD=<40char SHA>
   MAIN_SHA=<40char SHA>
   SOURCE_TEST_DIFF=<none|actual paths>
   CPU_FOCUSED_CORE_STUDIO=<commands counts imports>
   FORENSIC_ARCHIVE_AUTH=<21/21, exact raw and media digests>
   W01C_CONTEXT_QC=<source ids, timing, candidate, meaning uncertainty>
   S02_CONTEXT_QC=<source ids, negation, displaced evidence, candidate>
   NEGATIVE_CONTROLS=<clean/repetition/music/Mandarin>
   REVIEW_POLICIES=<policy hashes, tiers, candidates/deferred>
   TEXT_MODEL_PROBE=<executed/not_executed, budget, response integrity>
   OPTIONAL_AV=<executed/not_executed, bounded and reason>
   MEANING_FIRST_OUTPUT_QUALITY=<coherence grammar polarity readability event timing>
   APPLICATION_BOUNDARY=<advisory vs accepted, generic missing connector>
   PUBLIC_EVIDENCE_ROOT=<fullSHA:path>
   PUBLIC_MANIFEST_SHA256=<sha256>
   PUBLIC_MEMBERS_VERIFIED=<count>
   PRIVATE_REVIEW_ARCHIVE=<path sha256 bytes>
   HUMAN_WER_CER_DER=UNVERIFIED
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|records>
   omitted_sensitive_values=<yes|no>

One complete handback. No new inference campaign, PR, merge, deployment or issue closure. If actual source behavior contradicts synthetic results, preserve exact repro and return PARTIAL; Central owns production corrections and final exact-head review.
