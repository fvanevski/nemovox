# Issue 14 / draft PR 15: exact-head gate check and focused long-form timing evidence

## Objective and immutable authority

ONE consolidated host-bound campaign: establish the exact-head deterministic gate after a central test-contract fix, then collect bounded real ASR/CTC re-inference evidence plus archived speaker/translation attributions needed for Central to implement the remaining Issue 14 fixes. Not another full-episode transcription and not a local development delegation.

Repository: fvanevski/nemovox
Issue: #14 OPEN; existing PR: #15 OPEN/DRAFT
Branch: issue/14-longform-timing-boundaries
EXACT_TESTED_SOURCE_SHA: 66275bdb512977182ed52fddbb022f2fbac2e670
EXACT_TESTED_TREE: 343075870237fe39c7ee813cbcbb1d6e5032769f
MAIN_BASE: 8f6c03cbc70e84bf1970ba364207a02c9fe22b44
PREVIOUS_HOST_TESTED_SOURCE: 866aa734a29e609bd9761e1d7365e32aefd2b2f7
PREVIOUS_HOST_EVIDENCE: 99dce3f8bfa1a93670b28473209dbf8bd924a54e:staging/issue14/evidence/20261122T215418Z/
DIAGNOSTIC_BASELINE: 5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z/
Primary local agent: OpenCode build, host-bound execution only
EVIDENCE_ROOT: staging/issue14/evidence/<new-UTC-run>/
SEMANTIC_SOURCE_OR_TEST_EDITS: FORBIDDEN
REMOTE_EVIDENCE_WRITE: one evidence-only non-force commit on existing branch
NEW_PR_OR_ISSUE: FORBIDDEN
PRODUCTION_DEPLOYMENT, MERGE, CODEX_REVIEW, ISSUE_CLOSURE: FORBIDDEN
MODEL_CONFIG_CHANGES, UNRELATED GPU SERVICE EVICTION, WEIGHT_DOWNLOADS: FORBIDDEN

The only changed test contract since PREVIOUS_HOST_TESTED_SOURCE is a Central-authored assertion at tests/test_issue14_timing_boundaries.py:64, now requiring timing_evidence_status == cross_chunk_ctc_disagreement AND timing_evidence_reason == relative_sequence_shift_unverified. Application source is unchanged. This is not an assertion bypass. Prove actual git diff of src/ and tests/, not a summary.

Private original inputs, READ ONLY:
- /workspace/nvidia-transcribe/outputs/studio/jobs/980af31f014245c4b200e0af8b33ebca/
- /opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm
- /workspace/nemovox-mcog-private (prior bounded clips and raw diagnostics)
- production checkout /workspace/nvidia-transcribe and nemovox-studio.service
Original historic job-execution Git SHA is UNVERIFIED. Do not silently assign the current main SHA to that earlier run.

## A: admission and CPU verification

1. Freshly fetch and prove GitHub main, PR #15 exact branch head and Issue #14 state. Require MAIN_BASE unchanged, PR OPEN/DRAFT, issue OPEN, exact tested-source object/tree. The branch may have a single instruction-only descendant of EXACT_TESTED_SOURCE_SHA. Unexpected new source/test changes => stop STALE; no force, reset, stash, clean or rebase.
2. Use a new collision-safe detached worktree in an isolated scratch root, not the production checkout. Read AGENTS.md and test conventions. Use production .venv and .venv-studio only as installed read-only interpreters with PYTHONPATH explicitly bound to the isolated candidate src. Confirm import origins. Do not upgrade dependencies.
3. Recheck original video/job/manifest/16 cached chunks hashes against previous evidence. Record current production Git identity/status, Studio PID/InvocationID, active jobs, model units and lease ownership both before and after. No mutations to original media/jobs/outputs, service or source checkout.
4. Run real compileall, focused pytest (test_issue14_timing_boundaries.py, test_chunking_and_resume.py, test_utterances_alignment_speakers.py, test_recording_and_normalization.py), full pytest -q -m 'not gpu', and Studio pytest test_studio_exports.py test_studio_export_api.py test_studio_api.py. Preserve commands, actual exits/counts/skips/warnings and any traceback. Stop GPU phase on any remaining deterministic failure. Never weaken an assertion or skip a gate.
5. Since src/ remains byte-identical to the previously replayed source, reuse the 16-chunk W04c/W05a/W04b/W03a/W02c/C1/C2 data from PREVIOUS_HOST_EVIDENCE as immutable source-behavior evidence. DO NOT run another 16-chunk offline replay solely because a test assertion was corrected. Clearly distinguish reused evidence from new exact-head tests.

## B: targeted ASR + CTC timing experiment, only after A passes

Goal: determine whether SHORT source-audio re-inference restores correct timing for multiword/isolated 9-55s CTC drift without breaking clean dialogue, repeated speech or names. Current source-only timing audit flagged 63 review-only large gaps and quarantined zero of 2,415 words; it cannot acoustically correct an isolated timestamp from lexical structure alone.

Admit one existing Nemovox-owned ASR model/lease session only when GPU ownership, service compatibility and original model checkpoint are verified. Inspect original job request mode, language, CTC options, speaker mode, model revision, sample-rate and source-offset contract. Use the SAME ASR/CTC mode, not an alternate model or prompt, and do not retranslate/review/render. No external model startup, checkpoint download or global service configuration changes. Model service admission blocked => GPU_PROBE=BLOCKED and continue read-only phase C.

Extract at most SIX 8-30-second PRIMARY excerpts, plus TWO 8-30-second negative controls, from the original authenticated media or existing verified private clips. Suggested anchors (use previous catalog's acoustically checked exact times, not guesses):
- W01a around 00:28 (c00:111 falsely near 01:22.85);
- W01b/W01d around 00:43-01:25, at most two distinct short clips covering early and late multiword drift;
- W02a around 03:35-04:15 (c02 early timing);
- W03b around 16:55-17:10 (misplaced words and mixed speaker tags);
- W04a around 20:20-20:42 (bidirectional time split);
- Clean control C1 around 11:27-11:37 (29 reliable words);
- Repeat/overlap control C2 around 17:03-17:07 (must remain genuine).

Each excerpt must be <=30 s, source-bounded, with exact absolute audio offset, sample-rate, input/audio SHA256 and original unit IDs. Run newly generated ASR/CTC with verified original parameters. Record raw output privately, then convert new local CTC word offsets to absolute time using the actual excerpt seek origin. Compare word recovery/omissions, CTC timing residuals vs the previously independently audio-checked anchors, sequence order and speaker-tag behavior. A finite timestamp alone does not prove timing correctness. Do not infer speaker identity from model tag number alone.

If a narrow excerpt clearly loses context, at most THREE matched longer-context (30-60 s) re-inference comparisons are allowed, chosen from failed or ambiguous anchors. This is a bounded context-length experiment to inform Central's architecture, not a parameter sweep. Explicitly record clean C1/C2 degradation, proper-noun loss or ASR error as counterevidence to universal short chunking. Do not misrepresent historical Antigravity/listening observations as independent new listening.

Preserve raw requests/responses, WAV/PCM and full dialogue ONLY in chmod-700 local scratch; publish compact source-token IDs, timing offsets, coverage metrics, confidence, method and SHA fingerprints. Verify model ownership releases and GPU VRAM returns to baseline. No full episode reprocessing.

## C: existing speaker and translation evidence, read only

From original 270 diarization spans, 42 local to 23 global mapping, TitaNet/temporal reconciliation artifacts and original French/Riva/Qwen archived results, collect bounded case comparisons without new model calls:
- W02b, W03b, S05 plus clean/concurrent C2: local diarization boundaries, source units, tags, actual acoustically supported turns, embedding similarity and margins, 0.6536/0.6637 low-margin merges, near-ties and any contradicting simultaneous speaker evidence. Distinguish false merges, false splits and unknown identities. DO NOT assert exactly eight episode speakers.
- S01, S02 and W04c: source French -> Riva original translation -> Qwen review -> displayed text; demonstrate any negation inversion or fragment regression, plus counterexamples where review helped. Do not reveal full copyrighted text in public.
- Record safe implementation constraints and outstanding acoustic evidence gaps for Central. No speaker-model thresholds, prompts, glossary or reviewer code changed by host agent.

## D: accurate metrics and sanitized handback

Prior evidence's negative "legacy_1_5s_recovered_units" (-44, -4) is a derived accounting error. The repository's recovered_boundary_units counts legacy rescues plus extended rescues, while sequence_promoted_units is a separate quantity NOT added into that recovered total. Recompute legacy = recovered_boundary_units - extended_recovered_units. Last candidate 10-6=4 legacy, not -4. Present actual raw diagnostic values, definitions and denominators, not silent reconciliation.

Publish ONE sanitized manifest-backed evidence-only commit under EVIDENCE_ROOT, with:
- README.md; source-identity.json, validation/tests.md;
- validation/short-asr-probes.jsonl with exact window/source hash/model params, local+absolute CTC errors, ASR coverage, method and controls (NO raw transcripts);
- validation/ctc-strategy-evidence.md comparing narrow vs old 90s context and any qualified longer variant, acoustic uncertainties and failure modes;
- validation/speaker-review.md on W02b/W03b/S05/C2 and S01/S02/W04c;
- validation/metrics-corrections.json with correct counter arithmetic and baseline/prior candidate distinction;
- review/acceptance-matrix.md, review/findings.md and COMMAND_SHAPE_FRICTION.txt;
- manifest.json with every member's SHA256 and byte length, omitting its own hash.

Never publish original video, WAVs, screenshots of show, full transcript/ASS, raw chunk JSON/ASR replies, secrets, full service journal, or private host paths. Keep complete raw trace and source clips private, hashed and referenced by bounded case IDs. On exact-head drift or publication failure, return a sanitized local archive and BLOCKED rather than forcing changes.

After publication re-fetch the remote branch, verify the one evidence-only commit and complete manifest SHA256/bytes. PR remains draft; no Codex request, new PR, merge, deployment or Issue #14 closure.

LOCAL_AGENT_HANDBACK:
 RESULT=PASS|PARTIAL|BLOCKED|FAIL
 EXACT_TESTED_SOURCE_SHA=66275bdb512977182ed52fddbb022f2fbac2e670
 EXACT_TESTED_TREE=343075870237fe39c7ee813cbcbb1d6e5032769f
 REMOTE_MAIN_SHA=<full SHA>
 PR_HEAD_CARRIER=<evidence commit SHA>
 ORIGINAL_MEDIA_JOB_AND_PRODUCTION_INVARIANCE=PASS|FAIL
 FOCUSED_TEST=<count/exits/warnings>
 NON_GPU_TEST=<count/skips/exits>
 STUDIO_TEST=<count/exits>
 PRIOR_REPLAY_REUSE=<yes/no with source-diff evidence>
 GPU_PROBE=<performed|blocked|not_attempted, admission evidence>
 SHORT_ASR_PROBES=<case IDs, count, corrected/degraded/uncertain timings>
 CONTROLS_C1_C2=<status and acoustic limits>
 SPEAKER_EVIDENCE=<W02b/W03b/S05/C2 findings>
 TRANSLATION_REVIEW=<S01/S02/W04c attribution>
 RECOVERY_COUNTER_ACCOUNTING=<legacy/extended/sequence-promoted totals and formulas>
 REMAINING_TIMING_DEFECTS=<case IDs, severity, evidence>
 EVIDENCE_GITHUB_ROOT=<full carrier SHA:path>
 MANIFEST_HASHES=PASS|FAIL|BLOCKED
 PRIVATE_SUPPORT_REQUIRED=<no or bounded host-only artifacts>
 COMMAND_SHAPE_FRICTION=<encountered=no/events=[] or details>
 BLOCKING_FINDINGS=<none or exact repro>
 ISSUE14_PR15_STATE=OPEN/DRAFT

Central owns all ensuing substantive CTC timing, speaker-turn and translation-review code on this SAME PR. Do not disguise evidence gathering as permission to implement algorithms.
