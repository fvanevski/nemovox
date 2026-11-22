# Issue #14 / draft PR #15: source-preserving ASR integrity gate + three bounded acoustic recovery investigations

## Authority, objective, and strict boundaries

This is ONE cohesive host campaign. Central already authored the production source change and synthetic tests; the local agent executes source-bound validation and BOUNDED GPU observations. The purpose is to establish safe automatic recovery criteria for three residual episode defects, NOT to perform another full-episode inference or to delegate substantive code design.

REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14 (stay OPEN)
EXISTING_PR=15 (stay DRAFT/OPEN)
BRANCH=issue/14-longform-timing-boundaries
EXACT_TESTED_SOURCE_SHA=4c32acf659891fe999d2ae6e15f59ccb5750d92f
EXACT_TESTED_SOURCE_TREE=a4e866f53899fe159c99793ff8a78326a10b26f5
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
PREVIOUS_FULL_EPISODE_SOURCE=81704133cfec54e284f39adb8ad99b7cd1587bcb
PREVIOUS_FULL_EPISODE_EVIDENCE=a1812212f6e5c7a82bcd0a27769130fda420a3d3:staging/issue14/evidence/20261122T225739Z/
BASELINE_EVIDENCE=5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z/
EVIDENCE_ROOT=staging/issue14/evidence/<new-UTC-run>/
PRIMARY_HARNESS=OpenCode build
SOURCE_OR_TEST_EDITS=FORBIDDEN (Central ownership)
PRIVATE_DIAGNOSTIC_HARNESS=ALLOWED (read-only source, temporary scripts, normal execution fixes)
GPU=only owned Nemovox ASR lifecycle; no Qwen/Riva required
GPU_SAMPLE_LIMIT=6 initial excerpts, up to 3 conditional follow-up comparisons
REMOTE_WRITE=ONE sanitized evidence-only commit to existing branch after fresh CAS/ref checks
NEW_PR, CODEX_REQUEST, PR_READY, MERGE, ISSUE_CLOSURE, DEPLOYMENT=FORBIDDEN
PRODUCTION_STUDIO, ORIGINAL_MEDIA, ORIGINAL_JOB, MODEL_CONFIG, VENVS=READ ONLY

The source file src/nvidia_transcribe/asr_integrity.py, its recording.py integration, and tests/test_issue14_asr_integrity.py are NEW and NOT YET HOST TESTED. Do not represent staging as a test PASS.

Historical original job execution SHA is still UNVERIFIED. Full episode from the PREVIOUS_FULL_EPISODE_SOURCE passed CPU 59 focused/289 non-GPU, Studio 50 and Web 14/build and completed one managed-model 23-minute private run. It is the same application source except for this later ASR-integrity gate. That run was PARTIAL: full ASR, Riva and Qwen executed, but c28 left one untimed word due finish=length; W04b c31 had CTC 1253/1264s swaps; W05a's six source words were absent from c34 raw ASR. Do not replay 23 minutes merely to rediscover these three defects. The original 90s run and newer 35-chunk output are immutable references.

## Central production change under test

- audit_asr_generation(raw_response) preserves the response and CTC unchanged. It examines finish_reason and extreme loop-cleaner contraction. finish=length is INCOMPLETE regardless of valid-looking CTC, rather than a completed transcription. A large pathologically repeated text may also be INCOMPLETE even if finish=stop; brief genuine repeated interjections do not trigger the test. Unsupported finish reasons remain review-only, not fabricated acceptance.
- process_media writes the original raw response and manifest with asr_run_status=incomplete_generation and chunk.status=incomplete_generation before raising IncompleteASRGeneration. The rejected original is retained in chunks/rejected/<content-sha>.raw.json, which cannot be accidentally picked up as an ordinary completed chunk. On retry, completed neighboring chunks may resume, but the incomplete chunk MUST NOT be reused as complete. Old successfully accepted checkpoint entries must also undergo the same audit when read.
- Only a COMPLETE processing pass may mark asr_run_status=completed and publish a new source contract. Preexisting recording/transcript files in an output directory can remain byte-identical after a failed rerun; do not mistake those stale files for newly produced outputs. The manifest is the current run-status authority.
- This is a **fail-closed quality invariant**, not an automatic c28 speech repair. If the host discovers a semantic defect, STOP source edits and return a reproducer. No prompt/threshold/model changes to get tests green.

## Phase A: exact source admission and CPU checks (mandatory)

1. Freshly fetch GitHub main, Issue #14 and PR #15. Require main == BASE_MAIN_SHA; PR is OPEN/DRAFT; branch is an instruction-only child of EXACT_TESTED_SOURCE_SHA; no source modifications outside the Central source commit. Authenticate full Git tree. Read AGENTS.md and docs/model-resources/LIFECYCLE.md.
2. Use a freshly named, isolated detached worktree from the exact tested SHA, NOT the production checkout, not a preexisting dirty verification directory. Ensure PYTHONPATH imports from the candidate src while using installed production .venv and separate .venv-studio as read-only executors. Do not install, alter or repair model/venv components.
3. Verify baseline video, the original Studio job and its manifest, 16 original chunks, and the previous 35-chunk private full-run manifest/contracts/raw response hashes against the prior archived provenance. Inspect their exact locations only privately; retain chmod-700 scratch. Snapshot Studio PID/InvocationID, current production Git/status, running jobs, model-unit status and GPU VRAM before and after.
4. Run compileall, focused tests including NEW tests/test_issue14_asr_integrity.py and tests/test_recording_and_normalization.py/test_chunking_and_resume.py/test_issue14_timing_boundaries.py and other relevant speaker/review/planner regressions, full .venv pytest -q -m 'not gpu', and .venv-studio Studio tests. Report real exits, counts, skips and warnings. Web code was not changed since the prior exact host 14-test/build PASS; verify git diff, then re-use that specific UI evidence rather than rerunning build unnecessarily.
5. Specifically demonstrate (synthetic fixture) a length-exhausted response with apparently valid CTC is archived as incomplete, raised, not rendered; retry does not resume it; rejected raw is retained after a retry; existing completed neighbors can resume; and legacy already-"completed" but exhausted raw cannot bypass the audit. Preserve C1/C2 valid repeated speech and normal stop path. A failure here blocks GPU and must be returned with traceback, not repaired via local source/test edits.

## Phase B: PRIVATE source/response attribution, no new model calls

Using the original and previous 35-chunk PRIVATE outputs, produce exact source-bound comparisons for:
- **c28 generated loop**: original chunk core [1104.141292,1150.141292], inference [1100.141292,1154.141292]. Identify raw finish=length, exact raw-vs-cleaned character counts, CTC alignment/error/diarization presence, source-time interval and anchor words from *independent acoustic spot-check* if available. Do not assert the 1 untimed "Ah" equals the real spoken content. Identify neighboring c27/c29 recognized timing, speaker tags and coverage without leaking dialogue.
- **W04b / c31**: core [1230.195875,1263.699646], inference [1226.195875,1267.699646]. Compare W04b speech ~1253–1255.3 and W04c ~1264.3–1266.6 to c31 and c32 raw CTC and the old fixed-window reference. Separate lexical omission, source-word order inversion, CTC offset drift, and nominal core-exclusion. No silent time substitution.
- **W05a / c34**: c34 core [1344.119948,1379.901], inference [1340.119948,1379.901]. The six-word speech at 1352–1354 was present in old fixed c14 right-context ASR and absent in new c34 raw ASR. Compare old raw word IDs/times and new raw CTC/diarization to verify whether the new omission is lexical, timing or both. Check neighboring narration and S05 tail ~1375–1377 as independent non-loss controls.

For S05/W02b (speaker-turn failures) collect already-existing bounded diarization intervals, tag map, unit indices and acoustic support to distinguish independently evidenced alternating voice from mere tag-number guesses. Avoid any full-episode speaker truth claim. These are small auxiliary references, not a fresh speaker project.

If there is inadequate independently heard reference for one anchor, mark acoustic truth NOT_CHECKED and do not elevate that case from inference to VERIFIED.

## Phase C: one owned ASR model session, targeted independent candidate observations

Only after Phase A PASS and a safe owner/lease preflight. Use the original local Nemotron NVFP4+CTC model, original fr/speaker request mode, capture_ctc_timestamps=True, prompt/seed/quantization/CTC middleware parity. No external service eviction, weight downloads, global configuration edits, or Riva/Qwen calls. Use existing ModelLifecycle acquisition/release exactly as the previous successful evidence run. Never overwrite the previous raw job/output. Confirm each excerpt's SHA, absolute seek bounds, sample rate and exact source_offset_s. Raw requests/responses/WAVs stay PRIVATE.

At most **six primary clips**, each a single independent request:
- c28 left: 1099.5–1129.0 (29.5s); c28 right: 1123.0–1156.0 (33s). Compare 6s shared acoustic overlap, source lexical coverage and CTC units against neighboring c27/c29. Distinct speaker tags from independent clips are NOT globally identical by number. Do not concatenate raw speaker-tag strings and claim reconciled full-source speech.
- W04b: 1238.0–1266.5 (28.5s) and 1242.0–1274.0 (32s). Verify 1253s words actually sound and align in their originating window, and that W04c at 1264s is retained instead of displaced over W04b. Distinguish legitimate different speakers and music from CTC misalignment.
- W05a: 1331.0–1368.0 (37s) and 1345.0–1379.901 (34.901s). Seek independent lexical evidence of the precise six words at 1352–1354, check surrounding narrator and S05 end in the second clip.

If a primary pair leaves a high-impact ambiguity, permit **at most three extra matched context comparisons total**, one per problem:
- c28 wider 1100.1–1144.1 (44s), or another bounded split justified by the original acoustic boundaries;
- W04b 1231.0–1267.0 (36s), matching much of original c31;
- W05a 1338.0–1379.901 (41.901s), matching more episode-end speech.
Do not change model prompts or context duration arbitrarily; these are deliberate context/coverage comparisons, not a broad search.

For each inference: report whether finish_reason=stop, raw/cleaned lengths, CTC metadata validity, ASR lexical coverage with the previous acoustically checked source, timing error in seconds at independently supported anchors, missing/duplicated phrase source IDs, concurrent speaker evidence, hallucinations, and performance/VRAM. A finite CTC timestamp alone does not establish acoustic correctness.

**STOP** further GPU work on unsafe lease, model fingerprint drift, original-data mutation, service cleanup failure, or repeated pathological generation. If c28 short windows fail to recover actual speech, do not accept short texts merely because they are non-looping.

## Phase D: offline candidacy decision and no-scope-drift handback

This campaign does **not** modify production code beyond the pre-staged integrity guard. Do not locally invent speaker attribution, confidence, CTC corrections or automatically merge the candidate clips. From the observed candidates propose only a data-backed **selection/acceptance matrix** for Central: which alternate audio window is supported by independent source coverage, what overlapping lexical content can be reconciled safely, whether a second pass adds real words, and where ambiguity remains. Central will then implement source recovery with positive controls and one final integrated full-run qualification on this SAME PR.

Publish exactly one small sanitized manifest-hashed evidence packet under EVIDENCE_ROOT:
- README.md, source-identity.json (full source SHA/tree, parent carrier, original/prior job digests, no production changes)
- validation/tests.md (exact commands, exits, warnings, source import origins, generated incomplete-manifest assertions)
- validation/original-failures.json (c28 length/loop/CTC/diar status; c31 W04b timing/source ambiguity; c34 W05a missing source; indexes/digests, NO full transcript)
- validation/asr-recovery-probes.jsonl (up to nine cases; window offsets and hashes, mode/model/seed, word presence/timing vs source, positive/negative controls, no raw speech dump)
- validation/source-recovery-decision.md (causal distinctions, proposed safe acceptance criteria and rejection counterexamples; clearly NOT an implemented fix)
- validation/speaker-limits.md (W02b/S05 known vs uncertain diarization/tag splits)
- review/acceptance-matrix.md, review/findings.md, COMMAND_SHAPE_FRICTION.txt, manifest.json (all member exact SHA256+byte sizes)
- Optional bounded sanitized read-only probe harness if needed for reproducibility, never host-private paths or raw dialogue.

Keep complete copyrighted/raw trace, excerpts and outputs only in chmod-700 private scratch; no full transcripts/ASS, video, WAV, original raw model JSON, model weights, credentials or unredacted journals in GitHub. After freeze, fetch branch head, confirm only evidence-root paths are changed, publish ONE non-force evidence-only commit, read back HEAD/tree and verify SHA/byte parity for every member. On concurrent drift/publication block, return a sanitized local checksum archive instead of force/rebase.

### Required handback

LOCAL_AGENT_HANDBACK:
  RESULT=PASS|PARTIAL|FAIL|BLOCKED
  EXACT_TESTED_SOURCE_SHA=4c32acf659891fe999d2ae6e15f59ccb5750d92f
  EXACT_TESTED_SOURCE_TREE=a4e866f53899fe159c99793ff8a78326a10b26f5
  MAIN_SHA=<fresh full SHA>
  EVIDENCE_HEAD=<evidence-only new commit full SHA>
  PRODUCTION_MEDIA_AND_PRIOR_RUN_INVARIANCE=PASS|FAIL
  FOCUSED_NON_GPU_STUDIO=<actual counts/skips/exits>
  FRONTEND_REUSE=<previous tested SHA + no-web-diff verification>
  ASR_INCOMPLETE_CHECKPOINT_AND_SAFE_RETRY=PASS|FAIL; exact source case if FAIL
  ORIGINAL_FAILURES=<c28,c31,c34 source-linked diagnostic>
  GPU_SESSION=<owned|blocked|not attempted; lifecycle+model+VRAM>
  C28_ALTERNATE_WINDOWS=<lexical/CTC/finish; overlap integrity and negatives>
  W04B_ALTERNATE_WINDOWS=<real 1253s phrase vs 1264s phrase; core clipping>
  W05A_SIX_WORDS=<fully recovered by independently verified audio/CTC? which window>
  C1_C2_W04C_S05_CONTROLS=<reused source-bound and any new control observations>
  SPEAKER_W02B_S05=<raw diarization vs ASR turn IDs and truth limits>
  CENTRAL_REPAIR_ACCEPTANCE_CRITERIA=<positive/negative candidate evidence>
  EVIDENCE_GITHUB_ROOT=<new full commit:path>
  MANIFEST_HASHES=PASS|FAIL|BLOCKED
  PRIVATE_SUPPORT_REQUIRED=<none|bounded host artifacts and source index>
  COMMAND_SHAPE_FRICTION=<encountered=no/events=[] or details>
  BLOCKING_FINDINGS=<none|exact code/model reproducer>
  ISSUE14_AND_PR15=OPEN/DRAFT

A PASS here means the *gate and bounded probe evidence* are trustworthy, not that all Issue #14 source failures or the full 23-minute episode have been repaired. No new PR, independent Codex review, ready transition, merge, deployment or issue closure.
