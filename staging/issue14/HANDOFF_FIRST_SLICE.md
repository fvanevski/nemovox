# Issue #14 first cohesive slice: host qualification of Central-authored timing/boundary repairs

## Scope and authority

This is ONE bounded OpenCode Build evidence campaign. Central has already authored source changes and CPU regressions on the existing Issue #14 branch. Run and evaluate them against immutable cached source records; do not implement or redesign ASR/CTC algorithms locally. No final issue closure, PR approval or production deployment.

REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
BRANCH=issue/14-longform-timing-boundaries
TESTED_SOURCE_SHA=50b1408f73f3738f2f2ff5e59bbb421e4cb52e2f
TESTED_SOURCE_TREE=a0731a2b3440118795397cb96fa5fb79dc0d87d7
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
BASE_MAIN_TREE=21ae1c6d1b682f1061ae36f6896cd0ac87db1580
DIAGNOSTIC_SOURCE=5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z
LOCAL_HARNESS=OpenCode primary build
SEMANTIC_REMEDIATION=forbidden
MECHANICAL_REMEDIATION=allowed within exact meaning-preserving formatter/import/static normalization, one bounded source commit maximum, with complete diff review and re-validation at new SHA
SOURCE_REMOTE_UPDATE=forbidden except the permitted mechanical-only lane
EVIDENCE_REMOTE_UPDATE=allowed, exactly one sanitized non-force evidence-only commit
EVIDENCE_ROOT=staging/issue14/evidence/<UTC-RUN-ID>/
EVIDENCE_PACKET_REQUIRED=yes
PRODUCTION_CHECKOUT=/workspace/nvidia-transcribe (READ ONLY)
ORIGINAL_JOB=/workspace/nvidia-transcribe/outputs/studio/jobs/980af31f014245c4b200e0af8b33ebca/ (READ ONLY)
ORIGINAL_MEDIA=/opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm (READ ONLY)
PRIVATE_BASELINE=/workspace/nemovox-mcog-private (private, inspect before use)
PRODUCTION_STUDIO=nemovox-studio.service (READ ONLY)
MODEL_SERVICE_CHANGES=forbidden
NEW_PR=forbidden; NEW_ISSUE=forbidden; CODEX_REVIEW=not_yet

## Phase A: host/source admission

1. Re-fetch the GitHub main, issue branch and relevant immutable commits, prove full SHA/tree and Issue #14 state. The final handoff/instruction carrier is expected to be a direct child of TESTED_SOURCE_SHA; confirm changed files since that SHA are only instruction/evidence content or later explicitly permitted meaning-preserving mechanical changes. A stale main/branch is STOP.
2. Prepare one freshly named collision-safe isolated worktree under /workspace for the source branch. Use a fresh git fetch BEFORE local object/ref proof; do not reset, clean, switch or modify the existing production checkout, even if it is dirty. Distinguish remote authority from same-name local branches.
3. Read project AGENTS.md; verify source imports actually resolve into the isolated candidate src, using installed compatible interpreter at /workspace/nvidia-transcribe/.venv/bin/python. Studio tests may use /workspace/nvidia-transcribe/.venv-studio/bin/python. Do not reinstall or upgrade either venv. Inspect original manifest, contracts, video and private baseline digests. Historical original-job execution SHA is UNVERIFIED: never claim it came from new source.
4. Snapshot live Studio PID, InvocationID and health, the production checkout identity and dirty state, current running jobs and GPU/model unit status before and after. No live Studio write/API job, restarts, checkpoint downloads or third-party service eviction. No original media/job changes.

## Phase B: executable deterministic verification

In the isolated candidate, with PYTHONPATH explicitly set to candidate/src:
- Compileall of src, tests and scripts, capturing exit.
- Focused pytest on tests/test_issue14_timing_boundaries.py, tests/test_chunking_and_resume.py, tests/test_utterances_alignment_speakers.py and tests/test_recording_and_normalization.py, using the installed .venv Python.
- Full .venv pytest -q -m 'not gpu', actual count/skips/warnings.
- .venv-studio pytest -q tests/test_studio_exports.py tests/test_studio_export_api.py tests/test_studio_api.py.
No frontend/GPU requirement for this source-only CPU change. Never weaken fixtures, assertions, validation profiles or tests to obtain green. A semantic error means a precise FAIL handback with traceback. Fix only genuinely mechanical, meaning-preserving formatter or import-order issues within the explicitly permitted one-commit lane. On that lane inspect every changed line, follow current publication guards if any, and RE-RUN source-bound checks after publishing the mechanical edit.

## Phase C: one bounded offline episode replay (CPU only)

Use existing 16 raw CTC responses and asr/manifest.json from original job, without running the Nemotron model or overwriting the original job. It is permitted to build a small PRIVATE diagnostic driver, but NOT to invent algorithms. Required exact source function flow:

- For each archived chunk obtain infer_start_s, infer_end_s and core bounds from manifest, verify raw response SHA and source audio fingerprints. Parse raw choices[0].message.content and ctc_timestamps.
- Call recover_chunk_alignment(raw_text=..., ctc_payload=..., source_offset_s=infer_start_s, chunk_index=idx, model_path=None, subinterval_runner=None). When a source artifact used an alternate recovery mode that this replay cannot reproduce, mark incomparable rather than silently asserting exact replay.
- Call audit_ctc_timing_units(units, infer_start_s=infer_start_s, infer_end_s=infer_end_s), then normalize_chunk_speakers with parsed diarization and tag maps.
- Preserve original global-speaker mapping for this A/B-only replay by mapping each archived recording.contract.json speakers[].local_speaker_ids to its global_speaker_id, instead of loading/re-running TitaNet. This avoids conflating future speaker remediation with timing repairs.
- Supply assigned units and real ChunkSpecs to reconcile_chunk_boundaries, then reconstruct_speaker_utterances with authentic chunk infer windows. Place derived transcripts, full token trace and contracts ONLY in a new private chmod-700 scratch root, never the original job/output. Public traces must use case IDs and hashes instead of copyrighted dialogue.
- Measure the candidate against the immutable baseline cases.jsonl W01a/b/c/d, W02a/b/c/d, W03a/b/c, W04a/b/c, W05a, S03/S04, and controls C1/C2 plus the 1327/1329s genuine repeated interjections. Capture actual source-token provenance, time hypotheses vs quarantined timing, chunk ownership, sequence dedup, expanded recovery, and cue-affecting utterance shifts. Do not invent a new English ASS or claim rerender improvement without rendering.
- Required checks: W02c long displaced duplicate reduced to one occurrence if safely matched; W05a six-word right-context utterance recovered if supported by archive, no duplicate; W03a/W03c and W04b candidate outcome explicitly shown; source text preserved. C1 29-token clean passage, C2 genuine repetition, independent voices and true overlapping speech MUST NOT be over-deleted or wrongly joined. Compare micro-utterance, untimed, unique duplicate, recovered, ambiguous and speaker counts with exactly the baseline denominators.
- Know the intentional policy limits: W03c is a short displaced cross-speaker duplicate that the initial conservative matcher may leave unresolved; S03 is a one-word orthographic mismatch without acoustically validated equivalence; isolated large CTC gaps alone stay review-only. These limitations are NOT permission to silently widen thresholds or alter algorithms. Stage A/B may be PARTIAL without failing overall investigation.

Classify cases PASS/PARTIAL/FAIL. If any true clean control, legitimate repeated passage or authentic source coverage is harmed, flag a BLOCKING REGRESSION and stop the rollout path. Report concrete source indices/CTC offsets so Central can repair the substantive invariant. If a diagnostic driver requires technical repairs, agent may fix that PRIVATE driver without affecting source or changing model behavior.

## Phase D: optional narrow acoustic/ASR corroboration

Only when Phase C leaves a high-impact acoustic decision unresolved AND the existing Nemovox-owned model lease safely admits it, run at most THREE 8-30s excerpt transcriptions. Verify checkpoint/model/prompt/CTC contract and original media SHA, use the existing managed lifecycle, no change to production service. Do not download weights, alter GPU lease ownership, stop unrelated services or run a full episode. If lease/admission is blocked, record GPU_COMPARISON=BLOCKED and continue with the CPU handback. Private raw model responses/audio stay host-local; GitHub receives digests, timing differences, and honest verification status only. No new acoustic inference is required for a clean deterministic A/B qualification.

## Phase E: one manifest-verified public-safe handback

Prepare a single small packet under EVIDENCE_ROOT:
- README.md with source/tree/tested identities and limitations, plus a statement that this is not full Issue #14 acceptance.
- source-identity.json with production/main/branch/ref observations, original input/job digests, private fixture fingerprints and candidate import origin (redact host-personal paths/credentials).
- validation/commands.md: actual commands, exits, targeted/full pytest counts and warnings, any mechanical source change.
- validation/cases.jsonl: all targeted cases and controls, indexed source-unit IDs, counts, CTC hypothesis time/uncertainty, chosen/removed/recovered status, no raw copyrighted dialogue.
- validation/metrics.json: baseline/candidate on same denominator for coverage, duplicated passages, timing quarantine/gaps, fragmentation, boundary rescue and genuine repetition retention.
- validation/causal-assessment.md with case-by-case PASS/PARTIAL/FAIL, root cause, residuals, next CPU/GPU evidence decision. Include explicit unchanged cases, not only improved cases.
- validation/acoustic.md with precise method and lease outcome, or not_attempted; never pass off cached/fictional audio inference as fresh.
- review/acceptance-matrix.md and review/findings.md with uncovered risks, stable controls, source branch identity.
- COMMAND_SHAPE_FRICTION.txt: encountered=no/events=[] if no friction; otherwise record requested/submitted/effective commands, guard rewrite, actual result, and successful corrective events.
- manifest.json covering each member by relative path, exact SHA-256 and bytes. The manifest may omit its own self-hash.
Optional sanitized bounded audit-driver source if needed to reproduce logic. Keep media/WAVs, raw model chunks, all full transcripts, unredacted journals and model weights PRIVATE.

One evidence-only Git commit on the same issue branch after freezing and verifying the diff contains ONLY this packet. Re-fetch remote branch identity before non-force push, verify the remote parent/tree and manifest hashes after publication. Never use force/rebase/reset/stash/clean or open another PR. If publication blocked or branch unexpectedly advanced, return a local sanitized archive with checksum and status STALE/BLOCKED.

## Stop and decision

STOP with diagnostics (no source repair) on semantic failed tests, source drift, over-dedup or false speaker merge, damage to any input artifacts, unexpected model resource changes, or any public-private content leak. Normal host shell/tool invocation, permissions, scratch directory setup and authorized test-runner issues are agent-owned, without unnecessary Central roundtrips. A test failure is diagnostic evidence, not license to bypass. No production promotion, merge or Issue #14 closure.

Return:
LOCAL_AGENT_HANDBACK:
  RESULT=PASS|PARTIAL_A_B|FAIL|BLOCKED
  SOURCE_SHA=<tested full source SHA>
  SOURCE_TREE_SHA=<full tree SHA>
  MAIN_SHA=<fresh full SHA>
  EVIDENCE_CARRIER_SHA=<full SHA, source parent specified>
  IMPORT_ORIGIN=<isolated src>
  SOURCE_EDITS=<none|exact mechanical delta>
  PRODUCTION_AND_INPUT_INVARIANCE=PASS|FAIL
  TEST_FOCUSED=<exit/pass/fail/skips>
  TEST_NON_GPU=<exit/count/skips/warnings>
  TEST_STUDIO=<exit/count>
  CTC_TIMING_AUDIT=<counts/reasons and case effect>
  SEQUENCE_DEDUP=<W02c/W03a/W03c exact outcomes>
  EXTENDED_COVERAGE=<W05a words W04b/S03 outcomes>
  CONTROLS=<C1 C2 separate repeats and concurrent voices>
  CASE_MATRIX=<all case IDs and PASS/PARTIAL/FAIL>
  METRICS=<denominator-aware baseline vs candidate>
  GPU_COMPARISON=<not needed|performed|blocked; lifecycle evidence>
  EVIDENCE_GITHUB_ROOT=<carrier:path|BLOCKED>
  MANIFEST_HASHES=PASS|FAIL|BLOCKED
  PRIVATE_SUPPORT_REQUIRED=<none|bounded fields/paths>
  COMMAND_SHAPE_FRICTION=<canonical observations>
  BLOCKING_FINDINGS=<none|exact repro>
  PR_ISSUE_STATE=<PR draft; Issue #14 open>

Central will independently verify the returned exact-head source, case-level evidence and manifest, then own any remaining substantive fixes and later speaker/translation work on this SAME PR. Do not mistake candidate partial improvement for completed issue scope.
