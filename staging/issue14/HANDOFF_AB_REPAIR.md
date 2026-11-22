# Issue #14 / PR #15: corrective A/B host qualification after failed first replay

## Task identity and authority

This is the next **single coherent host campaign** for the *same* PR #15, not a new PR, issue or implementation workstream.

REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14 (must stay OPEN)
PR_NUMBER=15 (must stay DRAFT, OPEN)
BRANCH=issue/14-longform-timing-boundaries
TESTED_SOURCE_SHA=f0dfc994947c3319df76bf7d60563ef3352536c1
TESTED_SOURCE_TREE=d60cc4e53be49ecec7158a5776f076af9603cc43
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
PREVIOUS_FAILED_CANDIDATE_SOURCE_SHA=50b1408f73f3738f2f2ff5e59bbb421e4cb52e2f
PREVIOUS_FAILED_EVIDENCE=593dc41d7099ee7176b845c3dee24827981a7977:staging/issue14/evidence/20261122T212746Z
ORIGINAL_DIAGNOSTIC_BASELINE=5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea:staging/quality-audit/mcog-s04e01/evidence/20261122T205259Z
LOCAL_AGENT=OpenCode build (execution, bounded private replay harness, evidence publisher)
SEMANTIC_SOURCE_REMEDIATION=forbidden
MECHANICAL_SOURCE_REMEDIATION=meaning-preserving formatting/import-order only, at most one separately identified commit with complete source requalification
REMOTE_SOURCE_WRITE=forbidden except explicit mechanical lane above
EVIDENCE_PUBLICATION=one sanitized evidence-only commit on this SAME issue branch
EVIDENCE_ROOT=staging/issue14/evidence/<UTC-RUN-ID>/
PRODUCTION_MUTATIONS=forbidden
MODEL_SERVICE_MUTATIONS=forbidden
MODEL_INFERENCE=not required; do not start models for this CPU adjudication
NEW_PR=forbidden
CODEX_REVIEW=not yet
MERGE_AND_DEPLOY=forbidden

Original host inputs are immutable:
- production source checkout /workspace/nvidia-transcribe, running nemovox-studio.service (read-only);
- Studio job /workspace/nvidia-transcribe/outputs/studio/jobs/980af31f014245c4b200e0af8b33ebca/;
- media /opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm;
- private acoustic/raw-source reference /workspace/nemovox-mcog-private;
- existing prior isolated verification /workspace/nemovox-issue14-verify (do not overwrite any changed workspace).

The original job's historical execution commit remains UNVERIFIED. All 16 cached chunks are baseline data, not newly inferred model output.

## Central-owned correction that this run qualifies

Central replaced the previous global mixed-speaker SequenceMatcher+gap-bridging scheme with:
1. **speaker-local** sequence comparison from neighboring inference excerpts, which prevents interleaved other voices being consumed as a matched sequence;
2. **per-token relative-offset consistency**, splitting a phrase at discontinuous CTC shifts before applying any destructive deduplication; this directly addresses the W04c source-word deletion;
3. accent-tolerant lexical comparison for a highly constrained, multiword alternative observation;
4. **no unconditional unmatched left suffix promotion**. A right-hand continuation can only be promoted when a coherent >=2-word, index-contiguous continuation is observed and selected as a more complete alternative; an unmatched extra one-word token is never enough;
5. exact positive interval overlap rather than +/-0.30s endpoint padding, plus stronger phrase-copy evidence for the extended right-context omission check;
6. explicit no-deletion ambiguity for unknown voices and CJK character units; real repeated interjections and distinct voices remain controls.

These are hypotheses until host-tested. Do not assert the defects are fixed because new tests were written. The initial CTC timing plausibility auditor is largely unchanged; do not claim W01/W04 isolated timing shifts were repaired or re-timed simply because boundary output improved.

## Phase A: exact GitHub and host admission

- Obtain fresh refs for main, PR #15 head, and Issue #14, require main == BASE_MAIN_SHA and SOURCE commit/tree immutable. Expect one **handoff-only direct child** of TESTED_SOURCE_SHA at the branch PR head, optionally followed by this campaign's one evidence-only commit. A source change beyond the stated SHA invalidates this packet and requires Central requalification.
- Use fresh fetch + exact detached candidate from TESTED_SOURCE_SHA in a new collision-safe isolated worktree, or verified clean pre-existing host worktree with no uncommitted content to overwrite. Read AGENTS.md and repository test instructions. Do not reset, stash, clean, rebase, force or switch production.
- Prove import origins resolve to isolated candidate src with the existing /workspace/nvidia-transcribe/.venv/bin/python and .venv-studio/bin/python interpreters. No virtualenv/pip/model upgrades.
- Verify original video/job/manifests, 16 raw chunk digests and private reference hashes against the prior packet; snapshot original job metadata and live production Git identity, Studio PID/InvocationID, model units and current jobs before and after. Preserve all bytes; do not start Studio jobs, stop GPU units or change model lease settings.

## Phase B: run exact deterministic gates before deeper replay

1. python -m compileall -q src tests scripts against isolated candidate, record real exit code.
2. .venv Python pytest -q tests/test_issue14_timing_boundaries.py tests/test_chunking_and_resume.py tests/test_utterances_alignment_speakers.py tests/test_recording_and_normalization.py. This includes the new W04c, W03a mixed-speaker, accent-variant W04b, precise W05 and genuine repeated-phrase tests. Record names, count, full failure diagnostics and true exit.
3. .venv Python pytest -q -m 'not gpu'; record count/skips/warnings/exits.
4. .venv-studio Python pytest -q tests/test_studio_exports.py tests/test_studio_export_api.py tests/test_studio_api.py; record actual result.
5. No frontend/Chromium/Riva/Qwen qualification is needed for this Python-only iteration. No GitHub Actions workflow is configured; do not report CI PASS.

**If any test fails because of source semantics or tests, stop source remediation and return a precise failure handback.** Only genuinely meaning-preserving formatter/import-order corrections are within the bounded local mechanical permission. Do not modify fixtures, assertions, algorithms, thresholds, test collection, resource gates or retry skip rules to make green. A private replay-driver error may be repaired locally after verifying it is not production behavior.

## Phase C: offline, CPU-only 16-chunk A/B/C comparison

Re-run the EXACT baseline host diagnostic driver (a private mechanical harness; not production source) with source-bound imports of:
- recover_chunk_alignment on archived raw CTC, no fresh model call, no optional Qwen;
- audit_ctc_timing_units against each chunk's true inference bounds;
- normalize_chunk_speakers, using original archived local-to-global mapping to hold speaker identity constant;
- reconcile_chunk_boundaries, including revised sequence and extended recovery; and
- reconstruct_speaker_utterances using actual chunk inference windows.

Compare candidate against BOTH the original baseline and the immediately prior failed candidate with identical raw inputs and matching measurement definitions. Never write into the original job or produce a new subtitle file in place. Full raw token traces and any media/audio stay in private chmod-700 scratch.

**Mandatory case gates** (collect actual source-unit indices and chronology, not only totals):

- **W04c, blocking source-loss regression:** c13:108 has early CTC (~1254.61s), while c14:12 retains the corresponding correctly timed word (~1264.45s). Candidate must NOT delete c14:12; the c13 isolated early token may remain suspect but must not be misclassified as fixed. Verify original right phrase is intact. If the correct right word disappears or a new false phrase appears, RESULT=FAIL regardless of aggregate metrics.
- **Unintended suffix promotions:** c12:73 (S03 / ~1180s) and c03:299 (~370s) must not be promoted solely by a one-token continuation. Confirm no new artificial lexical output on these edges; record any other new candidate promotions with justification.
- **W05a:** check whether c14:248..253, all six recognized source words at 1352.01-1353.855s, survive, while c15:23 ending at 1351.95s does not erroneously count as overlap. Require no duplicated six-word text, no silencing of c15 narrator, and explicit source provenance; otherwise diagnose exact mismatch.
- **W03a:** previously a 21-token match crossed two other-voice SPK_19 interjections and failed; verify new per-speaker streams match the duplicated passage without deleting those separate interjections.
- **W02c:** compare c02:c03 multiword displaced duplicates under separate speaker streams. Record exact matched token runs, their source speakers, per-token offset stability, remaining duplicate fragments and timing uncertainty. A genuine repeated phrase or simultaneous distinct voice must not be deleted as a shortcut to a numerical improvement.
- **W04b:** compare c13:103..107 vs c14:3..9 with proper-name accent variation. A more complete right-hand sequence should be eligible without adding stray suffix words; record all source candidates and preservation of the original French words.
- **W03c:** 4-word cross-speaker displaced repetition remains intentionally ambiguous absent independent acoustic evidence. Record the case honestly; not grounds to widen the delete rule.
- **W01a/b/d, W02a, W03b, W04a, S04:** CTC drift still may remain a timing-uncertain or erroneous native result. Report exact no-change/diagnostic status; never invent improvement. W01c/S03 remain potential independent ASR errors.
- **C1:** all 29 source words, their alignment and single clean utterance remain intact.
- **C2:** both truly repeated interjections plus the concurrent other speaker remain intact.
- **1327/1329-second shouts:** both distinct acoustic occurrences remain; do not collapse them.
- **Other-language controls:** previously supported Mandarin CTC character units and unknown-speaker paths must not be newly deleted. Report actual deterministic test outcomes.

Quantify exactly: raw units=2415, total surviving units/utterances, legacy vs new seq dedup counts, extended recovery source words, falsely promoted tokens, valid/untimed/suspect timing, micro-utterance numerator and denominator, speaker counts (unchanged by intentional archive mapping), false speaker merges and stable control counts. Record every W/C/S case with PASS/PARTIAL/FAIL against a fixed factual criterion. Smaller fragment count alone is not an acceptance criterion. Never report 0 unresolved just because a shifted phrase was deleted without a reliable time.

### Stop / interpretation

If any source speech is lost, controls harmed, or original input mutated, stop and mark FAIL with native reproduction. A safe W04c+W05a fix and W03a/W02c partial improvement is a legitimate PARTIAL_A_B handback; it is not full Issue #14 closure. If deterministic tests fail, do not move on to GPU or try semantic host patches. Real acoustic comparison is optional only if the archived CPU artifacts fundamentally cannot distinguish a remaining high-impact case and legitimate owned-model admission is safe. No whole-episode model inference or new retranslation at this phase.

## Phase D: sanitized GitHub evidence carrier

Publish ONE self-contained manifest-hashed evidence-only commit to this same draft PR branch (no new branch/PR/issue). One source-bound packet under EVIDENCE_ROOT containing:
- README.md with scope and tested-source plus evidence carrier identities;
- source-identity.json with full SHA/tree, immutable baseline digest identities, private import/host isolation facts;
- validation/commands.md with full actual commands/exits/counts/failures/warnings;
- validation/cases.jsonl, one bounded row for each mandatory case and control, source token indexes and direct before/after;
- validation/metrics.json with same-denominator comparisons of original baseline, failed candidate and current source;
- validation/causal-assessment.md with root cause, remaining drift/ASR/diarization limits, unsafe output decisions and follow-on priority;
- validation/acoustic.md with model comparison NOT_ATTEMPTED or genuinely owned bounded inference evidence;
- review/acceptance-matrix.md, review/findings.md and COMMAND_SHAPE_FRICTION.txt;
- manifest.json (relative path, SHA256, exact byte size; no self-hash necessary).

Never publish copyrighted media, full transcripts, raw chunk JSON, WAVs, unredacted logs, private configuration or credentials. A small sanitized read-only replay script may be included if reproducibility requires it; do not hardcode private host paths. Keep private raw per-token traces in retained chmod-700 scratch. Do not accidentally commit generated __pycache__ or dependencies.

Before push, inspect diff and remote HEAD; permit only the evidence root paths, or the separately qualified mechanical lane. Use one ordinary non-force evidence commit, then read back GitHub branch SHA/tree, verify changed paths and 10/10 or actual manifest-member SHA256 and byte parity. On branch drift or publish denial, return an audited private sanitized archive and BLOCKED publication. Never force/rebase, mutate main, close Issue #14, request Codex or mark PR ready.

## Required handback

LOCAL_AGENT_HANDBACK:
  RESULT=PASS|PARTIAL_A_B|FAIL|BLOCKED
  TESTED_SOURCE_SHA=f0dfc994947c3319df76bf7d60563ef3352536c1
  TESTED_SOURCE_TREE=d60cc4e53be49ecec7158a5776f076af9603cc43
  MAIN_SHA=<fresh full SHA>
  PR_HEAD_SHA=<source-bound evidence carrier full SHA>
  SOURCE_EDITS=<none|bounded exact mechanical-only change>
  HOST_ISOLATION_AND_ORIGINAL_INPUTS=PASS|FAIL
  TEST_FOCUSED=<count/pass/fail/exit>
  TEST_NON_GPU=<count/skips/warnings/exit>
  TEST_STUDIO=<count/skips/exit>
  W04C_CORRECT_WORD_PRESERVED=PASS|FAIL; evidence by c13/c14 source index
  UNINTENDED_SUFFIX_PROMOTIONS=<none|exact source unit IDs>
  W05A_SIX_WORD_COVERAGE=PASS|FAIL|PARTIAL; source unit IDs
  W03A_INTERJECTION_AND_DUPLICATE_INTEGRITY=PASS|FAIL|PARTIAL
  W02C_DISPLACED_DUPLICATES=PASS|FAIL|PARTIAL
  W04B_COMPLETE_NEIGHBOR=PASS|FAIL|PARTIAL
  CONTROLS_C1_C2_SHOUTS_MANDARIN_UNKNOWN=PASS|FAIL; evidence
  TIMING_AUDIT=<quarantined/review-only counts; unresolved cases>
  THREE_WAY_METRICS=<baseline vs failed vs candidate>
  CASE_MATRIX=<W01..W05, S03/S04, controls; explicit unresolved>
  GPU_COMPARISON=<not_attempted|blocked|performed>
  EVIDENCE_GITHUB_ROOT=<carrier SHA:path|BLOCKED>
  MANIFEST_HASHES=PASS|FAIL|BLOCKED
  PRIVATE_SUPPORT_REQUIRED=<none|bounded reason/files>
  COMMAND_SHAPE_FRICTION=<encountered=no/events=[] or typed records>
  BLOCKING_FINDINGS=<none|exact IDs and reproducer>
  ISSUE14_AND_PR15_STATE=<OPEN/DRAFT>

Central will review this ONE host packet, then make further substantive changes for any remaining A/B defects and subsequent Issue #14 speaker/translation fidelity on PR #15. The agent does not own semantic source remediation or release decisions.
