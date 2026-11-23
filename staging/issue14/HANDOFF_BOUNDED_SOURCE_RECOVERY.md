# Issue #14 / PR #15: one bounded source-recovery + lineage host campaign

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CENTRAL_SOURCE_SHA=a46490266f5f161057eeb47daf2fee3a07bb2084
CENTRAL_SOURCE_TREE=960fc7abf24a8ad57db275e179e019bd23d6863e
CONTROLLING_DESIGN=staging/issue14/BOUNDED_SOURCE_RECOVERY.md
MODEL_EVIDENCE=cbebbc0e13b854bc866f76e96932936699f1dd67:staging/issue14/evidence/20261123T011017Z/
SOURCE_EVIDENCE=1bbcb3c8511e2bf2dfd5798d88d44789068549b5:staging/issue14/evidence/20261123T004035Z/
NO_NEW_PR=yes; NO_MERGE=yes; NO_CODEX=yes; NO_PRODUCTION_ACTIVATION=yes

## Mission and permitted scope

One integrated Build campaign: validate the Central-authored source_recovery.py recovery contracts, lineage-preserving translation and SRT/ASS changes, and authenticated offline French/Mandarin source-event plans. Then run one source-bound, independent, language-pinned Qwen3-ASR 1.7B bounded coverage pilot/sweep if safe GPU admission is available, and return a complete sanitized evidence packet. No new model comparison shopping, no unrelated project refactors, and no separately created PRs.

Central owns all substantive production algorithms, language-model routing, source-event acceptance, schema migrations and canonical reconciliation. Build owns exact-source host execution, test-only vendor SDK interfaces in private evaluation scratch, resource admission, localized mechanical repairs and complete results. Do not use green tests as a license to design an alternate production algorithm locally.

## Identity, isolated environment and admission

Launch a fresh Build process from the authenticated Nemovox host checkout, ordinarily /workspace/nvidia-transcribe. Read nearest AGENTS.md, current docs/model-resources/LIFECYCLE.md, BOUNDED_SOURCE_RECOVERY.md, source code and its tests. Fetch the live remote issue branch before relying on a cached object; prove that PR #15 is OPEN/DRAFT, issue #14 OPEN, the fetched branch equals the exact remote PR head, CENTRAL_SOURCE_SHA is an ancestor, and src/tests have not changed after CENTRAL_SOURCE_SHA unexpectedly. Record current base-main full SHA. Create a fresh collision-safe private worktree at the fetched SHA, leaving the operator's original checkout, media, /workspace/pyannote, model weights, host Studio and external services unchanged. No git reset, force, rebase, clean, stash or unrelated service stop.

Run (workdir: isolated repository root; verify actual Python interpreter path first):
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu" tests/test_issue14_source_recovery.py tests/test_issue14_source_to_renderer.py tests/test_viewing_quality_regressions.py tests/test_issue14_source_evidence.py tests/test_issue14_source_event_replay.py
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
  git diff --check

Test the explicitly changed behavior, not just exit status: distinct English from different source event IDs must appear in SRT and ASS; two render copies of the same event at the same audio time may collapse; v1 source contracts should still load; translation regrouping must retain distinct v2 source-turn IDs; no fabricated timestamp and no speaker assignment from unverified acoustic activity. Existing test_viewing_quality_regressions was substantively updated from a previously unsafe English-text dedup expectation; do not revert or waive that new source-preservation invariant.

Build may correct obvious test-only import/fixture/workspace or model SDK parameter mismatches. If a new defect is semantic, or would require rewriting source acceptance, producer contracts, translation logic, timestamps or tests to fit desired outcomes, return the specific blocker to Central rather than doing substantive development. Keep routine in-scope environment fixes in this single campaign.

## Cached source observations and soft acoustic activity

Authenticate the prior 16-chunk French, 35-chunk French and EP166 Mandarin private ASR sources using their own media fingerprints, manifests, chunk response digests and actual crop offsets. Do not fabricate missing data. Use source_event_replay in strict mode and explicit forensic-incomplete mode when legacy c28 and ZH c14 finish=length are rejected. Rejected responses remain incomplete despite any legacy completed manifest flag.

Adapt the previously measured FireRedVAD-AED, Silero, and pyannote short-clip results into a PRIVATE JSON:
  schema: nemovox-audio-activity-evidence/v1
  media_fingerprint_sha256: exactly the source-evidence media fingerprint
  intervals: [{id:<unique original source ID>,source:<actual model+revision>,kind:<speech|lexical_shout|singing|music|sfx|nonlexical_vocalization|unknown>,start_s:<absolute>,end_s:<absolute>}]
Do not present model activity as independently heard words. Only pass --auditioned-misses for independent acoustic annotations with IDs and acoustic_truth=independently_auditioned from the earlier direct-audition packet. Prepare an incomplete JSON array with exact c28 and c14 raw-response sha256 and actual infer bounds.

On each source create one non-overwriting private plan using:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.source_recovery --source-evidence <PRIVATE_SOURCE_LEDGER> --activity-evidence <PRIVATE_ACTIVITY_JSON> --incomplete <PRIVATE_INCOMPLETE_JSON> --auditioned-misses <PRIVATE_AUDITIONED_MISSES> --coverage-end-s <ACTUAL_MEDIA_DURATION> --max-requests 128 --output <NEW_PRIVATE_PLAN_JSON>

If a given optional evidence file is genuinely absent, omit its flag and report its missingness; never synthesize it. Compare selective-only and coverage-floor plans, with all planned/pending requests and reasons. Verify c28 multi-window coverage, W04c and H2 under ALL-detector-negative conditions, and the exact original media duration in French/Mandarin. No planned region or budget overflow may silently disappear. Never mark Qwen model output as an accepted source turn solely from model agreement.

## One bounded language-pinned challenger campaign

Reuse preexisting Qwen/Qwen3-ASR-1.7B @7278e1e70fe206f11671096ffdd38061171dd6e5 under /workspace/models. Do not download models or change owner-controlled services. Inspect nvidia-smi, actual GPU allocations, validated Nemovox-owned lease/admission and the approved host local test service process first. Run an 8-clip positive/negative pilot against known omissions and negatives: c28 rescue, W04c captain, H2 start, W05a, CTRL_song and CTRL_sfx; include one Chinese ordinary and one rapid handoff. Language must be explicit French or Chinese. Test cached model runtime bindings and preserve all raw results privately with input media/crop digests, model revision and status.

If the pilot succeeds safely, carry out one COMPLETE independent bounded Qwen observation sweep of the 23-min French source using the coverage floor, and optionally one Mandarin EP166 sweep on the same resource-safe footing; no repeat full-episode Nemotron or repeated tuning campaigns. A sequential Qwen-only challenger is appropriate because both VAD models shared true-voice blind spots. Preserve every generated response, excerpt bounds and failed/empty outputs separately, rather than recycling earlier code's guessed timestamps. Avoid unowned service eviction, model resource lifecycle bypass and implicit CPU device fallback. If GPU admission fails, return BLOCKED_RESOURCE for that phase while finishing CPU validation.

Reconcile Qwen candidate observations against independent audio EVENT IDs and prior Nemotron source observations; do not auto-select a canonical transcript by model majority, timeliness or forced alignment. Report genuine missed/hallucinated utterances and bounded case observations, explicitly WER/CER=UNVERIFIED unless privately verified verbatim references were actually created. Song/SFX Qwen false positives cannot become speech merely because AED missed or accepted another class.

If useful, validate inspect_forced_alignment against previous true/false text cases. Any zero-duration, overlong, pause crossing or false text remains rejected/structurally provisional; do not claim a structurally provisional alignment proves word existence. Reuse the prior 9-case aligner evidence when no new GPU evaluation materially changes the outcome. Scene Nemotron-3/episode pyannote evidence from prior campaign remains advisory; no new diarization benchmarking required unless genuine unexplained turn conflicts demand one short targeted test.

## Render preservation and compatibility

Re-render source-bound legacy v1 recording/translation fixtures read-only in scratch; and use project_source_turn_to_v1_utterance with separate source-turn IDs in translated fixture records. Track source event IDs from v2 projections through translate_utterances and build_display_cues/assign_subtitle_lanes/render_subtitles. Compare SRT/ASS counts and textual presence for repeated identical English from distinct speakers/events, exact source-copy duplicates and untimed unrenderable observations. Classify previously suppressed legacy duplicate-looking cues as possible unresolved ASR echo, not necessarily new true speech; upstream acceptance must solve false echoes later. Confirm default existing v1 checkpoint/resume behavior is unchanged beyond explicitly revised renderer content preservation.

## One public-safe handback packet

Create staging/issue14/evidence/<UTC-run-id>/ with exact source identity:
  README.md
  source-identity.json
  validation/cpu-gates.json
  validation/source-recovery-plan.json
  validation/cross-model-coverage.md
  validation/source-turn-lineage.json
  validation/runtime-and-resources.json
  review/architecture-reconciliation.md
  COMMAND_SHAPE_FRICTION.txt
  manifest.json

Include actual source/PR/base/tree, media identity, model revision, test commands/counts/exits, rejected c28/c14, full vs selective planned/executed/pending counts, Qwen real source positive/negative comparisons and resource nonmutation, unchanged/changed SRT/ASS counts, explicit uncertain conclusions and material unresolved source-event decision inputs. Public evidence is numeric or bounded descriptive only, no raw WAV, full transcripts, copyrighted episode material, large raw model payloads, credentials, weights or local secrets. Private detailed sample, logs and model responses remain under chmod-700 collision-safe scratch. Manifest SHA-256 and bytes must cover ALL public files except manifest. Check ignored *.jsonl and staged actual file inventory explicitly to prevent prior missing-member error.

One non-force exact-head evidence commit on existing branch, after fresh remote readback. Do not rebase on concurrent drift. If source fixes were legitimately needed and Central-authorized, record full source diff and rerun gates on the new exact source SHA before evidence publication. Do not open PRs, merge, request Codex or deploy.

Return:
  ISSUE14_BOUNDED_SOURCE_RECOVERY:
    RESULT=PASS|PARTIAL|BLOCKED
    ADMITTED_PR_HEAD=<40 sha>
    FINAL_SOURCE_SHA=<40 sha>
    FINAL_SOURCE_TREE=<40 sha>
    CURRENT_PR_HEAD=<40 sha>
    MAIN_SHA=<40 sha>
    FOCUSED_AND_FULL_TESTS=<exit/counts>
    FRENCH_ORIG16_AND_FULL35=<identities/plans/c28>
    MANDARIN_EP166=<identities/plans/c14>
    QWEN_FULL_COVERAGE=<actual model/crops/candidate omissions/false content>
    W04C_H2_JOINT_VAD_MISS=<proof of scheduled coverage floor>
    SOURCE_EVENT_ID_RETENTION=<translation/SRT/ASS counts and actual lineages>
    FORCED_ALIGNMENT_TRUTH_LIMIT=<result>
    RESOURCE_NONMUTATION=<evidence>
    NEW_SOURCE_DIFF=<none|exact diff and sha>
    PUBLIC_EVIDENCE_ROOT=<commit:path>
    MANIFEST_SHA256=<digest>
    REQUIRED_FILES_COMPLETE=<yes|no>
    WER_CER=<verified denominators|UNVERIFIED>
    NEXT_CENTRAL_OBJECTIVE=<one cohesive implementation>
    PR15=DRAFT ISSUE14=OPEN
  COMMAND_SHAPE_FRICTION:
    encountered=<yes|no>
    events=<[]|list of: scenario, requested_shape, effective_shape, rewrite_provenance, event_kind, execution_effect, correction_or_annotation, replacement_shape, recipe_reference, resource_reference, contract_conflict, durable_gap_candidate>
    omitted_sensitive_values=<yes|no>

The Build campaign ends at a source-grounded PASS/PARTIAL/BLOCKED handback. A green planner does not mean the new episode transcript is accurate or issue #14 is complete.
