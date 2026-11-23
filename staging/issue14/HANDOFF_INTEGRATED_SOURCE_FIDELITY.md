# Issue #14 / PR #15: consolidated source-fidelity hardening and integrated acceptance gate

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=e3b10a94e0a1214856aef7ade1258e1f4a5a4d41
CENTRAL_SOURCE_TREE=2382265517842fb562135d00eafc6d78a5442fb6
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CONTROLLING_DESIGN=staging/issue14/INTEGRATED_SOURCE_FIDELITY_DESIGN.md
PRIOR_HANDOFF_EVIDENCE=f1cb408b873379856c2f1803155379eebef2a2fe:staging/issue14/evidence/20261123T034336Z/
HOST_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-promotion-private/issue14-source-promotion-evidence-20261123T034336Z.tar.gz
HOST_PRIVATE_ARCHIVE_SHA256=a15ca6b60550eebc19cf11d619492262ccffd7ee9ba887eac0f1ec8c02f3bdf0
MODEL_QWEN=Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5
NEW_PR=FORBIDDEN; MERGE=FORBIDDEN; PROD_DEPLOY=FORBIDDEN; ISSUE_CLOSE=FORBIDDEN

## Mission and authorized workflow

Complete ONE source-fidelity and real subtitle acceptance campaign on the Central staged production code. Do not begin a generic ASR research, full-episode human-verbatim transcription, one-case micro-slice campaign, model retraining or renderer Issue #9 rewrite. Unlike the previous pass, do not count extra subtitle cues as recovered correct audio: use the exact original source observation identities and independently heard event evidence. Central has implemented the new model crop transcript-to-event plausibility screen and immutable legacy-to-reviewed-audio-event binding, with tests. Local work primarily involves host execution, fixed private evidence adapters, optional bounded audio listening, owned GPU admission, actual subtitle output, and one self-contained evidence handback.

The agent may solve routine environment, fixture, and test-only adapter friction autonomously, and independently inspect the raw/source/English outputs. It may add bounded host diagnostic scripts in private scratch. Do not rewrite Central acceptance algorithms, relax or skip a regression, invent positive audio references, silently modify default model policy, mask an ASR failure or replace a partial legacy source observation by English-text similarity. Escalate substantive source algorithm failures with exact traceback and a minimal reproducer.

## Preparation and source admission

Fresh OpenCode Build from /workspace/nvidia-transcribe (or authenticated canonical path). Refresh GitHub exact PR/main/head/issue states and SHA ancestry. Confirm CENTRAL_SOURCE_SHA unchanged under current PR head; no post-source src/tests drift. Read full AGENTS.md, INTEGRATED_SOURCE_FIDELITY_DESIGN.md, source_unit_lineage.py, source_promotion.py, source_recovery.py, source_reconciliation.py, audio_regions.py, source_event_replay.py, translation.py and all related tests. Use a new isolated worktree, not the operator's production checkout, and a fresh private output root. Never delete/reset/rebase/stash/force any operator data. Re-authenticate private tar SHA and relevant prior manifests. If path changed, discover the actual mounted prior evidence without falsifying provenance.

## Phase A: focused and full CPU gates

Run compileall and exact-source tests:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_source_promotion.py tests/test_issue14_source_unit_lineage.py tests/test_issue14_audio_regions.py tests/test_issue14_source_recovery.py tests/test_issue14_source_reconciliation.py tests/test_issue14_source_event_replay.py tests/test_issue14_source_to_renderer.py tests/test_viewing_quality_regressions.py
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  git diff --check

Record exact paths, source SHA/tree, exits, counts, traces and any warnings. The prior 379 passing non-GPU tests apply ONLY to earlier source 31529fc3e4fd37ca29b4d9d367483faeb5f44faf. If tests reveal a defect, diagnose and return the root cause, not a bypass. Focused CPU test contracts are acceptance guards.

## Phase B: replay real model/source-unit lineage

Use authenticated historic original16 FR and EP166 ZH v1 recording contracts, exact source_event_replay ledgers and their matching reconciliation dossiers. Run src/nvidia_transcribe/source_unit_lineage.py independently in private scratch:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m nvidia_transcribe.source_unit_lineage --recording-contract <PRIVATE_BASELINE_V1> --source-evidence <PRIVATE_EXACT_SOURCE_LEDGER> --reconciliation-dossier <PRIVATE_SAME_SOURCE_DOSSIER> --output <NEW_PRIVATE_LINEAGE_JSON>

First verify raw-response SHA and owning inference audio input bounds for EVERY source chunk represented in the ledger: different full35 vs original16 chunking MUST NOT be cross-joined just because media fingerprint and chunk ordinal match. Reproduce exact original-unit pairing and independently reviewed unit links. Prior archive data suggest FR original16 2051/2051 old lexical units exact matched; ZH 2832/2901 with 69 remaining unmatched due to original alignment source changes. Do not promote that expectation into PASS without fresh exact-code replay.

Classify FR c28 reviewed event (original16 utt-0193 all 9 units), W04c displaced/partial source units, separate C2 repeated speech, H2 originally missed opening, false c30 1203s audio bleed. Classify ZH 525.16s switch, 41 untimed, full vs partial/CTC-displaced legacy candidates (especially utt-0062/0063/0064/0065). Audio event mapping is NOT speaker identity, word-level timing or source reference correctness. Never delete partial legacy rows. Record full event/legacy IDs, reason labels, positive mapping evidence SHA and source observability.

## Phase C: targeted context-contamination adjudication

Re-run source_promotion.py with prior exact model results and audit source_evidence using the Central inputs. Compare pre-hardening 7 FR accepted (2 reviewed, 5 provisional) to new outcomes. Verify the known four contaminated model-provisional cases are now quarantined by model_text_scope_exceeds_audio_event_duration, retaining source text privately for review rather than displaying 18-25 words in 2-4s events. Confirm H1 short provisional (if otherwise justified) and reviewed W04c/H2 real speech survive, along with W05 omission evidence and both CTRL_c2 repeated turns.

Optional and bounded local acoustic recovery for four quarantined FR events (W05 S05, W05 closing, W04b question, distant call), plus c28 and positive/negative controls: listen directly to a few precisely cropped audio clips using available native audio review, compare corroborating historic Nemotron/Qwen hypotheses, and, if actual independent agent hearing verifies exact event words, supply a NEW agent-audio-review-attested event in a separately authenticated private dossier. This does not constitute human WER/CER truth. Do not simply re-label the old 8-10s entire Qwen output as an audition. Record event-level time, language, review method/evidence digest and actual bounded transcript; if ambiguity remains, quarantine and report. Only perform extra owned Qwen inference or forced aligner checks when a focused acoustic decision genuinely needs them; avoid downloading models or rerunning broad surveys.

For exactly matched reviewed legacy source events, issue explicit supersedes_legacy_utterance_ids only when every original unit maps to the same reviewed audio event, all responses/inference audio inputs are authenticated, original time is plausible and the complete eligible set is identified. Execute the opt-in source_promotion.py --source-evidence path. Confirm a fully covered already-present c28 event does not get appended as a duplicate without explicit replacement, true distinct repeated speech remains, ZH reviewed switch can replace only wholly matched same-event source if justified, and partially mapped utt-0062/early W04c CTC discrepancies are preserved as unresolved. Candidate source text must retain digest, language and model provenance. No automatic source truth labels, fabricated timing, speaker IDs or dropped original observations.

## Phase D: one comparable integrated subtitle acceptance

Run one complete French 23-minute episode-derived opt-in v1 recording-contract -> source translation -> actual English SRT/ASS gate in a fresh private output root, using the exact existing authenticated original16 recording as source baseline. Riva translation on GPU is permitted under existing Nemovox ModelLifecycle safe admission; preserve unrelated desktop+embedding services. Compare PREVIOUS FR baseline 234/309 Riva cues, previous 241/320 promoted Riva cues (unsafe context leakage), and NEW vetted contract. Use identical renderer/lane/translation configs for fair counts, and exact source/target lineages. Check W04b, W04c, H2, c28, S05, S02 negation, speaker handoffs, repeated C2, song/SFX, source timing, duplicate-looking cues, high CPS and unrenderable lines. Use reviewed annotated source samples as bounded tests; do not claim full WER/CER.

For Mandarin EP166, run the same source-lineage gate, preserve every untimed/unaligned unit without fabricated timestamps and test the rapid 525.16s turn handoff. IMPORTANT: previous 247 cues were generated by target backend none (source-language output), so do not compare them as English accuracy. Run a bounded Mandarin->English Riva sample and, resources permitting, one comparable full English subtitle render under the same lane config as the baseline, separately reporting inability to render the 41 untimed utterances. If a full English run is not practical, label it NOT EXECUTED/UNVERIFIED; never call source-language SRT an English pass. Primary full Nemotron ASR c28/c14 generation failures remain explicit evidence; run a fresh full primary-model episode only if it resolves a distinct uncertainty not already established by strict replay, and only in a new isolated job.

## Phase E: publish one complete auditable handback

Create one sanitized public evidence packet on the SAME PR branch under staging/issue14/evidence/<new UTC run>/, with canonical manifest including all member relative paths, byte counts and SHA-256. Include README.md, source-identity.json, validation/cpu-gates.json, validation/legacy-unit-lineage.json, validation/context-scope-gate.json, validation/positive-source-event-cases.md, validation/source-to-english-integrated.json, validation/mandarin-episode-integrity.json, validation/resource-nonmutation.json, review/issue14-acceptance-matrix.md, COMMAND_SHAPE_FRICTION.txt, and manifest.json. Private review-ready self-contained tar.gz should include bounded exact source snippets/response proofs, source-unit identity tables, changes vs original, host logs/test exits, translation comparison, limited source/audio review records, and checksum manifest. Never publicly upload actual WAV, media, copyrighted full source/translation transcript, raw complete vendor output, model weights or secrets.

One GitHub non-force exact-head evidence commit. Readback all public file hashes/bytes and manifest hash. Confirm source SHA/tree unchanged, issue OPEN and PR DRAFT. If substantive source production defects arise, return PARTIAL/BLOCKED with trace and specific Central-owned change, not silent "PASS". Do not create another PR, merge, request external review prematurely, close issue #14, change Studio service or deploy.

Return full:
ISSUE14_INTEGRATED_SOURCE_FIDELITY:
  RESULT=PASS|PARTIAL|BLOCKED
  ADMITTED_PR_HEAD=<fullSHA>
  VALIDATED_SOURCE_SHA=<fullSHA>
  VALIDATED_SOURCE_TREE=<fullSHA>
  BASE_MAIN_SHA=<fullSHA>
  CURRENT_PR_HEAD=<fullSHA>
  CPU_GATES=<compile/focused/full exits counts warnings>
  SOURCE_DIFF=<none or full path + diff>
  PRIVATE_ARCHIVE_INPUT_SHA=<verified exact SHA>
  UNIT_LINEAGE_FR_ZH=<raw digest, match counts, missing, event ownership, omissions>
  CONTEXT_TEXT_GATE=<old and new accept/quarantine, 4 high-density cases, controls>
  VERIFIED_SAME_EVENT_SUPERSESSION=<old IDs, event IDs, positive reviewed unit hashes, no deletion beyond proven event>
  W04C_H2_C28_W05_S02=<source positive/negative case findings; unresolved explicit>
  FRENCH_EPISODE_ENGLISH=<matched config, utterances, Riva SRT ASS counts, source text validity and ambiguity>
  MANDARIN_EP166_ENGLISH=<Riva vs none distinguished, overlap, 41 untimed and remaining deficits>
  REAL_MODEL_EXECUTION=<if any, checkpoint/VRAM/reason/result; otherwise reused>
  GPU_HOST_NONMUTATION=<actual state>
  PUBLIC_EVIDENCE_ROOT=<exactsha:path>
  PUBLIC_MANIFEST_SHA256=<digest>
  VERIFIED_PUBLIC_MEMBERS=<count/pass>
  PRIVATE_REVIEW_ARCHIVE=<path sha256 size>
  WER_CER=<UNVERIFIED unless real human reference>
  ISSUE14_ACCEPTANCE=<PASS|PARTIAL, exact unmet points>
  REMAINING_CENTRAL_BLOCKERS=<list or none>
  ISSUE14=OPEN PR15=DRAFT
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events=<[]|canonical records requested_shape, effective_shape, rewrite_provenance, event_kind, execution_effect, correction, replacement_shape, recipe_reference, resource_reference, contract_conflict, durable_gap_candidate>
  omitted_sensitive_values=<yes|no>

Do one consolidated campaign and hand back complete evidence. Keep a strict distinction between source semantics actually repaired and mere improvement in a display count. Substantive central algorithm failures require exact reproducible report, not an independently invented acceptance policy.
