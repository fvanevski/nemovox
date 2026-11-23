# Issue #14 / PR #15: validate reversible reviewed-turn translation and private subtitle rendering

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=17bb0239f7d8290d43475b06055cd95c2418a1d6
CENTRAL_SOURCE_TREE=1dded4da4ed118632724b69eb83b92f40a80153d
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CONTROLLING_DESIGN=staging/issue14/REVISION_SUBTITLE_INTEGRATION_DESIGN.md
PREVIOUS_PUBLIC_EVIDENCE=251d4b8471d62a49c2a3404a288b66369e7ff7bd:staging/issue14/evidence/20261123T194055Z/
PREVIOUS_PRIVATE_ARCHIVE=/workspace/nemovox-issue14-evidence-fusion-private/issue14-evidence-fusion-evidence-20261123T194055Z.tar.gz
PREVIOUS_ARCHIVE_SHA256=16ef680e419cc2bbe71fa9607e17a6c6323a104b7e23b468e13b5a915a7bc224
PREVIOUS_REVISED_CONTRACT=/workspace/nemovox-issue14-evidence-fusion-private/partitions/private_revised_recording.json
PRIOR_DEFAULT_SOURCE_ROOT=/workspace/nemovox-issue14-default-recovery-private/default_fr_job/
NO_NEW_PR=yes
NO_MERGE=yes
NO_DEPLOY=yes
NO_ISSUE_CLOSE=yes
NO_LIVE_PIPELINE_MODIFICATION=yes
NO_MODEL_DOWNLOAD=yes
NO_FULL_EPISODE_ASR=yes
NO_WHOLE_EPISODE_RETRANSLATION=yes

## Goal

ONE consolidated host qualification of the newly authored Central **source_fusion_downstream.py**, using the real French 2008-unit source evidence and already qualified six complete review components. Produce a private side-by-side revised subtitle result with only the nine independently reviewed replacement turns newly translated, preserving the 209 untouched English turns, plus a bounded positive/negative audit of the unresolved c11:0..3 raw overlap words. Evaluate whether a general, tunable evidence-to-correction scaffold now demonstrably improves the rendered subtitle, without per-case production patches or bypassed source lineage.

## A. Exact worktree authority and CPU regression

Fresh isolated OpenCode Build worktree at verified current DRAFT PR #15 HEAD; confirm Issue #14 OPEN, main, source SHA/tree, no source/test drift since CENTRAL_SOURCE_SHA, and read AGENTS.md plus design. Reverify previous private archive 28/28 SHA, public manifest 9/9 SHA and exact GitHub readback, source media, source-evidence and review dossier IDs. Do not overwrite operator's working tree or previous jobs. Use actual existing host core/Studio Python interpreters, CWD of isolated worktree, PYTHONPATH=src:. (not simply PYTHONPATH=src). Run:
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_issue14_source_fusion_downstream.py tests/test_issue14_source_fusion_application.py tests/test_issue14_evidence_fusion.py tests/test_issue14_source_unit_lineage.py tests/test_issue14_source_promotion.py tests/test_issue14_source_reconciliation.py
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'
  PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'
  git diff --check
Verify both new module __file__ imports resolve to isolated candidate, not main checkout. No test skipping, semantic weakening or local architecture redesign. A true test failure returns PARTIAL with traceback; do not quietly amend production source. Check any command path against actual file availability.

## B. Reversible local acoustic voice overlay (CPU)

Use the authenticated original recording, source-evidence.v1 and candidate_reconciliation_dossier, and compare source_fusion_revision on PREVIOUS_REVISED_CONTRACT to the original recording exact digest and event/revision identities. Recreate the six complete selected component plan if needed, but do not replace incomplete original components. Use existing W02b independent audio attestation and evidence review SHA to build a **private** nemovox-local-acoustic-voice-review/v1 with opaque labels voice_0 and voice_1 in a source/audio-review-specific scope. Link only positively attested W02b events (w02b_ev_3, w02b_ev_4_ord100, w02b_ev_5_ord101_127, w02b_ev_6_final) according to their independently reviewed same-voice assessments. The quiet breath [231.95,232.45] is UNRESOLVED; do not label it. Carry separate direct-audio review evidence SHA for new voice review (not a fabricated hash), exact reviewed event bounds and a scope SHA. Do not infer character names from subtitles/appearance or map local voices to global SPK numbers.

Run attach_local_acoustic_voices() on PRIVATE in-memory source. Assert all reviewed turns retain speaker_id=SPK_UNK and speaker_identity_accepted=False, original contract remains byte-identical, audited local voice metadata references the exact source event/revision and review scope, and wrong scope/unknown event/character-name labels fail. If the existing audio review cannot justify equality for any pair, leave the uncertain event without local label instead of forcing an alternation pattern.

## C. Revision-aware translation and private rendering

Obtain original translated.contract.json from PRIOR_DEFAULT_SOURCE_ROOT/subtitles/translated.contract.json, same-identity recording.contract.json and the previously generated 218-turn PRIVATE revised recording. If files are no longer extracted under PRIOR_DEFAULT_SOURCE_ROOT, retrieve ONLY these exact members from the previously authenticated /workspace/nemovox-issue14-default-recovery-private/issue14-default-source-recovery-evidence-20261123T174459Z.tar.gz into NEW private scratch paths after checking each member against the original archive checksum manifest. Do not assume a directory is permanently mounted or alter the original archive. Validate original and translated media fingerprints, original 217-utterance ID set, exact source text/time for 209 retained turns, and that the eight superseded IDs are removed from the revision. Use assemble_revised_translation(original_recording, prior_translated, revised, [],enable=False) to enumerate missing fresh translation IDs. The list must correspond to the nine new source-revision turns, not 209 untouched turns. No model access needed for preview.

If safe GPU admission is possible under Nemovox ModelLifecycle, perform ONE bounded existing-Riva translation pass ONLY over these nine French reviewed turns using translation.translate_utterances() with the actual configured Riva checkpoint/device/resource. Respect owned GPU lease, preserve Studio/embeddings, no unowned services stopped, record before/loaded/after VRAM and Riva results. Do not introduce paid/cloud API. Prefer fresh private checkpoint/root so stale prior English cannot contaminate revised source. Translate source_text as is; don't use visual hypotheses as French source words. Emit new translation receipt per turn with original ID, source_revision_id, SHA256 of its source_text, source/target language, status=translated, actual translated_text, and actual translation-result SHA256 from authentic worker output. If translation fails or no lease available, report NOT_EXECUTED/blocked and finish CPU structural preview without inventing accepted English output.

Call assemble_revised_translation(..., new_translations, enable=True) only if ALL nine fresh receipts pass; write the returned translated contract to a NEW PRIVATE root and preserve its SHA and source_fusion_revision. Never overwrite the original translated.contract.json, source contract, default subtitles or Studio history. Independently compare old/new IDs and English text for all 209 retained turns, verify removal of stale source versions and no grouped references to replaced IDs. New turns must carry separate local voice metadata if accepted, while retaining global SPK_UNK and unverified word alignment. Do not change source_promotion legacy 3s supersession rules.

With the accepted private translated contract, use existing subtitles.render_subtitles in a NEW PRIVATE output root to create side-by-side .srt/.ass (never directly in the user's library). Report exact cue lineage for W02b, W03b, W04a, S02, W05a, C2 and S05. Compare actual event timing against original acoustics, orphaned/overlapping cues, 18/22 CPS with same denominator, duplicate/omitted dialogue, speaker uncertainty, and English meaning/polarity. Do not modify Issue #9 ASS lane scheduler or claim word timing where only event bounds exist. If renderer errors, preserve traceback and test reproduction. A parseable SRT alone is not proof of correct English.

## D. Pending pre-v1 raw duplicate disposition (small evidence-only)

Read raw ledger c10:75..78 vs c11:0..3, the existing corr_547022f83ef1499b34f47816 record, current dossier event w03a_bateau_volant, and the authenticated [984,1013.979] WAV. They are possible overlapping ASR observations, but both correspondence evidence and dossier mark same-event identity **UNVERIFIED**. No assumption that lexical equality proves duplication. Independently listen specifically to the short ~985-989s phrase and inspect original overlapping source ASR inference bounds. Return one *proposed* general raw unit disposition record containing source/alternate raw unit identities, parent ASR SHA and inference windows, positive independently reviewed audio-event evidence SHA, actual audible repeat/one-occurrence result and explicit confidence/unknowns. Do not change core dedup or mark four orphan units resolved without positive evidence. If not independently confirmable, preserve the blocked component and move on.

## E. One evidence packet + contract verification

Return one public sanitized evidence-only commit on SAME PR #15 and one private self-contained archive with unique tar entries/checksums. Public: exact source/test identity, core/Studio gate counts, reviewed voice overlay positive/negative, nine-turn translation receipts and retained English hash comparison (no full copyrighted transcript), private render metrics and representative bounded case results, raw duplicate candidate disposition status, root cause of any failure, service VRAM nonmutation, command-shape friction. Private: full small revised source/translated contract, selected Riva output metadata, per-turn rendered cue side-by-side and audio annotations, minimal bounded sampled assets, test logs and SHA manifest. No full episode video/audio, weights, secrets or cloud endpoints publicly.

Required handback:
 ISSUE14_FUSION_DOWNSTREAM:
   RESULT=PASS|PARTIAL|BLOCKED
   ADMITTED_PR_HEAD=<full SHA>
   VALIDATED_SOURCE_SHA=17bb0239f7d8290d43475b06055cd95c2418a1d6
   VALIDATED_SOURCE_TREE=1dded4da4ed118632724b69eb83b92f40a80153d
   CURRENT_PR_HEAD=<full SHA>
   MAIN_SHA=<full SHA>
   SOURCE_DIFF=<none|paths>
   CPU_FOCUSED_FULL_STUDIO=<counts/exits/import identity>
   SOURCE_ARCHIVE_AND_PRIOR_PUBLIC_AUTH=<28/28, 9/9, source, translated and dossier digests>
   REVIEWED_LOCAL_VOICES=<W02b exact event scope/labels, unknown breath, global SPK_UNK and failure controls>
   REVISION_TRANSLATION_PREVIEW=<9 missing new IDs, 209 retained, input source/digest>
   NINE_TURN_RIVA=<EXECUTED/NOT_EXECUTED, model resource, result hashes, GPU/lease results>
   REVISED_TRANSLATED_CONTRACT=<SHA, 218 rows, 209 retained, 9 new, no stale English/IDs>
   PRIVATE_SRT_ASS_COMPARE=<cue counts, W02b/W03b/W04a source fidelity, polarity, CPS and unknowns>
   RAW_OVERLAP_DISPOSITION=<c10:75..78/c11:0..3 one-vs-two occurrence evidence, candidate/verified/unknown>
   ADOPTION_GAPS=<remaining generalized policies, required provenance, renderer/word timing, source-fidelity status>
   PUBLIC_EVIDENCE_ROOT=<sha:path>
   PUBLIC_MANIFEST_SHA256=<sha>
   PUBLIC_MEMBERS_VERIFIED=<number>
   PRIVATE_REVIEW_ARCHIVE=<path SHA256 bytes>
   HUMAN_WER_CER_DER=UNVERIFIED unless human reference
   ISSUE14=OPEN PR15=DRAFT
 COMMAND_SHAPE_FRICTION:
   encountered=<yes|no>
   events=<[]|records>
   omitted_sensitive_values=<yes|no>

Stop/escalate only on changed authoritative PR source SHA, unsafely unavailable GPU lease, ungrounded immutable audio/source identity, or genuine contract correctness failure; continue safe CPU portions and return PARTIAL when needed. Do NOT invent new production architecture, patch case IDs into source, weaken validations, start long-form ASR/Whisper, or create extra PRs.