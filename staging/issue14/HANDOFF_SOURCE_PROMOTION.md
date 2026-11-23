# Issue #14 / PR #15: source-turn promotion and first integrated subtitle gate

LOCAL_AGENT_PRIMARY=build
REPO_ID=fvanevski/nemovox
ISSUE_NUMBER=14
PR_NUMBER=15
BRANCH=issue/14-longform-timing-boundaries
CENTRAL_SOURCE_SHA=31529fc3e4fd37ca29b4d9d367483faeb5f44faf
CENTRAL_SOURCE_TREE=d5c502f0ce0a5f070f046218cd124b99e6727df0
BASE_MAIN_SHA=8f6c03cbc70e84bf1970ba364207a02c9fe22b44
CONTROLLING_DESIGN=staging/issue14/SOURCE_PROMOTION_DESIGN.md
LATEST_HOST_PUBLIC_EVIDENCE=874bd718ac4a42e87e55a54ebad54a44edc33c52:staging/issue14/evidence/20261123T022122Z/
PRIVATE_SOURCE_HANDOFF=/workspace/nemovox-issue14-reconciliation-private/issue14_reconciliation_private_20261123T022122Z.tar.gz
PRIVATE_SOURCE_SHA256=56317262ac2bee3a912abf1f372e4ca71a7b54dfb1c27b2a91548b07eaea3df8
MODEL_QWEN=Qwen/Qwen3-ASR-1.7B@7278e1e70fe206f11671096ffdd38061171dd6e5
NO_NEW_PR=yes; NO_MERGE=yes; NO_PRODUCTION_ACTIVATION=yes

## One consolidated campaign

Qualify three new Central-authored modules together and produce an ACTUAL opt-in, v1-compatible source-turn overlay that can be translated and rendered into English SRT/ASS. The modules are source_event_replay.py (contract-bound language fallback), audio_regions.py (model-bound bounded crop proposals), and source_promotion.py (audited/provisional turn selection). Use the archived Qwen sweeps, audio activity, source ledgers and reconciliation dossiers as source evidence. There is no requirement to launch another broad model discovery campaign. A bounded new Qwen inference batch is legitimate because it directly tests the separate-crop requirement that the old 24s sweeps could not establish.

Do not leave semantic production coding to the local agent under the guise of host integration. Central owns the model agreement policy, event promotion algorithm, source acceptance and v1 contract semantics. Build owns host/source verification, existing vendor model invocation, runtime admission, bounded mechanical evidence-adapter scripts in isolated private scratch, fixture corrections and one fully recorded actual integrated episode test. A semantic test failure returns blocker and exact traceback, not lowered gates or silently changed source.

## Exact-source admission and safe host preparation

Launch a fresh OpenCode Build process using the authenticated repository workdir (ordinarily /workspace/nvidia-transcribe). Fetch the existing issue branch, verify live PR #15 OPEN/DRAFT, issue #14 OPEN, main SHA, current remote PR head, and that CENTRAL_SOURCE_SHA is in ancestry with src/tests unchanged after this commit. Review complete AGENTS.md, SOURCE_PROMOTION_DESIGN.md, source_recovery.py, source_reconciliation.py, source_event_replay.py, audio_regions.py, source_promotion.py, tests and host resource lifecycle guide. Prepare fresh collision-safe isolated worktree; never reset/clean/stash/rebase/force the operator checkout or existing evidence. All scratch private outside repo; preserve existing models, original media, service units and running GPU services.

Record admitted source SHA/tree and exact installed host Python. Use the proven interpreter path if still applicable, otherwise discover and record replacement. Run compileall and focused tests:
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src/nvidia_transcribe
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu" tests/test_issue14_source_event_replay.py tests/test_issue14_source_promotion.py tests/test_issue14_audio_regions.py tests/test_issue14_source_recovery.py tests/test_issue14_source_reconciliation.py tests/test_issue14_source_to_renderer.py tests/test_viewing_quality_regressions.py
  PYTHONPATH=src /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m "not gpu"
  git diff --check

Do not claim the new code passed because the prior 351 tests passed: those applied to fc38ba614ec2e145c1efd6ba8389db27a80efde0. If a test exposes an ordinary import/signature/environment failure, perform the smallest provably mechanical integration fix within scoped authority or return exact root cause for Central. Never weaken unit assertions, source identity, negative song/SFX controls, source-lineage preservation or default recording schema.

## Immutable source language and replay proof

For original16 FR and Mandarin EP166, authenticate manifest media fingerprint, recording.contract.json schema/source.fingerprint_sha256/source_language, raw chunk hashes and replay SHA. Verify new replay infers source language from SAME-MEDIA contract only when manifest omits it; a stale/mismatched contract fails closed. Do not edit historic manifest. For full35 forensic replay c28 and EP166 c14, rejected finish=length must remain incomplete regardless of legacy completed flag. If full35 lacks a matching v1 recording contract, do not create fake trusted source-language authority: use the authenticated original16 v1 contract for the opt-in integrated output and retain full35 evidence as forensic. The prior test-only language-adapted ledgers are allowed as bounded evidence but not represented as fresh official source replay.

## AudioRegion and bounded inference

Read the existing 209 FR and 21 ZH activity events (FireRedAED/Silero/pyannote), previously reviewed source-event regions, episode duration/fingerprint, and existing 63/60 archived Qwen coverage windows. Run audio_regions.py as the Central-authored proposal producer in private scratch. Supply scene-bounded speaker tracks only where their exact local identity/clip chronology is proven; distinct simultaneous speaker channels MUST stay distinct. Treat unlocalized 24s Qwen text as context, not a 2-second acoustic turn. Flag long activity intervals as requires_subsegmentation and do not automatically assign one text/speaker.

Choose one consolidated, representative bounded Qwen experiment over the same audio: W04c [1261.5..1263.5], H2 opening [362.5..364.2], c28 rescued [1131.5..1136], S05/W05, song [25..40], SFX [1200..1216], French rapid/real repetition, Mandarin ordinary [96..114], Mandarin handoff [518..536], and 2-4 ordinary unreviewed dialogue turns for model-only selection. Use crop context typically 3s on both sides; model-only crop must be no longer than min(16 seconds, event_duration + 8 seconds) and actually contain the proposed event. The old isolated 3s W04c crop hallucinated: document whether adding context helps without assuming correctness. Never turn a full 24s Qwen result directly into a short candidate transcription. For longer regions use smaller scene-separated candidates, do not silently split into invented source words.

Use existing Qwen model with explicit French or Chinese pinning, verified exact local checkpoint, safe sequential Nemovox-owned ModelLifecycle / GPU admission. No unrelated service eviction, installation, model downloads or off-policy global device changes. Capture source media fingerprint, WAV crop SHA256, raw/stored model response SHA256 (distinguish canonical stored result from native raw object), exact text SHA256, model repo/revision, language, crop/time, finish status. Full outputs/private audio stay off GitHub. If GPU phase fails safely, finish CPU proof and return PARTIAL with exact blocker.

## Actual opt-in promotion and English subtitle path

Produce one private nemovox-bounded-source-segments/v1 file per authenticated FR/ZH media from the new bounded model results. Use only event-specific content, never clip-wide 24s text. Make explicit candidate id/audio_event_id/event start/end, source text, bounded_asr model+revision/source_text_sha256/crop SHA/response SHA/finish status/language and optional context_window_ids and positive activity IDs. The agent may adapt raw host outputs to this exact Central schema, but not design new ranking, acceptance or source reconciliation heuristics.

Run the Central source_promotion.py CLI against a true intact v1 recording contract, matching dossier and segment evidence, with activity sidecar. Write only NEW private output contract; leave baseline contract unchanged. Record accepted model_source_candidate_provisional, direct-audio-review-attested source turns, quarantined candidates and reason codes, legacy overlaps and source/event IDs. Validate distinct repeated speech events survive, no missing original utterance is silently removed, only positively matching event lineage AND plausible original audio time permit supersession, no new speaker identity or word-level timing is fabricated. Preserve the original raw observation ledger and rejected c28/c14. Explicitly test unsafe events: Qwen theme-song words, short SFX exclamations, CTC drift, 24s text falsely assigned to short crop, single-detector speech over independently auditioned SFX, nonmatching audio/model text SHA, different model revision and non-positive VAD.

Use generate-subtitles --recording-contract with the opt-in contract path in a fresh private output root. Run one genuine French episode-derived translation and SRT/ASS rendering if the existing Riva/local translation backend can be admitted safely and correctly; otherwise document exactly where a source-only or fixture-based rendering ends rather than claiming end-to-end English. Run an additional Mandarin EP166 bounded episode fixture with 3 handoff turns and a source-preserving playback/render path, using ordinary locally available resources. Compare original16 and qualified source overlay: source event count, missing audited speech, supported new words, unrenderable cue count, duplicate-looking SRT/ASS cues, overlap/speaker lanes, false singing/SFX and context-word hallucinations. Subtitle visual layout/CPS belongs to issue #9; do not broaden the coding task.

The difference between agent-attested audio review and a human verbatim reference must be explicit. No full human-reviewed WER/CER is available; qualitative source-case checks do not imply complete transcript fidelity. Existing prior independent audio observations are evidence of hearing, not blanket authorization to trust any new model-only text.

## One evidence carrier and handback

Publish exactly one sanitized evidence commit under staging/issue14/evidence/<new-UTC-run-id>/ on the existing draft PR, after refreshing exact remote SHA and confirming no source drift. Expected public members:
  README.md
  source-identity.json
  validation/cpu-gates.json
  validation/replay-language-authority.json
  validation/audio-region-proposals.json
  validation/bounded-qwen-provenance.json
  validation/source-promotion.json
  validation/episode-source-to-render.md
  review/production-readiness.md
  COMMAND_SHAPE_FRICTION.txt
  manifest.json

All public fields must be safe: only counts, statuses, bounded time spans, SHA identifiers and essential descriptive conclusions. Never commit WAVs/media/weights, full raw source or translated transcripts, copyrighted songs, user secrets or giant logs. Keep complete private diagnostics/artifact archive (tar.gz self-contained, small samples, exact input/response hashes, logs/tests, diff and private candidate/dossier/recording paths; no large media/weights/secrets). Return host private archive path, bytes/digest for operator to upload if Central needs detailed review. Verify every public member SHA-256 and bytes against manifest, manifest digest, exact Git object sha and required paths after the public non-force commit. Do not create another PR, merge, close issue #14, request Codex, deploy or activate the new default pipeline.

Required handback:
  ISSUE14_SOURCE_PROMOTION:
    RESULT=PASS|PARTIAL|BLOCKED
    ADMITTED_PR_HEAD=<full_SHA>
    VALIDATED_SOURCE_SHA=<full_SHA>
    VALIDATED_SOURCE_TREE=<full_SHA>
    CURRENT_PR_HEAD=<full_SHA>
    BASE_MAIN_SHA=<full_SHA>
    SOURCE_DIFF=<none|exact_path_and_diff>
    CPU_FOCUSED_FULL_COMPILE=<commands/exit/count>
    REPLAY_LANGUAGE_BOUNDARY=<FR/ZH provenance, stale reject, c28/c14 incomplete>
    AUDIO_REGION_PROPOSALS=<count/source, turn overlap, unlocalized floor, long spans>
    MODEL_BOUND_CROPS=<actual Qwen checkpoint, clip spans/crops, SHA receipts, runtime/VRAM>
    DIRECT_AUDIO_REVIEW_VS_MODEL_STATUS=<reviewed/provisional/human-WER-UNVERIFIED>
    SOURCE_PROMOTION=<accepted/provisional/quarantined/reason counts, original IDs, no silent loss>
    NEGATIVE_SONG_SFX=<specific controls, outcome>
    FR_EPISODE_CONTRACT_AND_RENDER=<source and English SRT/ASS counts, exact qualifiers>
    ZH_EP166_SOURCE_AND_RENDER=<same with CTC untimed and rapid speaker handoff>
    GPU_AND_HOST_NONMUTATION=<actual>
    PUBLIC_EVIDENCE_ROOT=<full_SHA:path>
    MANIFEST_SHA256=<sha>
    PUBLIC_MEMBER_COMPLETENESS=<yes|no>
    PRIVATE_ARCHIVE=<path sha256 bytes or none>
    WER_CER=<UNVERIFIED unless new independent human reference>
    REMAINING_CENTRAL_BLOCKERS=<specific>
    ISSUE14=OPEN PR15=DRAFT
  COMMAND_SHAPE_FRICTION:
    encountered=<yes|no>
    events=<[]|canonical records including requested/effective command shape, rewrite provenance, event kind/effect, correction, replacement, recipe and resource, contract conflict and durable gap candidate>
    omitted_sensitive_values=<yes|no>

Finish the complete source-to-render evidence campaign in this ONE handback. Do not repeatedly return for trivial adapter or environment choices. For actual source algorithm defects, return precise diagnostics to Central rather than making unapproved semantic changes.
