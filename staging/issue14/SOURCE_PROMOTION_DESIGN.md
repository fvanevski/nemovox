# Issue #14: source promotion and active subtitle overlay

Status: CENTRAL CODE STAGED, HOST VERIFY NOT YET RUN. Same draft PR #15, issue remains OPEN. The following design is bounded to source fidelity; do not broaden into Issue #9 display-layout work.

## Admitted evidence

Handback source: 874bd718ac4a42e87e55a54ebad54a44edc33c52:staging/issue14/evidence/20261123T022122Z/. Independently verified 9 public member digests and manifest SHA256 d01522115479f1d44556f9c6544cfe0dc8a429c9d01798068462580c9d0a470c. Uploaded private archive SHA256 56317262ac2bee3a912abf1f372e4ca71a7b54dfb1c27b2a91548b07eaea3df8 was independently inspected by Central. Archive contains original and forensic source-evidence ledgers, admitted planner, Qwen challenger windows, reconciliation dossiers, per-event review record, and provenance adapters. No raw WAV included, so Central did not independently listen to the clips. Native read_file audio-review records are LOCAL AGENT ATTESTATIONS, not human-verified verbatim references; WER/CER remain UNVERIFIED.

Host evidence: 55 focused/351 non-GPU passed at source fc38ba614ec2e145c1efd6ba8389db27a80efde0; no source drift, 63 FR and 60 ZH Qwen sweep windows mapped using canonical stored-result JSON hashes rather than native HTTP raw-response hashes. Partial French c28 and Mandarin c14 were rejected; original utterance evidence preserved. Full35 independent floor 63/63 at max128 with detector coalescing. Model-only text remains unaccepted; audio-reviewed same-event correspondences preserve raw source units.

## Root-cause work staged on this PR

1. src/nvidia_transcribe/source_event_replay.py:
   - Missing manifest source_language resolves from same-directory recording.contract.json ONLY when its schema is nemotron-recording-contract/v1 and source fingerprint matches the cached manifest media fingerprint. Conflicting languages or stale contract fingerprint fail closed. Replay provenance records authority. No hostile path supplied by the manifest is followed. No mutation to historic jobs.
2. src/nvidia_transcribe/audio_regions.py:
   - Generates AudioRegion advisory proposals, bounded ASR crop suggestions, direct-audition event intervals, scene-local separate speaker turn proposals, non-speech risk IDs and unlocalized Qwen coverage windows.
   - It does not invent speaker identity, make model speech spans canonical turns, erase overlap between channels or promote full-window Qwen words.
   - Long activity intervals remain needs_subsegmentation; do not blindly assign one speaker or one transcript across them.
3. src/nvidia_transcribe/source_promotion.py:
   - Offline OPT-IN conversion of existing recording contract v1 plus source reconciliation dossier and new bounded source segments into a new v1-compatible recording contract.
   - Input schema nemovox-bounded-source-segments/v1 contains media_fingerprint_sha256, source_language, and segments: id, audio_event_id, start_s/end_s, kind speech|lexical_shout, source_text, optional context_window_ids, optional speech_activity_ids, optional bounded_asr, optional supersedes_legacy_utterance_ids.
   - Bounded ASR record: model_repo_id, model_revision, language explicitly French/Chinese, crop_start_s/end_s, audio_crop_sha256, raw_response_sha256, source_text_sha256, finish_status completed/complete.
   - Model-only source requires: exact dossier model identity, separate materially narrow input crop containing turn, source-text digest match, finished inference, at least one positive speech activity event in exact media, and no conflicting same-event source candidate. This produces model_source_candidate_provisional, not acoustic truth. Model-only candidate collisions with existing legacy speech are quarantined.
   - Direct audio review can yield audio_review_attested_source if transcript, event bounds, review evidence and source lineage match the dossier. The existing direct review evidence was agent-attested and is NOT a general human-verbatim WER reference.
   - Independently auditioned non-speech-only regions override unsupported model hallucinations. Singing/SFX-dominant bands without independent speech model corroboration are held for review; this is NOT a global VAD exclusion of possible quiet speech. MUSIC-ONLY/BGM is not an absolute ban on actual dialogue.
   - Different audio_event_ids with identical text remain separate turns. Multiple conflicting hypotheses for the same event remain quarantined. Supersession of an old v1 utterance is permitted only if it already carries matching positive source_turn_id lineage AND acoustically plausible original event timing (within a conservative 3s bounding slack). Wildly displaced legacy content stays unresolved instead of being deleted; old baseline v1 utterances normally remain intact.
   - New turns carry stable source_turn_id, source_revision_id, evidence status, speaker unknown and only turn-level source bounds; no fabricated word timings. Existing v1 translation and SRT/ASS renderer can consume the resulting contract without a default pipeline switch.
   - JSON report attached under recording_contract.source_promotion includes accepted, held, source count, legacy supersessions and known uncertainty, with no full false model text in quarantine entries. Original source contracts and raw chunks are never overwritten.

## Architecture boundaries

This is a usable source-turn overlay path, not a claim that automated Qwen transcription is accurate. The 24s full-coverage source text remains unlocalized until a separate narrow model inference is actually run on a bounded region. The earlier 3s W04c crop hallucinated while the 24s contextual window succeeded; use the proposed 4-16s context-preserving crop, inspect negative controls, and label every new model-only turn PROVISIONAL. CTC monotonicity and Qwen word alignment cannot certify a source word. Speaker-local tags and scene diar are advisory; keep simultaneous voices separate.

The existing 23-minute original16 and partial full35 are NOT equivalent gold sources. For end-to-end media rendering, choose an intact authenticated v1 contract (normally original16); use partial full35 only as forensic model evidence and incomplete gap tasks. The new default recorder remains unchanged. User activation requires an explicit new recording-contract path; no release or service deployment is authorized by this stage.

## Host acceptance matrix

- Source identity: manifest language fallback works only on exact fingerprint and correct schema; strict mismatch and incomplete c28/c14 fail.
- CPU regression: compileall, focused source replay/audio regions/source promotion/reconciliation/translation/SRT/ASS and full non-GPU suite on exact source SHA, no weakening.
- Audio proposals: same model overlap merge only advisory, no merging two diar speaker channels or audited repeated speech events; old VAD-negative W04c/H2 independently labeled candidates remain proposed; no Qwen window text implicitly attributed to one region.
- Inference: use already owned Qwen3-ASR-1.7B checkpoint, pinned FR/ZH, one safe bounded crop admission batch; each output digest/region/media/provenance recorded. Avoid empty/subsecond hallucination and theme song/SFX false positives.
- Promotion: direct-reviewed source turn and bounded Qwen provisional turn pass; unlocalized 24s text, song/SFX, discordant model source, incomplete requests, collisions, false matching and different-source overlap are held. No synthetic word timestamps/speaker IDs.
- Translation/render: candidate source turn identities are preserved across translation, SRT and ASS; count deltas, source-turn IDs and unresolved legacy echo risks. Build at least one full French recording-contract overlay and one Mandarin representative fixture; include a real readable English subtitle path if Riva resource admission is safe.
- Privacy: never publish raw WAV, model transcripts, episode lyrics, full personal media or checkpoints on GitHub. Public evidence must be hash complete; private host archive may include limited bounded excerpts and tests if useful.
- Final: Host PASS may qualify the opt-in overlay, but Issue #14 closure requires one source-fidelity audit of the integrated actual episode and independent exact-head review. WER/CER remain unverified without independent verbatim reference.

## Next action after this handback

Central reviews exact-head host results and remaining false-negative/duplicate errors, fixes any substantive source selection or integration defect, and completes one integrated FR+ZH final episode gate before review. Do not proliferate PRs or create a parallel renderer rewrite.
