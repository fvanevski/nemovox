# Issue #14: source-audio reconciliation architecture and admission contract

Status: Central candidate staged on the existing issue/14-longform-timing-boundaries branch, NOT host-tested or automatically activated.

## Governing evidence

Current evidence handback: e681203de5eb03bb1fb6a88d34db6e32beb3ee6f:staging/issue14/evidence/20261123T013938Z. Its 9 files and manifest SHA-256 08786e4d7ddc5677df222432313cfd16fe0e510aac212faf89aa0d36634ce53c were independently checked by Central. Host reported 41 focused and 337 non-GPU tests passing at the prior code. It completed one 63-window French and one 60-window Mandarin pinned Qwen3-ASR sweep on the original media without external GPU service eviction. Those are credible execution receipts, but not measured French WER or Mandarin CER, and no full-audio human-verbatim reference exists.

Acoustic root cause of missed source dialogue cannot be solved by detector-positive-only work: Silero VAD, FireRed AED and pyannote miss portions of W04c and H2. The independent sweep recovers candidate words there but emits lyrics and short hallucinated interjections over music/SFX. Every word candidate is *model testimony*, not proof of spoken source. An isolated 3-second W04c crop hallucinated while a context-preserving 24-second crop did substantially better. Prior 35-chunk c28 remains generation-incomplete. Cross-chunk text matches can link SFX to speech 15 seconds away and must never authorize deletion by string similarity or CTC monotonicity.

## Central implementation staged for this slice

1. src/nvidia_transcribe/source_recovery.py now coalesces overlapping detector speech proposals by reason, carrying all contributing evidence IDs, and protects the opt-in independent coverage floor ahead of redundant model activity requests. Priority is incomplete-generation gaps, independent human-auditioned omissions, full-source floor, then model-only activity. It retains explicit pending requests and reports floor selected/pending separately. It DOES NOT conflate independently heard overlapping speech events: those are left separate.
2. src/nvidia_transcribe/source_reconciliation.py and tests/test_issue14_source_reconciliation.py provide an offline, CPU-only, private evidence dossier consuming:
   - immutable nemovox-source-evidence/v1 ledger;
   - nemovox-source-recovery-plan/v1 returned by the planner;
   - nemovox-challenger-observations/v1 with exact source-media fingerprint, language, pinned model revision, request ID, crop interval and crop SHA, raw-response SHA, completion status and model source text;
   - optional nemovox-audio-activity-evidence/v1 model labels (speech/lexical_shout/singing/music/sfx/etc);
   - optional nemovox-audited-source-events/v1 from direct human listening, where independently heard speech events have stable IDs, actual audio bounds and optional separately human-verified source text.
   It must fail closed on media/language/request/crop mismatches, missing/duplicate IDs, unsupported model statuses and unverifiable attestation fields.
3. A Qwen answer for a 24-second crop is WINDOW-WIDE and unlocalized. Do not assign its entire text to one 2-second shout merely because the two time intervals overlap. Dossier records potential contextual links to audited audio events, but does not fabricate per-word timing, speaker identity or acoustic event boundaries. No automatic source turns accepted by model agreement.
4. A source turn is projected to the v1 translation/renderer interface only when a positive independent verbatim audio review attests source event ID, audio bounds, source language, source text and review-evidence SHA. That is an auditable positive sample gate for qualification, NOT a new requirement for humans to transcribe every episode. After this campaign Central will design the automatic, provenance-aware provisional acceptance policy for ordinary personal subtitle operation.
5. Lexical correspondences between overlapping ASR windows remain unresolved unless independently listened source-event mapping positively links exact source unit IDs. If both sides are assigned to ONE reviewed acoustic event, mark same_audio_event_reviewed_keep_observation_lineage; if they are mapped to DIFFERENT reviewed events, mark distinct_audio_events_reviewed_preserve_both. In no case is a source observation silently deleted.
6. Existing source_evidence sidecar, v1 recording contract, default Nemotron process_media and translation pipeline remain unchanged by this new module. No runner/download/global GPU service changes. The new reconciliation CLI writes a fresh private chmod-600 dossier only.

## Non-negotiable semantic boundaries

- Never infer that detected music means no simultaneous quiet speech.
- A detected Qwen phrase is not automatically localized within a 24-second crop, and can be hallucinated over singing/SFX; no automatic source-turn promotion.
- Positively linked source unit IDs are audited source-event identity evidence, not text similarity. Even true duplicate model observations are preserved in the ledger; later canonical source-turn output may use the unique reviewed event.
- A missing verbatim reference remains UNVERIFIED; previous 42/42 temporal event hits cannot be written up as 100% word accuracy.
- The source-turn reviewed projection can have event-level time but not accepted word timestamps or canonical global speaker ID.
- Full independent audio coverage is protected against per-detector proposal multiplicity. If global max_requests is insufficient for all floor intervals, they remain listed, not hidden.
- Do not touch Issue #9 ASS lane/layout optimization.
- This source dossier is a step toward the active hybrid pipeline, not its replacement. One final Central-owned model-bound provisional selection / audio-region segmentation stage, followed by a new integrated full-episode acceptance, remains.

## Next host gate

Run source-bound CPU tests, replay the authenticated cached originals and Qwen private 63 FR/60 ZH sweep without re-running GPU inference, map exact model outputs to the new declarative challenger schema, inspect real W04c/H2/c28/S05/OP/ED controls, and provide a small positive direct-audition sample for 2-5 unit/event mappings. Record exact count of coalesced activity proposals, independently covered floor windows at budgets 32/64/128, unexecuted/deferred source requests, same/different event correspondence decisions, WER/CER uncertainty, and license/VRAM evidence from prior receipts. The Build role may handle test-only source-interface adapters but cannot design the production event-selection algorithm.

Use the remaining evidence to author the final centralized automatic candidate-selection and bounded source-turn construction on the same PR before a genuinely new end-to-end episode and independent review.
