# Issue #14: source-evidence layer, transcript-first architectural slice

Status: CENTRAL-AUTHORED CANDIDATE; HOST INTEGRATION AND ACOUSTIC ACCEPTANCE PENDING.

## Scope of this slice

Build an immutable observation record from the real Nemotron chunk responses **before** legacy core-first boundary deletion, turn reconstruction, translation or rendering. The recorder writes `source-evidence.v1.json`, with an integrity pointer in its ASR manifest. This is a useful source-bound replay/diagnostic foundation, not a replacement transcript accepted from unverified CTC.

This slice is one architectural integration campaign on existing draft PR #15, not a new PR. It does not install models, change GPU lifecycles or alter CLI/Studio defaults. The v1 recording contract, existing chunk checkpoints and downstream translation/ASS behavior remain unchanged. A future source-turn v2 acceptance/projection may be activated only after source-audited event matching and aligner/speaker evidence are sufficient.

## Typed record relationships

- `asr_observations`: stable immutable observation ID rooted in media fingerprint, original raw SHA and excerpt boundaries; retains recognized source text and the actual inference window.
- `lexical_observations`: one source unit per upstream unit, with stable input-ordinal ID, its original Unicode text, local tag, non-authoritative global speaker hypothesis and separate CTC hypothesis/status. Never assigns accepted acoustic word time or speaker identity.
- `local_turn_candidates`: source-local contiguous runs, *not* established physical speaker turns; useful for comparing source transcription to legacy utterances.
- `cross_chunk_correspondence_candidates`: lexical sequence correspondence only when *actual inference windows overlap*. Four or more contiguous normalized units; original observations always survive, including repeated phrases, Chinese characters and potentially separate speakers. A proposed correspondence is never a deletion instruction, especially if CTC chronology is shifted or speakers disagree.
- `diarization_activity_hypotheses`: native model diar spans, including unexplained spans. Neither a positive span nor absence of a span certifies speech presence/absence. Independent audio annotations remain necessary.

IDs are stable for identical raw source + media + excerpt identity, not asserted stable across modified input or source-text revision. The ledger is shadow-only; `source_units_dropped_by_ledger=0` is a structural invariant, **not** a source recall/precision metric. Raw response/checkpoint original source remains the primary provenance.

## Acceptance for this slice

1. CPU deterministic tests cover stable/different revisions, genuine repeats, CJK, unknown timing, displaced CTC, shared-vs-disjoint audio, speaker hypothesis disagreement and false activity flags; full CPU, Studio and frontend checks remain green.
2. Existing 16-chunk and 35-chunk private cached runs can generate this ledger with exact input source/raw digests and without changing existing translation/ASS; compare before/after observable counts and mark coverage unverified. If raw source outputs are missing or mismatched, BLOCKED with evidence, not fabricated replay.
3. Review the 22 primary W/C/S cases, H1-H4 and two Mandarin controls against immutable source-unit and correspondence IDs; identify recoverable event matches, missing-model words not represented by any observation, and false candidate matches. Report per-case precision/recall limits; avoid treating agreement among two Nemotron requests as independent acoustic validation.
4. Perform a bounded **existing-model** host comparison only if it materially resolves one of those uncertainties, using Nemovox resource admission and no additional third-party weights or unowned GPU service disruption. No 23-minute full-episode rerun is necessary at this slice.
5. Publish a complete sanitized, manifest-hashed evidence packet and exact source identity. Private WAV, transcriptions of full episodes, raw model payloads, credentials, model weights and proprietary media stay host-local.

## Architecture decisions reserved to Central

Canonical source-event acceptance, audio-region VAD authority, forced-aligner adoption, episode-level speaker adjudication, source-turn v2 conversion, replacement of legacy boundary deletion, and eventual renderer suppression changes. The local Build agent may fix a demonstrated in-scope adapter/fixture/wiring problem without changing these semantics, but returns novel architectural tradeoffs to Central. Do not claim Issue #14 complete merely because the ledger is generated.

## Source/evidence provenance

Design governing context: `staging/issue14/ARCHITECTURE_REDIRECTION.md`. Acoustic audit: `staging/issue14/evidence/20261123T000342Z/`. Legacy complete recorded-run tests and the source-fidelity issue body remain controlling; renderer layout remains Issue #9.
