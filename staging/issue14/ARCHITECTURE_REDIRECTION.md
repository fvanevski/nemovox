# Issue #14: architecture redirection after independent fixture and nine-probe audit

Status: CENTRAL DESIGN BASIS / NOT AN IMPLEMENTATION OR HOST-TESTED REPLACEMENT
Source inspected: fvanevski/nemovox@4c32acf659891fe999d2ae6e15f59ccb5750d92f (tree a4e866f53899fe159c99793ff8a78326a10b26f5)
Base: main@8f6c03cbc70e84bf1970ba364207a02c9fe22b44
Evidence: 5bf6f37d173cfcc160fdf03b60b051cd0f5e7dea (original 22-case acoustic trace), a1812212f6e5c7a82bcd0a27769130fda420a3d3 (35-chunk full run), 11a0ab8bfa0a9be4c67fcc2b822cd529abce5461 (nine targeted probes)
Scope: #14; Issue #9 owns rendering/readability improvements, but #14→#9 must preserve every accepted source event.

## Decision

STOP extending overlapping word-by-word deletion and rescue rules in \`chunking.py\` and \`boundary_evidence.py\` as the primary solution. Retain the proven bounded acoustic-context inference and \`asr_integrity.py\` fail-closed *generation status* check, but evaluate a cohesive replacement based on source-region activity, immutable ASR candidate observations, and a single evidence-based event-correspondence/selection pass. Existing tests are regression evidence, NOT immutable algorithm specifications. This is a design candidate, contingent on independent acoustic anchors and held-out controls below.

## What the nine probes do and do not establish

- c28 old response: \`finish=length\`, 32,774→14 chars, zero CTC words across 11 model diarization intervals. A 1123–1156s retry produced 27 words and overlaps c29 within 0.06s at its tail. BUT earlier ~1100–1123s acoustic content remains NOT_CHECKED. Two other retries ending with \`stop\` returned 1–2 words over 6–7 diarization spans. Therefore a clean \`stop\`, high lexical density on one portion, or overlap agreement is NOT proof of complete audio coverage.
- W04b / W04c: 1242–1274s candidate places both phrases in expected order and speaker turns. The 1238–1266.5 and 1231–1267 candidates degraded. Do NOT generalize that all sources require >=8s right context. The successful W04c end ~1266.4 is only ~7.6s before end, and other successful probes end closer to source speech. The salient independent factor may be clean start/end relative to speech activity.
- W05a: archived c34 diarization has an unpopulated 1352.63–1353.81s span; alternative excerpts recover 5 or 6 tokens and speaker separation, plus S05. Model diarization is generated within the same ASR system and is NOT independent source truth. A missing-spoken-word trigger must be calibrated against silence, music/SFX, overlaps, and held-out clear dialogue.
- Apparent improvements in micro-utterance count (90→40), short shards (49→12), and ASS count (308→245) are useful proxies, NOT speech coverage or transcript accuracy. Speaker ID count (23→28) is not a quality target.
- Old 22-case archive contains host-audio-heard reference anchors, but S01/S02/S03 are NOT_ACOUSTICALLY_CHECKED. Do not silently promote their alleged text ground truth.

## Proposed minimal architecture (Central-owned production implementation later)

1. \`AudioRegion\`: track/media fingerprint, source interval, candidate speech/non-speech/uncertain activity mask, acoustic evidence/provenance, optional overlapping regions. Region boundaries are hypotheses, not forced utterance or speaker labels. Never treat silence detection, energy, Nemotron diarization or VAD alone as proof of spoken words.
2. \`ASRObservation\`: immutable individual request excerpt + source offset, model identity, request settings, raw hash, finish/error status, local speaker tags, model CTC, lexical order, diar spans, and acoustic-coverage flags. Multiple observations may refer to same underlying speech. The status of a response is distinct from whether its content covers all source events.
3. \`EventCorrespondence\`: source-interval-based matching of observations, monotonic lexical correspondence with bounded uncertainty, explicit alternative candidate sets, independently overlapping-speaker support. Never infer global speaker identity from local \`<spk:N>\` numerical equality. If it is unclear whether similar text represents one or two events, preserve both candidates as unresolved, not duplicate subtitles.
4. \`AcceptedSourceTurn\`: a selected, acoustically supported speech event with source text, timing quality, speaker evidence, observation lineage, and optional review markers; competing observations are retained non-destructively. One accepted acoustic occurrence should appear once in downstream subtitles, while genuine repetition/two simultaneous speakers remain distinct. No numeric confidence fabricated without calibration.
5. \`RecoveryController\`: budgeted targeted alternative excerpts for an incomplete observation, acoustic-coverage gap, genuine cross-observation contradiction, or unresolved speech-turn split. Choose candidate cuts using actual audio activity and duration constraints; no per-episode hardcoded times, >=1s-unpopulated or >=8s-context rule. Stop and mark explicit unresolved intervals when corroboration remains inadequate.

Retain media extraction, Nemotron client/CTC parser, lifecycle/lease, immutable source contract, Riva and contextual Qwen as adapters. Keep \`fixed\` as a backward-compatible experimental policy if needed; \`pause_aware\` stays opt-in until outcome validation. Replace the three-step boundary deletion/recovery in \`chunking.py\` and \`boundary_evidence.py\` *as a unit*, with focused adapter migration instead of layering a fourth matcher. Align speaker/turn decisions after (or iteratively with) source-event correspondence so erroneous CTC times cannot decide global speaker merges. Translation and renderer must not silently delete source events based only on text similarity; cross-issue invariant with #9.

Avoid elaborate generic DAG infrastructure or additional model dependencies as a default. Use minimal typed records, deterministic matching/selection, and a bounded selective retry policy. Preserve accessible manifests and stage-specific cache signatures based on semantically effective inputs, not incidental diagnostic counters.

## Fixture migration classification

- KEEP the *observable claims*: original audio/source not mutated; genuine C1/C2 repeats and different-speaker utterances retained; source W05a once; W02c/W03a not rendered twice; W04b/W04c correct chronology; W02b/S05 alternating voice retained; no invented timestamps; raw unfinished generation rejected; compatible resume; safe resource ownership.
- REWRITE *oracles* that assert internal \`sequence_promoted_units\`, \`sequence_deduplicated_units\`, which neighboring chunk wins, \`detected_pause_early_long\` at exactly 64.241427s, hard-coded 4/6/8-token match lengths, 0.05 cosine margin, "must preserve both" just because CJK character sequence matching is disabled, or 2048 loop bytes as product truth.
- RETIRE obsolete fixture/implementation paths only after a stronger corpus passes: untranslated Mandarin particle suppression/English-clause duplicate deletion without source-event proof, renderer's 0.85 text-similarity duplicate suppression on distinct simultaneous speakers, legacy ordinal \`utt-XXXX\` as stable cross-version identity, and absolute host file paths inside portable CPU tests. A test may remain temporarily to prove backward compatibility but must not veto a strictly better outcome.
- NOTE \`subtitles.py::_deduplicate_concurrent_echo_cues\` currently drops similar concurrent text without speaker/source-event equality. This is a cross-stage source preservation risk, not merely an ASS layout issue. Scope code remediation to the right issue/PR and prove the #14→#9 contract.

## Evaluation gate for selecting a replacement

Primary scores on independently heard/annotated small reference set and later held-out windows: coverage (events present, omitted, duplicated), word/character accuracy where independently labeled, timings against independent anchors, speaker turn boundaries and actual overlap, genuine repeats preserved, invalid accept/reject decisions, downstream translation meaning preserved. Track inference latency, number of retries and GPU peak. Present per-case error classes and confidence limits, not fabricated episode-wide accuracy from a failure-enriched sample.

Adversarial controls: music/singing/SFX, silence, genuine simultaneous repeated text, extremely sparse \`stop\`, wrong same-text different-speaker pairing, CTC displaced multword, CJK overlapping characters, valid end-of-media speech, clean speech with no reason to retry.

**No production rewrite is authorized by this memo alone.** First close critical acoustic label/calibration gaps with HANDOFF_ARCHITECTURE_EVIDENCE.md; then Central drafts a compact replacement with implementation and acceptance tests. One PR (#15) preferred. Avoid new issues/special-case campaign proliferation.
