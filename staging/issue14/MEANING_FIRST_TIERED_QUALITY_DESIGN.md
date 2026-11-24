# Issue #14: Meaning-first subtitle-quality objective and tiered review

## Governing objective

Per operator clarification, Nemovox optimizes for coherent, grammatical, readable natural-English subtitles that accurately convey intended meaning and narrative continuity. This is NOT a requirement for word-for-word English, perfect French WER/CER, complete recovery of every background interjection, episode-wide character identification, or verified timestamp for every individual word. Idiomatic translation, condensation, reordered syntax, and sensible contextual phrasing are valid when polarity, causal claims, essential events, and important speaker distinctions remain faithful. Usable utterance-level timing may suffice.

Source integrity is a distinct contract: output-language flexibility does not license hallucinating acoustic events, reversing negations, deleting separate utterances or claiming source observations independently verified when they are merely ASR/CTC/diarization/VLM hypotheses.

## Forensic root causes

Authenticated read-only host archive issue14-omission-diagnostic-20261124T010009Z.tar.gz: SHA256 52568c845ab3e7c2541e3711b8950cccde93a11164cb7eadf3c706d4bf923b8e; 21/21 checksums, 27 members, no duplicates. Same exact 16kHz chunk WAV inputs and inference config in two runs, but different stochastic completions despite temperature 0 and seed 0.

S02 (~355-360s): both ASR chunks contain dialogue including important negation, yet prior chunk CTC shifted it ~19-58s early and left its accurate neighbor copy in nominal non-core context, which was discarded. HIGH semantic/timing impact. W01c: chunk 0 omitted overlapping speech and dialogue at ~77-83s; chunk 1 contains the later portion accurately in its left context but lost it to nominal core filtering. The 64-67s interval itself is musical transition and should not be treated as unheard dialogue. These are distinct generation plus reconciliation failures, not proof that literal source tokens must be perfect.

## Context-dependent review ladder

- T0 deterministic lightweight QC: record coherent neighboring left-context words, competing core CTC, prospective unrepresented diarization activity, and semantic risk markers. This is evidence discovery, not correction acceptance.
- T1 cheap text QC: inspect grammar, idiomatic English, continuity, readability and contextual plausibility. A bounded low-effort LLM call may propose English postediting for an accepted source turn.
- T2 high reasoning: when materially important dialogue is missing/displaced, negation reverses, or acoustic/ASR/CTC/speaker evidence conflicts, reason across original acoustic review, adjacent hypotheses, source/translation context, diarization and contrary observations. The model may abstain.
- T3 audiovisual only if seeing the scene could materially resolve remaining ambiguity, such as active speaker identity. Discount animated mouth-motion synchrony relative to live action when appropriate. No fixed universal modality hierarchy; applicability is claim- and scene-dependent.

Reviewer prompt, thresholds, model effort and budgets are tunable and versioned. They are not calibrated accuracy probabilities. A text-only model's statement cannot certify new source words. A corrected source event still requires factual raw-unit ownership and independently reviewed acoustic occurrence; a nonliteral English phrasing decision does not require literal one-to-one word mapping.

## Central staged source (host unverified)

Source/test commit 622715afc00dd6d215fe27c9b5844f615d762971, tree 1317b67868baec61852e9d84d087b0e0e3994d5d, on existing draft PR #15:
- New semantic_review.py: configurable plan_boundary_semantic_reviews; identity-pinned validate_semantic_review; injected run_tiered_text_reviews with low/high reasoning effort and bounded token/call budgets. Advisory output only, no implicit model service calls or output changes.
- chunking.py: pre-filter boundary review discovery, before destructive core ownership decisions. Reports candidates and limitations without promoting source.
- recording.py: passes existing immutable diarization activity hypotheses from source-evidence ledger into review planning without treating them as acoustic truth.
- configs/meaning-first-review.example.json: tunable QC instructions and constraints; default metadata-only policy remains conservative.
- tests/test_issue14_semantic_review.py: six synthetic focused Central local CPU checks across candidate discovery, activity/displacement, no automatic promotion, negative controls, stale source, limited model response and budgets. Local 6/6 PASS for the separately authored module/test; complete host candidate suite PENDING.

Previously qualified seven-component reviewed CLI remains the opt-in accepted-correction path. New QC supplies candidates and bounded advisory model assessments, but does NOT automatically fix the W01c or S02 subtitles yet. Its purpose is to identify high-impact semantic defects for informed review, not pursue universal literal transcription.

## Acceptance

Prefer meaning, polarity, narrative sequence, natural English, relevant turn-taking and usable cue timing. Report remaining factual uncertainty, false speech/music positives, genuine repetitions and output legibility; retain exact source hashes for audit. Compare the two forensic cases and negative controls, not every isolated word. Reuse existing archive and model resources; no new full-episode ASR is justified merely to qualify this diagnostic layer. If model review provides evidence for a correction, only the established evidence-fusion/source-revision application gate may change accepted source. Keep Issue #9 ASS lane architecture out of scope.
