# Issue #14: source-audio recovery and source-turn lineage

Status: Central-authored candidate, host integration not yet validated. Existing PR #15 only.

## Governing measurements

The previous model challenge is bound to cbebbc0e13b854bc866f76e96932936699f1dd67:staging/issue14/evidence/20261123T011017Z/. Central independently verified all nine SHA-256/byte-count members and the manifest (f2e7fedda03b22d1ebea3a62d41057c165fcc0aa4ba0cf4dba2a0bc97ce2b603).

French word error rate, Mandarin character error rate and complete episode source recall are UNVERIFIED: the 18-region reference lacks verbatim human transcripts. Qwen3-ASR's 42/42 event overlap is not 100% word recognition. FireRedVAD-AED improved observed event recall but shared W04c/H2 blind spots with every tested detector. Forced aligner produced monotonic times for false text, so monotonicity and confidence cannot certify lexical truth. Scene-bounded Nemotron-3-Diarization complements episode pyannote but is unsuitable as an unlimited global speaker cache.

## New Central code in this source slice

- src/nvidia_transcribe/source_recovery.py: bounded 24-second source-language recovery requests with 2-second overlap; authoritative raw-response identity; finish=length coverage; optional model activity proposals; audited omissions; explicit unserved-budget reporting. No model is started by this CPU module, and no candidate text is silently accepted. CLI accepts source-evidence, optional activity/incomplete/manual inputs and optional --coverage-end-s.
- Crucial architectural addition: opt-in whole-episode independent Qwen coverage floor. Only using VAD positive gaps cannot recover audio that every VAD misses, including W04c/H2. A bounded challenger observation over complete audio reduces this shared-detection blind spot. The returned transcript remains model testimony, not acoustic ground truth.
- Forced alignment structure QC distinguishes rejected from structurally provisional; positive output never proves its text was spoken. Versions source event ID separately from source revision ID, preserving provenance across corrected source text. Provides explicit legacy-v1 utterance projection with unknown timing/speaker status honestly retained.
- src/nvidia_transcribe/subtitles.py and translation.py: propagate source_turn_ids through cue creation/lane assignment, and prevent deleting different source turns because their translated English resembles another cue. Only exact repeated render copies of the SAME source event and same acoustic bounds may suppress. Adjacent translation regrouping cannot erase distinct v2 source identities.
- tests/test_issue14_source_recovery.py, tests/test_issue14_source_to_renderer.py, and the necessary semantic update in tests/test_viewing_quality_regressions.py enforce new observable requirements.

## Acceptance and boundaries

The new planner is not yet an active alternate ASR backend. Original process_media still generates its v1 recording contract and source-evidence sidecar without changing default model loading or legacy checkpoint behavior. Host Build must verify deterministic tests, offline French/Chinese source plans, one safe, language-pinned independent Qwen coverage pilot/sweep and source-turn lineage through SRT/ASS. Do not infer transcript truth from two ASR models or force-alignment plausibility. No service eviction, full 23-minute Nemotron rerun, production deployment, new PR or issue closure.

A retained duplicate-looking English subtitle is a conservative content-preservation result, not yet proven better viewing quality. Upstream audio-event correspondence must later resolve true duplicates without lexical similarity suppression at render time. New default renderer content counts may change; document exact differences.

Next Central implementation after host proof: qualify the model-bound audio-region adapter and reconcile multiple source ASR observations against bounded acoustic events, assigning stable accepted/provisional source turns with explicit event-ID decisions and independent timing/speaker advisories, then complete one new end-to-end episode gate.
