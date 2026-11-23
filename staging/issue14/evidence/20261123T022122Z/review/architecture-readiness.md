# Issue #14 Architecture Readiness & Central Next-Stage Inputs (`20261123T022122Z`)

- **Validated Source SHA:** `fc38ba614ec2e145c1efd6ba8389db27a80efde0` (PR head `52bc363b0d402a13da4ba678e518bcdc66cf31a7`, zero `src/`/`tests/` drift).
- **Verdict:** **PASS for qualification of `source_recovery.py` budget/coalescing policy and `source_reconciliation.py` offline evidence dossier.**

## Validated Improvements at `CENTRAL_SOURCE_SHA`

1. **Coalescing & Protected Coverage Floor:** Coalescing overlapping `activity_without_timed_source_hypothesis` proposals reduced FR `full35` activity requests from `73 -> 44` (`orig16`: `44 -> 29`, `zh_ep166`: `6 -> 4`) with exact detector evidence ID union preservation (`100%`). Placing `independent_coverage_probe` (priority `2`) ahead of `activity_without_timed_source_hypothesis` (priority `3`) allows `full35` at `max_requests=128` (even with `c28` + `6` misses) to fit `100%` of the 63-window floor (`116/128` selected, `complete_proposals`), whereas budgets `32` and `64` fail transparently to `partial_budget` with all missing floor windows listed in `unplanned_due_to_budget`.
2. **Fail-Closed Provenance & Acoustic Window Linkage:** `source_reconciliation.py` rejected raw `"unverified"` language ledgers, rejected deferred budget-32 requests, rejected disjoint owning ASR inference windows (`c00` -> `W04c`), kept all `model_only_source_turns_automatically_accepted=0`, and kept `source_unit_observations_deleted=0` across `fr_orig16`, `fr_full35`, and `zh_ep166`.

## Host Integration Note for Central (Non-Blocking for Qualification, Needed for Seamless Chaining)

- **`source_event_replay.py` Legacy Manifest Language Fallback:** Legacy `manifest.json` files for `fr_orig16`, `fr_full35`, and `zh_ep166` omit top-level `source_language` (storing `"fr"` / `"zh"` in `recording.contract.json`), so `source_event_replay.py` emits `"source_language": "unverified"`, causing `source_recovery.py` to emit `"challenger_language": null` and `source_reconciliation.py` to fail closed. For seamless CLI chaining without a host adapter, Central should consider reading `recording.contract.json` or accepting `--source-language` in `source_event_replay.py` (while keeping fail-closed rejection when language is truly unknown).
- **In-Process Qwen Worker Provenance:** `run_qwen_worker.py` invoked `Qwen3ASRModel.transcribe()` in-process and stored per-window result objects in `*_qwen_sweep_results.json` without a precomputed `raw_response_sha256` field; future managed Qwen challenger runners should emit immutable per-request raw JSON + SHA-256 directly.

## Remaining Inputs for Central Final Active Source-Turn Stage

1. **Non-speech gating (`AudioRegion` + `FireRedAED`):** Gate Qwen/Parakeet/Nemotron hypotheses where `FireRedAED` reports `singing`/`music` (`CTRL_song`, OP/ED) or pure SFX (`CTRL_sfx`, `H3`) so singing lyrics and 1-2 token fillers are never auto-promoted.
2. **Window-to-turn segmentation & multi-ASR provisional selection:** Segment 24s Qwen coverage windows into bounded utterances, cross-check pause-crossing forced alignment against acoustic silence anchors (`[1262.9, 1264.5]`), and promote provisional source turns without deleting raw `ASRObservation` lineage.
3. **Fresh End-to-End Episode Acceptance:** Qualify the final active hybrid pipeline on a genuinely new episode once implemented.
