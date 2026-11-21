# Cache Correctness & Zero-Inference Re-render (Work Package B)

## 1. Stale Cache Bug Addressed
Before this remediation, `refinement.checkpoint.json` and `translation.checkpoint.json` cached results purely by `utterance_id`. Any upstream edit to:
- Utterance ordering / reconstructed IDs (Work Package A),
- Refined source text feeding translation,
- Neighboring context utterances (`before`/`after`),
- Glossary entries, prompt version, profile/style, or target language

would silently reuse stale cached outputs when resuming against an existing directory.

## 2. Content-Addressable Cache Key Design
- **Refinement ([refinement.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/refinement.py)):** `_batch_cache_key` computes a SHA-256 digest over prompt version, backend, model, profile, language, ordered glossary, every batch item's `{utterance_id, speaker_id, source_text, untimed}`, and surrounding context window. Cache entries are stored under `f"{uid}:{bkey}"`.
- **Translation ([translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py)):** `_translation_batch_cache_key` computes a SHA-256 digest over prompt version (`v2_echo_fix`), backend, model, target language, style, glossary digest, ordered batch items `{utterance_id, speaker_id, source_text, duration}`, and surrounding `before`/`after` context.
- **Condensation (`condensation.checkpoint.json`):** Keys include `_CONDENSE_PROMPT_VERSION`, `uid`, current English `en`, `src`, `prev`/`next` neighbors, character budget, and `style`.

## 3. Observed Selective Reuse vs. Targeted Invalidation

| Scenario | Observed Behavior | Evidence |
| :--- | :--- | :--- |
| CPU Unit Test (`test_b_cache_correctness_invalidation_selective_reuse_and_rerender`) | Unchanged batch `B1` hits cache (`0` extra API calls); editing one utterance in `B2` invalidates only `B2` (`1` API call); neighboring context change invalidates affected batch cleanly | Passed in `pytest -v -m "not gpu"` |
| EP166 Regeneration after A1 Reordering | `refinement` made `3` targeted API calls + `279` cache hits; `translation` made `3` targeted API calls + `285` cache hits | `comparison_metrics.json` |
| Cached Final Pass (`utt-0012` condensation `max_tokens=4096`) | Condensation reused `30/32` cached items (`cache_hits: 30`, `api_calls: 2`, `errors: []`, `16.145s`) | `comparison_metrics.json` |
| CLI `--re-render` Smoke Verification | Re-rendered EP166 SRT & ASS (`raised_main`, `margin_v=150`, `max_lanes=3`) directly from `translated.contract.json` in `0.059s` with `reused_translated_contract` status | Live CLI execution on host |
