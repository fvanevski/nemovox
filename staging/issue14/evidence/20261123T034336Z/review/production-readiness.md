# Production Readiness Assessment — Issue #14 Bounded Source Promotion

**Assessment Outcome:** Conditionally ready as an **opt-in** artifact/pipeline path; **NOT** certified for full automatic source fidelity or default unattended cutover.

## Proven Safe Invariants (Verified on Head `5501adad9582`)
1. **Contract Language Authority (`source_event_replay.py`)**: Legacy FR (`fr_orig16`, `fr_full35`) and ZH (`zh_ep166`) caches with missing/untrusted manifest language resolve cleanly from the authenticated `recording.contract.json` (`source_language_provenance="recording_contract_verified"`). Stale fingerprints and manifest/contract disagreements hard-fail; incomplete chunks (`c28`, `c14`) fail strict mode and remain quarantined under `--forensic-incomplete`.
2. **Bounded Proposal Accounting (`audio_regions.py`)**: 100% of regions stay strictly <=18.0s unless explicitly split, scene speaker channels remain distinct (`audio_regions_zh_scene.json`), and unlocalized model windows remain explicitly uncertified (`63` FR / `60` ZH).
3. **Bounded GPU Inference (`Qwen/Qwen3-ASR-1.7B` @ `7278e1e`) & Translation (`Riva-Translate-4B-Instruct-v2`)**: Cooperative `ModelLifecycle` admission ran cleanly on RTX 5090 (`4.48 GiB` Qwen peak VRAM; `8.02 GiB` Riva peak VRAM), releasing VRAM to exact pre-run baseline (`3101 -> 3103 MiB`) with zero disruption to `kwin_wayland`, `text-embeddings-router`, or `nemovox-studio.service`.
4. **Source Promotion & Anti-Hallucination Guards (`source_promotion.py`)**:
   - No silent deletion of legacy utterances (`234 -> 241` FR; `285 -> 288` ZH with all `41` untimed units preserved).
   - Over-long 24s crop relabeling, text/hash tampering, wrong model revisions, unpinned language, and direct reviewed non-speech ID reuse hard-fail (`ValueError`).
   - Song hallucinations (`CTRL_song_25_40`), SFX hallucinations (`CTRL_sfx_1200_1216`), single-detector speech over dominant singing/SFX, incomplete `finish_status`, model-only supersession attempts, displaced bounds (`legacy_supersession_audio_bounds_unverified`), unlinked supersession (`legacy_turn_lacks_positive_same_event_lineage`), colliding model text, and disabled provisional mode all quarantine deterministically across `17/17` test cases.

## Remaining Non-Blockers & Honest Limitations
1. **`WER / CER` = `UNVERIFIED`**: Prior independent audio observations are agent-attested (`read_file`), not an independent full-episode human verbatim transcript reference.
2. **Provisional Crop Context Leakage**: Bounded ~8–10s Qwen crops on `C_w05_S05_omitted` and `C_w05_tail_omitted` included adjacent dialogue / spurious proper nouns in model output. The contract correctly labels them `model_source_candidate_provisional` (`source_text_human_reference_verified=false`) rather than attested truth, underscoring why unreviewed model crops cannot auto-supersede legacy utterances or bypass `source_text_human_reference_verified=false`.
3. **Unaligned Speaker Identity & Word Timing**: Promoted turns remain `SPK_UNK` with `alignment_status="turn_bounded_word_timing_unverified"` and `source_fidelity_certified=false`.
4. **Raw Legacy Contracts Lack `source_turn_ids`**: Because pre-Issue-14 legacy contracts do not populate `source_turn_id`/`source_turn_ids` on legacy rows, reviewed turns without overlapping legacy gaps insert safely, whereas in-place legacy replacement (`supersedes_legacy_utterance_ids`) strictly requires positive event-ID lineage first (demonstrated in `zh_ep166_bounded_fixture_promoted`).
