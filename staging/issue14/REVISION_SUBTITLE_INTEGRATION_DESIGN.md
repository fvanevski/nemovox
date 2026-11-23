# Issue #14: reversible revision-specific subtitle integration (Central design)

Current development: fvanevski/nemovox, Issue #14, existing draft PR #15.
Latest host-qualified source: 420173025dd28837adc91dfdb970227f3795124e, tree 3c98b4f72ab7d68924abcb3a5d1fe8fec2fc42b8.
New Central source after this handback: 17bb0239f7d8290d43475b06055cd95c2418a1d6, tree 1dded4da4ed118632724b69eb83b92f40a80153d (host verification PENDING).
No production integration, merge, or issue closure authorized.

## Evidence admitted

User-supplied private evidence issue14-evidence-fusion-evidence-20261123T194055Z.tar.gz: SHA256 16ef680e419cc2bbe71fa9607e17a6c6323a104b7e23b468e13b5a915a7bc224, 158617 bytes, 28/28 unique internal checksums, no duplicate entries. Public packet root 251d4b8471d62a49c2a3404a288b66369e7ff7bd:staging/issue14/evidence/20261123T194055Z/, manifest SHA256 c2cc55cd41f464774c558e4573f391df63103a935123dc6c9c44fb2b25efc529 and 9/9 exact remote Git blobs/sha256/bytes independently verified.

Host CPU qualification of prior Central source: focused 70 PASS, full core 425 PASS, Studio 425 PASS at identical source SHA under PYTHONPATH=src:.; source diff none. Real French source-unit replay: 2008/2008 original legacy units matched, 149 positive event links, 13 audio-reviewed source turns, 213 connected components total, six complete reviewed components and three incomplete reviewed components. Private opt-in revision replaced 8 legacy utterances with 9 reviewed source events; revised contract 218 turns (209 retained + 9 new), 94 event-linked raw unit references, raw ledger unchanged, global speaker unknown, word-level timing unverified. Synthetic local model probe 3/3 structured response parses: semantic accuracy/calibration UNVERIFIED. Review-support labels refer to the PROPOSED ACTION (including a proposed action to abstain), not direct permission to change source.

## Three general remaining capabilities

### A. Raw observations excluded before v1 assembly

Original raw units are NOT guaranteed to have an active legacy-utterance owner after chunk/core/overlap reconciliation. The blocked W03a event demonstrates this: c11:0..3 "Nous avions déjà vu" occur in the original raw ASR, but were pruned before v1 utterance assembly. They are listed in the reviewed event region and therefore correctly block a fully owned connected component. The ledger's corr_547022f83ef1499b34f47816 associates a 10-unit c10/c11 lexical correspondence but explicitly labels it candidate_only_unverified and preserve_both_observations. The dossier likewise reports cross_chunk_echo_decision=unresolved_without_positive_same_event_identity.

Do NOT silently label those four missing units as duplicates or erase their ledger entries. A future general disposition record must distinguish:
  retained source owner | independently reviewed alternate observation of the SAME event | separate genuine repeated speech | unreviewed/unknown.
It must bind every original raw unit identity, inference window, exact parent ASR SHA and audio event/crop review SHA; a lexical model match or shared speaker tag alone cannot certify same acoustic occurrence. Accept only positively reviewed duplicate-disposition records in the partition plan. The next host evidence phase should directly verify the four units' status from existing original audio and source-assembly traces. No case-name routing, tolerance widening or deletion bypass.

### B. Local voice equivalence without global character claims

W02b split source events carry acoustically attested local Voice_A/Voice_B distinctions. New in-memory source revisions currently set speaker_id=SPK_UNK because local turn review does not certify global episode speaker or character identity. This is correct for the global speaker field, but need not discard the local evidence.

Central added src/nvidia_transcribe/source_fusion_downstream.py: attach_local_acoustic_voices(revised, dossier, voice_review) produces another private in-memory contract that retains SPK_UNK while attaching opaque, scope-bound voice_0/voice_1 labels only to event turns whose review identity and exact event bounds match the direct-audio-reviewed dossier. No character names or episode-wide global speaker merge. Reject stale/contradictory labels. This is **separate from inferential visual character recognition** and is not wired into the translator or renderer as a global ID. It may later inform constrained context/grouping after a distinct acceptance test.

### C. Revision-specific translation/render, not whole-episode retranslation

The existing subtitle_pipeline.generate_subtitles already loads an independent recording_contract_path and the renderer uses utterance source_start/source_end when word timing is absent. The unresolved hazard is mistakenly reusing stale English text for a source utterance replaced by an independently reviewed new event, or wrongly preserving text/group linkage to removed legacy IDs.

Central added src/nvidia_transcribe/source_fusion_downstream.py: assemble_revised_translation(original_recording, previous_translated, revised_recording, new_translations, enable=False|True).
- Assert revised source_fusion_revision identifies the EXACT original recording digest and identical media/language, that prior English translated contract IDs match the original recording IDs, and that removed/new IDs match the reviewed source revision.
- Preview (enable=False) lists the exact missing new-translation IDs; it does not create a translated contract.
- Acceptance (enable=True) requires new translation evidence for EACH added event, bound to source_revision_id, original source_text SHA, language and provenance digest. It reuses previous English translations only for retained IDs with identical source text/time/speaker, rejects grouping that references superseded IDs, and marks newly translated turns as NOT post-edited. Original jobs and old translated contracts are deep-copy preserved.
- New reviewed turns retain their event-level source bounds, empty source_words/characters, SPK_UNK and local voice evidence where attested. Do NOT fabricate word timestamps or assume the global speaker identity.
- Returned translated contract is a PRIVATE in-memory artifact, not automatically saved to Studio, not a registered production CLI path. It can be rendered in a separate private root by existing render_subtitles only after a completeness audit.

For real English semantics, local agent may run one bounded Riva translation of the NINE changed event turns under existing owned GPU lease, then assemble the 218-turn translated contract from the existing archived 217-utterance English baseline, rather than retranslating the whole episode. No mandatory model usage if local service cannot be admitted: CPU preview and structural render can still be assessed with clearly marked non-English placeholder output, but NEVER assert translated-English quality from placeholders.

## Tuning and application architecture

Keep separate layers: immutable observations and dependency graph; context-sensitive heuristic/model applicability policy and reasoned decisions, including counterevidence/abstention; event-unit partitioning and evidentiary ownership; opt-in, reversible source revision; translated source-revision receipts and private render. Different strengths for animated vs live-action lip synchronization are examples of a configurable starting policy, not fixed rules of speech truth. Tune prompt/rules on real and held-out controls, not by adding a series of per-episode patches. Do not treat schema-valid model choices or a model's confidence as an acoustic truth oracle.

The final production opt-in must bind evidence/policy/review contracts, source/revision hashes, resource ownership, translation provenance, render source IDs, and before/after metrics. A source revision that can be generated in memory but is not proven to render correctly is not yet a complete Issue #14 solution. Preserve current source_promotion's older conservative 3s same-event gate for that path; the new event-partition adapter is distinct and may apply ONLY complete components under its own positive evidence gate.

## Acceptance plan / when to stop

1. Host test the exact new source, focused + full nonGPU and Studio with PYTHONPATH=src:.. Verify input source-identity digests, 28/28 private checksums, 9/9 public packet members, clean src/tests head and no unauthorized changes.
2. Reproduce private six reviewed complete components without mutation; attach local audio voice labels to only independently confirmed event IDs. Show global SPK_UNK preserved and unknown breath not attributed. Probe wrong review scope and event identity as negative controls.
3. Compare revised source to previous translated contract. Preview missing new translation receipts; if safely possible, translate just nine new reviewed French turns using owned local Riva. Validate actual source text and response SHA, fresh revision IDs and content, absence of stale legacy English; assemble private translated v1 contract.
4. Render private SRT/ASS by existing public renderer and examine changed W02b/W04a/W03b cues, S02 negation, W05a, genuine repetitions and overlapping speech; report source events/English meaning/time/unknown speakers and CPS counts with SAME denominator as baseline. Do not edit renderer (Issue #9 separate). If evidence does not justify a correction or rendering fails, preserve the failed output separately and report PARTIAL with exact reason.
5. Independently audition the specific c11:0..3/c10:75..78 overlap (already available) to determine whether a general pre-v1 disposition record can be safely authenticated. Do not add automatic duplicate removal in this campaign or force it into a complete component without documented positive same-event evidence.
6. Return one public sanitized evidence packet and self-contained private archive, with complete SHA manifests, unique tar entries, exact PR head and source tree, a clear recommendation on integrating this general opt-in pathway. No new PR, live model restarts, whole-episode ASR, wholesale translator rerun, model download, Studio deployment or merge.

A full episode WER/CER/DER and global character identity remain UNVERIFIED. A successful bounded render is narrower than proof of complete episode speech fidelity.