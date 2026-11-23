# Issue #14: independent audit of downstream subtitle assembly handback

**Disposition:** Prior host candidate at 17bb0239f7d8290d43475b06055cd95c2418a1d6 PASS on CPU/static and bounded Riva/local-voice qualification; overall downstream acceptance PARTIAL because the **canonical** in-memory 218-utterance translation assembly refused a retained-source mismatch. The temporary rekeyed diagnostic render was only evidence, not the accepted output. Issue #14 OPEN; existing PR #15 DRAFT.

## Evidence admitted

- Uploaded issue14-downstream-subtitles-evidence-20261123T195952Z.tar.gz: SHA256 d171504e8cfd361351348406b84f596973b3971e7692e15e4ad7f554e650e467, 254507 bytes, 35 unique regular tar members including CHECKSUMS, 34/34 checksums valid, 0 duplicate names.
- Previous host public evidence-only GitHub commit a6eadf27e83e3e50df7b5be574cb10199d859c18; manifest staging/issue14/evidence/20261123T195952Z/manifest.json SHA256 7c8cbde661d9a1914d9ab44ebcf57353e8aeed22422d36478e9dbd7c0417009b. Independently compared all 10 uploaded public file Git-blob identities with authenticated GitHub remote, 10/10 exact matches.
- Prior source 17bb0239f7d8290d43475b06055cd95c2418a1d6: compileall PASS; focused 71/71; core 431/431; Studio 431/431; correct isolated worktree imports. New source below has NOT yet been host-validated.
- Original source recording file SHA256 a67056da45130ae59bed594857372023c11e3c18a5f5b9bc85e167457b3b550e; original translated contract SHA256 9422b3861743be135bf91baca40b43ef6de39b662f7032d78a1322b744df94fd. Diagnostic renewed translation is non-canonical.
- Private Riva nine-turn receipts are actual existing, resource-safe translated responses. Fresh model inference is NOT necessary merely to verify the identity fix.

## Root cause proven from original archived input

Central independently extracted BOTH original 217-utterance recording.contract.json and 217-row translated.contract.json from the earlier authenticated private default-FR archive. Every translated row carries a unique original_utterance_id that matches precisely one recording source ID. When joined on original_utterance_id, the source_text/source_start/source_end/speaker_id fields of ALL 217 original and translated rows agree: **0 mismatches**.

The chronology repair audit_and_order_untimed_utterances in utterances.py lines 659-664 sorts the translated rows and reassigns each display id. Five changed rows have the following (translated display id -> original source id) permutation:
  utt-0148 -> utt-0150
  utt-0149 -> utt-0151
  utt-0150 -> utt-0152
  utt-0151 -> utt-0148
  utt-0152 -> utt-0149
The previous source_fusion_downstream.assemble_revised_translation() joined on display id. It therefore falsely compared the content of utt-0150/0151 with other legacy utterances, then correctly failed its strict source-equality gate. No indication the original untranslated source units were corrupt; changing the all-units gate or accepting mismatched text would have been the wrong remedy.

## Central correction staged on existing PR #15

Central source commit **59a1b6a07613c1d7901f549af9c61db1d5db8696**, tree **14aabba2a241fdec636970ffb6a2a8bc07fff82e**:
- Build an exact bijection from translated display id to stable original_utterance_id (fall back to id only for older translated contracts that never had the audit alias); reject unknown/duplicate/missing source identity before any translation reuse.
- Keep strict source text/bounds/speaker comparison on the stable identity and reject reusing stale English for revised source.
- In copied retained English rows, restore canonical original source ID; map grouped_utterance_ids, merged_into and probable_duplicate_of through the source identity bijection. Reject removed source grouping and merge dependencies. Retain any historical probable-duplicate reference to a superseded source in explicit historical_probable_duplicate_of_superseded metadata, not an active potentially dangling pointer.
- Report the historical translation identity-map digest and count of renumbered rows (five for the archived FR input), keeping the original contracts byte-for-byte unchanged.
- Add general regression coverage in tests/test_issue14_source_fusion_downstream.py for chronology permutation, stale references, conflicting aliases, duplicate hypotheses and unchanged no-alias contracts; record invariant in AGENTS.md.
- This changes NO raw source evidence, ASR, source reconstruction, text model, Riva, diary of speaker labels, cue renderer or ModelLifecycle policy. No case-specific utt-0150 fix.

## Bounded diagnostic output already available, not certified final

Non-canonical in-memory rekeying passed a 218-row translated render: 209 unchanged English legacy rows and nine newly Riva-translated reviewed source turns, with 289 renderable cues versus 287 baseline, unrenderables 2 versus 3, 18+ CPS 60 versus 61, 22+ CPS 27 unchanged, ASS overflow zero. W02b turns now distinguish local Voice_A/B events, while global speaker remains SPK_UNK; W04a SFX/displaced legacy cues are excluded from the revised rendered component. This is promising but NOT proof of exact canonical source identity, independently human-approved English, complete episode fidelity or correct word timing.

Raw overlap observation c11:0..3 versus c10:75..78 remains proposed only. The independently heard phrase in [985.75,986.90] may be one occurrence, but the original [980,984] audio coverage was not furnished for a complete negative repeat test. Do not auto-dispose the raw units. Three incomplete reviewed source components remain blocked. Riva word-choice concerns ('nacale', 'langage mu') should be treated as general glossary/contextual review tuning candidates, not as a justification to modify the acoustic source.

## Next acceptance and disposition

Perform ONE CPU-only host campaign: validate exact Central source; use the authentic original/translated/218 revised contracts and already captured nine Riva receipts to call canonical assemble_revised_translation(..., enable=True) WITHOUT temporary external rekeying. Require 209/209 retained source/English identity matches, nine changed translation receipts, independent copy immutability, and all active grouped/merged/duplicate pointers either mapped to surviving source IDs or explicitly historical/inactive. Independently validate private renderer on canonical contract and compare with the prior diagnostic and baseline. Reuse same 94 reviewed source units and local acoustic labels; no ASR, VLM or Riva GPU inference. On success, Central can consider a separate explicit opt-in pipeline path; on failure return exact repro, not a relaxed test.

New controlling handoff: staging/issue14/HANDOFF_CANONICAL_REVISION_SUBTITLES.md.