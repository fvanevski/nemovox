# Translation Review Polarity Guard Qualification (`4c6c7ac5`)

## Guard Behavior Verification (`src/nvidia_transcribe/translation_review.py:97–111`)
- `_unsupported_polarity_inversion(source, original, candidate, language="fr")`:
  - **`S02` negation-drop reproducer:** French negative (`C'est pas ...`) + Riva negative (`It wasn't ...`) + candidate affirmative (`Our ... would have ...`) -> **REJECTED (`True`)**, preserving Riva English MT and recording `review_rejected_polarity_preserved_machine_translation` (`rejected_polarity=1`, `changed=0`).
  - **Equivalent negative rephrasing:** French negative + Riva negative + candidate negative (`... would not have ...`) -> **ALLOWED (`False`)**.
  - **Helpful lexical/pronoun repairs (e.g. archived `idx 11, 21, 61, 63, 180`):** No French+Riva negation drop triggered (`False`) -> **ELIGIBLE & PRESERVED**.
- **Full-Episode Cached Review Status:** Not executed (`STOPPED_PHASE_A_CPU_GATE_FAILURE`); original job `980af31f014245c4b200e0af8b33ebca` outputs (`translated.contract.json`) remain unmodified.
