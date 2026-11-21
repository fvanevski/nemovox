# Subtitle Quality Review

## Deliverables Inspected

- **Complete French Episode (`221` utterances, ~23 min):** `Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].en.srt` and `.en.ass` under `outputs/subtitle_runs/fr_full_episode/`
- **Four Repaired Mandarin Fixtures:** `zh_fleurs_01_2006`, `zh_fleurs_05_1805`, `zh_2spk_overlap`, `zh_multichunk_synthetic_qwen` under `outputs/subtitle_runs/zh_samples/`

## Summary Metrics

| Recording / Sample | Utterances | Renderable Cues | ASS Displayed Cues | Top-Lane Overlaps (`Layer 1`) | Overflow (`>=3`-way) | Conflict Cues | Empty / Unrenderable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **FR Full Episode (`S04E01`)** | 221 | 286 | 281 | 21 | 5 | 61 | **0** |
| `zh_fleurs_01_2006` | 1 | 2 | 2 | 0 | 0 | 0 | 0 |
| `zh_fleurs_05_1805` | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| `zh_2spk_overlap` | 2 | 2 | 2 | **1** | 0 | 0 | 0 |
| `zh_multichunk_synthetic_qwen` | 7 | 12 | 12 | 0 | 0 | 0 | 0 |

- **Vetoed-regrouping orphan bug caught and resolved:** Initial inspection of `fr_full_episode` surfaced 6 utterances (`utt-0036`, `utt-0065`, `utt-0114`, `utt-0118`, `utt-0147`, `utt-0159`) where `SwiftU` merged into the previous utterance (`translated_text == ""` + `grouped_with_prev=True`), `_apply_safe_adjacent_regrouping` cleanly vetoed the merge due to `>1.5s` gap or `>12s` duration, and `subtitles.py` fell back to French `refined_text`. We fixed `translate_utterances` in [translation.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/translation.py) to automatically re-translate any vetoed orphan pair with `allow_regrouping_override=False` and keep `_grouped_with_prev` intact until final regrouping (`Cleanly merged utterances: 4`, `Orphan empty translations remaining: []`).

## Qualitative Dialogue & Regrouping Highlights

1. **Full French Episode (`fr_full_episode`)**:
   - **Names & Lore Consistency:** Opening theme (`00:00:04–00:00:22`), narrator (`Esteban, Zia, and Tao have discovered three new Cities of Gold...`), antagonists (`Ambrosius`, `Zarès`, `Laguerra`, `Gaspard`), places (`Kumla`, `Lalibela`, `Mu`), and Tao's bird (`Pichu`, recovered from ASR's `"Pikachu"`) read consistently throughout the episode.
   - **Adjacent Same-Speaker Regrouping:** 4 true continuous splits merged safely (`utt-0049 -> utt-0048`, `utt-0137 -> utt-0136`, `utt-0140 -> utt-0139`, `utt-0156 -> utt-0155`), keeping unbroken phrasing within single cues.
   - **Remaining Known Limitations Observed:**
     - **Upstream ASR chunk duplicate (`280.0s–291.5s`):** `utt-0042` (`SPK_07`) and `utt-0045` (`SPK_10`) in the frozen input `recording.contract.json` both carried `"L'intérieur du continent est encore inexploré."`, so Stage 4 faithfully placed one in `Main` and one in `Top` (`shot_06`).
     - **Missing clan word in glossary (`utt-0152`):** `"Naacal"` was absent from [mcog_fr.yaml](file:///workspace/nvidia-transcribe/configs/glossaries/mcog_fr.yaml), so Stage 2 converted `"Akâa"` to `"Tao"` -> `"A Tao doesn't touch things!"`.

2. **Mandarin Samples (`zh_samples`)**:
   - **`zh_fleurs_01_2006`**: `"Calcium, potassium, and other elements are metals. Silver and gold are, of course, metals too."` (2 clean cues, `0` conflicts, OOV `钾` timed accurately via localized Qwen fallback).
   - **`zh_fleurs_05_1805`**: `"Scotturb Route 403 runs regularly to Sintra, stopping at Cape Roca."` (1 cue, `0` conflicts, recovered from ASR `"Scouter...新特拉"`).
   - **`zh_2spk_overlap`**: Two overlapping speakers cleanly separated onto `Main` (`"This isn't a farewell, just the end of one chapter and the beginning of a new one."`) and `Top` (`"The Scotturb 403 bus runs regularly to Sintra, with a stop at Cape Roca."`).
   - **`zh_multichunk_synthetic_qwen`**: Adjacent split `utt-0002` + `utt-0003` regrouped seamlessly into `"The vertical clearance under the bridge is fifteen meters. The project was completed in August 2011, but it didn't open to traffic until March 2017."` with exact combined acoustic span `[22.45s, 30.915s]`.
