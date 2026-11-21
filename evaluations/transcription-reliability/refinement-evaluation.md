# Phase C Report: Experimental Localized Transcript Refinement (`C0` vs `C1` vs `C2`)

Phase C evaluated three localized refinement strategies across **39 French and Mandarin cases** (`19` verified natural clips, `8` controlled error/clean probes, `12` French episode excerpts with `reference_verified=False`). All outputs remain strictly experimental (`status="proposed_only"` in [src/nvidia_transcribe/refinement.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/refinement.py)); no canonical transcript was modified.

---

## 1. Reasoning Backend Findings: Nemotron-3.5 vs `/workspace/ninfer/swiftu`

1. **Nemotron 3.5 Transcribe NVFP4 (`enable_thinking=True` and `False` on `127.0.0.1:8190`):**
   - Explicitly probed per stop condition. Failed text-only reasoning cleanly and repeatably:
     - Pass 1 (`enable_thinking=True`): emitted a JSON block repeated 3× consecutively with self-contradictory `"changed": false` alongside an edited `"proposed_text"`.
     - Pass 2 (`enable_thinking=True`): returned `"content": null`, dumping malformed JSON with a stray trailing `}` into `reasoning` (`parse_error`), again with `"changed": false`.
     - Pass 3 (`enable_thinking=False`): emitted `changed: false`, keeping the misspelled target unchanged while arguing in its rationale that `"Estebane"` should be corrected and hallucinating `"from previous segment: Esteban doit retrouver Mendoza"`.
   - **Disposition:** Nemotron 3.5 text-only `C2` was **stopped and rejected** for reasoning duties.
2. **Alternative Text Reasoning Backend (`/workspace/ninfer/swiftu`, `Qwen3.8-27B-swift15` NVFP4 + DFlash2 on `127.0.0.1:8001`):**
   - Passed the gate immediately (`0.774s`, clean CoT `reasoning_content`, `changed=true`, `0` repetition).
   - Across 39 cases: **`36/39` valid JSON parses (`92.3%`)**, `0` degenerate repetitions, `2` harmless Unicode quote normalization mismatches, `3` `<think>` token budget exhaustions cleanly caught via graceful fallback to `c0_text`, and **`14` proposed changes** (`C2a`) forwarded to independent Nemotron audio re-examination (`C2b`).

---

## 2. Quantitative Comparison (`C0` Baseline vs `C1` Contextual Audio vs `C2a` Swiftu Text vs `C2b` Audio-Verified)

### Track A — Controlled Error-Detection & False-Correction Probe (`8` Cases: `4` Injected Errors + `4` Clean Negatives)

| Strategy | Precision | Recall | False-Correction Rate (Clean Controls) | Net Edits Reduced | Mean WER/CER (`Before -> After`) | Outcomes (`I / D / N / U / FC / M`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`C0` (Baseline)** | — | `0.000` | `0.000` | `0` | `0.0300 -> 0.0300` | `0 improved, 4 unchanged_correct, 4 missed_error` |
| **`C1` (Nemotron Contextual Audio)** | `0.2857` (`2/7`) | `0.500` | **`0.500` (`2/4`)** | **`-4`** (worse) | `0.0300 -> 0.0424` | `2 improved, 1 degraded, 2 neutral, 1 unchanged, 2 false_correction` |
| **`C2a` (`/workspace/ninfer/swiftu`)** | **`1.000` (`4/4`)** | **`1.000`** | **`0.000` (`0/4`)** | **`+6`** | **`0.0300 -> 0.0000`** | **`4 improved, 4 unchanged_correct, 0 errors`** |
| **`C2b` (`C2a` + Nemotron Audio Gate)** | **`1.000` (`2/2`)** | `0.500` | **`0.000` (`0/4`)** | `+3` | `0.0300 -> 0.0116` | `2 improved, 4 unchanged_correct, 2 missed_error` |

### Track B — Verified Natural Clips (`19` Cases: `6` Mandarin FLEURS, `4` French FLEURS, `7` `O1–O4`/`S1–S3`, `2` Synth Overlap)

| Strategy | Precision | Recall | False-Correction Rate | Net Edits Reduced | Mean WER/CER (`Before -> After`) | Notable Observations |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`C0` (Baseline)** | — | `0.000` | `0.000` | `0` | `0.2680 -> 0.2680` | `4 unchanged_correct, 15 missed_error` |
| **`C1` (Contextual Audio)** | `0.600` (`9/15`) | `0.600` | `0.000` | `+69` | `0.2680 -> 0.1626` | Helped overlap/glossary (`ov_O4`, `fr_2spk_overlap`), **but** dropped 2nd speaker on `zh_2spk_overlap` and hallucinated glossary override (`ov_O2` `"Mandfoura" -> "Mendoza !"`) |
| **`C2a` (`/workspace/ninfer/swiftu`)** | **`0.800` (`4/5`)** | `0.2667` | **`0.000`** | `+17` | `0.2680 -> 0.2460` | Fixed genuine ASR homophone **`“跳下垂直近空” -> “桥下垂直净空”`** (`zh_fleurs_02_1883`) + deinterleaved `fr_2spk_overlap`; `0` degraded |
| **`C2b` (`C2a` + Nemotron Audio Gate)** | — | `0.000` | `0.000` | `0` | `0.2680 -> 0.2680` | Global string distance tie (`d_c0=13 == d_prop=13`) on Chinese vs Arabic digits (`二零一一` vs `2011`) vetoed valid `zh_fleurs_02` fix |

---

## 3. Concrete Behavioral Highlights & Audio-Verification Lessons (`episode_unverified_audit` + `controlled_probe`)

- **Why audio verification matters (`C2b` catches text-only hallucinations):**
  - `ep_utt-0005`: `C0 = "chèvres"`; `swiftu` guessed `"Cités d'Or"` from episode glossary/context alone -> Nemotron audio recheck transcribed `"En Chine, au Japon et en Afrique."` and **vetoed the hallucination (`corrob=False`)**.
  - `ep_utt-0027`: `C0 = "Diablement efficace, n'est-il pas?"`; `swiftu` proposed `"n'est-ce pas?"` -> Nemotron audio recheck confirmed `"n'est-il pas ?"` (`d_c0=0 < d_prop=1`, **vetoed `corrob=False`**).
- **Where `C2a` + `C2b` succeeded together:**
  - `ctrl_err_zh_oov_homophone`: `'钙、甲等'` -> `'钙、钾等'`, corroborated identically by Nemotron audio (`d_c0=1, d_prop=0`).
  - `ep_utt-0003`: `C0 = "Tincio, Carlos, Diego, Zia et Tato..."` -> `swiftu` fixed `'Tato' -> 'Tao'`, corroborated by Nemotron audio recheck (`"Esteban, Zia et Tao"`, `d_c0=5, d_prop=4`).
- **Why `C1` must NOT run automatically:**
  - `ctrl_neg_fr_clean_02`: `C1` corrupted valid `"au rugissement ample du lion"` -> `"au rugissement amplifié"` (deleting `"du lion"`).
  - `ep_utt-0000`: `C1` corrupted `"Enfant du soleil"` -> `"Enfant du sommeil"`.
- **Implementation flaw identified in naive global `C2b` comparator:** Comparing whole-utterance Levenshtein distances (`d_prop < d_c0`) suffers when (a) audio recheck alternates script normalization (`"二零一一"` vs `"2011"` -> `13 == 13`), or (b) custom audio prompts return empty `""` (`ep_utt-0030`, `ctrl_err_fr_compound`), making shorter proposed text falsely satisfy `d_prop < d_c0`. Future production promotion requires span-aligned acoustic confidence scoring and human review (`status="proposed_only"`).
