# ASR Comparison: `Qwen3-ASR-1.7B` vs Nemotron 3.5 NVFP4 & WhisperX/Pyannote (Experiment C)

Evaluated via [`run_asr_eval.py`](file:///workspace/nvidia-transcribe/experiments/modular/asr/run_asr_eval.py) in `outputs/modular-eval/.venv` (`qwen-asr==0.0.6`, `transformers==4.57.6`, `bfloat16`, `max_new_tokens=4096`). Load time: **`1.9331s`**, load VRAM: **`4480.79 MiB`** peak.

## 1. Clean Read Speech (FLEURS French & Mandarin Benchmarks)

| Benchmark | Model | Aggregate Error Rate | Breakdown & Key Observations |
| --- | --- | --- | --- |
| **French FLEURS** (`4` clips, `98` words) | **Qwen3-ASR-1.7B** (`explicit` & `auto`) | **`0.0816` WER** (`8` edits) | `fr_fleurs_00: 0.0000`, `01: 0.1538`, `02: 0.0000`, `03: 0.1538` |
| French FLEURS | Nemotron 3.5 NVFP4 | `0.1020` WER (`10` edits) | `fr_fleurs_00: 0.0000`, `01: 0.1538`, `02: 0.0769` (`"en plus"` vs `"ample"`), `03: 0.1538` |
| **Mandarin FLEURS** (`6` clips, `178` chars) | Qwen3-ASR-1.7B (`explicit` & `auto`) | `0.1461` CER (`26` edits) | `00: 0.0000`, `01('钾'): 0.0000`, `03: 0.0000`, `04: 0.0690` (`"大气大气"` repeat); wrote Chinese numerals `"十五"`, `"二零一一"` on `02` (`12` subs), nailed **`"Scoturb"`** on `05` (`0.3243`) |
| Mandarin FLEURS | Nemotron 3.5 NVFP4 | **`0.1236` CER** (`22` edits) | Nailed Arabic numerals on `02` (`0.0270`), `'钾'` exact on `01` (`0.0000`), misrecognized `"Scotturb"` as `"School tour bus"` on `05` (`0.4324`) |
| `zh_2spk_seq` (`27.08s`, `81` chars) | Qwen3-ASR-1.7B | **`0.1111` CER** (`2S / 7D / 0I`) | Recognized ABA return (`"Scoturb 403 ... 钙、钾..."`) cleanly |
| `zh_2spk_seq` (`27.08s`) | Nemotron 3.5 (`speaker`) | `0.2469` CER (`5S / 6D / 9I`) | Emitted native `<spk:0>/<spk:1>` tags (`4` speaker runs) |

Language Identification (`language=None`): Qwen3-ASR detected `'French'` / `'Chinese'` with **100% accuracy** across all 30 clips (`auto` WER/CER virtually identical to `explicit`, except `S1` balanced overlap where `auto` degraded from `0.7273` to `1.0` vs A).

## 2. Real French Overlap/Noise Controls (`O1`–`O4`) & Synthetic Overlap (`S1`–`S3`, `zh/fr_2spk_overlap`)

| Case | Class | Qwen3-ASR-1.7B Output & Metric | WhisperX/Pyannote Baseline | Nemotron 3.5 Baseline |
| --- | --- | --- | --- | --- |
| `O1` (`9s`) | Speech + groan | `WER = 0.0435` (`1D`: dropped `n'` in `on a`) | `WER = 0.0000` | Nailed dialogue in `100s` & episode |
| `O2` (`9s`) | Rapid turn `"Mais oui !"` | `WER = 0.2143` (`4S/2D`: `"Acad..."`, `"Ah oui"`, `"Manfura"`) | **`WER = 0.0357`** | Clean `"Mais oui !"` in `F4` |
| **`O3` (`9s`)** | **Music + laugh (negative control)** | `WER = 0.2941` (`2S/3I`: hallucinated **`"Un jour, envers Shangri-La."`**) | `WER = 0.1176` | — |
| `O4` (`9s`) | Speech + yelp | `WER = 0.2778` (`2S/3I`: `"ancêtres muets"`, `"ça se coule pas vrai"`) | `WER = 0.2222` | Nailed `"ancêtres muins"` |
| **`S1` (`4.8s`)** | **Balanced 0dB 2-spk overlap** | **`WER_AB = 0.8500` (`6S/9D/2I`)**: blended A+B hallucination **`"Je suis sûr qu'il fait de l'escalade sur la côte occidentale."`** | `WER_A=0.0, WER_B=1.0` (only A) | Native overlap channel capability in SoT mode |
| **`S2` (`5.18s`)** | **Dominant + `-8 dB` secondary** | **`WER_A=0.0, WER_B=1.0` (`7D`)**: `0%` of secondary speaker Zia recovered | `WER_A=0.0, WER_B=1.0` | Same single-stream omission risk without diarization |
| `S3` (`7.42s`) | Interrupted `0.9s` overlap | `WER_AB = 0.1111` (`1S/1D`: `"Passons à la phase deux"`, dropped `"Bien,"`) | `WER_AB = 0.0000` | Same combined turn |
| **`zh_2spk_overlap`** | Mandarin `7.6s` overlap | **`CER_AB = 0.5167` (`31D`, `0%` overlapped B)**: dropped `"Scotturb 403 路... 前往辛特拉"`, kept only `"在罗卡角停靠"` | — | **`multispeaker_sot` transcribed BOTH speakers** (`49` chars, `<spk:0>/<spk:1>`) |
| `fr_2spk_overlap` | French `6.28s` overlap | `WER_AB = 0.2826` (`3S/10D`): dropped B's `"L'accident a eu lieu en terrain montagneux"` | — | `multispeaker_sot` captured words from both (`45` hyp words, `6D`) |

## 3. French Animation Multi-Speaker Speech (`F1`–`F7`, `F_100s`, `F_300s`)

- **Short clips (`F1`–`F7`):** Providing domain terms via `context="Koumlar, Nat'cal, Ambrosius, Tao, Zia, Esteban, Mu"` on `F1` corrected `"Kumla"` -> `"Koumlar"` and `"Thaou"` -> `"Tao"`, but left `"natkal"` (`Nat'cal`). Short clips `F5` (`2s`: `"fera de moi le"`), `F6` (`"Mûin. Ça se."`), `F7` (`"c'est mieux."`) show Qwen's single-stream transcription without diarizer context.
- **`F_100s` (`100s`, `8.21s`) and `F_300s` (`300s`, `18.21s`):** Lexical quality degraded noticeably compared to Nemotron 3.5 NVFP4:
  - In `F_100s`: `"comme des gamins"` -> `"comme des diamants"`, `"Tao"` -> `"t'a eu"`, `"Nat'cal"` -> `"nappale"`, `"baiser / Zia"` -> **`"Daisy Meadows"`**, `"Zia"` -> `"Zira"`.
  - In `F_300s` (`max_new_tokens=4096`): **`"comme des gamins"` -> `"comme des chiens"`**, `"Isabella"` -> **`"Abel"`**, `"Mendoza"` -> `"mais de ça"`, `"Grand Condor"` -> `"comte"`, `"nous nourrit pas"` -> `"nous ne l'oublions pas"`, `"Gaspard"` -> `"Garde pas"`.
- **Single-stream architectural gap:** Because Qwen3-ASR has **zero native speaker-turn output**, it provides no speaker tags during rapid dialogue (`F2`) and loses overlapping speech (`zh_2spk_overlap`, `fr_2spk_overlap`, `S1`, `S2`) — confirming integrated Nemotron 3.5's decisive ASR advantage on our target audio.
