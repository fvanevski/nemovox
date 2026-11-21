# Continuous Nemotron 3 Diarization Comparison (Experiment A)

Executed via [`run_diarization_eval.py`](file:///workspace/nvidia-transcribe/experiments/modular/diarization/run_diarization_eval.py) in `outputs/modular-eval/.venv-diar` (`transformers==5.23.0`, `float32`). Load time: **`0.324s`**, load VRAM: **`383.16 MiB`** peak.

## 1. Synthetic & Verified Controls (`S1`–`S3`, `O1`–`O4`, Overlap, Mandarin)

| Case | Mode | Spk Count | Segs | Overlap Detected (s) | Runtime (s) | DER | JER | Overlap F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S1` (balanced 0dB overlap, `4.8s`) | Streaming `low_latency` | 2 | 2 | 1.84 | 0.093 | **0.1386** | 0.1481 | **0.8578** |
| `S1` | Offline | 2 | 2 | 1.86 | 0.015 | 0.1368 | 0.1463 | 0.8631 |
| `S2` (`-8 dB` secondary overlap, `5.18s`) | Streaming `low_latency` | 2 | 3 | 1.84 | 0.101 | **0.1868** | 0.1777 | **0.9293** |
| `S2` | Offline | 2 | 3 | 1.61 | 0.014 | 0.2189 | 0.2054 | 0.8633 |
| `S3` (`0.9s` interruption, `7.42s`) | Streaming `low_latency` | 2 | 2 | 0.58 | 0.137 | **0.0737** | 0.0684 | **0.7838** |
| `S3` | Offline | 2 | 2 | 0.59 | 0.015 | 0.0681 | 0.0658 | 0.7919 |
| `zh_2spk_seq` (`27.08s`, ABA return) | Streaming | 2 | 8 | 0.00 | 0.513 | 0.4314 | 0.3818 | 1.0000 |
| `zh_2spk_overlap` (`10.38s`) | Streaming | 2 | 5 | 3.69 | 0.193 | 0.4272 | 0.4060 | 0.6525 |
| `fr_2spk_overlap` (`12.2s`) | Streaming | 2 | 4 | 2.84 | 0.247 | 0.3149 | 0.3060 | 0.6228 |
| `O3_music_neg` (music + laugh, `9.0s`) | Streaming / Offline | **1** | 2 / 1 | **0.00** | 0.177 / 0.015 | — | — | — |
| `F1` isolated (`03:16–03:55`, `39s`) | Streaming / Offline | **2** | 14 / 7 | 0.68 / 0.00 | 0.744 / 0.036 | — | — | — |
| `F2` isolated (`04:22–04:39`, `17s`) | Streaming / Offline | **2** | 8 / 5 | 0.00 | 0.323 / 0.016 | — | — | — |

Key observations on short/controlled clips:
- **Strong acoustic overlap detection & `-8 dB` sensitivity:** Detected `S2`'s `-8 dB` quiet secondary speaker with `ovF1 = 0.9293` (where Pyannote/WhisperX + MossFormer2 produced 0% lexical recovery on speaker B).
- **Clean negative-control immunity:** On `O3_music_neg` (where Pyannote falsely flagged a second speaker over orchestral music), Nemotron 3 Diarization reported **1 speaker (`0.0s` false overlap)**.
- **Note on `O1`/`O4`:** On `O1` (`3` speakers streaming vs `2` offline, `0.34s` overlap streaming) and `O4` (`1` speaker), short nonlexical groans/yelps differed between 1.04s streaming lookahead and full 9s offline buffers.

## 2. Continuous 5-Minute Passage (`F_300s`, `02:30–07:30`) vs Full 23-Minute Episode (`F_full`, `1379.9s`)

| Run | Mode | Runtime (s) | RTF | Peak VRAM (MiB) | Active Speakers | Segments | Overlap Frames (s) | Uncertain `[0.25, 0.50)` (s) | `F1` (`03:16–03:55`) Separated? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `F_300s` (`300s`) | Streaming `low_latency` | 5.9727 | 0.0199 | 419.33 | `6` (`0..5`) | 60 | 0.00 | — | **Yes (`1` & `2`)** |
| `F_300s` (`300s`) | Offline | 0.5200 | 0.0017 | 447.74 | `6` (`0..5`) | 32 | 0.00 | — | **Yes (`1` & `2`)** |
| `F_full` (`1379.9s`) | Streaming (`1917` steps) | **27.4232** | 0.0199 | **419.33** | **`7` (`0..6`)** | 250 | **0.05** | **113.18** | **Collapsed mostly to `5` (`2` blip `0.05s`)** |
| `F_full` (`1379.9s`) | Offline | **1.4602** | 0.0011 | **579.10** | **`6` (`0..5`)** | 162 | **0.00** | **81.82** | **Collapsed mostly to `1` (`3` blip `0.04s`)** |

## 3. Side-by-Side Episode Identity Comparison (`F_full`, 23 Minutes)

| System | Active Speaker IDs | Segs | `F1` (`03:16–03:55`) Behavior | `F5` (`10:31–10:33`) | `F_full` Overlap Frames |
| --- | --- | ---: | --- | --- | --- |
| Pyannote (root `result.json`) | `5` (`SPEAKER_00`..`04`) | 415 | Collapsed to single `SPEAKER_03` | Single `SPEAKER_01` | — |
| Pyannote (`20261120` run) | `7` (`SPEAKER_00`..`06`) | 280 | Collapsed to single `SPEAKER_04` | Single `SPEAKER_03` | `0.0s` |
| Nemotron 3.5 Sortformer (single pass) | `7` (`0`..`6`) | 229 | Clean multi-seg (`DER` ref vs below) | Single | `7.63s` |
| Nemotron 3.5 Chunked 90s + TitaNet | `22` global (from `44` local, `14` provisional) | 213 utt | Collapsed Esteban/Tao into `SPK_06` (+ 1-word tags) | Clean | — |
| **Standalone Nemotron 3 Diar. (Streaming)** | **`7` (`0`..`6`)** | **250** | **Collapsed onto `5` across 23m (worked in `F_300s`!)** | **Single `3` (`625.68–643.4s`)** | **`0.05s`** |
| **Standalone Nemotron 3 Diar. (Offline)** | **`6` (`0`..`5`)** | **162** | **Collapsed onto `1` across 23m (worked in `F_300s`!)** | **Single `2` (`624.07–643.77s`)** | **`0.00s`** |

- Cross-system agreement on `F_full`: Streaming Nemotron-3 vs Nemotron-3.5 Sortformer has **`DER = 0.4968` (`218.91s` confusion, `106.87s` missed, `14.12s` FA)**; Offline Nemotron-3 vs Sortformer has `DER = 0.4201`; Streaming vs Pyannote (`20261120`) has `DER = 0.6248`.

## 4. Eight-Speaker Ceiling & Cache Saturation Findings

- Channel `7` (8th speaker slot) reached only **`max_prob = 0.2192` (streaming)** and **`0.0559` (offline)** — `channel_8_saturated = False`.
- **Critical cache dynamics:** First-arrival timestamps in streaming `F_full` were:
  `{"0": 4.32, "1": 28.0, "2": 43.92, "3": 77.76, "4": 81.74, "5": 196.76, "6": 265.52}` seconds.
  After `t = 265.52s` (`4:25`), **zero new speaker channels opened for the remaining 18.5 minutes** even as new episode characters entered; later voices were absorbed into slots `0..6`, similarity pressure flattened child-voice `F1` separation compared to the 5-minute `F_300s` run, and `113.18s` of speech sat in `[0.25, 0.50)` uncertain activity with near-zero detected overlap (`0.05s`).
- **Architectural implication:** Standalone Nemotron-3-Diarization is much better suited for **~3–5 minute scene-bounded continuous streaming** (`F_300s`, where `F1` separated cleanly and all `60/90/120s` ASR chunk seams stayed seamless) than unchunked 23-minute multi-scene episodes that exceed ~7 distinct speakers.
