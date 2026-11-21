# Phase D Report: Scene-Bounded `Nemotron-3-Diarization` Evaluation

Standalone `Nemotron-3-Diarization` (`/workspace/models/Nemotron-3-Diarization`, `local:Nemotron-3-Diarization:88f392c3e4da`) was evaluated in isolation (`outputs/modular-eval/.venv-diar`) over **scene-bounded intervals** (`60–120s` and `240–300s`) with **explicit `speaker_cache` resets between scenes**, plus 10 ground-truth overlap/sequence clips across `offline` (`is_streaming=False`) and all 3 streaming modes (`low_latency [9,4]`, `very_low_latency [6,2]`, `ultra_low_latency [3,1]`).

---

## 1. Scene-Bounded Runtime, VRAM & Adapter Reconciliation

Model load: `0.4952s`, `378.52 MiB` allocated GPU memory (RTX 5090).

| Scene ID | Scale / Interval | Offline Elapsed / Peak VRAM / Speakers | Streaming (`low_latency`) Elapsed / Speakers | Canonical Repaired Speakers | Adapter Reconciliation (`evidence_adapter_offline`) |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `scene_fr_100s_child_rapid` | `60–120s` (`185–285s`) | `1.095s` / `431.7 MiB` / `6` ch (`74.33s`) | `2.073s` / `6` ch (`72.41s`) | `4` (`SPK_07..10`) | `consistent: 1`, `ambiguous_many_to_one: 4`, `split_and_merge: 1`, `advisory: 2` |
| `scene_fr_90s_overlaps_0530_0700` | `60–120s` (`330–420s`) | `0.087s` / `430.9 MiB` / `4` ch (`35.06s`) | `1.794s` / `4` ch (`32.26s`) | `3` (`SPK_07,09,10`) | `ambiguous_many_to_one: 2`, `one_to_many: 1`, **`unmatched_acoustic_activity: 1`** |
| `scene_fr_90s_music_0930_1100` | `60–120s` (`570–660s`) | `0.132s` / `430.9 MiB` / `3` ch (`57.76s`) | `1.775s` / `3` ch (`58.94s`) | `2` (`SPK_12,13`) | `consistent: 1`, `ambiguous_many_to_one: 2`, `advisory: 1` |
| `scene_fr_90s_yelp_1700_1830` | `60–120s` (`1020–1110s`) | `0.084s` / `430.9 MiB` / `2` ch (`40.93s`) | `1.785s` / `2` ch (`41.96s`) | `2` | `one_to_many: 1`, `many_to_one: 1`, `advisory: 1` |
| **`scene_zh_multichunk_70s`** | `60–120s` (`0–69.78s`) | **`0.070s` / `427.5 MiB` / `3` ch (`35.10s`)** | `1.373s` / `3` ch (`35.05s`) | **`3` (`SPK_00..02`, up from prior `2`)** | **`consistent_support: 3`, `advisory: 0` (exact 1:1 agreement!)** |
| `scene_fr_300s_early_0230_0730` | `3–5m` (`150–450s`) | `0.360s` / `442.8 MiB` / `6` ch | `~6s` / `6` ch (`~140.4s` speech) | `6` | `one_to_many: 2`, `many_to_one: 2`, `split_and_merge: 2`, `advisory: 2` |
| `scene_fr_240s_mid_0730_1130` | `3–5m` (`450–690s`) | `0.254s` / `442.8 MiB` / `4` ch (`91.21s`) | `4.764s` / `4` ch (`92.67s`) | `5` | `consistent: 2`, `one_to_many: 1`, `many_to_one: 1`, `advisory: 1` |
| `scene_fr_300s_late_1200_1700` | `3–5m` (`720–1020s`) | `0.337s` / `447.7 MiB` / `5` ch (`147.86s`) | `6.006s` / `6` ch (`144.39s`) | `7` | `consistent: 2`, `split_and_merge: 2`, `one_to_many: 1`, `advisory: 2` |

---

## 2. Sub-Region Comparisons: Child Voices, Overlaps & Music False Positives

- **Child voices (`F1_child_voices`, `196–235s`) & rapid turns (`F2_rapid_turns`, `262–279s`):**
  - Reference `pyannote/result.json` collapsed BOTH `F1` (`6` segments) and `F2` (`3` segments) into a single `SPEAKER_03`.
  - Scene-bounded `Nemotron-3-Diarization` (both inside `100s` and `300s` scenes, offline & streaming) cleanly separated **`2` distinct child voices** in `F1` (`[0, 1]`, `34.98s` / `34.63s` in `100s`; `34.99s` / `33.41s` in `300s`) and **`3` speakers** in `F2` (`11.15s` in `100s`). Our experimental adapter in [src/nvidia_transcribe/scene_diarization.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/scene_diarization.py) detected that multiple diarization channels mapped onto canonical `SPK_07` / `SPK_08` and emitted **`inspect_possible_collapsed_speaker` advisory flags (`authoritative_override_applied=False`)** without forcing merges.
- **Music false positives (`O3_music_neg`, `627.5–636.5s`; `F5_music_fp`, `631–633s`):**
  - Both standalone `Nemotron-3-Diarization` (`offline` `9.00s` active as `[0]`; `low_latency` `9.02s` as `[0, 1]`) and `pyannote` (`SPEAKER_01`) triggered speaker activity over pure background music/singing. Scene bounding does **not** eliminate music FPs by itself.

---

## 3. Ground-Truth Overlap & A-B-A Sequence Metrics

| Clip | Offline DER | Offline Overlap Recall | `low_latency` DER | `low_latency` Overlap Recall | `very_low_latency` DER | Speakers Detected |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `zh_2spk_seq` (`27.08s`, A-B-A) | `0.4314` | `1.000` | `0.4314` | `1.000` | `0.4314` | `2` |
| `zh_2spk_overlap` (`10.38s`) | `0.4294` | `0.4829` | `0.4272` | `0.4843` | `0.4289` | `2` |
| `fr_2spk_overlap` (`12.20s`) | `0.3144` | `0.4538` | `0.3149` | `0.4522` | `0.3149` | `2` |
| `S1` (`4.80s` balanced overlap) | `0.1368` | `0.7592` | `0.1386` | `0.7510` | `0.1349` | `2` |
| `S2` (`5.18s` quiet secondary) | `0.2189` | `0.7594` | `0.1868` | `0.8679` | `0.1604` | `2` |
| `S3` (`7.42s` interrupted) | `0.0681` | `0.6556` | `0.0737` | `0.6444` | `0.0737` | `2` |

---

## 4. Integration Recommendation

Retain **`Nemotron-3-Diarization` strictly as an offline scene-bounded supplemental acoustic evidence stream** (`scene_diarization.py`, `authoritative_override_applied=False`). At `0.07–0.36s` per 1–5 min scene (`~448 MiB` VRAM offline), it excels as a **diagnostic voice-separation detector** (confirming `3/3` Mandarin speakers and flagging collapsed child voices inside `SPK_07`/`SPK_08`), while persistent music FPs (`O3_music_neg`) and chunk-less cross-scene ID resets make it unsuitable as an unattended global override.
