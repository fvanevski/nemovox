# Quality Review & Ablation Analysis

All figures below come directly from recorded benchmark outputs in [`benchmark.json`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/benchmark.json) and [`supplemental.json`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/supplemental.json), and artifact inspections on `/workspace/nvidia-transcribe/outputs/`.

## 1. French 300s Slice Window-Size Ablation (`40s` vs `60s` vs `90s` vs `120s`)

Source excerpt: `outputs/audio/fr/fr_0230_0730_300s.wav` (`300.0s`, auto language detection `fr` via `"model_detected"`, `context_seconds=10.0`, `min_tail_seconds=15.0`, TitaNet CPU clustering enabled).

| Window | Chunks | Total Wall Time (s) | Raw Local Spk | Global Spk | Unresolved Spk | Ambiguous Matches | Utterances | Timed / Total Units | Boundary Dedup | Boundary Recovered | Alignment Failures | Child-Voices (`03:16-03:55`) Speakers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `40s` | 8 | 16.7491 | 16 | 9 | — | — | 47 | — | 55 | 3 | — | `["SPK_01", "SPK_02"]` |
| `60s` | 5 | 16.8267 | 12 | 8 | 6 | 1 | 40 | 553 / 553 | 15 | 5 | 0 | `["SPK_00", "SPK_01"]` |
| **`90s`** | **4** | **10.1317** | **12** | **7** | **3** | **0** | **42** | **525 / 525** | **20** | **0** | **0** | **`["SPK_01", "SPK_02"]`** |
| `120s` | 3 | 11.1195 | 9 | 6 | 3 | 0 | 42 | 526 / 526 | 19 | 0 | 0 | `["SPK_01", "SPK_02"]` |

*Note: `40s` metrics come from `supplemental.json` / `benchmark.json` `ablation_300s.chunk_40s` where unresolved/ambiguous/timed subsets were not separately tabulated.*

### Selection Rationale (`chunk_seconds=90.0`, `context_seconds=10.0`)

- **Lowest latency (`10.1317s` vs `16.75s` / `16.83s` / `11.12s`)** and **0.034x real-time factor** on the 300s slice.
- **Minimal boundary churn:** `0` recovered boundary omissions and `0` ambiguous boundary matches (vs `3`–`5` omissions at `40s`/`60s`), because shorter windows split more mid-clause transitions.
- **Stable child-voice speaker separation (`SPK_01`, `SPK_02`)** matching `120s` and `40s`, while staying shorter than `120s` (each `90s + 10s + 10s = 110s` max window remains safely below the ~2+ minute threshold where long-form LLM speaker drift accelerates).

## 2. Full 23-Minute French Episode Comparison

Source: `/opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm` (`1379.901s`, explicit `"fr"`, peak GPU VRAM `27813 MiB` of `32607 MiB`).

- **Single-pass Nemotron baseline (`outputs/responses/longform/fr_full_episode.raw.json`):**
  - Sortformer `diarization` maintained `229` continuous segments over speakers `'0'`..`'6'`, but the LLM decoder produced `1992` words with **`481` word-level speaker switches** (severe `<spk:N>` ping-ponging inside sentences).
- **Chunked Core (`90s + 10s`, 16 chunks, `31.0437s` total wall time):**
  - **Ping-ponging eliminated:** `213` coherent utterances over `2016/2016` CTC-aligned words (`0` untimed units, `0` alignment failures), `67` boundary duplicates removed cleanly, `4` boundary omissions recovered, `0` boundary ambiguities.
  - **Window `04:22-04:39` (`utt-0050`..`utt-0053`):** Reconciles cleanly into 4 uninterrupted multi-sentence turns across `SPK_06` and `SPK_08`.
  - **Window `03:16-03:55` (`utt-0030`..`utt-0035`):** Reconciles into 6 utterances (`SPK_06` carries Esteban/Tao monologue; tiny overlapping artifacts `Il` -> `SPK_07` and `Diablement` -> `SPK_08`).
  - **Honest residual limitation:** In `chunk_002.raw.json` (`180`–`270s`), Sortformer `diarization` detected `5` speakers (`'0'..'4'`), whereas the LLM only emitted `word_spks=['0', '1', '2']`, merging Esteban and Tao into `SPK_06`. Separately, stray 1-word tags (`Il`, `Diablement`) occur with valid CTC timings (`error: null`). This confirms NVIDIA's upstream warning: **valid CTC alignment does not guarantee accurate speaker attribution**.
  - **Global lexical consistency:** Chunked WER vs Pyannote transcript is `0.196` (`404` edits over `2061` ref words: `136S / 65D / 203I`, partly reflecting Nemotron's higher song/ad-lib recall), and `0.1924` vs single-pass Nemotron (`420` edits over `2183` ref words).

## 3. Mandarin Overlap Reconstruction (`zh_2spk_overlap`)

- **Phase-1 global-sort contract:** `32` broken 1–2 character utterances interleaved between `<spk:0>` and `<spk:1>`.
- **Phase-2 per-speaker reconstruction (`benchmark.json` `mandarin.zh_2spk_overlap`):** Reduced to **2 coherent overlapping speaker streams** (`SPK_00` and `SPK_01`):
  - `SPK_00`: `"这并不是告诫这是一个片章的结束也是新篇章的开"`
  - `SPK_01`: `"斯科特四零三路公交车车去定期检查确保车轱前方清洁无垃圾在罗卡角停靠"`
- Note that Nemotron hallucinated `'诫'` and `'轱'` (both outside the CTC vocabulary), triggering `diarization_bounded_only` with `0/55` timed units and explicit `oov_characters: ["诫", "轱"]` without losing the text.

## 4. Mandarin `'钾'` OOV Isolation & Partial Sub-Interval Recovery

- **Single clip (`zh_fleurs_01_2006`):** `source_language="zh"` (`model_detected`), preserves exact verbatim source text `"钙、钾等元素属于金属，银和金等元素当然也是金属。"` with `alignment_status="diarization_bounded_only"`, `failure_cause="ctc_vocab_oov"`, `oov_characters=["钾"]`, `3` diarization intervals, and `start=null, end=null` (no fabricated timings).
- **Multi-chunk synthetic (`zh_multichunk_synthetic`, `69.78s`, 2 chunks, 3 local -> 2 global speakers, 7 utterances, 3 boundary dedups):**
  - Chunk `1` aligned `100%` (`alignment_failures=0`).
  - Chunk `0` hit `'钾'` in `[11.68, 17.84]` (`ctc_vocab_oov`). Partial sub-interval recovery grouped non-OOV Sortformer intervals and re-ran CTC alignment, **recovering `83` of `107` units in chunk `0`** (`utt-0001`, `utt-0002`, `utt-0003` marked `subinterval_ctc_recovered`, bringing total recording coverage from `87/194` to **`170/194` timed units**), while keeping only the `24`-unit `'钾'` utterance (`utt-0000`) explicitly `diarization_bounded_only` with its exact source text intact.

## 5. Prompt Mode Comparison (`supplemental.json`)

- **French context vocabulary (`fr_0316_0355_child_voices`):** Supplying domain terms converted `"Kumla"` -> `"Koumlar"` and `"natcal"` -> `"Nat'cal"` while preserving `<spk:0>/<spk:1>` tags.
- **Mandarin script-control vs `mode="speaker"` baseline:**
  - `zh_fleurs_05_1805`: Both achieved identical `CER = 0.3784` (`14` edits / `37` ref chars); `speaker` produced `"School tour bus 403路公交汽车定期发车前往新特拉在罗卡角停靠。"` while `script_control` produced `"School bus 403 路公交汽车定期发车前往新特拉在罗卡角停靠。"`.
  - `zh_2spk_overlap`: `script_control` introduced English words (`"Skull"`, `"Terrier"`, `"San"`) and shifted `<spk:1>` text into `<spk:0>` (`"再挪卡脚停靠"`), making `mode="speaker"` clearly stronger and justifying its retention as the default pipeline prompt.
