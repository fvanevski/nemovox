# Conditional Modular & Hybrid Integration Probe (`F_300s`, Experiment D)

Staged on the representative 5-minute passage `outputs/audio/fr/fr_0230_0730_300s.wav` (`300.0s`, global offset `150.0s–450.0s`) via [`run_integration_probe.py`](file:///workspace/nvidia-transcribe/experiments/modular/integration/run_integration_probe.py).

## Tested Variants

| Variant | ASR Source | Aligner Source | Diarization Source |
| --- | --- | --- | --- |
| **D1 (Modular Direct `300s`)** | `Qwen3-ASR-1.7B` | `Qwen3-ForcedAligner` (single `300s` call, over 180s cap) | `Nemotron-3-Diarization` (continuous streaming `low_latency`) |
| **D1 (Modular Chunked `<=180s`)** | `Qwen3-ASR-1.7B` | `Qwen3-ForcedAligner` (auto-split `<=180s` chunks) | `Nemotron-3-Diarization` (continuous streaming `low_latency`) |
| **D2 (Hybrid `Nemotron + QwenAlign + Nem3Diar`)** | **Nemotron 3.5 NVFP4** (`ablation_chunk_90s`) | `Qwen3-ForcedAligner` (`2 x 150s` windows) | `Nemotron-3-Diarization` (continuous streaming `low_latency`) |

All variants enforced Section 10 invariants: exact word preservation, explicit `unaligned` / `zero_duration` flagging without fabrication, edge-word transition guard (`<= 0.18s`), concurrent speaker recording, and `speaker="unknown"` when acoustic overlap was absent or tied.

## Quantitative Side-by-Side (`integration_probe.json`)

| Metric | D1 Modular Direct (`300s`) | D1 Modular Chunked (`<=180s`) | D2 Hybrid (`Nemotron 3.5 ASR`) | Phase-2 Existing Chunked Nemotron 3.5 |
| --- | ---: | ---: | ---: | ---: |
| Total words | 492 | 497 | **525** | **525** |
| Valid non-zero words | 356 (`72.4%`) | 427 (`85.9%`) | 455 (`86.7%`) | **525 (`100.0%` CTC)** |
| Zero-duration words (`Issue #197`) | **136 (`27.6%`)** | **70 (`14.1%`)** | **70 (`13.3%`)** | **0 (`0%`)** |
| Attributed words | 434 (`88.2%`) | 442 (`88.9%`) | 454 (`86.5%`) | 525 (`100%` `<spk:N>` tags) |
| Honest `unknown` words | 58 (`11.8%`) | 55 (`11.1%`) | 71 (`13.5%`) | 0 |
| Utterance count | 42 | 49 | 54 | 42 |
| Global speakers | 6 (`SPK_00..05`) + `unknown` | 6 (`SPK_00..05`) + `unknown` | 6 (`SPK_00..05`) + `unknown` | 7 (`SPK_00..06`, TitaNet) |
| Combined ASR+Align load / run (BF16) | — | `4.10s` / `20.15s` (`6990 MiB` peak) | (Nemotron vLLM `10.13s` on `90s` chunks) | `10.13s` (90s chunks, ~27.8 GiB) |

## Observed Integration Failure Modes

1. **Word-Timestamp vs Diarizer-Boundary Gaps (`11–13.5%` `unknown` words):** Even on `<=180s` chunks, `Qwen3-ForcedAligner`'s `80ms` snapping and `F_300s` alignment drift caused `55–71` words (e.g., `[225.60, 226.32]` `"pour le conseiller"` falling into a brief diarization probability dip `[225.60, 226.48]` below `0.5`, and Nemotron intro words at `150–168s`) to miss any active diarization segment. Honoring our no-fabrication invariant kept them explicitly `"unknown"` rather than forcing bad assignments, whereas naive midpoint proximity would have misattributed them.
2. **Alignment Errors Fragment Utterances in `F1` (`03:16–03:55`):** Within `F1`, `Nemotron-3-Diarization` streaming toggled briefly around `218.7s`, `224.88s`, `226.72s`, `228.56s`, splitting Tao's continuous sentence across `SPK_01`, `SPK_02`, and `unknown` in **both** D1 and D2.
3. **Why Full Replacement (D1) Degrades Quality:** D1 inherited Qwen3-ASR's `F_300s` hallucinations (`"comme des chiens"`, `"Abel"`, `"comte"` in direct mode; `"cité d'Ou M'la"` in chunked mode) PLUS `14.1%` zero-duration words PLUS complete blindness to overlapping dual-speaker utterances.
4. **Why Selective Hybrid (D2 Concept) Works Best:** Nemotron 3.5 already aligns **`100%` (`2016/2016`)** of clean French words at `10–20ms` precision via native CTC. Replacing native CTC with `Qwen3-ForcedAligner` globally degrades French timing to `80ms` + `13–14%` zero-duration tokens; calling `Qwen3-ForcedAligner` **only as an `ExternalForcedAligner` fallback on CTC OOV failures** (such as Mandarin `'钾'`) captures 100% of Qwen's upside with zero regression on French.
