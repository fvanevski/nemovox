# Architecture Decision — Modular Multilingual Transcription Evaluation

## Decision: **Category B — Hybrid Architecture Preferred**

**Nemotron 3.5 Transcribe 30B-A4B NVFP4 remains the primary ASR, dual-speaker overlap, and clean CTC timing foundation, augmented selectively with `Qwen3-ForcedAligner-0.6B` for CJK CTC-OOV fallback and bounded `Nemotron-3-Diarization` as a medium-horizon acoustic diarization anchor.**

A full migration to **Modular (`Qwen3-ASR-1.7B` + `Qwen3-ForcedAligner-0.6B` + `Nemotron-3-Diarization`) is rejected** on lexical quality, overlap preservation, and forced-alignment granularity grounds. Keeping **pure single-pass/chunked Nemotron 3.5 unchanged without targeted hybrid augmentation and bug fixes is also suboptimal**, given verified CJK CTC-OOV aborts and chunk-boundary speaker instability.

## Decisive Evidence Matrix

| Dimension | Nemotron 3.5 NVFP4 (Baseline) | Qwen3-ASR-1.7B | Qwen3-ForcedAligner-0.6B | Standalone Nemotron-3-Diarization | Decision Weight |
| --- | --- | --- | --- | --- | --- |
| Clean FLEURS French WER | `0.1020` (`98` words) | **`0.0816`** | N/A | N/A | Near parity |
| Clean FLEURS Mandarin CER | **`0.1236`** (`178` chars, Arabic numerals) | `0.1461` (`0.0000` on `00/01/03`; wins `Scoturb`) | N/A | N/A | Parity (style diffs) |
| **French Animation Multi-Speaker (`F_100s`/`F_300s`/`O2`/`O3`)** | **Strong (`~0.19` full episode, real proper nouns preserved)** | Degraded (`"comme des chiens"`, `"Daisy Meadows"`, `"Abel"`, `"Shangri-La"`) | N/A | N/A | **Decisive for Nemotron 3.5** |
| **Simultaneous Overlap ASR (`S1`, `zh_2spk_overlap`, `fr_2spk_overlap`)** | **Native dual-stream `<spk:N>` recovery** | **Dropped secondary speaker (`31D` ZH, `100%` S2 B) or blended hallucination (`S1` `0.85` WER)** | Fails (`41–49%` zero-duration) | Strong frame overlap on short/med clips (`S2 ovF1=0.93`) | **Decisive for Nemotron 3.5** |
| **CJK OOV Timestamp Resilience (`'钾'`)** | Aborts chunk alignment (`5679`-char vocab gap) | N/A | **`21/21` valid timings (`'钾'`: `1.36–1.60s`)** | Independent of vocab | **Decisive for Qwen Aligner fallback** |
| **Clean Non-OOV Word Timing Precision** | **`10–20ms`, `0%` zero-duration (`2016/2016` FR)** | N/A | **`80ms` bins, `12.5%` zero-duration (`331/2644`)** | N/A | **Rejects replacing Nemotron CTC** |
| **Multi-Hour / Full-Episode Speaker Identity** | Single-pass LLM switches `481/1992` words; chunked TitaNet `44->22` (`14` provisional) | Zero speaker concept | Zero speaker concept | Excels `<=5 min` (`F_300s`); **23m `F_full` saturates/stalls at `7` slots (`t=265.5s`), collapses `F1` & loses overlap** | **Supports scene-bounded anchoring, not naive global oracle** |
| Runtime & Memory Footprint (RTX 5090 32 GB) | `~27.8 GiB` VRAM via vLLM; `31.0s` per 23m episode | `4.48 GiB` BF16; `18.2s` for `300s` | `1.75 GiB` disk; `~2.2 GiB` incremental (`6.99 GiB` combined with Qwen ASR); `<=180s` limit | **`0.42–0.58 GiB` VRAM**; `27.4s` streaming / `1.46s` offline for `23m` | Fits cleanly when staged or fallback-only |

## Next-Stage Implementation Implications (Separate Phase — Do Not Start Here)

1. **Remediate known chunked-core defects first:** Resolve the 8 identified correctness bugs in the Nemotron chunked pipeline (boundary dedup with unknown speakers, lexical substitution in alignment recovery, timed/untimed Chinese unit ordering, misleading partial-timing status, ambiguous speaker auto-merging, stale extracted-audio reuse, mixed CJK/Latin spacing, and factorial speaker matching) before adding external models.
2. **Attach `Qwen3-ForcedAligner-0.6B` strictly behind `ExternalForcedAligner`:** Invoke it on demand (subprocess in its pinned `transformers==4.57.6` env or isolated worker when GPU VRAM is freed after vLLM, or CPU/sequenced two-pass workflow) **only** when `recover_chunk_alignment` hits unaligned CTC OOV sub-windows (preserving zero-duration diagnostics explicitly rather than clamping `80ms` ties).
3. **Evaluate bounded 3–5 minute `Nemotron-3-Diarization` acoustic anchoring:** Use `Nemotron-3-Diarization` (`0.42 GiB`, `0.5s` offline / `6s` streaming per `300s` window) within ~3–5 minute scenes as an acoustic boundary/anchor oracle alongside TitaNet embeddings, **without** relying on unchunked 23-minute cache rollout (`speaker_cache` stagnation past ~7 speakers) and **without** sacrificing Nemotron 3.5's native `<spk:N>` overlap recognition.
