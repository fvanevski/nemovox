# Campaign Handback — Full Mandarin Episode EP166 Baseline

- **Repository:** `/workspace/nvidia-transcribe`
- **Starting / Expected HEAD:** `e63a68b2ac45a120c36d79967279eae17faf33e4`
- **Reference `/workspace/pyannote` HEAD:** `2b40b1c0c17ce324178af765d8d2c79234870acd` (unmodified)
- **Target Media:** `/workspace/pyannote/outputs/sources/e4g3i65Kyf8/✨ A Will Eternal Final Season EP166 Full Version [MULTI SUB] [e4g3i65Kyf8].webm` (`1318.88s`, `217,173,697` bytes, `sha256: 2cfcffb70b4d168239afc47db3f763f2cac1d858512309bfc90085c4375ca0aa`)

## Execution Summary

1. **Preflight & Audio Stream Selection ([source-media.md](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/source-media.md)):** Verified single `opus` Mandarin audio stream (`ffprobe index=1` -> `--audio-stream 0`), zero embedded subtitle tracks, burned-in simplified Chinese bottom hardsubs across dialogue scenes, a promo trailer at `1204–1222s`, and `96.56s` of trailing `-91 dB` silence (`1222.28–1318.84s`).
2. **Stage A — Nemotron 3.5 ASR + Alignment ([run-configuration.md](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/run-configuration.md)):**
   - Verified `quantization=modelopt_mixed` on `http://127.0.0.1:8190/v1`.
   - Initial `--alignment-fallback qwen` run completed all `15/15` full-chunk Nemotron ASR passes (~2 min) then spent `~10m20s` reaching `chunk_004` due to per-span `Qwen3ForcedAligner.from_pretrained()` cold subprocess restarts.
   - Gracefully interrupted Qwen and resumed (`no --fresh`) with `--alignment-fallback off`: reused all `15/15` cached chunk WAVs and `.raw.json` files with zero repeated full-chunk ASR requests, completing alignment & speaker reconciliation in **`36.21s`** (`285` utterances, `22` speakers, `175` `full`, `69` `partial`, `41` `failed`).
3. **Stages B & C — Glossary, Balanced Refinement & Natural English Translation:**
   - Created [a_will_eternal_zh.yaml](file:///workspace/nvidia-transcribe/configs/glossaries/a_will_eternal_zh.yaml).
   - Ran SwiftU (`ninfer-serve` on `http://127.0.0.1:8001/v1`) `generate-subtitles`: `94/285` utterances refined (`0` rejections), `284/285` translated (`15` merges, `0` batch errors, `1` single-char `'的'` fallback), generating complete `.en.srt` (`307` cues) and `.en.ass` (`304` cues, `50` Top-lane, `3` overflow) in `576.624s`.
4. **Stage D — Viewing-Quality Diagnostics & Visual Verification ([quality-review.md](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/quality-review.md), [french-mandarin-comparison.md](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/french-mandarin-comparison.md)):**
   - Computed all 15 diagnostics identically for FR & ZH (`timing-analysis.json` / `benchmark.json`) and inspected `10` FFmpeg ASS screenshots.

## Remediation Priority List Observed for Next Campaign

1. **Burned-in Chinese hardsub collision:** Add optional opaque backdrop box (`BorderStyle=4`/`3`) or raised `MarginV` mode for hardsubbed sources.
2. **Reading-speed spikes (`34.9% > 20 cps`, max `153.45 cps`) & `<1.0s` micro-cues (`11.4%`):** Add CPS-aware display extension into trailing silence and minimum-gap duration balancing.
3. **Mandarin single-char CTC speaker-interleave fragments (`'你'`, `'峰'`, `'前'`, `'的'`):** Coalesce/re-attribute 1-character interleave fragments & block raw-CJK fallback leaks.
4. **Chunk 14 silence-tail hallucination (`34` untimed utterances) & OP/recap coverage:** Pre-trim or VAD-gate silence tails before chunking and support batched/persistent Qwen aligner worker instead of cold per-span subprocesses.
5. **FR baseline negative-duration clamp bug (`cue 113`, `-0.33s`):** Enforce `end > start` invariant after sequential collision clamping.

## Evidence Reports (`evaluations/mandarin-ep166/`)

- [source-media.md](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/source-media.md)
- [run-configuration.md](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/run-configuration.md)
- [quality-review.md](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/quality-review.md)
- [french-mandarin-comparison.md](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/french-mandarin-comparison.md)
- [timing-analysis.json](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/timing-analysis.json)
- [benchmark.json](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/benchmark.json)
- [artifact-manifest.json](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/artifact-manifest.json)
