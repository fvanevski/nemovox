# Comparative Viewing-Quality Baseline: French S04E01 vs Mandarin EP166

Measured identically across `outputs/subtitle_runs/fr_full_episode/` and `outputs/subtitle_runs/zh_full_episode_ep166/` by [analyze_ep166_baseline.py](file:///workspace/nvidia-transcribe/experiments/subtitle/analyze_ep166_baseline.py) (full machine-readable metrics in [timing-analysis.json](file:///workspace/nvidia-transcribe/evaluations/mandarin-ep166/timing-analysis.json)):

| Metric | French Baseline (`MCOG S04E01`) | Mandarin EP166 (`A Will Eternal`) |
| --- | --- | --- |
| Source duration (`s`) | `1379.88` (`23m 00s`) | `1318.88` (`21m 59s`) |
| Input utterances (ASR) | `221` | `285` |
| Untimed / unrenderable utterances | `0` (`0.0%`) | **`41` (`14.4%`)** *(Chunk 0, 11, Chunk 14 tail)* |
| Partial-timing utterances | `150` | `69` |
| Long raw utterances (`> 12s` span) | `4` (max `21.2s`) | **`10`** (max **`45.4s`**, unpunctuated `116`-char `utt-0015` = `31.1s`) |
| Total SRT cues | `286` | `307` |
| Total ASS `Dialogue` events | `281` (`260` Main, `21` Top) | `304` (`254` Main, `50` Top) |
| Invalid start/end ordering cues | **`1`** (`cue 113: 504.21 > 503.88, -0.33s`) | **`0`** |
| Cues `< 1.0s` duration | `25` (`8.7%`) | **`35` (`11.4%`)** |
| Reading speed `p50 / p90 / p95 / max` (`cps`, valid durations) | `14.31 / 27.68 / 40.24 / 148.89` | **`16.67 / 30.75 / 39.77 / 153.45`** |
| Cues `> 20 / > 25 / > 30` `cps` | `51 (17.8%) / 34 (11.9%) / 23 (8.0%)` | **`107 (34.9%) / 69 (22.5%) / 37 (12.1%)`** |
| Line-wrap violations (`> 2 lines` or `> 42 chars`) | `0 / 0` (`42x2` held cleanly) | `0 / 0` (`42x2` held cleanly) |
| Concurrent duplicate/similar cues (`sim >= 0.80`) | `1` (`282s` ASR overlap duplicate) | `0` |
| Genuine 2-speaker overlaps (`Main` + `Top`) | `21` | **`43`** |
| ASS omissions (`>= 3`-way overflow) | `5` (`1.7%`) | `3` (`1.0%`) |
| Translation omissions / source leaks | `0` (`0` empty unmerged) | **`1`** (`utt-0241` single `'的'` leaked cleanly) |
| Representative visual conflicts | Minor opening credit overlap (`10s`), duplicate overlap (`284s`) | **Direct bottom-center collision with burned-in Chinese hardsubs**; `153 cps` bursts |

## Key Language-Contrast Takeaways for Remediation Campaign

1. **CPS inflation under Mandarin -> English expansion:** Mandarin packs dense classical/xianxia clauses into tiny acoustic windows (and unpunctuated 15–45s monologues), doubling the rate of `> 20 cps` cues (`34.9%` vs `17.8%` in French) and driving `153 cps` spikes whenever proportional text splitting hits tight partial CTC boundaries (`268.5s–269.9s`).
2. **Untimed fallback rate under native CTC (`41` vs `0`):** Without Qwen alignment, CTC OOV / silence-tail failures (`Chunk 14`) dropped `14.4%` of utterances outright; however, running Qwen subprocesses per span took `~10m20s` for just 4 chunks vs `36.21s` for native CTC across all 15. Future remediation should support **persistent/batched Qwen alignment on true OOV spans + VAD trimming before Chunk 14 silence tails**.
3. **Burned-in hardsub visual collision (Mandarin-only):** Unobtrusive white outlines suffice on French broadcast media (`no hardsubs`), whereas Chinese donghua with bottom-center hardsubs needs an opaque backdrop box (`BorderStyle=4`/`3`) or raised `MarginV`.
4. **Negative-duration clamping bug (seen in French `cue 113`):** Must guard display-cue clamping (`end = max(start + min_step, end)`) in `subtitles.py` during remediation.
5. **Single-character CTC interleave fragments (`你`, `峰`, `前`, `的`):** Mandarin overlaps produce isolated 1-char turns on alternate speakers (`Top`-lane micro-cues & 1 `fallback_empty_translation` leak).
