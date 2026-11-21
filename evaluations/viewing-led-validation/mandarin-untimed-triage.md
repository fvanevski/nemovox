# Mandarin EP166 Untimed Dialogue Triage

## 1. Summary Breakdown of All 41 Untimed Records
Cross-referencing all 41 untimed items in [ep166_untimed_audit.json](file:///workspace/nvidia-transcribe/evaluations/viewing-quality/ep166_untimed_audit.json) against chunk raw ASR transcripts, pre-validation playback clips, audio activity, and read-only EP166 reference boundaries resolved every item in [untimed-decisions.json](file:///workspace/nvidia-transcribe/evaluations/viewing-led-validation/untimed-decisions.json) (`0` remaining `needs_review`):

| Final Decision | Count | Included? | Rationale |
| :--- | ---: | :--- | :--- |
| `recover_coarse_timing` | **19** | Yes (`+19` cues) | Verified spoken dialogue (`18` from 22 `plausible_independent` + `1` `'天哪！'` panic opener `utt-0223` from `uncertain`) placed cleanly into silent/dialogue gaps |
| `suppress_duplicate` | **6** | No | Chunk overlap duplicates (`utt-0007/0008` of `utt-0012`, `utt-0205..0208` of `utt-0223..0226`) explicitly suppressed |
| `leave_unresolved_short_interjection` | **9** | No | Ultra-short interjections/fragments (`嗯。`, `也对。`, `是神。`, `亏待。`, `哎呀。`, `啊，`) left unfabricated |
| `reject_artifact_or_overlap_fragment` | **3** | No | Non-independent/hallucinated lines verified absent or duplicated (`utt-0032`, `utt-0040`, `utt-0180`) |
| `confirm_duplicate_unresolved` | **2** | No | Confirmed chunk-overlap echoes of timed speech (`utt-0035` -> `utt-0038`, `utt-0247` -> `utt-0249`) |
| `reject_theme_song_bleed` | **1** | No | Chunk 0 `utt-0000` inside `0..99s` opening theme/credits with no spoken dialogue |
| `reject_hallucination_loop` | **1** | No | Chunk 14 `utt-0284` ending repetition loop |
| **Total** | **41** | **19 recovered** | Zero synthetic timestamps fabricated on unresolved lines |

## 2. Major Recovered Dialogue Windows
- **Chunk 1 (`173.4..177.8s`):** `utt-0006` (*"My motto as Bai Xiaochun: whatever happens, it works in my favor."*) eliminates a 7.8s recap gap.
- **Chunk 3 (`305.3..339.6s`):** `utt-0023` (*proclamation*), `utt-0026` (*"What difference does the answer make to me or to you?"*), `utt-0029` (*"No matter who the successor is, you need only obey the decree."*).
- **Chunk 7 (`633.1..685.1s`):** `utt-0102` + `utt-0103` restore a 13.5s blocking coronation prologue gap; `utt-0107` (*"Bai Hao! It really is Bai Hao!"*) and `utt-0112` (*"All karmic bonds are ordained by fate."*) restore crowd and Tomb Keeper reactions.
- **Chunk 10/11 Overlap (`932.0..1071.5s`):** `utt-0201..0203` restore the Bai Xiaochun door reveal; `utt-0223..0226` restore Zhou Zimo's 10s inner monologue (`985.2..994.8s`); `utt-0234, 0244, 0245, 0246` plus relocated `utt-0243` (`1071.9..1078.8s`) restore the entire emotional confrontation climax.
