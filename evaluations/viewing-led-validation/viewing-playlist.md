# Human Review Playlist (Optional Spot-Check Priorities)

Although all **22** representative playback excerpts (`22` MP4 clips totaling ~12 minutes across both episodes) were directly inspected audiovisually via multimodal video+audio inspection, the 5 focused excerpts below are the highest-value clips for optional human playback confirmation:

| Priority | Clip File | Episode Time | What to Confirm | FFmpeg / Player Verification |
| :--- | :--- | :--- | :--- | :--- |
| **1 (High)** | `outputs/subtitle_runs/viewing_validation/previews/zh_08_1620_1655_chunk10_11_monologue.mp4` | `16:20–16:55` | Recovered 10s Zhou Zimo panic monologue (`utt-0223..0226`) timing, comedic tone, and `150` px clearance above Hanzi | Check no residual duplicate and comfortable reading pace across `16:25–16:35` |
| **2 (High)** | `outputs/subtitle_runs/viewing_validation/previews/zh_09_1740_1815_chunk11_rebuke.mp4` | `17:40–18:15` | Recovered `utt-0246` + relocated `utt-0243` + `"I, Bai Xiaochun, will wait"` (`utt-0251`) during confrontation climax | Check lip/hardsub sync across `17:46–17:59` |
| **3 (Medium)** | `outputs/subtitle_runs/viewing_validation/previews/zh_06_1028_1058_chunk7_missing.mp4` | `10:28–10:58` | Recovered 13.5s coronation prologue (`utt-0102`, `utt-0103`) | Check timing cleanly hands off to `"The heavens align..."` at `10:46.7` |
| **4 (Medium)** | `outputs/subtitle_runs/viewing_validation/previews/fr_06_1320_1355_fast_cues.mp4` | `13:20–13:55` | Corrected `"It's raining rats!"` gag (`utt-0117`) and comfort of rapid `<0.5s` overlapping banter at `13:31–13:50` | Decide if fast overlapping argument around `813s` (`max 148.89 CPS`) needs manual condensing or is fine in comedic context |
| **5 (Low)** | `outputs/subtitle_runs/viewing_validation/previews/zh_05_0912_0936_3spk_topalt.mp4` | `09:12–09:36` | Three-speaker (`Main` + `Top` + `TopAlt`) background crowd fragments during Chen Manyao's spear training | Confirm whether `TopAlt` background chatter is acceptable or better omitted in a future pass |

## Exact FFmpeg Preview Reproduction Command
Example (French, standard ASS):
```bash
ffmpeg -y -ss 800.0 -t 35.0 -i "/opt/mcog/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].webm" \
  -vf "setpts=PTS+800.0/TB,ass=filename=<ASS_COPY>,scale=1280:720,setpts=PTS-STARTPTS" \
  -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -c:a aac -b:a 128k fr_06_1320_1355_fast_cues.mp4
```
*(Copy `.ass` to a bracket/apostrophe-free temporary filename so FFmpeg's libass filtergraph parses cleanly without path-escaping collisions.)*
