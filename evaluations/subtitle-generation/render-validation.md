# Visual Render & Qwen Fallback Validation

## 1. Visual ASS Overlay Verification (FFmpeg `ass` filter on `S04E01 .webm`)

Rendered using `--render-screenshots` in [run_live_subtitles.py](file:///workspace/nvidia-transcribe/experiments/subtitle/run_live_subtitles.py) with `outputs/subtitle_runs/fr_full_episode/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].en.ass` (`1920x1080`, `DejaVu Sans`, `font_size=52`). All 6 PNG frames in `outputs/subtitle_runs/visual_screenshots/` were directly visually inspected:

| Frame | Timestamp | Active ASS Style(s) | Rendered Subtitle Text | Visual Inspection Result |
| --- | --- | --- | --- | --- |
| `shot_01_intro_theme_10s.png` | `10.00s` | `Main` (`Layer 0`) | `"Find your path, it's your life, / it's your destiny."` | Clean wrap & contrast; overlaps burned-in French TV credit (`En association avec MoviePlus BV`) inherent to the opening credit video region. |
| `shot_02_narrator_48s.png` | `47.50s` | `Main` (`Layer 0`) | `"and Tao have discovered three new Cities / of Gold, in China, Japan, and Arabia."` | High legibility even over bright golden scene; crisp dark outline and shadow; balanced 2-line split. |
| `shot_03_child_voices_f1_210s.png` | `210.00s` | `Main` (`Layer 0`) | `"But it's not just about being able / to open them, you still have to find them."` | Clean phrase break (`being able / to open`), bottom safe margin respected, zero character obstruction. |
| `shot_04_rapid_turns_f2_268s.png` | `268.00s` | `Main` (`Layer 0`) | `"I'm sure Isabella made it out. / Gaspard took her before the city sank."` | Clean sentence-boundary break; crisp readability during fast exchange. |
| `shot_05_simultaneous_overlap_1_175s.png` | `175.20s` | `Main` (`0`) + **`Top` (`1`)** | Main: `"Devilishly effective, isn't it? Sure, it's / amazing! Right, let's move to phase two."`<br>Top: `"Oh no!"` | Simultaneous speakers cleanly separated at top-center and bottom-center with zero collision and zero clipping near the broadcaster logo (`Okoo`). |
| `shot_06_simultaneous_overlap_2_284s.png` | `283.58s` | `Main` (`0`) + **`Top` (`1`)** | Top & Main: `"The interior of the / continent is still unexplored."` | Proves 2-lane top/bottom rendering renders crisply without wrapping errors; surfaced upstream duplicate chunk-overlap utterance in `recording.contract.json` (`utt-0042` + `utt-0045`). |

## 2. Live GPU Proof of Work Package A (Qwen Localization Fix)

Before subtitle generation, [verify_qwen_fix_gpu.py](file:///workspace/nvidia-transcribe/experiments/subtitle/verify_qwen_fix_gpu.py) re-verified Mandarin OOV alignment live against `vLLM` + `Qwen3-ForcedAligner-0.6B` (`outputs/subtitle_runs/zh_qwen_fix_verify/qwen_fix_summary.json`):

- **`zh_multichunk_synthetic_qwen` (`wall_time_s=24.182s`, `7/7` full timing)**:
  - `utt-0000` (`[1.83s..7.281s]`): `"这并不是告别，这是一个篇章的结束，也是新篇章的开始。"` — zero contamination from `utt-0001`.
  - `utt-0001` (`[12.0s..17.44s]`): `"钙、钾等元素属于金属，银和金等元素当然也是金属。"` — **`21/21`** `external_qwen_recovered` characters cleanly bounded inside `[12.0s, 17.44s]` (`'钾'` localized at `[12.24s, 12.48s]`, never migrated to `utt-0000`).
  - `utt-0002` (`[22.45s..27.041s]`): recovered OOV tokens inside `[22.45s, 27.041s]` without cross-utterance drift.
- **`zh_fleurs_01_2006` (`wall_time_s=11.86s`, `1/1` full timing)**: all **`21/21`** characters including `'钾'` (`1.36s..1.60s`) recovered inside `[1.12s, 6.56s]`.
