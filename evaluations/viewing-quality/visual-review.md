# Visual Subtitle Verification (FFmpeg ASS Render Check)

Rendered frames were extracted directly against source video + regenerated `.en.ass` using FFmpeg (`libass`) across **14 timestamps** (10 Mandarin EP166 frames using `ass_layout="raised_main"`, 4 French frames using `ass_layout="standard"`).

## 1. Visual Adjustment from First Check (`MarginV: 135 -> 150` px)
- **Initial pass (`raised_main_margin_v=135` px):** Inspection of the 1080p Mandarin frames showed bottom Chinese hardsubs reaching `~115–130` px high, allowing descenders on two-line `Main` cues (`y/g/p`) to lightly touch the top strokes of tall white Hanzi glyphs (`zh_02_181s`, `zh_05_411s`, `zh_08_1045s`, `zh_09_1201s`).
- **Adjusted default (`raised_main_margin_v=150` px):** Raising `raised_main_margin_v` by `15` px in [subtitles.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/subtitles.py) and [cli.py](file:///workspace/nvidia-transcribe/src/nvidia_transcribe/cli.py) and re-rendering confirmed clean vertical breathing room between the bottom line of English `Main` and the burned-in Chinese hardsubs across all inspected frames, with zero collisions with `Top`/`TopAlt`.

## 2. Frame-by-Frame Visual Inventory

### Mandarin EP166 (`ass_layout="raised_main"`, `MarginV=150`, `ass_max_lanes=3`)

| Frame PNG | Time (s) | Lanes Active | Observation |
| :--- | :--- | :--- | :--- |
| `zh_01_102s_opening_hardsub_102s.png` | `102.0` | `Main` | Cleanly raised above Chinese opening hardsub; condensed 2-line cue crisp |
| `zh_02_181s_recap_hardsub_181s.png` | `181.2` | `Main` | Clear air gap above `从没人看好的质子`; `utt-0012` condensed cleanly after `max_tokens=4096` fix |
| `zh_03_270s_echo_fix_270s.png` | `270.0` | `Main` | Echo-free `utt-0019/0020` continuation (`269.64..271.77s`) clearly legible |
| `zh_04_368s_two_spk_368s.png` | `368.0` | `Main`, `Top` | Proper top/raised-bottom speaker separation without overlap |
| `zh_05_411s_lore_two_spk_411s.png` | `411.0` | `Main` | Clean air gap above `与守陵人的使命不同` |
| `zh_06_729s_emotional_729s.png` | `729.0` | `Top`, `Main` | Rapid confrontation split across `Top` + raised `Main` |
| `zh_07_942s_rapid_942s.png` | `942.0` | `Main` | Single-line `"How come it's you?"` centered crisply above `怎么会是你` |
| `zh_08_1045s_confrontation_1045s.png` | `1045.0` | `Main` | Two-line confrontation clearly separated from `难道是我凭空得来的吗` |
| `zh_09_1201s_final_1201s.png` | `1201.0` | `Main` | Clean air gap above `去魁皇城` |
| `zh_10_topalt_3spk_563s.png` | `563.28` | `Main`, `Top`, `TopAlt` | Formerly omitted 3rd-speaker cue (`#114` baseline) preserved in `TopAlt` below `Top` |

### French Baseline Episode (`ass_layout="standard"`, `MarginV=42`, `ass_max_lanes=3`)

| Frame PNG | Time (s) | Lanes Active | Observation |
| :--- | :--- | :--- | :--- |
| `fr_01_48s_narrator_48s.png` | `47.5` | `Main` | Standard bottom-aligned narrator caption, crisp outline & 2-line wrap |
| `fr_02_175s_2spk_175s.png` | `175.2` | `Main`, `Top` | Bottom `Main` + top `"Oh no!"` separation unobstructed |
| `fr_03_284s_dup_fix_284s.png` | `283.58` | `Main` | Duplicate concurrent cue resolved; single `Main` line `"The interior of the continent is still unexplored."` |
| `fr_04_306s_3spk_topalt_306s.png` | `306.0` | `Main`, `Top`, `TopAlt` | Formerly omitted cue (`#76` baseline) displayed cleanly via `TopAlt` (`"Zia, you're amazing!"`) |
