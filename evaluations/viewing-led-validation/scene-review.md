# Scene-by-Scene Audiovisual Review

All **22 bounded preview clips** (**8** French excerpts spanning `~4.0` min, **14** Mandarin excerpts spanning `~7.7` min) were rendered at `1280x720` with original audio + burned-in `.en.ass` subtitles and directly reviewed in multimodal playback (`review_modality: "audiovisual"`). Machine-readable judgments live in [review-scenes.json](file:///workspace/nvidia-transcribe/evaluations/viewing-led-validation/review-scenes.json).

## 1. French Episode (*Mysterious Cities of Gold* S04E01)

| Scene | Time | Modality | Ratings (C/F/T/Co/P) | Key Finding & Action Taken |
| :--- | :--- | :--- | :--- | :--- |
| `FR-01` | `00:40–01:10` | `audiovisual` | All `acceptable` | Removed two spurious `0.215s` flashes (`"The"`, `"goats"`) overlapping opening French narrator |
| `FR-02a` | `02:50–03:18` | `audiovisual` | All `acceptable` | Clear villain dialogue & `Top` reaction; smooth timing |
| `FR-02b` | `03:16–03:56` | `audiovisual` | All `acceptable` | Fixed ASR typo `"a nat'cal"` -> `"a Naacal"` matching spoken audio |
| `FR-03` | `04:35–04:57` | `audiovisual` | All `acceptable` | Duplicate `282.69s` cue cleanly gone; minor background `TopAlt` tail remains readable |
| `FR-04` | `05:00–05:22` | `audiovisual` | All `acceptable` | Clean 3-speaker `Main` + `Top` + `TopAlt` separation |
| `FR-05` | `08:15–08:37` | `audiovisual` | All `acceptable` | Fixed chopped `"Slowly, everything will"` + 7s-late `"fine."` flash + nonexistent `"ogre."` hallucination |
| `FR-06` | `13:20–13:55` | `audiovisual` | `T=issue`, others `acceptable` | Fixed `"It's going to rain!"` -> `"It's raining rats!"` matching visual rat gag; rapid `0.45s` sub-chunk argument remains fast |
| `FR-07` | `21:25–22:05` | `audiovisual` | All `acceptable` | Clean documentary narration through episode end |

## 2. Mandarin Episode (*A Will Eternal* EP166)

| Scene | Time | Modality | Ratings (C/F/T/Co/P) | Key Finding & Action Taken |
| :--- | :--- | :--- | :--- | :--- |
| `ZH-01` | `01:35–01:55` | `audiovisual` | All `acceptable` | Clear `150` px `raised_main` above golden Chinese OP karaoke |
| `ZH-02` | `02:50–03:20` | `audiovisual` | All `acceptable` | Recovered Bai Xiaochun motto (`utt-0006`) filling a 7.8s blackout (`173.4..177.8s`) |
| `ZH-03` | `04:20–04:45` | `audiovisual` | `T=issue`, others `acceptable` | Stripped 8s-early `"hidden hand"` duplicate bleed (`utt-0019`); fixed `"tragic end"` -> `"unknown chapter"` (`utt-0024`) |
| `ZH-03b` | `05:18–06:04` | `audiovisual` | All `acceptable` | Recovered missing royal court answers (`utt-0023, 0026, 0029`); fixed `"Xu Zheng"` garble |
| `ZH-04` | `06:05–06:55` | `audiovisual` | All `acceptable` | Clean 2-speaker Tomb Keeper lore & bright portal legibility |
| `ZH-05` | `09:12–09:36` | `audiovisual` | `F=issue`, others `acceptable` | 3-speaker layout legible; minor background crowd chatter fragments in `TopAlt` |
| `ZH-06` | `10:28–10:58` | `audiovisual` | All `acceptable` | Recovered 13.5s blocking coronation prologue gap (`utt-0102`, `utt-0103`) |
| `ZH-06b` | `11:00–11:32` | `audiovisual` | All `acceptable` | Recovered crowd shout (`utt-0107`) + Tomb Keeper `682s` on-camera line (`utt-0112`) |
| `ZH-07a` | `14:50–15:20` | `audiovisual` | All `acceptable` | Suppressed stray `Top` `"Far, no one..."` (`utt-0182`); fixed `utt-0186` & `utt-0189` |
| `ZH-07` | `15:25–16:05` | `audiovisual` | All `acceptable` | Recovered 8s Bai Xiaochun reveal entrance (`utt-0201..0203`) |
| `ZH-08` | `16:20–16:55` | `audiovisual` | All `acceptable` | Recovered 10s blocking panic monologue (`utt-0223..0226`) + `"Why?"` (`utt-0234`) cleanly |
| `ZH-09a` | `17:04–17:42` | `audiovisual` | All `acceptable` | Recovered Song Que shout (`utt-0244`) & Bai Xiaochun rebuke (`utt-0245`) |
| `ZH-09` | `17:40–18:15` | `audiovisual` | All `acceptable` | Recovered `utt-0246`, relocated 7s-early `utt-0243` to `1071.9..1078.8s`, fixed `utt-0251` |
| `ZH-10` | `19:45–20:12` | `audiovisual` | All `acceptable` | Fixed `"Gazing toward you from afar, Master..."` (`utt-0280`); removed stray `"Thank you."` split |
