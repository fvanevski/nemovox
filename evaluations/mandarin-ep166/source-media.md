# Source Media Inspection — *A Will Eternal* EP166

## Identity & File Metadata

- **Source path:** `/workspace/pyannote/outputs/sources/e4g3i65Kyf8/✨ A Will Eternal Final Season EP166 Full Version [MULTI SUB] [e4g3i65Kyf8].webm`
- **File size:** `217,173,697` bytes (~`207.1 MiB`)
- **File SHA-256:** `2cfcffb70b4d168239afc47db3f763f2cac1d858512309bfc90085c4375ca0aa`
- **Container & duration:** `matroska,webm`, `1318.880s` (`21m 58.88s`)

## Stream Layout (`ffprobe`)

| FFprobe Stream Index | `audio_stream_index` | Codec | Details | Language Tag | Spoken Content |
| --- | --- | --- | --- | --- | --- |
| `0` | — | `vp9` | `1920x1080`, `25/1` fps | `eng` | Video + burned-in Chinese hardsubs |
| `1` | **`0`** | `opus` | `48,000 Hz`, stereo (`2` ch) | `eng` (mislabelled container tag) | **Mandarin Chinese (`zh`) audio** |

- **Embedded subtitle streams:** None (`0` tracks inside `.webm`).
- **Sidecar reference file observed on disk:** `/workspace/pyannote/outputs/sources/e4g3i65Kyf8/✨ A Will Eternal Final Season EP166 Full Version [MULTI SUB] [e4g3i65Kyf8].en.srt` (`31 KB`, `489` cues spanning `34.86s` to `1203.76s`). Never fed into ASR/refinement/translation; inspected solely post-hoc as optional comparison context.

## Burned-in Subtitles & Audio Structure Checks

- **Preflight visual inspection** ([preflight_180s.png](file:///workspace/nvidia-transcribe/outputs/subtitle_runs/zh_full_episode_ep166/review/preflight_180s.png), [preflight_450s.png](file:///workspace/nvidia-transcribe/outputs/subtitle_runs/zh_full_episode_ep166/review/preflight_450s.png), [preflight_800s.png](file:///workspace/nvidia-transcribe/outputs/subtitle_runs/zh_full_episode_ep166/review/preflight_800s.png)):
  - The source video carries **burned-in simplified Chinese hardsubs** at bottom-center across dialogue scenes (as well as `"前情提要"` recap headers around `180s–275s`).
- **Audio continuity (`ffmpeg silencedetect=-35dB:d=2.0` & `volumedetect`)**:
  - Speech/BGM active regions: `178s–184s` (`mean=-16.6 dB`, `max=-2.0 dB`), dialogue continues up to **`1203.6s`**, promo preview (`《斩神2》`) plays `~1204s–1222.28s`.
  - **Trailing silence tail:** Continuous `-91.0 dB` digital silence from **`1222.276s` to `1318.840s`** (**`96.564s`**).
