# Viewing-Led Subtitle Validation — Final Handback

**Repository:** `/workspace/nvidia-transcribe`  
**Starting Baseline HEAD:** `c93962bd6897727326d69b625ce761f59801e242`  
**Reference Repo (`/workspace/pyannote`):** Untouched at `2b40b1c0c17ce324178af765d8d2c79234870acd`  
**Validation Approach:** Reusable overlay infrastructure (`apply_validation_overlay`, CLI `--validation-overlay`) + CPU-only `--re-render` over verified audiovisual playback excerpts (zero Nemotron/SwiftU reruns).

---

## 1. Scenes Reviewed & Modality
- **Total clips reviewed:** **22 bounded 720p excerpts** (~`11.7` minutes total playback: `8` French scenes `~4.0` min, `14` Mandarin scenes `~7.7` min), each with original audio and burned-in `.en.ass` subtitles.
- **Modality:** `audiovisual` (direct multimodal video + audio inspection via `read_file` on rendered `.mp4` clips, supplemented by 1080p FFmpeg frame verification across fixed scenes). No scene claims were inferred from metadata alone.

## 2. Assessment Summary

### French Episode (*Mysterious Cities of Gold* S04E01)
- Watchable, natural, and well-synced across narration, ship/village banter, 3-speaker overlaps (`Main` + `Top` + `TopAlt`), and the closing Lalibela documentary.
- **Fixes applied (`fr_s04e01_validation.json`):** Suppressed stray overlapping ASR hallucinations (`"The"`, `"goats"`, `"ogre."`), unified split `"Easy now, everything will be fine."`, fixed `"a Naacal"`, and fixed `"It's raining rats!"`.
- **Remaining:** Fast rapid overlapping banter around `13:31–13:50` (`max 148.89 CPS`) stays tight due to rapid multi-speaker pace (`1` scene marked `timing: issue` for optional operator check).

### Mandarin Episode (*A Will Eternal* EP166)
- All `raised_main` (`150` px) captions clear burned Chinese hardsubs cleanly across single-line, two-line, 2-speaker, and 3-speaker shots.
- **Blocking omissions fixed:** Recovered **19 missing spoken dialogue utterances** (`+20` cues, `281 -> 301`) across Chunks 1, 3, 7, 10, and 11—eliminating multi-second subtitle blackouts at `02:53–02:57`, `05:21–05:27`, `10:33–10:46`, `15:32–15:40`, `16:25–16:35` (Zhou Zimo panic monologue), and `17:46–17:58` (climax rebuke). Relocated 7s-early `utt-0243` (`1064.88s` -> `1071.9..1078.8s`), stripped `utt-0019` 8s forward spoiler echo, suppressed `9` chunk-overlap duplicates & spurious background words (`"Thank you."` at `19:55`), and fixed `"Xu Zheng"`, `"unknown chapter"`, `"victory is mine"`, `"Underworld Emperor Palace"`, `"will wait"`, and `"Gazing toward you from afar, Master..."`.
- **Remaining:** Background crowd speech fragments leak mildly into `TopAlt` at `09:16–09:34` (`ZH-05`) and `主宰大墓` renders `"great veil"` at `06:11`.

## 3. Untimed Mandarin Triage (`41 -> 22` without `source_start`)
- **`19` `recover_coarse_timing`:** Included chronologically with coarse bounds & `timing_status="coarse_recovered"`.
- **`6` `suppress_duplicate` & `2` `confirm_duplicate_unresolved`:** Chunk overlap duplicates prevented from duplicating captions.
- **`9` `leave_unresolved_short_interjection`, `3` `reject_artifact_or_overlap_fragment`, `1` `reject_theme_song_bleed`, `1` `reject_hallucination_loop`:** Preserved unresolved with `source_start=None` rather than fabricating timings.

## 4. Updated Output Paths
- **French SRT/ASS:** `outputs/subtitle_runs/viewing_validation/fr_full_episode/Les mystérieuses cités d'or Saison 4 Episode 1 [YlVGUDNbyXk].en.{srt,ass}`
- **Mandarin SRT/ASS:** `outputs/subtitle_runs/viewing_validation/zh_full_episode_ep166/A Will Eternal Final Season EP166 [e4g3i65Kyf8].en.{srt,ass}`
- **Evidence Package:** `/workspace/nvidia-transcribe/outputs/handbacks/viewing-led-validation-evidence.tar.gz`

## 5. Final Conclusion
**B. Ready with known limitations**

Both full-length English subtitle tracks are comfortable, coherent, well-positioned, and ready for regular personal viewing after closing the blocking Mandarin missing-dialogue gaps and distracting French/Mandarin playback glitches; only minor rapid overlapping exchanges (`FR-06` `13:31–13:50`) and minor background-chatter leakage (`ZH-05` `09:16–09:34`) remain in [viewing-playlist.md](file:///workspace/nvidia-transcribe/evaluations/viewing-led-validation/viewing-playlist.md).
