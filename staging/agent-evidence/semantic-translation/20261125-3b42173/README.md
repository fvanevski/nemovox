# Sanitized Qualification Evidence — Integrated Semantic Adjudication & Contextual Translation (`20261125-3b42173`)

- **Tested Candidate Commit:** `3b4217331d717059143d4677df4abf0cc3bd6866` (tree `5a89b7ff20d59d5b76903744dbbe099c3687f7d8`)
- **Base (`origin/main`):** `4dd4599e51775ca510002783ea650ecd25fd7393` (tree `dfbc53906300996373c34631fe33029b87bbf313`)
- **Prior Evidence Carrier:** `da9f99b367ba109dba5ce0f2432894c44567c2ab`
- **Result:** `PASS` (no source changes made; owner checkout and deployed config untouched)

## Summary of Qualification Phases

- **Phase A (CPU & Studio):** `compileall` PASS, `git diff --check` PASS, focused regression `53 passed`, verbose scene/review `39 passed`, full CPU `535 passed` (1 `anyio` deprecation warning), Studio `50 passed`. Both `.venv` and `.venv-studio` verified bound to the detached candidate worktree.
- **Phase B (Integrated Real-Video CLI Job):** 110s authenticated French video excerpt (compression region `260.0s–370.0s`, `h264` + `pcm_s16le`), exited `0` in `428.686s` (`26` utterances, `36` rendered SRT/ASS cues).
  - Recording context: 1 native NInfer Messages call covering `26/26` rows, `4` validated advisory nominations (`0` rejected).
  - Shared scheduler: `boundary_triage.mode=shared_structural_scheduler` (`0` redundant boundary text votes), `6` windows observed, `4` deferred by budget.
  - Acoustic acquisition: `20` native Nemotron requests + `4` independent CUDA Whisper challenger targets (`8` observations, `faster-whisper==1.2.1`, `ctranslate2==4.9.1`, `cuda`/`int8_float16`).
  - Acoustic mutations: `2` corroborated acoustic mutations (`replaced_corroborated_source_hypothesis`, `extended_corroborated_turn`); `0` scene-driven mutations.
  - Scene adjudication & real-video vision: `6` decisions assessed, `2` deferred, `0` applied, `0` errors. Changed-reference guard explicitly flagged `1` nomination (`source_reference_changed_or_retired`). Single-model acoustic alternatives did not authorize source mutations. Real video vision inspected `6` PNG frames (`requested_seek_s` `56.72s–75.91s`), read the broadcast logo only (no dialogue subtitles), and correctly excluded non-matching visual text from accepted semantic context (`0` rows exported).
  - Dedicated Riva primary MT (`cuda:0`, `peak_vram_mib=8024.05`, `900` native input tokens, `30.084s`) followed by 12-row English review (`52.234s`, `6` API calls, `20` changed, `2` comparative rejections, `0` polarity/dependent-passage rejections).
- **Phase C (Matched 6-Row vs 12-Row English Review):** Executed on exact unchanged Phase B `translated.contract.json` (`1debbd2f3a81e65502abd1f88151997d02c1c4c38e17a55a1a7e0b598a4fa067`). Candidate 12-row reduced API calls `10 -> 6` (`-40.0%`), wall time `77.177s -> 52.234s` (`-32.3%`), input tokens `21,528 -> 16,583` (`-23.0%`), output tokens `19,485 -> 12,785` (`-34.4%`), and reasoning tokens `14,740 -> 8,496` (`-42.4%`). Baseline 6-row triggered `2` dependent-passage guard rejections on a 3-row split continuation across a batch boundary; 12-row kept the passage in one batch and resolved it coherently with zero polarity or referent regressions.
- **Phase D (Controlled FR/ZH MT & Known-Pixel Vision):** Exited `0` in `40s`. French (`4` rows, `3` Riva example rows) and Mandarin (`3` rows, `2` Riva example rows) controls preserved negation, restriction (`ne...que`), idioms, names, and genuine repetition with native Riva example IDs and input token counts. Known-pixel vision control sent `2` PNG frames via `ninfer_anthropic_messages`, accurately read the rendered literal, accepted matching literal context (`entity_identity_accepted=false`, `human_verified=false`), rejected mismatched spelling (`null`), and left source unmutated.
- **GPU Lifecycle & Restoration:** All managed GPU scopes (`nemovox-asr.service`, `nemovox-text-review.service`, CUDA Whisper worker, CUDA Riva worker) acquired and released sequentially. Post-run VRAM restored to `3,046 MiB` used (`29,081 MiB` free), managed units `inactive/dead`, unrelated desktop/embedding GPU processes preserved.

## Private Review Archive (Host-Retained)

- **Path:** `/workspace/nvidia-transcribe/outputs/handbacks/semantic-translation-20261125-3b42173/semantic-translation-review-3b42173.tar.gz`
- **Bytes:** `3712002`
- **SHA-256:** `6aeb764bf7e8368a71dc4fc1a8eeb4473831951c2c3723e8b990d7830863181c`

*(Private dialogue, detailed qualitative passage reviews, raw model answers, and extracted frame PNGs are retained exclusively in the host private review archive and omitted from this public commit.)*
