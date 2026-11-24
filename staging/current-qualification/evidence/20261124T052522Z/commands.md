# Commands and native evidence

All commands ran with the existing interpreter/dependencies and exact candidate source. Shell entry used `rtk proxy`. Complete native logs and exact private paths are retained in the host-local evidence manifest. In the shapes below `$cli` and `$studio` are the existing project Python interpreters and `$run` is the protected fresh output root.

| Check | Observed result |
| --- | --- |
| `$cli -m pytest -q tests/test_source_quality.py tests/test_offline_rerender.py` | Exit 0, 30 passed |
| `$cli staging/current-qualification/evidence/20261124T043313Z/reproduce.py` | Exit 0, expected inner CLI error, 0 acquisitions and explicit displaced hypothesis |
| `$cli staging/current-qualification/evidence/20261124T045622Z/reproduce_overlap.py` | Exit 0, 3.5 s in SRT and ASS, 18 chars/s and two ASS lanes |
| `$cli -m pytest -q tests/test_source_quality.py tests/test_subtitles.py tests/test_translation_review.py tests/test_offline_rerender.py` | Exit 0, 53 passed |
| Focused handoff suite and both full `-m 'not gpu'` suites | All exit 0, 70 focused and 387 in each environment |
| Compileall, Git diff check, npm test and npm run build | All exit 0, 14 frontend tests in 6 files |
| Tracked consumer grep for removed commands, configs and modules | Exit 1, no matches |
| Independent entrypoint and checkpoint controls | Exit 0. CLI defaults and alternate translator, explicit stream/embedding forwarding, Studio defaults and forwarding, no-media import skip, invalidation of older recording signature and current resume verified. |
| Checkpoint preflight | Exit 0, local ModelOpt mixed-precision asset and CTC weight present |
| French question and normal-control CLI, Mandarin Studio driver | All exit 0 |
| Current-file rerender for French and Mandarin | Both exit 0, original contract and both output files byte-identical, no Riva start or owned-unit journal change |
| Runtime sampler and per-invocation journal readback | Exit 0 |

Bounded clips were freshly extracted from the existing original local audio stream at starts 770 s (French question), 195 s (Mandarin) and 235 s (French normal control) for 78, 72 and 50 s respectively. These offsets map clip-relative evidence onto the original. The selected stream was audio index 0. The crop streams and full originals have separate hashes.

```sh
rtk proxy ffmpeg -hide_banner -v error -nostdin -n -ss "$start" -i "$original" -t "$duration" -map 0:a:0 -vn -c:a flac "$fresh_clip"
rtk proxy env PYTHONPATH="src:.:$run/observe-child" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NEMOVOX_QUALIFICATION_CHILD_RECEIPT="$receipt" "$cli" -m nvidia_transcribe.cli generate-subtitles "$fresh_clip" --output-dir "$job_root" --source-language fr --audio-stream 0 --speaker-embeddings none --chunk-policy pause_aware --source-quality on --translator riva --translation-review balanced --refinement off --formats srt,ass --fresh
rtk proxy env PYTHONPATH="src:.:$run/observe-child" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NEMOVOX_QUALIFICATION_CHILD_RECEIPT="$receipt" "$studio" "$run/studio_driver.py" "$run" "$fresh_mandarin_clip" "$job_root"
rtk proxy env PYTHONPATH="src:.:$run/observe-child" NEMOVOX_QUALIFICATION_CHILD_RECEIPT="$receipt" "$cli" -m nvidia_transcribe.cli generate-subtitles --output-dir "$rerender_root" --formats srt,ass --re-render
```

The Studio request used fixed 38 s cores, 10 s context, Mandarin to English, Riva/balanced, audio index 0, embeddings and refinement off, both formats, and no export. The Riva observation helper hashes the actual subprocess stdin by row ID and records the worker exit. It leaves request and result untouched. CPU and live artifact identities bind to the candidate. The prior campaign's output was unused as an input.
