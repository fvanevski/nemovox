# Qualification commands and native exits

Run from the authenticated candidate worktree. The retained native transcripts and exact paths are listed by hash in the private archive. The command shapes below use local path placeholders for media and the protected run directory. `$cli` and `$studio` are the existing `.venv/bin/python` and `.venv-studio/bin/python` under `/workspace/nvidia-transcribe`.

| Verification | Native result |
| --- | --- |
| `rtk proxy env PYTHONPATH=src:. $cli -m pytest -q tests/test_source_quality.py tests/test_offline_rerender.py` | 0 · 26 passed |
| `rtk proxy env PYTHONPATH=src:. $cli staging/current-qualification/evidence/20261124T043313Z/reproduce.py` | 0 · expected inner CLI exit 1 with zero ASR acquisitions, and explicit `unresolved_possible_displaced_or_repeated_turn` |
| `rtk proxy env PYTHONPATH=src:. $cli -m compileall -q src tests` | 0 |
| Focused suite named in the handoff using `$cli -m pytest -q -m 'not gpu'` | 0 · 62 passed |
| `$cli -m pytest -q -m 'not gpu'` and `$studio -m pytest -q -m 'not gpu'`, both with candidate `PYTHONPATH=src:.` | 0 each · 376 passed in each |
| `rtk proxy npm test` and `rtk proxy npm run build` in `web` | 0 each · 14 tests, 6 files, build succeeded using the existing dependencies |
| `rtk proxy git diff --check` and candidate versus base diff check | 0 each |
| Tracked current docs, config, scripts, source, web and tests grep for removed command, config and module names | 1 · no matches |
| Independent fixture checks in both Python environments | 0 each · default Riva/balanced, alternative OpenAI Chat/off, no-media import skip, original-media and recording forwarding, format and source-quality event conversion |
| Incompatible recording checkpoint seed and matching current resume check | 0 · one stubbed ASR call after old-signature rejection, zero on matching current resume |
| `rtk proxy env PYTHONPATH=src:. $cli -m nvidia_transcribe.cli validate-checkpoint` | 0 · expected native CTC weight, ModelOpt mixed precision, no missing checkpoint files |
| French fresh `process-media` | 0 · 35 fresh chunks, completed manifest |
| French `generate-subtitles` | 0 · source quality partial, translation and review completed, both formats rendered |
| Fresh read-only rerender from copied current translated contract | 0 · byte-identical rendered files and unchanged translated contract |
| Studio submission controller | Python process exit 1 after durable job complete, due to harness error when reading artifact response. The native children completed and a fresh API and disk readback exited 0. |
| Synthetic overlap reproduction | 1 · reproduced an unreadable cue with native CPU rendering in both output formats |
| Runtime sampler and per-invocation journal collection | 0 · 747 observations, device and both managed units |

Core French command sequence. All live CLI processes used the existing interpreters and resource/lifecycle bindings. Offline Hugging Face and Transformers settings were enabled. The short-lived Python observation helper hashed Riva stdin and recorded the worker exit without changing the request or result. Studio's private driver submitted the same type of subtitle job through the public API and ran the actual CLI children with the same helper.

```sh
rtk proxy env PYTHONPATH=src:. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$cli" -m nvidia_transcribe.cli process-media "$french" --output-dir "$run/french/asr" --source-language fr --audio-stream 0 --chunk-policy pause_aware --speaker-embeddings none --alignment-fallback off --fresh
rtk proxy env PYTHONPATH="src:.:$run/observe-child" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NEMOVOX_QUALIFICATION_CHILD_RECEIPT="$run/fr-child-receipts.jsonl" "$cli" -m nvidia_transcribe.cli generate-subtitles "$french" --recording-contract "$run/french/asr/recording.contract.json" --output-dir "$run/french/output" --source-language fr --source-quality on --translator riva --translation-review balanced --refinement off --chunk-policy pause_aware --formats srt,ass --fresh
rtk proxy env PYTHONPATH="src:.:$run/observe-child" NEMOVOX_QUALIFICATION_CHILD_RECEIPT="$run/rerender-child-receipts.jsonl" "$cli" -m nvidia_transcribe.cli generate-subtitles --output-dir "$run/french/rerender" --formats srt,ass --re-render
rtk proxy env PYTHONPATH="src:.:$run/observe-child" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NEMOVOX_QUALIFICATION_CHILD_RECEIPT="$run/zh-child-receipts.jsonl" "$studio" "$run/studio_control.py" "$run"
rtk proxy env PYTHONPATH=src:. "$studio" "$run/studio_readback.py" "$run"
rtk proxy env PYTHONPATH=src:. "$cli" staging/current-qualification/evidence/20261124T045622Z/reproduce_overlap.py
```

The Studio driver submitted a new local FLAC file extracted at 195–267 s from the original Mandarin audio track. Its explicit options were fixed 38-second chunks, 10-second context, audio stream 0, `zh` to `en`, Riva, balanced review, no refinement, optional speaker embeddings off, SRT/ASS, and export disabled. Both source files exposed only one audio track, tagged `eng` in container metadata. Explicit source languages were used based on the identified media, and the fresh ASR emitted the corresponding source text.
