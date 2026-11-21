# Test Results & Live GPU Validation Summary

## 1. CPU Unit & Regression Suite (`-m "not gpu"`)

Command:

```bash
.venv/bin/pytest -v -m "not gpu"
```

Recorded verbatim in [`pytest_cpu.txt`](file:///workspace/nvidia-transcribe/evaluations/chunked-core/pytest_cpu.txt): **`30 passed, 1 warning in 0.54s`** (anyio pytest plugin deprecation warning).

| Test File | Passed | Coverage Summary |
| --- | ---: | --- |
| [`tests/test_checkpoint.py`](file:///workspace/nvidia-transcribe/tests/test_checkpoint.py) | 4 | NVFP4 checkpoint validation, BF16 rejection, missing-shard failure, local checkpoint metadata check |
| [`tests/test_client.py`](file:///workspace/nvidia-transcribe/tests/test_client.py) | 4 | Instruction non-French default, generation-loop cleanup, OpenAI payload shape, missing CTC metadata error |
| [`tests/test_metrics.py`](file:///workspace/nvidia-transcribe/tests/test_metrics.py) | 2 | French WER normalization, Mandarin Chinese CER calculation |
| [`tests/test_parser_and_contract.py`](file:///workspace/nvidia-transcribe/tests/test_parser_and_contract.py) | 5 | Speaker-tag & event extraction, French Unicode + `source_offset_s`, Mandarin char timing, null/invalid timestamp preservation, source transcript preservation |
| [`tests/test_chunking_and_resume.py`](file:///workspace/nvidia-transcribe/tests/test_chunking_and_resume.py) | 4 | Exact gap-free chunk planning, boundary dedup preserving simultaneous repetitions, boundary omission recovery, manifest resume & invalid-media handling |
| [`tests/test_recording_and_normalization.py`](file:///workspace/nvidia-transcribe/tests/test_recording_and_normalization.py) | 4 | Language resolution (`explicit`, `model_detected`, `ambiguous`), code-switching & event isolation, archived French overlap/child-voice regressions, partial crash-resume |
| [`tests/test_utterances_alignment_speakers.py`](file:///workspace/nvidia-transcribe/tests/test_utterances_alignment_speakers.py) | 7 | Mandarin interleaved overlap reconstruction, French overlap + pause splitting, CJK OOV CTC failure preservation without timestamp fabrication, partial sub-interval recovery (recovering non-OOV windows and isolating OOV windows), ABA cross-chunk speaker reconciliation + `cannot_link` collision guard, explicit `unknown` speaker retention |

## 2. Live GPU Pipeline & CLI Validation

Server: `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` via vLLM on `127.0.0.1:8190` (`quantization=modelopt_mixed`, CTC timestamp middleware enabled). Hardware: NVIDIA RTX 5090 (32607 MiB VRAM; peak during chunked runs: `27846 MiB` used; returned to idle `3445 MiB` after shutdown).

| Validation Target | Result | Key Metrics |
| --- | --- | --- |
| CLI `nvidia-transcribe process-media` (`fr_0316_0355_child_voices.wav`) | Exit `0` (`1.01s`) | `source_language="fr"`, 1 chunk, emits `recording.contract.json`, `transcript.txt`, `manifest.json` |
| French 300s ablation (`40s`/`60s`/`90s`/`120s`) | Completed | `90s` fastest (`10.1317s`, `4` chunks, `525/525` timed units, `20` dedup, `0` ambiguities) |
| French 23-min episode (`90s`, TitaNet) | Completed | `31.0437s`, `16` chunks, `44 -> 22` speakers, `213` utterances, `2016/2016` timed, `67` dedup, `4` recovered |
| French 23-min episode (`90s`, temporal-only `speaker_embeddings="none"`) | Completed | `44 -> 28` speakers (`18` unresolved vs `14` with TitaNet, `0` collisions prevented vs `3` with TitaNet) |
| Mandarin `'钾'` OOV (`zh_fleurs_01_2006.wav`) | Completed | `source_language="zh"`, verbatim text preserved, `diarization_bounded_only`, `oov_characters=["钾"]`, `start=null, end=null` |
| Mandarin 2-spk overlap (`synthetic_zh_2spk_overlap.wav`) | Completed | Reconstructed into `2` concurrent utterances (`SPK_00`, `SPK_01`) instead of 32 interleaved fragments |
| Mandarin multi-chunk synthetic (`zh_multichunk_synthetic`, `69.78s`) | Completed | `2` chunks, `170/194` timed (`83` recovered in chunk 0 via `partial_subinterval_recovery`), only `[11.68,17.84]` `'钾'` segment left `diarization_bounded_only` |
| Prompt comparisons (`zh_fleurs_05_1805`, `zh_2spk_overlap`, `fr_0316_0355`) | Completed | Preserved `speaker` baseline superiority for Mandarin; confirmed domain context vocabulary fixes proper nouns in French |

## 3. Repository Integrity Verification

- `/workspace/pyannote` verified untouched at `2b40b1c0c17ce324178af765d8d2c79234870acd`.
- `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` untouched.
