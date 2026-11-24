# Verification commands

Commands ran from the exact candidate unless marked otherwise. The host RTK hook requires the `rtk` prefix. `rtk proxy` forwarded the native command and producer exit. Output capture used shell redirection without a filtering pipeline.

| Command | Native exit and observation |
| --- | --- |
| `rtk proxy env PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m compileall -q src tests` | 0 |
| `rtk proxy env PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu' tests/test_source_quality.py tests/test_issue14_semantic_review.py tests/test_translation_review.py tests/test_chunking_and_resume.py tests/test_resource_integration.py` | 0 · 55 passed and one AnyIO pytest deprecation warning |
| `rtk proxy env PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python -m pytest -q -m 'not gpu'` | 0 · 360 passed and the same warning |
| `rtk proxy env PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv-studio/bin/python -m pytest -q -m 'not gpu'` | 0 · 360 passed |
| `rtk proxy npm test` from `web` | 0 · 14 tests in 6 files |
| `rtk proxy npm run build` from `web` | 0 · TypeScript and Vite build passed |
| `rtk proxy git diff --check` and `rtk proxy git diff --check origin/main...HEAD` | Both 0 |
| `rtk proxy git grep -n -F -e apply-reviewed-revision -e prepare-semantic-review-queue -e meaning-first-review.example.json -e evidence-fusion.example.json -e staging/issue14 -- .` at candidate | 1 · no matches |
| `rtk proxy rg` over active files for 14 deleted module stems | 0 · only an unrelated test name containing the substring `source_event_replay` in `source_event_replayed_cue`. No imports, commands, or file references matched. |
| `rtk proxy env PYTHONPATH=src:. /workspace/nvidia-transcribe/.venv/bin/python staging/current-qualification/evidence/20261124T043313Z/reproduce.py` | 1 · two reproduced invariant failures, with inference intercepted |
| `rtk proxy env PYTHONPATH=src:. <host Python> <private root>/positive_controls.py` in each of the two environments above | Both 0 on the corrected probe |
| `rtk proxy env PYTHONPATH=src:.:tests /workspace/nvidia-transcribe/.venv/bin/python <private root>/stale_checkpoint.py` | 0 · one stubbed request against incompatible manifest and zero against current matching manifest |

The positive controls exercised CLI Riva/balanced and explicit OpenAI Chat/off defaults. An imported synthetic contract with a path recorded inside it skipped source quality and kept the original bytes. The resulting current synthetic translated contract regenerated SRT and ASS byte-for-byte, including the 2.000–4.000 second cue, with tripwires on model acquisition and translation. A Studio JobRunner fixture confirmed media and recording forwarding for a current-media job, recording-only forwarding for an imported contract, source-quality event persistence from synthetic child output, and disabled export. The checked-in source-quality pipeline test also exercises canonical text reaching the translation call boundary with the prior recording preserved.

The stale checkpoint control used the pre-consolidation signature function to seed a synthetic current-format manifest in a temporary directory. Current code rejected the signature and called the stub ASR once. The next invocation with the current signature reused the chunk with no calls.

Frontend Node v26.13.1, npm 12.2.3, Vitest 4.1.12, Vite 8.3.8. The operator's installed dependency tree was copied into the isolated workspace. All 111 installed lock entries matched the candidate by version and integrity, with no required entry absent and 36 optional entries absent. Candidate and operator package lock SHA-256 matched (`fb10700313530722db534794c0a564b13235e89add7c54b69edd4b3aad29a49c`).

The Studio web README describes the text resource as English-review only. The current shared pipeline also uses it for bounded source advice on original-media Riva jobs. The current GUI workflow guide and top-level README describe both roles.
