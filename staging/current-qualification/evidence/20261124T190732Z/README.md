# Issue 14 final completion qualification — French & Mandarin end-to-end complete

**Final completion qualification passed.** Tested application `ad768220a3a2240889461f0d10994a9a532c0375`, tree `f6a18b2b070fbf76deedc7a719ec934ad6bcec20` in the isolated detached worktree.

## CPU qualification (PASS)

- Re-ran the two previously failing config-order tests (`tests/test_translation_review.py::test_programmatic_postedit_rejects_legacy_primary_translator` and `tests/test_resource_integration.py::test_programmatic_riva_condensation_requires_explicit_text_resource`): **2 passed** (`exit 0`).
- Affected 9-file suite (including `test_resource_integration.py`): **131 passed** (`exit 0`).
- Full non-GPU suite in CLI `.venv`: **429 passed** (`exit 0`).
- Full non-GPU suite in Studio `.venv-studio`: **429 passed** (`exit 0`).
- `git diff --check origin/main...HEAD`, `git diff --check 94c9b8205d996d3be49591a6afc6cda4f47dd203..HEAD`, `python -m compileall -q src tests`, and CLI/Studio module imports all exited `0`.

## French episode downstream completion (PASS)

Continued from the preserved `canonical-source.contract.json` (`--recording-contract`, `--source-quality off`, omitting positional media, retaining prior output root, `riva`, `balanced` review, `refinement off`, `natural` style, `srt,ass`, no `--fresh`/`--fresh-translation`):

- **Preserved inputs unchanged:** `canonical-source.contract.json` and `source-quality.json` matched prior digests byte-for-byte; ASR (`28/28` completed primary checkpoints reused + `chunk_028` bounded subdivision into `2` leaves -> `36` contiguous effective chunks, `2,309` raw CTC words) was not re-run.
- **Exact cache reuse:** `183/183` primary Riva `fr-en` translation cache hits (`0.001 s`, `0` Riva worker calls) and `177/183` contextual review cache hits (`0` diffs across the prior `177` utterances).
- **Compact comparative recovery exercised cleanly:**
  - `wire[0]`: Source-first review on `batch[72:78]` completed (`end_turn`, `2,313` output tokens, `1,908` thinking tokens, `6.61 s`), proposing a change only on `utt-0077`.
  - `wire[1]`: Comparative review (`msg_2d478153ea5f5ec5`, `max_tokens: 8192`) hit the `\n ` runaway whitespace loop (`stop_reason: "max_tokens"`, `696` thinking tokens, `18.21 s`). `retry_suppressed_reason: "runaway_output_whitespace"` suppressed the futile expanded retry.
  - `wire[2]`: Compact ID-bound adjudication (`msg_c40c18d466984775`, `max_tokens: 8192`) completed cleanly (`end_turn`, `525` output tokens, `461` thinking tokens, `2.55 s`) with `verdicts: ["utt-0077=original"]`, preserving valid primary MT (`review_rejected_comparative_preserved_machine_translation`).
  - `translation_review`: `status: "completed"`, `reviewed: 183`, `changed: 97`, `comparative_rejections: 46`, `completion_retries: 0`, `deferred_comparison_ids: []`, `errors: []`.
- **Rendered outputs & offline rerender:** Rendered `251` SRT cues and `251` ASS Dialogue events (`Main: 238`, `Top: 13`, `overflow: 0`, `outside: 0`, `max_end: 1376.93 s <= 1379.901 s`). Offline `--re-render` verified byte-identical `translated.contract.json`, `.srt`, and `.ass` for both French and Mandarin (`exit 0`, zero owned-service journal activity).

## Same-denominator French before/after vs baseline (`980af31f`)

| Metric | Baseline (`980af31f`) | Candidate (`ad76822`) |
| --- | --- | --- |
| Effective ASR chunks (`stop`) | `16` (`16`) | `36` (`36`) |
| Raw CTC words | `2,415` | `2,309` |
| Reconstructed utterances | `234` | `183` |
| Micro-utterances (`<=0.60 s` or `<=2` words) | `90` | `42` |
| `0.19–0.24 s` fragments | `49` | `13` |
| Boundary deduplicated / recovered units | `53` / `4` | `200` / `5` |
| ASS events / SRT cues | `308` / `308` | `251` / `251` |
| ASS > 18 CPS / > 22 CPS / max CPS | `63` / `30` / `82.35` | `52` / `7` / `29.79` |
| Metadata `high_reading_speed_flags` | `29` | `7` |

## Independent Mandarin Studio evidence & resource restoration

- **Retained Mandarin Studio pass (`eb5bc5243677ede15e7b04c836cfabf84d260955`, carrier `ece833d875b00f754cf2d429fad004c5e70eb14e`, evidence `20261124T183928Z`):** `job_id` `00360c7e88554e57a7aa171254c9cf11`, `200.0 s` (`offset_s=150`), `5/5` ASR chunks (`317` raw CTC words), `36` utterances, Riva `36/36` (`zh-en`), balanced review `36/36` (`26` changed, `0` errors), `56` SRT/ASS cues (`outside: 0`, `max_end: 200.0 s`), offline rerender byte-identical. Not re-run.
- **Host restoration:** `ad76822` continuation started only `nemovox-text-review.service` (`36` telemetry samples, `peak_vram_mib: 30,679`); `nemovox-asr.service` and `nemovox-text-review.service` ended `inactive`, ports `8190`/`8001` have `0` listeners, `model-lifecycle.lock` is acquirable, and production `nemovox-studio.service` (`InvocationID=6cd5cf36a6774b0693d37a196c088464`, `MainPID=1121444`) remained unchanged and healthy (`/api/health` `"status":"ok"`).

[Original gate matrix](gate-matrix.md) · [Metrics](metrics.json) · [Commands](commands.md) · [Command friction](COMMAND_SHAPE_FRICTION.txt) · [Checksums](SHA256SUMS)
