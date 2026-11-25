# Issue #19 Review Completion Evidence (`20261125-cabed83d`)

Sanitized aggregate qualification packet for candidate `cabed83dc5685d18acccc354d1519b4e6505cf8e` (tree `99afccd2bc11e1b2036137f1d79aeb94be11535e`) against base `4dd4599e51775ca510002783ea650ecd25fd7393`.

No private dialogue text, raw model responses, private source excerpts, or credentials are included in this public packet. Full private per-row comparisons, translated contracts, SRT/ASS outputs, and raw rejected/recovery receipts are retained in the host-local private review archive (`review-completion-cabed83d.tar.gz`, `3764304` bytes, SHA-256 `6f3f0e308b227e69248e4121f9064c5e7e3e9bb543a6fe554d7e6426866a592a`).

## Qualification Summary

- **Phase A (CPU & Studio)**: `compileall` PASS, `git diff --check` PASS, focused regressions `91 passed`, full CPU `551 passed`, Studio `51 passed`, `web/` unchanged from prior qualified source `3e6f8d8244f818c304bf5d692e4db864a6ae9a1e`.
- **Retained Input Preparation**: Authenticated prior C3 archive (`49e31f742cae801672de81ea604144769d1e7e8f3114b6b4f5f83d07bcb2e1df`, `21867540` bytes); prepared `bounded` (26 rows), `mandarin` (2 rows), and `french` (221 rows) with exact Riva/review/glossary identity match and zero upstream invocations (`PRIMARY_ASR=0`, `RECORDING_CONTEXT=0`, `LOCALIZED_ASR=0`, `WHISPER=0`, `RIVA_INFERENCE=0`, `VISION=0`).
- **Phase B (Bounded French & Mandarin)**:
  - `bounded`: `status=completed`, `26/26` Riva cache hits, `6` NInfer calls (`66.129s`), `13` changed, `4` comparative rejections, `3` unresolved-source rejections, `3` dependent-passage rejections, `0` direction-grammar rejections, `35` renderable cues (`0` unrenderable, `0` overflow).
  - `mandarin`: `status=completed`, `2/2` Riva cache hits, `2` NInfer calls (`23.93s`), `1` changed, `1` comparative rejection, unsupported geographic substitution prevented on the unresolved bus-route row, `2` renderable cues (`1` top lane, `0` overflow, `0` conflicts). Independent reference manifest `overlap_manifest.json` authenticated (SHA-256 `d0d977411063ddf1cc457750c674df25cc4598a842dd93ce478ed054a07cc2cc`).
- **Phase C (Full French Episode, 221 rows)**:
  - `status=completed`, `221/221` Riva cache hits (`0.002s`), `39` NInfer Messages calls (`19` proposal, `19` standard comparison, `1` compact comparison recovery), `233258` input / `96570` output / `60367` thinking tokens (`0` missing usage), `396.323s` review wall time.
  - **Live Malformed Comparative Recovery**: Batch `[84:96]` emitted a completed malformed comparative response (`duplicate_decision_id:9`, `finish_reason=stop`, `msg_3603cb76ab1533b7`); bounded compact recovery (`msg_7556234fe2b06420`) succeeded on identical evidence (`outcome=complete`, `0` deferred rows).
  - **Former Failed Batch (`[204:216]`)**: Completed on first standard comparative attempt (`6` corrected, `4` unchanged, `1` polarity rejection, `1` unresolved-source rejection).
  - **Dispositions**: `119` changed, `27` comparative rejections, `10` unresolved-source rejections, `2` dependent-passage rejections, `1` polarity rejection, `0` joined-passage rejections, `0` direction-grammar rejections, `0` deferrals, `0` errors.
  - **Rendering**: `296` renderable cues (`48` top lane, `0` overflow, `0` unrenderable, `6` conflicts); SRT (`2f4f09a1...`) and ASS (`73e11676...`) verified.
- **Phase D (Offline Evaluator & Rerender)**:
  - `evaluate_final_integration` exited 0 (`render_status=rendered`, `0` wholly untimed rows, reference support unchanged).
  - `--re-render` exited 0 with zero model activation/inference and 100% byte-identical `translated.contract.json`, SRT, and ASS.
- **Resource Restoration**: GPU returned from `30625 MiB` peak to exact `3049 MiB` baseline; `nemovox-text-review.service` and `nemovox-asr.service` `inactive/dead`; `nemovox-studio.service` undisturbed; owner checkout unchanged.
