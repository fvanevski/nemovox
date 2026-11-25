# Normal subtitle workflow and final qualification

## Defaults and scope

The default public CLI, Studio API and frontend select fixed 90-second
primary chunks, 10-second overlap context, 15-second minimum tail,
speaker-attributed ASR, current-media source quality (auto on managed Riva
jobs), CUDA independent Whisper, Riva dedicated translation, balanced
twelve-row English review, SRT/ASS and no condensation. Source rewriting,
TitaNet and Qwen stay explicitly optional. The lower-level Python config
classes remain explicit composition interfaces, not a replacement for
public job resource admission.

The accepted matched geometry archive
`805ab0e43275171b16c0beebbc6c938d15ed5b055d4a6e5b342180f46d61eefe`
contains six same-model reference events. Fixed supported 2/6, pause-aware
3/6 (one shared, one fixed-only, two pause-only, two neither). Neither
establishes dominance; the pause-aware arm lost one previously supported
event and compression had more unlocalized units. The low-flag and late
regions had no positive reference event denominator. Retain fixed 90/10
as the current normal policy; 46/4 pause-aware remains opt-in. This is
not evidence that fixed is globally optimal or that silence is missing
speech. Do not run another full geometry grid to select a default.

Use the existing lifecycle and scheduler; no new model or additional
unconditional text stage. Source-quality budgets and actual deferred
dispositions remain visible. Twelve-row review is retained based on
the previous matched 26-row comparison (10 to 6 calls, 77.177 to 52.234s);
the joined-passage contract is strengthened and must be requalified.
Visual text is a model observation with one literal slot and explicit
category, never an independent spoken-word rewrite.

## Current-version resume

A current-job source-quality checkpoint binds the exact original recording
and its manifest-authenticated ledger, full media file SHA-256, selected
stream, policy, configured resource identities, declared owned services and
current Python implementation bytes. It stores the completed canonical
contract and original private quality report with a content digest.
Unresolved or budget-deferred evidence is a legitimate partial result;
provider/worker/scene errors and missing visual frame payloads are not
reusable. A fresh attempt invalidates the previous checkpoint before
execution. No historical manifest editing or reuse across changed source
or media. Configured resource fingerprints do not authenticate served
weight bytes; use fresh after an actual model/environment replacement.

To resume exactly the original current recording, supply its original
`--recording-contract` and the same media and output directory to
`generate-subtitles`, with normal resume enabled. On a valid checkpoint
the metadata field `source_quality_resume` is `reused_exact_input`.
The managed clients are lazy; no ASR/context/acoustic/scene inference
is needed for a stage hit. MT and review checkpoints remain independently
keyed. A plain media-only rerun can reconstruct the recording from primary
chunk checkpoints but recording runtime metadata may differ; it is not
guaranteed to reuse the exact source-quality stage. Offline rerender is
a separate entrypoint, never an inference retry. Current partial reports
are not represented as successful complete semantic resolution.

## Final bounded and episode qualification

Use one matched bounded compression-region current public job on the
prior 110-second excerpt if exact media bytes are still present
(SHA-256 `f1f6984d0c35d8df782285d80ce29a478d859029c29fc438b0a24a8d6dd1bec6`).
The accepted C2 archive
`6aeb764bf7e8368a71dc4fc1a8eeb4473831951c2c3723e8b990d7830863181c`
includes its original recording, manifest and ledger. Reuse only if
authenticated bytes and media identity match; do not modify its manifest.
If that video is unavailable, recreate only the registered compression
interval from the authenticated original source; record offset, extraction
and actual identities. Do not claim exact C2 matching on different bytes.
Run the new known-text/known-pixel controls via
`scripts.evaluate_semantic_translation --controls` (includes split
direction phrase and explicit disambiguating idiom context). Compare
actual primary and final passages; a successful model verdict is not
semantic truth. A matched six/twelve comparison of the new review
contract is optional if needed to disambiguate a material performance
or quality regression; the already accepted C2 comparison remains the
baseline, not a mandatory extra campaign.

The offline evaluator `python -m scripts.evaluate_final_integration --help`
reads a current job, original recording, media and optional sealed
geometry/C2 archives. It verifies actual ledger, quality and translation
lineage, retains private row differences and compares all six fixed
positive reference events for the full French episode, or the registered
bounded region reference events with the explicit original-media offset.
It reports original versus final event-level evidence using the same
reference denominator; do not equate event-level support with the
earlier strict word-locality criterion. Unmatched exact lexical evidence
is not a missing-dialogue label. For exact C2 comparison supply
`--prior-semantic-archive`; it requires identical media bytes and
discloses whether original recording/ledger and primary MT are identical.
Use `--geometry-archive` with `--region compression` for the
bounded control, and `--diagnostic-archive` for the original full
French episode. For an independent multilingual control omit French
reference arguments.

After bounded integration, current resume and offline rerender pass,
run one complete original French diagnostic episode under normal
defaults and one independent authentic multilingual control. Preserve
actual primary, localized native, independent, scene, MT and review
receipts. Examine registered compression, boundary, late-gap and low-flag
regions plus genuine silence, repeats and overlap. Compare with prior
diagnostic original and translated evidence on comparable original
intervals; no episode-specific dictionary or automatic source changes
from external metadata. Use the original observations and reference
events to assess gross timing displacement, omissions, duplicates,
fragmentation and semantic quality. Do not manufacture improvement
from changed row counts, omitted untimed rows or renderer suppression.
Return actual quality residuals and resource costs; no claim of perfect
ground truth. No new PR or deployment in this campaign; exact-head
independent review and authorized merged-main acceptance follow separately.
