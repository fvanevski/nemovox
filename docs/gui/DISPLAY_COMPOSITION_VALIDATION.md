# Display composition: implementation and host qualification

Issue #30, parent #29. This is the first cohesive renderer implementation, not issue closure.
Source baseline: `c568ff7714584e2d814a6e9e7770590ca908523b`.
Admitted private baseline archive SHA-256:
`ef5423913d548583aadcbc4cfa44ec4a65204733f6af574580703466d3a0155a`.

## Central implementation and boundary

- One working branch `issue/30`; no new PR until both functional packages are ready.
- This package changes split partition/allocation, guarded jointly selected continuation grouping, deterministic ASS display occupancy, conservative SRT scheduling, actual emitted event identity and lineage. ASS overflow is explicitly unplaced (null display interval); SRT retains saturation text with explicit conflicts. No implicit change to selected English, canonical source, MT, review or model calls; typographic straight-quote normalization is display-only.
- Stacking is allowed for successive non-concurrent turns when clipping creates reading pressure and the tentative simultaneous burden is bounded (two cues, 100 characters, 1.8 times target CPS). These are initial engineering defaults, not human reading thresholds. Prefer free trailing time; preserve actual distinct overlap and expose saturation. Source-event intervals, allocation fragments and display intervals are different.
- Adaptive geometry/scene constraints, source-caption observations, real font/block bounds, Studio presentation, opt-in condensation disposition and final full issue acceptance remain Central's next package. Current fixed styles remain in this first package. Never claim automatic placement or measured scene detection from it.
- CPU-only. No model inference, downloads, service restarts, runtime deployment, original-media modifications, target subtitle extraction or reference-string tuning.

## Execution contract

Role: caller-owned Build for safe preparation, CPU validation and narrowly mechanical corrections. Semantic remediation forbidden; mechanical remediation allowed; remote update allowed only for one authorized semantics-preserving fix commit on the existing branch, with exact non-force ref readback and full revalidation of the new identity. Independent PR review, PR creation/merge, issue mutation and live deployment forbidden. Keep source and existing user checkout intact; a source defect is evidence to Central, not permission to redesign. Normal private scratch outputs, verified lockfile-based frontend dependency installation inside the isolated candidate only if necessary, and safe runtime-path correction are allowed. No global dependency installs, dependency upgrades, lockfile changes, guard bypass, unrelated service edits, or deletion of owner state.

### 1. Authenticate and prepare

Work from authenticated owner checkout normally `/workspace/nvidia-transcribe`. Verify origin matches fvanevski/nemovox and inspect status without altering owner state. The first-message kickoff supplies the exact staged commit; call it TARGET. Read live remote heads with native git/gh and require `issue/30` == TARGET; require main still matches baseline for paired comparison (if main moved, preserve and report, do not rebase). Use a fresh unique permitted workspace directory under the repository or other allowed project workspace, e.g. a collision-safe `mktemp -d` on an admitted path, not recursive cleanup. Create separate candidate and baseline worktrees with native Git, without switching/resetting the owner checkout. Use literal commands with actual paths, not unexpanded variables across independent shell invocations:

    git fetch origin refs/heads/issue/30
    git rev-parse FETCH_HEAD
    git cat-file -t TARGET
    git worktree add --detach CANDIDATE TARGET
    git fetch origin refs/heads/main
    git rev-parse FETCH_HEAD
    git cat-file -t c568ff7714584e2d814a6e9e7770590ca908523b
    git worktree add --detach BASELINE c568ff7714584e2d814a6e9e7770590ca908523b

FETCH_HEAD must match each respective immutable identity before object/worktree operations. Record actual worktree HEAD/status and complete baseline-to-target changed paths. Use current harness-native permission shapes; don't assume historical V1 admission gateway. Verify owner interpreter locations and Python version, package import path and any scripts import path. Use owner .venv/bin/python and .venv-studio/bin/python if present and authentic; if names differ identify actual repository-owned venv from current instructions. Never substitute system Python simply to obtain PASS.

### 2. CPU deterministic gates

From CANDIDATE root, run using the authenticated owner orchestration interpreter PY, not an editable install pointing to main:

    PYTHONPATH=src:. PY -c 'from pathlib import Path; import nvidia_transcribe; print(Path(nvidia_transcribe.__file__).resolve())'
    PYTHONPATH=src:. PY -m compileall -q src/nvidia_transcribe scripts/evaluate_display_schedule.py
    PYTHONPATH=src:. PY -m pytest -q tests/test_subtitles.py tests/test_issue14_source_to_renderer.py tests/test_connected_units.py tests/test_viewing_quality_regressions.py tests/test_viewing_validation_regressions.py tests/test_offline_rerender.py -m "not gpu"
    PYTHONPATH=src:. PY -m pytest -q -m "not gpu"

Check Studio with the verified Studio interpreter from the same candidate root; run `PYTHONPATH=src:. STUDIO_PY -m pytest -q tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py -m "not gpu"`. From CANDIDATE/web use installed lockfile-compatible frontend dependencies or `npm ci` scoped to candidate, then `npm test` and `npm run build`. Record true exits, counts, skips, import binding, Node/npm versions; if a dependency is absent, resolve ordinary isolated host setup without updating repository dependencies or downloading models. Do not substitute fixture counts for proof or rerun the entire suite repeatedly for reassurance. Run `git diff --check BASELINE_SHA TARGET` and verify tracked workspace remains clean.

### 3. Frozen paired display comparison (required)

Locate the admitted HOB completed baseline and AWE failed-review salvage contracts/configuration from the retained #29 private archive and job receipts. Authenticate archive hash, exact contract bytes, media identities, original selected-English status, configuration, render input after any already accepted overlay, and stage identity. Do not regenerate or translate. HOB media SHA-256 cf7fa8f758ab6e75ca8249bcd672826ea8fc359c5db0988e9d28e0c486ee02a0; AWE media SHA-256 5a21d721d43401355c824ae3aff2d0ff681d7586589218ca4bcc76d18dfcff7a. If exact contract unavailable, keep the tests and return paired comparison BLOCKED rather than substitute another recording. Freeze corrected reserved comparison windows excluding known development HOB_H1/AWE_H2/AWE_H3 and any other overlap identified by the original partition ledger; do not call them blind. Keep whole-recording metrics.

Materialize a private SubtitleConfig JSON object for each recording from authenticated original config, preserving HOB raised override and AWE standard, actual fonts, sizes, line limits, lane limits, display budgets and media duration. Compare identical inputs/config, not new defaults. In each source worktree root run the *same* Central script from its absolute candidate path with explicit PYTHONPATH:

    PYTHONPATH=src:. PY CANDIDATE/scripts/evaluate_display_schedule.py --repo BASELINE --contract HOB_CONTRACT --config HOB_CONFIG --output NEW_BASE_HOB_DIR
    PYTHONPATH=src:. PY CANDIDATE/scripts/evaluate_display_schedule.py --repo BASELINE --contract AWE_CONTRACT --config AWE_CONFIG --output NEW_BASE_AWE_DIR

The above two commands execute with workdir BASELINE. Then from CANDIDATE:

    PYTHONPATH=src:. PY scripts/evaluate_display_schedule.py --repo CANDIDATE --contract HOB_CONTRACT --config HOB_CONFIG --output NEW_CAND_HOB_DIR
    PYTHONPATH=src:. PY scripts/evaluate_display_schedule.py --repo CANDIDATE --contract AWE_CONTRACT --config AWE_CONFIG --output NEW_CAND_AWE_DIR

Each output directory must be previously absent. Keep private output and lineage, hashes and metrics. Compare original admitted outputs with new baseline replay; if not byte-identical, reconcile inputs/config/title differences and compare actual dialogue events before making claims. Explain if receipt identity differs. Report actual emitted ASS/SRT CPS tails, short dwell, positive/negative allocation shift, peak simultaneous cues/chars/sum CPS, high simultaneous pressure time, same-style overlap time, overflow/unrenderable, exact selected occurrence coverage, grouped and split event accounting, and determinism. Check candidate lineages against selected rows: grouped rows share event id; count unique emitted events, not duplicate lineage rows; each split fragment and grouped constituent must be recoverable, ignoring only documented typography/whitespace normalization. Inspect representative useful improvements, adverse regressions, legitimate short replies, true overlap, dense saturation, and reserved windows. Do not tune from target references. No arbitrary universal numerical reduction requirement; identify material regressions with private reproducer.

### 4. Actual renderer smoke (required, not adaptive-placement acceptance)

Authenticate FFmpeg/libass version, available font and selected video stream. Use a small set of actual media frames (at least one meaningful changed HOB sequence and one AWE, plus a two-line/stack/saturation case if present; synthetic test frame if absent). Use exact existing source frame receipt or decode selected video only, never subtitles; confirm actual frame-time/seek semantics. Copy ASS privately to simple scratch names base.ass and candidate.ass. From the scratch directory, use e.g.:

    ffmpeg -hide_banner -nostdin -ss T -i MEDIA -map 0:v:VIDEO_ORDINAL -an -sn -dn -vf showinfo -frames:v 1 frame.png
    ffmpeg -hide_banner -nostdin -i frame.png -vf "setpts=PTS+T/TB,ass=base.ass" -frames:v 1 base.png
    ffmpeg -hide_banner -nostdin -i frame.png -vf "setpts=PTS+T/TB,ass=candidate.ass" -frames:v 1 candidate.png

Replace T and VIDEO_ORDINAL with actual verified values; ensure T in the ASS media timeline, distinguish requested seek from actual PTS and use identical frame for each comparison. Use new names/roots per sample, no overwrites. Inspect actual pixel occupancy and legibility, overlap between Main/Top/TopAlt and clipping. Quantify changed-pixel bounding box or equivalent independently from intended lane assignments. Geometry/caption avoidance is not yet acceptance: record fixed-layout effects and residuals for next package. Only bounded private frames/excerpts. Do not extract or read embedded English soft-subs. Preserve old job outputs and external services.

### 5. Handback (one private archive; no PR)

Evidence matrix (all required unless marked supporting):

| Criterion | Artifact and primary observation | identity / privacy |
| --- | --- | --- |
| Candidate identity | remote ref, worktree Git SHA, base-to-head changed paths, import file and renderer hashes, command/exit receipts | exact full SHAs; public-safe summary |
| Regressions and UI compatibility | focused and full CPU test logs, Studio tests, frontend test/build, skips and exits | exact tested code/interpreter; sanitized private |
| Frozen input independence | contract/config hashes, prior job/artifact receipts, corrected reserved-window ledger | both admitted inputs; private |
| Actual display quality | four metrics.json reports, paired rendered ASS/SRT and private-lineage.json, criterion comparison with denominators and actual samples | each original source SHA and contract identity; private |
| Actual libass | frame source/time identity, font/libass/FFmpeg readback, exact commands/exit, bounded frame images and pixel-occupancy notes | private, no media dump |
| Retention | input hashes before/after; event-to-occurrence lineage comparison, grouped/split allocation checks, overflow and unrenderable accounting, reproducibility | private; no silent drops |
| Resource/state preservation | start/end source/job checks, no inference/service changes, tracked cleanliness, final remote readback | host-attested; public-safe summary |
| Residuals | unresolved timing/display/geometry issues, source defects, observed behavior tradeoffs and smallest repro | private as necessary |
| Command friction | complete canonical block below | sanitized |

Create one controlled evidence directory with required receipts, report, manifest (criterion, relative path, SHA-256, byte size) and checksums, then tar.gz with stable filenames. Include bounded private subtitle/lineage evidence for these two recordings only, sanitized logs and representative frame crops; no original/full media, destination reference dump, model assets, secrets, environment dumps, unrelated logs or backups. If complete private dialogue artifacts are necessary for Central comparison use the private archive only, never GitHub. Verify tar listing and all manifest payload digests/counts, compute archive SHA-256 and size. Return actual absolute path and operator-attachable archive, not just a summary. No public GitHub artifact upload; Central will post compact non-content metrics/decisions later. Do not commit diagnostic outputs. Retain isolated worktrees and evidence until Central consumes them; no destructive cleanup required.

Return in conversation:
- RESULT=PASS_FOR_NEXT_PACKAGE | PARTIAL | BLOCKED | FAIL | STALE (not issue PASS).
- exact BASELINE, TARGET, tested SHA(s), remote branch readback, candidate tree, changed path inventory, any mechanical commit and revalidation.
- test matrix with exits/skips, before/after results for both formats/recordings, sample dispositions, explicit missing evidence reasons and material regressions.
- evidence archive path, bytes, SHA-256, manifest coverage; runtime/resource restoration and input/owner-worktree nonmutation.
- COMMAND_SHAPE_FRICTION with COMMANDS_MATCH_HANDOFF.

Canonical friction form (no material events):
    COMMAND_SHAPE_FRICTION:
      encountered: "no"
      events: []
      omitted_sensitive_values: "no"
    COMMANDS_MATCH_HANDOFF: "yes"

For each material event include all fields:
    scenario
    requested_shape
    effective_shape
    rewrite_provenance=none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown
    event_kind=pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other
    execution_effect=not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown
    correction_or_annotation
    replacement_shape
    recipe_reference
    resource_reference
    contract_conflict=yes|no|uncertain
    durable_gap_candidate=yes|no|uncertain
Set encountered=yes, events list, omitted_sensitive_values truthfully; preserve successful recoveries and requested/effective distinctions. Do not manufacture events. Raw/native exact identities, failures, full diffs and runtime receipts; RTK for routine filtered output only; Serena bounded semantic trace for actual failures, GitNexus read-only derived impact if useful, Probe supplemental search, OpenViking prior decisions only. Never use derived memory as current authority.

Stop after this consolidated CPU campaign. Resolve ordinary path/runtime/fixture setup and canonical mechanical formatting locally, recording any actual changes; if a semantic defect is exposed, preserve native failure, isolate minimal reproduction, complete unaffected evidence where meaningful and return it. Do not implement adaptive placement, modify source/MT/review, alter test meaning, start inference, open PR or merge. If live guard requires post-edit publication Fresh-review and Verify for the one mechanical commit, authorize those bounded child gates before publication; not independent PR review. Re-read remote exact head immediately before non-force push; if moved, stop STALE; do not force/rebase. Any new commit invalidates earlier exact-head tests; rerun required gates, preserve original run identities.
