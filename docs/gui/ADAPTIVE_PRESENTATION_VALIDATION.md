# Adaptive presentation: consolidated CPU qualification

Issue #30, continuation of accepted display composition. Central owns design and implementation; local Build owns host integration, execution, bounded mechanical repair and evidence. The exact target commit is supplied by the kickoff; authenticate it before proceeding. The prior accepted candidate is 3289fb8c7006752caebc5c41338bd6544a569dc0 and the original main baseline is c568ff7714584e2d814a6e9e7770590ca908523b. Read AGENTS.md and the earlier DISPLAY_COMPOSITION_VALIDATION.md for retained input provenance, not as an instruction to repeat that campaign.

## Execution authority

- PRIMARY=build; ADMISSION=caller-owned ordinary validation, current installed harness rules, not strict evidence or a typed gateway assessment.
- MECHANICAL_REMEDIATION=allowed; SEMANTIC_REMEDIATION=forbidden; REMOTE_UPDATE=allowed; SOURCE_REMOTE_UPDATE=allowed solely for one consolidated qualifying mechanical correction commit; EVIDENCE_REMOTE_UPDATE=forbidden; EVIDENCE_PACKET_REQUIRED=yes; EVIDENCE_ROOT=outputs/handbacks/adaptive-presentation-issue30.<unique>/evidence (private owner-host scratch, never Git-staged).
- PUBLICATION_FRESH_REVIEW=required and PUBLICATION_VERIFY=required if making a correction commit under the live guard; INDEPENDENT_PR_REVIEW=forbidden. Do not claim a local Verify is independent Central review.
- Authority includes ordinary dependency/path/fixture setup, isolated npm lockfile installation, unambiguous mechanical import/type/format/reference corrections that preserve Central's invariants, and scratch-only projection of a custom model registry to remove the retired subtitle_condensation binding. Preserve all other bindings. No production registry edits, dependency upgrades, test weakening, new algorithms, threshold tuning, source/MT/review changes, model inference, embedded subtitle access, model downloads, service changes, owner checkout edits, PR creation, merge or issue closure.
- Serena for bounded symbol/reference inspection and diagnostics when useful; Probe supplemental only; RTK for routine verbose output where evidence survives. Native git/gh, FFmpeg, ffprobe, tests and file hashes are authority. Raw exact refs, diffs, failures and actual render evidence must remain unfiltered. OpenViking is optional bounded prior context only, never mutable authority. No secrets/private frames/dialogue in durable memory.
- Resolve ordinary in-scope execution friction autonomously in the same campaign. If a genuine algorithm/semantic defect occurs, preserve minimal reproducer and actual output, complete unaffected work, return precise blocker. Do not patch heuristics or use per-episode exceptions to make metrics green.

## 1. Exact host integration

Launch directory /workspace/nvidia-transcribe. Inspect owner status and current repo identity; do not switch/reset/clean owner. Set OWNER to that authenticated checkout, TARGET to full SHA from kickoff, BASE to original main SHA, PRIOR to accepted candidate. Use native commands (shell variables below are illustrative shell assignments to actual paths, not literal placeholders).

    git -C "$OWNER" remote get-url origin
    git -C "$OWNER" status --short --branch
    git -C "$OWNER" ls-remote origin refs/heads/main refs/heads/issue/30
    git -C "$OWNER" fetch origin refs/heads/issue/30
    git -C "$OWNER" rev-parse FETCH_HEAD

Require branch ref and FETCH_HEAD == TARGET; if main changed, report actual main and assess integration drift rather than claim baseline current. Record ancestry and complete changed paths BASE..TARGET and PRIOR..TARGET. Create fresh private directory with mktemp -d under existing owner outputs/handbacks; no pre-emptive recursive deletion. Use git worktree add --detach "$ROOT/candidate" "$TARGET"; verify actual HEAD and clean state. CAND is that absolute worktree. Run tests there; use authenticated owner .venv/bin/python and .venv-studio/bin/python (or actual documented equivalent), explicit PYTHONPATH=src:. and confirm imports from CAND rather than installed/owner code. Record versions, libass library discovery, FFmpeg build/filter availability, actual font resolution (fc-match if available), Node/npm versions. No global installs or changes. Ordinary isolated dependency installation using current lockfile allowed; record failures and no silent skips.

## 2. Deterministic qualification

From CAND, with actual PY and STUDIO_PY:

    git diff --check "$BASE" "$TARGET"
    PYTHONPATH=src:. "$PY" -m compileall -q src/nvidia_transcribe scripts/evaluate_display_schedule.py
    PYTHONPATH=src:. "$PY" -m pytest -q tests/test_adaptive_presentation.py tests/test_subtitles.py tests/test_issue14_source_to_renderer.py tests/test_connected_units.py tests/test_viewing_quality_regressions.py tests/test_viewing_validation_regressions.py tests/test_offline_rerender.py tests/test_translation_review.py tests/test_resource_integration.py tests/test_model_resources.py -m "not gpu"
    PYTHONPATH=src:. "$PY" -m pytest -q -m "not gpu"
    PYTHONPATH=src:. "$STUDIO_PY" -m pytest -q tests/test_studio_api.py tests/test_studio_exports.py tests/test_studio_workflow.py -m "not gpu"

For Studio use actual tracked test filenames if the above named scope differs; record exact discovered names and actual command. In candidate web directory run npm ci if dependencies absent, npm test and npm run build using tracked lockfile. Record unmodified lockfile, exits and counts. No substituting mocks for native proof, no broad skips. Check tracked active code/config/docs for stale condensation consumers; report unrelated historical evidence separately. Verify public CLI no longer accepts retired option and API rejects explicit true; no credentials required for removed pass.

## 3. Frozen comparison inputs and public offline rerender

Reuse admitted prior campaign evidence rather than regenerate recognition/translation or replay original main. Locate prior archive:
nemovox-issue30-display-composition-3289fb8c-evidence.tar.gz
SHA-256 3b956c7304de8f45274a7b70327934dc235dd5b696e0904c38d483a82c9b6029, 8393281 bytes. Original #29 archive SHA-256 ef5423913d548583aadcbc4cfa44ec4a65204733f6af574580703466d3a0155a. Verify manifest and checksums; obtain original selected contracts from authenticated retained local jobs/inputs (do not assume prior archive contains complete contracts). Required identities:

| Input | Selected contract SHA-256 | Full original media SHA-256 |
|---|---|---|
| HOB | 6b2d881c77f793bada36965734b20fe072f3fcc4b1110f63365941c41cb14bb2 | cf7fa8f758ab6e75ca8249bcd672826ea8fc359c5db0988e9d28e0c486ee02a0 |
| AWE | 2b12ee77854362216670d379033a985144388d94eba534886baeaada578b2ea4 | 5a21d721d43401355c824ae3aff2d0ff681d7586589218ca4bcc76d18dfcff7a |

HOB complete; AWE explicitly failed-review salvage, not ordinary success. Keep HOB 422 and AWE 223 selected occurrences. Existing accepted HOB ASS/SRT 399 and AWE 234 emitted events. Compare candidate automatic output against accepted fixed (HOB raised, AWE standard) output, with old actual config unchanged except ass_layout=auto. Correct strict CPS threshold by parsing actual emitted bytes in integer milliseconds; old seconds helper now returns integer ms. Do not assume old threshold counts from rounded display summaries. Preserve corrected reserved windows HOB_H2 [840,890] (empty), HOB_H3 [1245,1295], AWE_H1 [450,490]; HOB_H1/AWE_H2/AWE_H3 development overlap excluded. No blind or generalization claim.

For each input copy exact translated.contract.json to new scratch job directory. Keep source paths explicit; no automatic dereferencing arbitrary contract paths. Public CLI (read --help first, preserve original relevant arguments and layout config; exact invocation with actual paths in receipt):

    PYTHONPATH=src:. "$PY" -m nvidia_transcribe.cli generate-subtitles "$MEDIA" --output-dir "$JOB" --re-render --ass-layout auto --formats srt,ass --episode-stem comparison

Use the same command without positional media for second run; preserve first outputs and observation bytes before it. Bind original selected-contract hash before/after, observation hash and actual source media fingerprint/full SHA-256, actual input stream. If custom host registry still binds retired role, use private projected copy via documented registry option/env only; no host config edits or credentials in evidence. Explicitly confirm no inference stage or model acquisition, no new recognition/translation/review requests, retained AWE failed-review receipt semantics. If default settings differ from frozen comparison config, separate public pipeline smoke from exact config evaluator. Reuse observations; no new inspection on media-free rerender. Show ASS/SRT byte reproducibility, observation identity and lineage consistency (metadata runtime/path fields may differ; compare meaningful fields).

For actual paired metrics use candidate script with original authenticated SubtitleConfig JSON, auto-only edit and persisted observation document, per recording:

    PYTHONPATH=src:. "$PY" scripts/evaluate_display_schedule.py --repo "$CAND" --contract "$CONTRACT" --config "$AUTO_CONFIG" --observations "$JOB/presentation.observations.json" --output "$METRICS_DIR"

Output directory must not already exist. Retain both first/repeat emitted files plus private lineage and JSON metrics. Use current evaluator events/metrics functions to independently calculate accepted archived fixed output metrics, not rerun old model workflow. Run manual standard and raised controls on frozen contracts (same relevant config) and compare old fixed outputs/lineage; explain title-only metadata changes if present. Confirm SRT remains independent. Verify all selected occurrences and exact selected English accounted for, only documented safe grouping/splitting/normalization; compare displayed unique event IDs, grouped constituents, actual source intervals, allocation and constituent onset shifts separately. No unnoticed overflow, invalid duration, same-style overlap, actual simultaneous bbox collision, or unaccounted dialogue. Report material differences and real read-pressure tradeoffs, not universal improvement promises. No destination reference extraction.

## 4. Actual visual and public surface proof

Automatic path must produce native measured geometry on this host, not merely fallback. Observe actual generated document (sample count/support, source hash, non-attached stream, dimensions/SAR/rotation, bar geometry, candidate union, unknown/confirmed status, bounds, requested seek vs actual PTS). Do not inject hand-labeled regions into production inputs to force PASS. Synthetic confirmed intervals/cuts are valid controls only. Actual caption-candidate detection is heuristic, not OCR, source-language classification or episode-wide absence. No scene detector is claimed; test positively supplied cuts and unobserved-cut fallback. Check mismatched media/cache rejection, audio-only/no-video, dark/unknown, present and absent interval controls, manual layout, non-16:9, long/wrapped cues, opaque box if configured, real concurrency and saturation with deterministic synthetic fixtures. Record unavailable proof rather than silently substituting it.

Use the original admitted visual samples (HOB around 498.05 and 1385.20, AWE around 71.69 and 731.00; locate exact prior frame receipts), plus a small representative single-line/double-line example. HOB measured bottom bar about 141 pixels in admitted 1080p evidence; AWE source captions around y954..1005. These are evaluation observations, not renderer constants. Use exact same undecorated decoded frame for accepted and candidate overlay comparison. Decode only actual selected non-attached video (map 0:V:0 or verified container index, -an -sn -dn), record frame time/PTS and pixel size, render ASS through actual FFmpeg libass with matching original timing and font. Follow earlier DISPLAY_COMPOSITION_VALIDATION.md frame command pattern. Quantify changed-pixel boxes, picture/bar penetration and caption-band intersection independently of proposed bbox; distinguish band overlap from exact glyph collision. For overlaps inspect fixed positions before/during/after, reading order, clipping, real concurrency and three-lane/saturation synthetic control. Include bounded original/paired render frames and small ledger, not full video. Avoid visual claims beyond inspected samples.

For Studio, run exact candidate API tests using isolated TestClient/fixtures and frontend suite/build; confirm default auto/manual controls, truthful diagnostic status, actual emitted ASS timeline versus SRT fallback, observation artifact export and no old condensation control. Exercise isolated preview/export for actual retained rendered artifact metadata if practical, without deployment or live service restart. Explicitly distinguish browser actual rendering if exercised from API/build proof. No obligation to start GPU work or requalify unrelated integrations.

## 5. Qualification decision and evidence

One campaign, no new PR. PASS_REVIEW_READY requires all mandatory deterministic gates, both exact frozen input public offline runs with native measurement/reuse, complete selected English/occurrence lineage, useful measured placement on actual frames without material regression, manual overrides, truthful unknown/fallback/overflow and Studio evidence. No material new ASS loss on these inputs; record actual clipping/overlap/pressure and any valid bounded tradeoff. If material defect or missing proof, return PARTIAL/FAIL/BLOCKED/STALE with exact reason, not an unqualified PASS. No universal numeric reduction target. Do not launch additional campaigns for semantic review or source timing (subsequent issues).

| Criterion | Private evidence |
|---|---|
| Exact code identity | native remote refs before/after, actual HEAD/tree/status, complete baseline/prior delta, any mechanical diff and new commit |
| Deterministic gates | commands/exits/counts, versions, source-bound imports, native library/font, Studio/frontend logs |
| Frozen input independence | prior archive hash, contract/media/config hashes before/after, selected-stream receipt, no-inference and owner/service nonmutation evidence |
| Automatic and manual | persisted observation documents, public CLI commands/status, manual controls, first/repeat ASS/SRT bytes and metadata/lineage |
| Quality and correctness | accepted vs candidate metrics, exact accounting and interval changes, reserved windows, representative regressions/tradeoffs |
| Visual evidence | bounded actual/synthetic frame evidence and source/frame/command/measurement ledger |
| UI/export | API test receipts, artifact and timeline proof, frontend tests/build, explicit browser proof limits |
| Closure | report, residuals, friction ledger, manifest/checksums, final ref proof |

Create one private evidence tar.gz in fresh ROOT (never Git-stage it). Manifest criterion/path/bytes/SHA-256 for all payloads and checksum verification; record archive path, bytes, SHA-256, inventory and omissions. Include only necessary private dialogue artifacts for these two inputs, sanitized logs, bounded frame comparisons and first/repeat output bytes; omit full media, target reference text, model weights, secrets/credential config/env dumps. Preserve worktrees and owner data. Deliver direct conversation handback and archive for operator attachment to Central; no public evidence commit, no duplicate public packet.

If a qualifying mechanical fix is genuinely needed, gather all same-cycle repairs, inspect full diff and verify invariants. Run appropriate required guard publication review/verify, re-run all relevant gates at new identity, then confirm remote issue branch still equals kickoff TARGET immediately before ordinary non-force push of the single commit to existing issue/30. Read back exact branch SHA; if moved, stop STALE, never force/rebase. No semantic rewrite or test weakening. If no changes, no commit. Recheck final exact refs and input identity in all cases.

## 6. Required handback fields

RESULT=PASS_REVIEW_READY|PARTIAL|FAIL|BLOCKED|STALE; repo/branch, kickoff SHA and final tested SHA/tree/ref, main ref, prior/base delta, test matrix, frozen input provenance, both recording metrics and actual placement observations, public CLI and Studio proof, archive path/bytes/SHA-256 and manifest coverage, material blockers/omissions, mechanical changes and state preservation. Report actual measured results, not predictions. Explicitly note AWE salvage. Include canonical command-shape friction ledger even for zero events:

```
COMMAND_SHAPE_FRICTION:
  encountered: "no"
  events: []
  omitted_sensitive_values: "no"
COMMANDS_MATCH_HANDOFF: "yes"
```

For each material event replace empty events with entries recording scenario, requested_shape, effective_shape, rewrite_provenance (none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown), event_kind (pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other), execution_effect (not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown), correction_or_annotation, replacement_shape, recipe_reference, resource_reference, contract_conflict (yes|no|uncertain), durable_gap_candidate (yes|no|uncertain). Report sensitive omissions without copying secrets; preserve meaningful native failure output. Use current GHDEV tool routing, not obsolete version-specific command assumptions. Stop after this integrated campaign.
