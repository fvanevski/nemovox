# Evidence acquisition response and transport qualification (CPU only)

## Authority and acceptance

Repository fvanevski/nemovox, issue #25 under #23, existing PR #27, branch issue/25, base/main 75d8534b3e3ae12ebf2a50dd76c5fd5e8f5b6227. Previously reviewed/qualified head 802cf38f10b835f3a2332890ea39ac060850dfad (tree 619989200c171925406e7cec2b0fe46b1696042f). Codex review 6092220760 at that exact head: C1 https://github.com/fvanevski/nemovox/pull/27#discussion_r4722747741 (non-retryable HTTP errors treated as deferred evidence); C2 https://github.com/fvanevski/nemovox/pull/27#discussion_r4722747748 (valid caption-only response rejected). Both Central-confirmed blockers. Central implemented the shared text-review/acquisition retryability policy and independent scene/text validation, with no successful empty observation. Prior B1/B2 admission correction unchanged. No new model, sampling, prompt, response-schema, checkpoint identity/version, inference allocation, resource lifecycle or rendering redesign. This is one existing-PR remediation, not a new issue or new PR.

LOCAL_AGENT_PRIMARY=build
VALIDATION_ROLE=caller-owned
MECHANICAL_REMEDIATION=allowed
SEMANTIC_REMEDIATION=forbidden
REMOTE_UPDATE=forbidden
PR_CREATION=forbidden
DEPLOYMENT=forbidden
GPU_INFERENCE=forbidden

Read AGENTS.md and this complete historical instruction file at kickoff's authenticated commit/blob. Native Git/gh, Bash, pytest, npm, systemctl and nvidia-smi authoritative; Serena for bounded source navigation and authorized mechanical edits, Probe supplemental only, RTK for lossless routine output (raw native for SHA/diffs/failures/evidence), OpenViking not authority for mutable source/runtime, no private content in durable retrieval. No historic operational-schema gateway assumption. Ordinary setup, safe path/import/format mechanics and diagnosis allowed in isolated candidate, not new semantic implementation, test weakening, dependency/lockfile changes, service activation, GPU inference, real media extraction, configuration changes, model installation, or external service eviction. Never modify owner checkout, active jobs/media/outputs, /workspace/pyannote or protected Nemotron checkpoint. If a mechanical source correction is essential, retain raw failure, complete diff and actual local SHA/tree; rerun affected/full verification, no push. Central owns its adjudication. Continue independent permitted work despite other failures. Stop on wrong identities, substantive defect or unauthorized scope. No Codex request/thread resolution, merge or deployment.

## A. Admit final candidate

Use Bash semantics, exact TARGET_SHA/TARGET_TREE/INSTRUCTIONS_SHA/INSTRUCTIONS_BLOB from kickoff. Owner /workspace/nvidia-transcribe. Check owner state/origin before fetch. Use collision-safe scratch, not a fixed shared location.

    cd /workspace/nvidia-transcribe
    git status --short --branch
    git rev-parse HEAD HEAD^{tree}
    git remote get-url origin
    git fetch origin refs/heads/issue/25
    test "$(git rev-parse FETCH_HEAD)" = "$TARGET_SHA"
    test "$(git rev-parse "$TARGET_SHA^{tree}")" = "$TARGET_TREE"
    git merge-base --is-ancestor "$INSTRUCTIONS_SHA" "$TARGET_SHA"
    test "$(git rev-parse "$INSTRUCTIONS_SHA:experiments/ACQUISITION_QUALIFICATION.md")" = "$INSTRUCTIONS_BLOB"
    umask 077
    PRIVATE_ROOT=$(mktemp -d /tmp/nemovox-acquisition-review.XXXXXXXX)
    OWNER=/workspace/nvidia-transcribe
    PY="$OWNER/.venv/bin/python"
    STUDIO_PY="$OWNER/.venv-studio/bin/python"
    BEFORE=802cf38f10b835f3a2332890ea39ac060850dfad
    git worktree add --detach "$PRIVATE_ROOT/worktree" "$TARGET_SHA"
    mkdir -p "$PRIVATE_ROOT/evidence/logs" "$PRIVATE_ROOT/replay"
    cd "$PRIVATE_ROOT/worktree"
    git status --porcelain=v1
    git rev-parse HEAD HEAD^{tree}
    git diff --name-status "$BEFORE" HEAD
    git diff --check "$BEFORE" HEAD
    git ls-tree -r HEAD -- experiments
    git diff "$BEFORE" HEAD > "$PRIVATE_ROOT/evidence/remediation.diff"

Expected final delta six files: src/nvidia_transcribe/review_evidence.py, review_execution.py, translation_review.py; tests/test_recording_review.py; README.md and docs/model-resources/CONFIGURATION.md. Active experiments tree absent. Confirm owner and owned service/GPU pre/post state read-only. Record full IDs and candidate Python/Studio actual package and any scripts.* import identities (run from candidate root using PYTHONPATH=src:.). Record raw logs and actual exit codes; pipefail if tee. Do not copy unrelated logs.

## B. Focused and final deterministic verification

    PYTHONPATH=src:. "$PY" -m pytest tests/test_recording_review.py tests/test_connected_units.py tests/test_deferred_review.py tests/test_translation_review.py tests/test_resource_integration.py tests/test_resource_lifecycle.py -m 'not gpu'
    PYTHONPATH=src:. "$PY" -m compileall -q src/nvidia_transcribe
    PYTHONPATH=src:. "$PY" -m pytest -m 'not gpu'
    PYTHONPATH=src:. "$STUDIO_PY" -m pytest tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py -m 'not gpu'
    (cd web && npm ci && npm test && npm run build)
    git diff --check 75d8534b3e3ae12ebf2a50dd76c5fd5e8f5b6227 HEAD

New regression families: test_visual_http_failure_respects_shared_retry_policy (400/401/403/404/422 explicit error, 408/429/500/503/transport timeout deferred, corrected/recovered endpoint completes only unfinished unit); test_caption_only_real_provider_refinement_and_durable_reuse (real provider, deterministic frame and model transport doubles, caption provenance, comparison, selected English, unchanged source/MT, durable reuse); test_visual_collections_reject_empty_or_malformed_evidence (both empty, forged scene/caption IDs, blank text, wrong collection type, excess scene/caption counts). Existing scene-only and combined observations, distinct/shared resource ownership and all B1/B2 regressions must remain green. Prior counts 344 non-GPU, 49 Studio, 14 frontend; 18 new parameterized cases expected, counts are not acceptance gates. A fatal HTTP acquisition must cause completed_with_errors (required review fails before render, primary preserved), never completed_with_deferred_evidence; transient errors remain resumable. Valid caption-only response needs no fabricated scene observation; both empty or malformed collections cannot be persisted as observed. Empty responses remain ordinary failed acquisition, not a new terminal disposition. Test evidence uses actual provider validation and reviewer control flow with deterministic doubles, not actual model calls. Do not weaken assertions or bypass provider. No need to redesign failure handling or run a new model benchmark.

## C. Scoped live-evidence reuse: candidate-only recorded replay

Original immutable private archive /tmp/nemovox-resumable-review.jxJ6RE1p/nemovox-issue25-resumable-review-qualification-0965afe3.tar.gz (or exact relocated copy), 7302520 bytes, SHA256 cacf2df53ad58c724a7d286ab2d165e734421b3fc85430c345480a499e4e4390. Previous exact baseline/target response replay PASS at 802cf38f10b835f3a2332890ea39ac060850dfad, private handback 22114 bytes SHA256 5fe9bf83cd018f9ecd9124b748f4fc38578e03e6c949a2919e6dd8559bc6aab8. No need to rerun the old baseline. Use the previously authenticated historical replay script unchanged, not a new experiment.

    REPLAY_SHA=a94335feeaffa9e2680a88cba756b813985f448c
    REPLAY_BLOB=683d8c9bbd8cfaca74e73a316898d6a6b9d974af
    test "$(git rev-parse "$REPLAY_SHA:experiments/replay_review_admission.py")" = "$REPLAY_BLOB"
    git show "$REPLAY_SHA:experiments/replay_review_admission.py" > "$PRIVATE_ROOT/replay/replay_review_admission.py"
    test "$(git hash-object "$PRIVATE_ROOT/replay/replay_review_admission.py")" = "$REPLAY_BLOB"
    PYTHONPATH=src:. "$PY" "$PRIVATE_ROOT/replay/replay_review_admission.py" \
      --archive "$ARCHIVE" --scratch "$PRIVATE_ROOT/target-replay" --root "$PRIVATE_ROOT/worktree"

Set ARCHIVE to actual verified absolute path before run; script verifies archive and individual member identities. Expected phase digests (canonical digest, not raw file SHA):

    fresh:
      rows=329
      row_sha256=1e802adea90fb1059bda4312fd9a0c0da82603ccf4c5373cb2618e2da0b1f7fe
      checkpoint_sha256=5104d1d200f21d9d9be46eb1e843683d4f8cb61c11793efbcdb22ed3a4ed21ac
      requests=70 acquisitions=15 restored=0 hits=0
    resume:
      rows=329
      row_sha256=59bcfdc25f0ee5936405df3049805780cacbb38c1fbbfd846ff30eabc880fc4a
      checkpoint_sha256=3ec01452e4859de9659df3540e12d5372d9513baf506a466f66237fb05163e42
      requests=4 acquisitions=3 restored=15 hits=325
    complete_replay:
      same row/checkpoint digests as resume
      requests=0 acquisitions=0 restored=18 hits=329

Check raw progress.json SHA256 against prior 1b519f317828ed21cd2a6fc48434f59ebfe400bb5ab0c4e5ef984299943fdb6d (6908295 bytes) as corroboration; canonical equality is enforced by replay. This script reexecutes reviewer using recorded provider receipts, NOT the real media acquisition provider. Thus it alone cannot certify changed provider behavior. The provider regressions in B exercise the changed code; Central separately confirmed all 18 successful prior integrated visual receipts contained both nonempty scene observations and visible text, and the only initially unobserved requests were three positive-budget deferrals. Prior live acquisition behavior is reusable only for those unaffected accepted response paths; no claim that novel caption-only live behavior has been measured. No GPU calls, fresh episode, rerender or second baseline. Never relax actual request or checkpoint equality. If mismatch, preserve actual diagnostic, report blocker; do not start GPU or modify product to force equality.

## D. Handback

Verify live issue/25 and main at handback via git ls-remote and PR #27 via gh pr view. Exact target and unchanged main expected; moved head/base STALE. Record both exact Git identities, full remediation diff, clean candidate/owner state and read-only owned service/GPU state. Report criterion dispositions for C1, C2, prior B1/B2 and deterministic gates, candidate-only archived replay/equivalence and scope of evidence reuse. No remote push, PR mutation, review-thread resolution, Codex request, merge or deployment. Central owns final review and the remaining one manual Codex request under cap 2.

One private mode-0600 tar.gz under PRIVATE_ROOT, complete report, manifest relative paths/bytes/SHA256, SHA256SUMS, raw bounded logs/exit codes for exact-head and import identity, regression/full/Studio/frontend/build/compile/diff checks, authenticated archive and replay JSON receipt, actual diff, owner/remote and resource state. Include prior archive/qualification digests by reference, not original private data. Exclude full progress.json, source dialogue, inference messages, original media, screenshots, weights, secrets, venvs, unrelated logs and archive itself. Return absolute path, byte size, SHA256, attach directly to Central. PASS requires actual current tests and recorded replay pass, exact candidate, resource preservation; report PARTIAL/FAIL/BLOCKED/STALE as observed. No relabeling prior live inference as fresh.

Canonical command-shape friction ledger:

    COMMAND_SHAPE_FRICTION:
      encountered=<yes|no>
      events:
        - scenario=<short repository/task scenario key>
          requested_shape=<bounded sanitized submitted command>
          effective_shape=<bounded effective command|same-as-requested|unknown>
          rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
          event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
          execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
          correction_or_annotation=<bounded correction/error/annotation>
          replacement_shape=<successful corrected shape|none>
          recipe_reference=<GHDEV-10 recipe key|none>
          resource_reference=<plugin/repository resource|none>
          contract_conflict=<yes|no|uncertain>
          durable_gap_candidate=<yes|no|uncertain>
      omitted_sensitive_values=<yes|no>

If none:

    COMMAND_SHAPE_FRICTION:
      encountered=no
      events: []
      omitted_sensitive_values=no

Also COMMANDS_MATCH_HANDOFF=yes/no and actual deviations. Distinguish product failure from command-shape friction.