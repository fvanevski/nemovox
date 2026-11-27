# Evidence admission and retryability: final PR remediation qualification

## Authority

Repository fvanevski/nemovox; issue #25, parent #23; existing PR #27, issue/25 -> main.
Previously reviewed head 4e06a49012ec538bd1cb610532090a64b575bb4e (tree 8f5772bd82984aea3e4d10a7b16e7230119b3987); base/main 75d8534b3e3ae12ebf2a50dd76c5fd5e8f5b6227. Original live-qualified production 0965afe3f5d829bfdf8ad1e95f34d0e114fbcd0b. Exact final target, historical instruction/replay commit and blobs are in kickoff. Central review B1/B2: unknown occurrence IDs could be admitted as reusable proposals; provider then returned invalid_nomination forever. Zero budget and permanently unavailable modality/overlong acoustic request similarly became unfinishable acquisition work. The accepted prior positive-budget episode did NOT exercise these failures.

Central corrected actual direct/keyed proposal admission against the full admitted recording, retained-record validation, zero-budget text-only routing (including pipeline provider construction), explicit terminal vs retryable acquisition outcomes, and successor regressions/docs. Valid positive-budget prompt, response schema, provider implementation, comparison geometry, checkpoint version and admitted identity remain unchanged. Already malformed retained nominations fail visibly; fresh translation can regenerate them. The fix does not silently accept malformed requests, force none for bad responses or declare every acquisition failure terminal. No semantic prompt redesign. No new PR.

LOCAL_AGENT_PRIMARY=build
VALIDATION_ROLE=caller-owned
MECHANICAL_REMEDIATION=allowed
SEMANTIC_REMEDIATION=forbidden
REMOTE_UPDATE=forbidden
PR_CREATION=forbidden
DEPLOYMENT=forbidden
GPU_INFERENCE=forbidden

Use native Bash/Git/pytest/npm and authoritative host state. Read AGENTS.md. Serena for bounded symbol inspection/mechanical corrections; Probe supplemental only; RTK only lossless routine output; raw native output for exact IDs/diffs/failures and evidence. OpenViking not authority for mutable state; never store private recording data there. No historic gateway assumption. Do not alter owner checkout, existing jobs/media/models, protected paths, service state, config, weights or install models. No GPU/model service invocation. No substantive implementation or acceptance/test relaxation. Safe command/path/import mechanics and routine environment preparation permitted; preserve exact evidence and rerun affected gates. One narrowly mechanical local commit if truly necessary, no push; Central decides source changes and evidence reuse. Continue independent checks if blocked. Stop on wrong remote head/base, actual unauthorized scope or substantive defect.

## A. Authenticate exact target and baseline (CPU only)

From /workspace/nvidia-transcribe (owner), Bash semantics. Set TARGET_SHA, TARGET_TREE, REPLAY_SHA, REPLAY_BLOB, INSTRUCTIONS_BLOB from kickoff.

    cd /workspace/nvidia-transcribe
    git status --short --branch
    git rev-parse HEAD HEAD^{tree}
    git remote get-url origin
    git fetch origin refs/heads/issue/25
    test "$(git rev-parse FETCH_HEAD)" = "$TARGET_SHA"
    test "$(git rev-parse "$TARGET_SHA^{tree}")" = "$TARGET_TREE"
    git merge-base --is-ancestor "$REPLAY_SHA" "$TARGET_SHA"
    test "$(git rev-parse "$REPLAY_SHA:experiments/EVIDENCE_ADMISSION_QUALIFICATION.md")" = "$INSTRUCTIONS_BLOB"
    test "$(git rev-parse "$REPLAY_SHA:experiments/replay_review_admission.py")" = "$REPLAY_BLOB"
    umask 077
    PRIVATE_ROOT=$(mktemp -d /tmp/nemovox-evidence-admission.XXXXXXXX)
    OWNER=/workspace/nvidia-transcribe
    PY="$OWNER/.venv/bin/python"
    STUDIO_PY="$OWNER/.venv-studio/bin/python"
    BEFORE=4e06a49012ec538bd1cb610532090a64b575bb4e
    git worktree add --detach "$PRIVATE_ROOT/before" "$BEFORE"
    git worktree add --detach "$PRIVATE_ROOT/worktree" "$TARGET_SHA"
    mkdir -p "$PRIVATE_ROOT/evidence/logs" "$PRIVATE_ROOT/replay"
    cd "$PRIVATE_ROOT/worktree"
    git status --porcelain=v1
    git rev-parse HEAD HEAD^{tree}
    git diff --name-status "$BEFORE" HEAD
    git diff --check "$BEFORE" HEAD
    git ls-tree -r HEAD -- experiments
    git show "$REPLAY_SHA:experiments/replay_review_admission.py" > "$PRIVATE_ROOT/replay/replay_review_admission.py"
    test "$(git hash-object "$PRIVATE_ROOT/replay/replay_review_admission.py")" = "$REPLAY_BLOB"

Expected final diff vs BEFORE confined to src/nvidia_transcribe/translation_review.py, subtitle_pipeline.py, cli.py; tests/test_recording_review.py; README.md and docs/model-resources/CONFIGURATION.md. No active experiments/ tree. Compare complete source delta and actual import identities. Verify owner and owned service/GPU state read-only before and after; do not activate any model service. Record native Git full IDs, all raw exit codes; use pipefail when tee. Prior qualified CPU counts 333/49/14 are evidence not count gates.

## B. Focused regression and full deterministic verification

From candidate root:

    PYTHONPATH=src:. "$PY" -m pytest tests/test_recording_review.py tests/test_connected_units.py tests/test_deferred_review.py tests/test_translation_review.py tests/test_resource_integration.py tests/test_resource_lifecycle.py -m 'not gpu'
    PYTHONPATH=src:. "$PY" -m compileall -q src/nvidia_transcribe
    PYTHONPATH=src:. "$PY" -m pytest -m 'not gpu'
    PYTHONPATH=src:. "$STUDIO_PY" -m pytest tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py -m 'not gpu'
    (cd web && npm ci && npm test && npm run build)
    git diff --check 75d8534b3e3ae12ebf2a50dd76c5fd5e8f5b6227 HEAD

Required new tests cover direct invalid reference -> bounded keyed correction, persistently invalid reference -> explicit failure with no durable successful proposal and later corrected continuation, valid reference outside owned batch, zero-budget text-only schema and zero provider calls, all five terminal limitations, transient acquisition failure and positive-budget deferral preserving completed rows, invalid pre-fix checkpoint failing closed. Retain all existing response/atomic/source/lineage/cache/resource invariants. No skipping, weakening or changing asserted meaning. Confirm actual Python/Studio package imports from candidate root with PYTHONPATH=src:. . A failing substantive regression is Central-owned; report it, do not implement around it.

## C. Replay real archived positive-budget paths (CPU only)

Original private archive /tmp/nemovox-resumable-review.jxJ6RE1p/nemovox-issue25-resumable-review-qualification-0965afe3.tar.gz, or an exact relocated copy. Set ARCHIVE absolute path. Verify 7302520 bytes and SHA-256 cacf2df53ad58c724a7d286ab2d165e734421b3fc85430c345480a499e4e4390. No media, model service, actual API or external network calls. Script reads only four authenticated members in memory and uses recorded response/acquisition adapters, never edits archive. First replay BEFORE to establish fidelity to original outputs; then replay target with identical archived inputs and exact request identity, not permissive fixture lookup:

    cd "$PRIVATE_ROOT/before"
    PYTHONPATH=src:. "$PY" "$PRIVATE_ROOT/replay/replay_review_admission.py" \
      --archive "$ARCHIVE" --scratch "$PRIVATE_ROOT/before-replay" --root "$PRIVATE_ROOT/before"
    cd "$PRIVATE_ROOT/worktree"
    PYTHONPATH=src:. "$PY" "$PRIVATE_ROOT/replay/replay_review_admission.py" \
      --archive "$ARCHIVE" --scratch "$PRIVATE_ROOT/target-replay" --root "$PRIVATE_ROOT/worktree"

Expect both PASS: fresh 329 exact rows, 70 recorded text requests, 15 recorded acquisitions, 0 cache hits; continuation 329 exact rows, 4 recorded requests, 3 recorded acquisitions, 325 cache hits; fully complete replay no text or acquisition requests, 329 cache hits. Exact progress-record and request digests, full request messages/schema matching, complete row equality and final progress dictionary equality are required, including stable context snapshots, original proposal reuse and retained observations. Original captured Riva translated vs translated_cached metadata preserved by using the corresponding original input rows. Both scripts compare exact original archive; compare phase results and row/checkpoint digests between baseline and target, ignoring only import path. This reexecutes the actual production reviewer under deterministic recorded responses/observations; it is NOT fresh model inference, a new time benchmark, or proof of novel media acquisition. If baseline cannot reproduce archive, report the precise projection difference (no silent weakening, no invented responses). Mechanical interface/path fixes to private replay tooling allowed only if preserving exact comparisons; no product implementation or changing evidence. If target differs materially from faithful baseline, stop before promotion; do not start GPU to make it green.

## D. Replay already corrected continuation/lineage verifier

The historical corrected verifier at bbee9bf51809749da75c2a8efdee09e49608a050 (blob b622de34f46adad339463070c73908ed5e7e3b5a), lineage verifier blob 1061ba0fd274a815ac151f469647479238796f33, and prior final CPU archive SHA 6f1d51ee24261cbc5a555e265ef07d79740149f2a7f8aa3034800113966deb32 remain valid prior receipts. The new full recorded-response replay checks exact rows and checkpoints plus deferred pipeline validation; unchanged rendering code, original actual cue-lineage and offline receipts support scoped reuse. Do not repeat the previous full offline verifier campaign or render new media; if a replay difference affects lineage/selected text, report it to Central, do not claim reuse. This avoids duplicate qualification.

## E. Handback

Return exact baseline/target/replay commit/tree/blob identities, complete actual diff, raw logs/exit codes and import identities, test/build receipts, authenticated input digest, both replay JSON receipts including actual failures, phase-level equivalence and limitations, final remote refs/PR head/base via native git ls-remote and gh pr view, clean candidate and unchanged owner/service state. No remote push/PR/merge/deployment. Preserve original qualification evidence at its original SHA, do not relabel it. Central decides criterion-level evidence reuse after actual receipts, fresh independent review, then late Codex if clean. No manual Codex request authorized here.

One small mode 0600 private tar.gz under PRIVATE_ROOT, manifest relative path/bytes/SHA-256 and SHA256SUMS, report plus bounded logs. Do not copy original archive, full replay checkpoints, private dialogue, request messages, model weights, full media, secrets, venvs or unrelated logs; never upload private evidence to GitHub. Include no unredacted secrets. Return path, byte size, SHA-256 and attach directly to Central. PASS requires faithful baseline and successor replay, current regression/full gates, exact remote identity and restoration. PARTIAL/FAIL/BLOCKED/STALE as observed; distinguish actual failures from harness friction.

Required canonical friction ledger:

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

Also COMMANDS_MATCH_HANDOFF=yes/no and actual deviations. Distinguish ordinary product/test failures from command-shape friction. No further GPU campaign or new PR.
