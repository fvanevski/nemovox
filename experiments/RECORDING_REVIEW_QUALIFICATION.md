# Final review qualification and promotion handback (CPU only)

## Authority and scope

Repository fvanevski/nemovox, issue #25 under #23, branch issue/25, owner checkout /workspace/nvidia-transcribe. Exact final candidate HEAD/tree, archived instruction commit/blob and replay verifier blob are supplied in Central kickoff. Prior live-qualified production head 0965afe3f5d829bfdf8ad1e95f34d0e114fbcd0b (tree d3d0a50afbaed953b083bc73fd717f9bd84abef2); main 75d8534b3e3ae12ebf2a50dd76c5fd5e8f5b6227. Read AGENTS.md. Central accepted current semantic architecture and operational performance, with residual limitations. Do not perform another inference campaign.

The 7,302,520-byte private archive SHA-256 cacf2df53ad58c724a7d286ab2d165e734421b3fc85430c345480a499e4e4390 proves 333 non-GPU tests, 49 Studio, 14 frontend, full integrated fresh and resume with 70+4 text calls, zero comparison recoveries, 325 complete review rows retained, 4 pending resolved, offline lineage and restoration. Independent Central replay authenticated 78 manifest payload entries, 79 checksum records, 29 progress snapshots, 70 retained requests, 15 retained acquisitions and four new owned requests. The unmodified prior verifier flagged all 325 completed rows because Riva records successful cache reuse as translated_cached instead of translated. Central's narrow verifier correction allows only that exact one-field transition; all other row fields, identities and checkpoint checks remain strict. Production source is unchanged. Central independently replayed the predicate and full archive invariants; this host run executes the actual corrected repository verifier and final tree test suite. Residual semantic limitations are documented, including imperfect nomination/adjudication at ZH 74 and 265, and unqualified live cross-batch scene semantic benefit. No new model or prompt changes.

LOCAL_AGENT_PRIMARY=build
VALIDATION_ROLE=caller-owned
MECHANICAL_REMEDIATION=allowed
SEMANTIC_REMEDIATION=forbidden
REMOTE_UPDATE=forbidden
PR_CREATION=forbidden
DEPLOYMENT=forbidden
GPU_INFERENCE=forbidden

Use a fresh Build-capable agent in the owner repo. Native git/gh, Python/pytest, npm, systemctl and nvidia-smi are authoritative. Serena for narrowly scoped source navigation; Probe only supplemental; RTK only for lossless routine output, raw native output for SHA/diff/actual failures and evidence; OpenViking not authority for mutable state and no private data there. No historical operational-schema gateway assumption. No GPU/model invocation, new model installation, global config mutation, source/media/job mutation, service restart, external service eviction, test weakening or semantic development. Never modify /workspace/pyannote or protected Nemotron checkpoint. No remote publication/PR/merge/deployment; Central owns final review and publication. May resolve genuine environment/command/path problems and equivalent formatting/import mechanics in isolated worktree if essential; any source change must be isolated, explicitly recorded, and requalified, never hidden. Return substantive failure to Central, continue independent checks.

## A. Authenticate final candidate and retirement boundary

Use Bash semantics, explicit absolute owner and candidate paths. Set TARGET_SHA, TARGET_TREE, REPLAY_SHA, REPLAY_VERIFIER_BLOB, INSTRUCTIONS_BLOB from kickoff. Before fetch inspect owner status/origin. Commands:

    cd /workspace/nvidia-transcribe
    git status --short --branch
    git rev-parse HEAD HEAD^{tree}
    git remote get-url origin
    git fetch origin refs/heads/issue/25
    test "$(git rev-parse FETCH_HEAD)" = "$TARGET_SHA"
    test "$(git rev-parse "$TARGET_SHA^{tree}")" = "$TARGET_TREE"
    git merge-base --is-ancestor "$REPLAY_SHA" "$TARGET_SHA"
    test "$(git rev-parse "$REPLAY_SHA:experiments/RECORDING_REVIEW_QUALIFICATION.md")" = "$INSTRUCTIONS_BLOB"
    test "$(git rev-parse "$REPLAY_SHA:experiments/verify_review_continuation.py")" = "$REPLAY_VERIFIER_BLOB"
    umask 077
    PRIVATE_ROOT=$(mktemp -d /tmp/nemovox-final-review.XXXXXXXX)
    git worktree add --detach "$PRIVATE_ROOT/worktree" "$TARGET_SHA"
    OWNER=/workspace/nvidia-transcribe
    PY="$OWNER/.venv/bin/python"
    STUDIO_PY="$OWNER/.venv-studio/bin/python"
    cd "$PRIVATE_ROOT/worktree"
    mkdir -p "$PRIVATE_ROOT/evidence/logs" "$PRIVATE_ROOT/replay" "$PRIVATE_ROOT/inputs"
    git status --porcelain=v1
    git rev-parse HEAD HEAD^{tree}
    git diff --name-status 0965afe3f5d829bfdf8ad1e95f34d0e114fbcd0b HEAD
    git diff --exit-code 0965afe3f5d829bfdf8ad1e95f34d0e114fbcd0b HEAD -- src tests web configs systemd-user pyproject.toml
    git ls-tree -r HEAD -- experiments

The final candidate must have removed exactly the four development-only paths (RECORDING_REVIEW_QUALIFICATION.md, qualify_recording_review.py, verify_cue_lineage.py, verify_review_continuation.py under experiments/) and changed only README.md outside them relative to qualified head. The historical staging commit retains corrected verification and these instructions, but neither active package nor suite depends on it. No deletion of host files, jobs, evidence archives or old commits. Check surviving tracked consumers with native git grep; an expected no-match exit 1 is success, an actual consumer blocks promotion:

    git grep -n -E 'qualify_recording_review|verify_review_continuation|verify_cue_lineage|RECORDING_REVIEW_QUALIFICATION|experiments/' HEAD -- . ':!README.md'

Inspect any actual match in context; historical references are not active consumers. Do not choose new cleanup targets. Verify import identities for both interpreters from candidate root; nvidia_transcribe package __file__ must point into worktree/src, any root-level scripts imported by suite must resolve there. Record final diff and exact SHA/tree. Confirm owned service and GPU pre/post state without changing it.

## B. Replay the exact retained private evidence (CPU only)

Locate original archive /tmp/nemovox-resumable-review.jxJ6RE1p/nemovox-issue25-resumable-review-qualification-0965afe3.tar.gz or exact relocated copy. Verify byte size 7302520, exact SHA above before extraction. Assign ARCHIVE absolute path. Extract only the needed files, never arbitrary tar contents:

    git show "$REPLAY_SHA:experiments/verify_review_continuation.py" > "$PRIVATE_ROOT/replay/verify_review_continuation.py"
    git show "$REPLAY_SHA:experiments/verify_cue_lineage.py" > "$PRIVATE_ROOT/replay/verify_cue_lineage.py"
    test "$(git hash-object "$PRIVATE_ROOT/replay/verify_review_continuation.py")" = "$REPLAY_VERIFIER_BLOB"
    for phase in zh-fresh zh-resumed; do
        mkdir -p "$PRIVATE_ROOT/inputs/$phase"
        for name in translated.contract.json translation-review.checkpoint.json subtitles.metadata.json; do
            tar -xOf "$ARCHIVE" "./$phase/$name" > "$PRIVATE_ROOT/inputs/$phase/$name" || exit 1
        done
    done
    tar -xOf "$ARCHIVE" ./MANIFEST.json > "$PRIVATE_ROOT/inputs/MANIFEST.json"
    PYTHONPATH=src:. "$PY" - "$PRIVATE_ROOT/inputs" <<'PYTHON'
    import hashlib,json,sys
    from pathlib import Path
    root=Path(sys.argv[1])
    records={x["path"].removeprefix("./"):x for x in json.loads((root/"MANIFEST.json").read_text())["files"]}
    for phase in ("zh-fresh","zh-resumed"):
        for path in (root/phase).iterdir():
            rel=str(path.relative_to(root))
            record=records[rel]
            assert path.stat().st_size==record["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record["sha256"]
    print("REPLAY_INPUT_DIGESTS=PASS")
    PYTHON
    PYTHONPATH=src:. "$PY" "$PRIVATE_ROOT/replay/verify_review_continuation.py" \
      --fresh-contract "$PRIVATE_ROOT/inputs/zh-fresh/translated.contract.json" \
      --resumed-contract "$PRIVATE_ROOT/inputs/zh-resumed/translated.contract.json" \
      --fresh-progress "$PRIVATE_ROOT/inputs/zh-fresh/translation-review.checkpoint.json" \
      --resumed-progress "$PRIVATE_ROOT/inputs/zh-resumed/translation-review.checkpoint.json"
    PYTHONPATH=src:. "$PY" "$PRIVATE_ROOT/replay/verify_cue_lineage.py" "$PRIVATE_ROOT/inputs/zh-fresh"
    PYTHONPATH=src:. "$PY" "$PRIVATE_ROOT/replay/verify_cue_lineage.py" "$PRIVATE_ROOT/inputs/zh-resumed"

Record output/exit codes. Expected: continuation PASS, 329 occurrences, 4 original pending IDs event-00018/19/20/00303, remaining pending empty, changed_row_ids those 4, unexpected empty, cache_status_transition_row_ids 325, 29 unchanged snapshots, 70 original requests retained, 4 new requests only owned pending, 325 cache hits, 4 calls. Lineage both 329/329, fresh 339 records, resumed 338 records. This is replay of immutable outputs, not live generation.

Run the following negative controls directly against the actual extracted verifier, in memory only (do not mutate archived inputs). Exact cache transition is accepted, but semantic/provenance changes and fallback status still fail. Full audit on modified in-memory row must report the changed completed ID.

    PYTHONPATH=src:. "$PY" - "$PRIVATE_ROOT" <<'PYTHON'
    import copy,runpy,sys
    from pathlib import Path
    root=Path(sys.argv[1])
    v=runpy.run_path(str(root/"replay/verify_review_continuation.py"))
    f=v["read"](root/"inputs/zh-fresh/translated.contract.json")
    r=v["read"](root/"inputs/zh-resumed/translated.contract.json")
    old=v["records"](root/"inputs/zh-fresh/translation-review.checkpoint.json")
    new=v["records"](root/"inputs/zh-resumed/translation-review.checkpoint.json")
    pending=v["pending_ids"](f)
    i=next(i for i,x in enumerate(f["utterances"]) if x["id"] not in pending)
    a,b=f["utterances"][i],r["utterances"][i]
    same=v["same_completed_occurrence"]
    assert a["translation_status"]=="translated" and b["translation_status"]=="translated_cached"
    assert same(a,a) and same(a,b) and not same(b,a)
    for key,value in (("translation_status","fallback_source_after_error"),("translated_text","invented"),
                      ("machine_translated_text","invented"),("source_text","invented"),
                      ("speaker_id","invented"),("translation_review_reason","invented"),
                      ("translation_unit",None),("unexpected_field",True)):
        assert not same(a,{**b,key:value}),key
    corrupt=copy.deepcopy(r)
    corrupt["utterances"][i]["translation_review_reason"]="unapproved semantic review change"
    receipt=v["audit"](f,corrupt,old,new)
    assert receipt["status"]=="FAIL"
    assert "completed_row_changed:"+a["id"] in receipt["errors"]
    print("NARROW_TRANSITION_NEGATIVE_CONTROLS=PASS")
    PYTHON

Do not waive the old failed verifier exit or replace old evidence. The replay receipts are a new, separate validation result.

## C. Final exact-head CPU and Studio/frontend verification

No model service start. Run from candidate root, with raw logs/exit codes (pipefail if tee).

    PYTHONPATH=src:. "$PY" -m compileall -q src/nvidia_transcribe
    PYTHONPATH=src:. "$PY" -m pytest -m 'not gpu'
    PYTHONPATH=src:. "$STUDIO_PY" -m pytest tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py -m 'not gpu'
    (cd web && npm ci && npm test && npm run build)
    git diff --check 75d8534b3e3ae12ebf2a50dd76c5fd5e8f5b6227 HEAD
    git status --porcelain=v1
    git rev-parse HEAD HEAD^{tree}

Existing full suite includes focused successor regressions; no duplicate focused suite required unless diagnosing failure. Prior 333/49/14 are evidence, not test-count gates. No ordinary inference, no additional fresh episode, no repeated offline generation necessary. If genuine mechanical fix needed, preserve exact diff and rerun affected/full gate; no remote push. A source or behavior change invalidates production-equivalence claim; return it for Central review rather than silently inheriting prior live evidence. Recheck live issue/25 branch at handback with git ls-remote origin refs/heads/issue/25 and main; any movement means STALE, no remote mutation. Owner and owned service state unchanged. Do not restart, deploy or merge.

## D. Handback (private, bounded)

Return one conversational receipt plus one small mode-0600 tar.gz containing report, exact final/qualified/replay SHA/tree/blob identities, complete final diff inventory, exact import/tool/runtime identity and raw logs/exit codes for archive authentication, continuation/negative controls/lineage replay, all deterministic tests/build, diff check, owner/remote and service pre/post state. Manifest relative paths/bytes/SHA-256 for all payload and SHA256SUMS. Include concise criterion dispositions: qualified product source unchanged, development files absent and no live consumers, corrected verifier PASS with exactly 325 legitimate metadata transitions and no unexpected completed-row changes, negative controls PASS, current candidate deterministic gates PASS, resources/owner unchanged. Reference original immutable archive by digest; no need to duplicate its private contracts, progress, captions or media, or copy model logs/weights. Exclude secrets, venvs, original media and archive itself. Return path, byte count, SHA-256; attach directly to Central. No GitHub evidence upload. Report PASS/PARTIAL/FAIL/BLOCKED/STALE and exact blockers. Central retains independent final PR review and publication. Prior live performance and residual semantic limitations are not newly measured here; no claim that every translation is correct.

Required command-shape friction ledger:

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

Also COMMANDS_MATCH_HANDOFF=yes/no, explain actual deviations. Distinguish ordinary product/test failures from command shape friction. Stop on unauthorized scope or source/identity mismatch; complete independent permitted checks. No additional PR.
