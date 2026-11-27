# Recording-wide semantic review: integrated qualification

## Authority and scope

Use the exact issue/25 commit/tree in the Central kickoff. This tracked file controls execution. Repo fvanevski/nemovox, owner checkout /workspace/nvidia-transcribe. Previous qualified architecture, not merge authority: 2b4f7be112fd421d1d2dfb7710ecaa59ceaeba67 (tree 446134f03437b74d4f43dd6ae91568fa971e4017). Main was 75d8534b3e3ae12ebf2a50dd76c5fd5e8f5b6227. Issue #25 under #23, accepted #24 MOSS source. Read AGENTS.md. Central has implemented recording context, discourse hints, adaptive visual/acoustic provider, conditional refinement, decision-specific comparison recovery, CLI/resource wiring, regression tests and qualification runners. Do not redesign locally. Production bilingual default is provisional, not an accepted comparison result. One existing branch, no PR.

Build-capable host integration and bounded mechanical remediation, not strict evidence-only. Select a harness that will not occupy GPU capacity needed by managed Nemovox stages. Use native git/gh, pytest, FFmpeg, systemd, GPU commands for authoritative facts. Serena for bounded symbol/reference navigation and mechanical fixes, Probe only if helpful, RTK only for lossless routine output. Raw native output for SHAs, diffs, failures, lifecycle and evidence. OpenViking not authority for live state; no transcripts/logs/secrets in durable retrieval. Do not presume old operational-schema gateway availability.

Allowed: isolated worktree, private scratch, existing owned resource lifecycle, read-only media/contract admission, narrowly demonstrated mechanical integration fixes (paths/imports/config/format/static issues), exact retirement inventory below. Preserve diff, rationale, regression evidence and actual local SHA. No semantic policy redesign, source observation changes, weakened tests/selection gates, model/engine substitution, downloads, CPU fallback, host-global config edits, changes to existing jobs/library/media/models, unrelated services, sidecars, deploy or #9 layout optimization. Do not modify /workspace/pyannote or the Nemotron checkpoint identified in AGENTS.md. Continue independent work when a dependent step blocks.

## A. Exact checkout and CPU-only qualification

Use Bash semantics and literal candidate workdir. Set TARGET_SHA and TARGET_TREE from kickoff. Read owner state first.

    cd /workspace/nvidia-transcribe
    git status --short --branch
    git rev-parse HEAD
    git remote get-url origin
    git fetch origin refs/heads/issue/25
    test "$(git rev-parse FETCH_HEAD)" = "$TARGET_SHA"
    test "$(git rev-parse "$TARGET_SHA^{tree}")" = "$TARGET_TREE"
    umask 077
    PRIVATE_ROOT=$(mktemp -d /tmp/nemovox-recording-review.XXXXXXXX)
    git worktree add --detach "$PRIVATE_ROOT/worktree" "$TARGET_SHA"
    OWNER=/workspace/nvidia-transcribe
    PY="$OWNER/.venv/bin/python"
    STUDIO_PY="$OWNER/.venv-studio/bin/python"
    cd "$PRIVATE_ROOT/worktree"
    git status --porcelain=v1
    git rev-parse HEAD HEAD^{tree}
    mkdir -p "$PRIVATE_ROOT/evidence/logs" "$PRIVATE_ROOT/inputs"

Do not reset, clean or change the owner checkout. Verify both interpreter and import identities with PYTHONPATH=src:. from candidate root. Run focused tests, then exact retirement inventory D if appropriate, then full deterministic suite on final actual candidate. Preserve raw logs and exit codes; tee must not mask failure.

    PYTHONPATH=src:. "$PY" -m pytest tests/test_recording_review.py tests/test_connected_units.py tests/test_translation_review.py tests/test_model_resources.py tests/test_resource_integration.py tests/test_resource_lifecycle.py -m 'not gpu'
    PYTHONPATH=src:. "$PY" -m compileall -q src/nvidia_transcribe experiments
    PYTHONPATH=src:. "$PY" -m pytest -m 'not gpu'
    PYTHONPATH=src:. "$STUDIO_PY" -m pytest tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py -m 'not gpu'
    (cd web && npm ci && npm test && npm run build)

Prior candidate: 321 non-GPU, 49 Studio, 14 frontend tests. Counts are not gates. Explain legitimate retirement; never remove valid regressions to get green. Material deterministic defects block expensive dependent inference. Mechanical defects may be fixed and rechecked; semantic failures return to Central.

## B. Authenticated inputs and matched language-context comparison

Use existing private NInfer service/resource, Riva/MOSS resources. Authenticate effective user unit/resource config, installed runtime, exact loaded checkpoint identity and bounded immutable checkpoint manifest/digest, context limit, image capability and GPU/ownership before/after. Expected checkpoint kaushikvira/Qwen3.8-27B-swift15-uncensored-ajgazin-nvfp4full-dflash2-NInfer-v3; operator confirms vision works. Prior review resource fingerprint 7c4ddcaf9e48cbcb1e675f1484eef3d9e08aadc9fc6b4a921c8f3f6c0a03cb2a; Riva 51e174c84e775b5a5628be1117811fc126df47af95526fa9645492bda3096210. Balanced/natural, no glossary digest 0b845b4dad8d. Report actual mismatch, no false matched claim. Use actual absolute RESOURCE_CONFIG from NEMOVOX_MODEL_RESOURCES or existing owner config; never archive credentials/private config. Do not stop an unrelated live agent/model.

Authenticate prior private archive (previous host path /tmp/nemovox-connected.WkmrlgcR/nemovox-issue25-connected-integration-2b4f7be1.tar.gz, or relocated actual copy): 613667 bytes, SHA-256 8b5f03fadaf2fba858641eb1588103f73893c6e225d56feed5d9dce9a490eac2. Exact member extraction only:

    tar -xOf "$ARCHIVE" ./outputs/fr/translated.contract.json > "$PRIVATE_ROOT/inputs/fr.translated.json"
    tar -xOf "$ARCHIVE" ./outputs/zh/translated.contract.json > "$PRIVATE_ROOT/inputs/zh.translated.json"
    sha256sum "$PRIVATE_ROOT/inputs/"*.translated.json

FR translated digest 047ed03365eeda341ee8a3e65298c695b7276737d7f849b8b84f74fd95172ba3, 257 events. ZH d65cd9b5d74ee50129fa4d450ac02d048383d7b25c8774ed0963020f2813e752, 329 events. Authenticate retained #24 recording contracts FR affe54d99cbed6c4e23870646a96eca19d723790ac7d011684867192cf38b9c3 and ZH ac577bdebf534a5e89b33b845616fc20b9418e6798fa97a6b43a7a5ebd19de56. Prior root /workspace/nvidia-transcribe/outputs/handbacks/native-promotion-49a7c13f.NFd9iM2U/review_stage/runs with fr_s04e01/recording.contract.json and zh_ep166/recording.contract.json. Resolve actual paths/digests, not broad discovery. FR media SHA 0e77ce3c809e498b97ffe48b53575c29e10d8b08a039b78e2150cd0c845c81b0; ZH a5084539d29cb3fd8a9e2190821315c5fea96290ae27d2420bd83708f457d588. Obtain actual source path from the authenticated previous contract and operator-authorized media location, verify full SHA and selected audio stream; explicitly supply media to pipeline, no arbitrary embedded-path access by production imports. Missing media blocks live media evidence, not text comparison. Missing archive is a real blocker; do not fabricate or substitute baseline.

Run the staged runner from candidate root. All three arms use identical current implementation, model and source/primary rows, evidence disabled. A is new implementation local-only, not the exact old candidate; compare the prior archive separately. Runner expands whole connected groups. No ASR/Riva reruns for this comparison.

    PYTHONPATH=src:. "$PY" experiments/qualify_recording_review.py \
      --contract "$PRIVATE_ROOT/inputs/fr.translated.json" \
      --contract-sha256 047ed03365eeda341ee8a3e65298c695b7276737d7f849b8b84f74fd95172ba3 \
      --model-resources "$RESOURCE_CONFIG" --output-dir "$PRIVATE_ROOT/evidence/fr-context" \
      --target 37:44 71 114:118 125 227:237 253 --modes local english bilingual
    PYTHONPATH=src:. "$PY" experiments/qualify_recording_review.py \
      --contract "$PRIVATE_ROOT/inputs/zh.translated.json" \
      --contract-sha256 d65cd9b5d74ee50129fa4d450ac02d048383d7b25c8774ed0963020f2813e752 \
      --model-resources "$RESOURCE_CONFIG" --output-dir "$PRIVATE_ROOT/evidence/zh-context" \
      --target 82:83 87:103 173:174 191:198 203:205 260:284 --modes local english bilingual

Assess actual source meaning, primary, prior proposal/decision/final, new proposals and decisions, named terms and adverse changes. FR 39–41, 115–116, 231 eleven, 236–237, controls 71/125/227/253; ZH 82–83, 94–103, 173–174 vocative, 191–195 repeated title, 196–198 interruption, 203–205 Nether River, 260–264 idiom, 265–266 honorific/bond, 273–284 former deferred names/idioms. Do not grade by edit counts, exact old English, or same-model self-approval. Distinguish source-text assessment from audio-confirmed facts; record indeterminate cases. Compare source/geometry/receipt, semantic fidelity, recurring entity consistency, malformed/recovered/deferred outcomes. Capture actual NInfer usage (prompt/output/thinking, all retries), elapsed, context coverage/bytes, GPU peak and any available runtime prefix-cache counters; if unavailable mark UNVERIFIED, no server reconfiguration/benchmark campaign. Byte cap is not a tokenizer count. Recommend best context mode per language or no clear winner. Select provisional comprehensive mode for integrated qualification; do not unilaterally change production policy.

## C. Adaptive media evidence and integrated confirmation

Run bounded auto-evidence qualification with selected comprehensive mode on genuine visually/acoustically relevant cases, e.g. ZH 260–277, FR 115–116. Same runner and exact contract digest, new output dir, one --modes value, --adaptive --media "$ZH_MEDIA" or "$FR_MEDIA". Observe actual intelligent nominations; no forced evidence or source alteration. If no useful automatic nomination, retain negative evidence and use explicit diagnostic --probe-visual N --probe-question 'neutral question about actual visible interaction' for a real timed event. --probe-acoustic N only if a genuine spoken-word uncertainty justifies and owned GPU admission allows it. Same-MOSS crop is an alternative observation, not independent recognizer confirmation. Combine probes in bounded adaptive run if useful. Explicit probe demonstrates transport, not successful automatic selection. Record media digest, requested frame seeks and frame hashes, cited frame IDs, observations and uncertainty, model receipts, role identity and ownership, revised proposal and comparison, why evidence helped or not. Small private matching-hash frame samples only if useful/feasible; no media in Git. No separate vision model merely to prove alternate role: deterministic distinct-resource lifecycle regression suffices. If auto does not acquire useful evidence, report limitation, not PASS for semantic utility.

Run normal CLI with explicit media and authenticated retained recording, balanced/natural, auto evidence, no condensation. Prefer full ZH retained (former deferred batch); also run FR if distinct language mode or material FR findings, both allowed if proportionate. Prior EP167/JA are evidence for retained architecture, not final exact-head acceptance; no duplicate full ASR campaign. Example:

    PYTHONPATH=src:. "$PY" -m nvidia_transcribe.cli generate-subtitles \
      "$ZH_MEDIA" --recording-contract "$ZH_CONTRACT" \
      --output-dir "$PRIVATE_ROOT/evidence/zh-integrated" \
      --model-resources "$RESOURCE_CONFIG" --translator riva --translation-review balanced \
      --style natural --source-quality auto --review-context "$ZH_MODE" --fresh
    PYTHONPATH=src:. "$PY" experiments/verify_cue_lineage.py "$PRIVATE_ROOT/evidence/zh-integrated"

Use analogous FR values as warranted. Retain full translated.contract.json and raw subtitles.metadata.json, exact SRT/ASS, verify occurrence_id, real source intervals and selected-unit IDs/membership, complete allocated English, no empty/missing/extra IDs, silent loss or duplication, no invented timing. Independently compare source/primary/geometry against retained inputs; actual SRT/ASS content and ordered coverage, conflicts/overflow/readability, no #9 work. If lineage verifier reports mismatch, investigate against actual schema; don't alter to obtain green. Untimed/overlap/repetition deterministic controls, live limitations explicit. Repeat one successful normal command without --fresh, verify resume/cached evidence and text/unit identity; failed/deferred evidence may legitimately retry, report actual calls. Then offline --re-render with same formats, byte-identical contract and SRT/ASS/lineage, service inactive/no inference. Restore GPU, owned service and owner checkout; no Studio deployment. CPU Studio tests confirm shared child CLI path.

## D. Clean-forward retirement, exact publication boundaries

Primary Central content-commit route cannot delete files atomically. Explicit bounded local deletion inventory, verify each staged blob with git rev-parse HEAD:path and git grep consumers. Only these obsolete experiment-only files:

    8dfb8071d0ff75b7793ebd893f9c34283bb6a099  experiments/connected_translation.py
    a1a8b316768f2475215991fc3469b1622c15c3bd  tests/test_connected_translation_experiment.py
    c976ada38455f00b2ad81706ce36839ca66f2a26  experiments/CONNECTED_TRANSLATION_QUALIFICATION.md
    f48afbe8b2d16c0a71fb288ac38c7a0d816ca3d3  experiments/CONNECTED_TRANSLATION_INTEGRATION.md

After focused regressions pass, git rm only inventory; retire obviously obsolete references with evidence, report any real consumer. Do not remove valid current tests; run final full deterministic checks on resulting candidate (consolidate with A). Retain current qualification runners/handoff for Central inspection. One local mechanical/retirement commit at most; exact diff and commit/tree. Authorize non-force branch advance ONLY for exact inventory deletion and clearly mechanical fixes after successful qualification, owner untouched, remote head re-fetched exactly kickoff SHA, complete diff reviewed, no unexpected files. Native git push origin HEAD:refs/heads/issue/25, no force, fetch/read actual new head/tree. If mismatch, stop, do not rebase or blindly retry. If material semantic repair needed or evidence doesn't support promotion, no push; return diff/local SHA and blockers. Never open PR, merge or deploy. Central decides context default and promotion, final retirement of diagnostic runners, independent review and one PR.

## Handback

One private self-contained tar.gz mode 0600 in collision-safe scratch. Include manifest relative paths/bytes/SHA-256 for staged files, SHA256SUMS, concise report, raw deterministic logs and exit codes, input and model/resource identities, complete per-mode review contracts/reports, case-level comparison (prior, A/B/C proposals/selections, source-supported adjudication, adverse/indeterminate), raw integrated lineage metadata plus SRT/ASS, evidence acquisition receipts and selected bounded private samples, actual source/MT/selected identity, lifecycle and GPU before/after, review/vision/acoustic usage separately, continuation/offline evidence, actual owner/branch/commit/tree, full diff/retirement inventory, criterion acceptance ledger. Exclude full media, weights, secrets, venvs, huge raw logs, unrelated data and archive itself. Do not upload private transcript/media evidence to public repo. Return absolute archive path, bytes, SHA-256 and attach directly to Central ChatGPT; conversational RESULT=PASS/PARTIAL/FAIL/BLOCKED, context mode recommendation per language, adaptive utility (demonstrated or unverified), residual material issues, resource restoration, mutations/PR count, blockers. Preserve actual evidence even on partial failure. Stop for material source/identity/lifecycle violation, unadmitted media/resource, architecture decision, destructive/unrelated scope. No serial handoffs for benign integration issues.

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

Also COMMANDS_MATCH_HANDOFF=yes/no and deviations. Distinguish product failure from actual command/interface friction.
