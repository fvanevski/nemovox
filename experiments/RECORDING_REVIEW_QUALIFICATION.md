# Efficient, resumable audiovisual semantic review: qualification

## Authority and scope

Repository fvanevski/nemovox, issue #25 under #23, owner /workspace/nvidia-transcribe, branch issue/25. Exact candidate head, tree and this file blob are supplied in the Central kickoff. Prior qualified-but-operationally-blocked head 8ab0fdc2721ed5f3110c6a5c43cc60ffcf54e9a3, tree ac80b8f2f77a997a434a0674653778792f8a12d5; main 75d8534b3e3ae12ebf2a50dd76c5fd5e8f5b6227. Read AGENTS.md and current issue. Accepted architecture: native MOSS occurrences, Riva primary and connected hypotheses, comprehensive bilingual review (FR/ZH), adaptive acquisition and source-cited discourse, current NInfer shared vision/text role. Previous fresh episode 2060.5s, 140 text calls (74 comparative recovery), 4 deferred comparisons and 11 supplement-error rows; continuation 1200.09s, 120 calls (73 recovery), 90/329 cache hits, 121 changed rows outside reported unfinished set and a new deferred occurrence. Prior comprehensive bilingual reference 750.3s, 71 calls, 13 recovery, zero deferred. These are measured comparators, not universal speed targets.

Central owns all substantive implementation. New native progress records authenticate immutable input and exact admitted discourse/scene snapshot, persist proposals and acquisition before further stages, persist exact keyed comparative decisions and refine only owned complete linguistic units. Completed decisions are not invalidated by newly accumulated advisory context. Ordinary new batches consume current bounded advisory context; continuing admitted batches preserve their actual snapshot. New checkpoint version is clean-forward; do not migrate old v1 cache. General scene observations are categorized, separate direct descriptions/interpretations/uncertainty, frame-bound, task-adaptive sampling, bounded nearby scene reuse. Shared NInfer default retained. No new inference pass for ordinary adequate cues.

LOCAL_AGENT_PRIMARY=build
VALIDATION_ROLE=caller-owned
MECHANICAL_REMEDIATION=allowed
SEMANTIC_REMEDIATION=forbidden
REMOTE_UPDATE=forbidden
PR_CREATION=forbidden
DEPLOYMENT=forbidden

Use Build-capable local agent, preferably non-competing inference environment for RTX 5090 stages. Native git/gh, pytest, FFmpeg, systemd, GPU and repo-native commands authoritative. Serena for bounded symbol/reference inspection and mechanical repairs; Probe supplemental only; RTK for lossless routine output, raw native output for SHA/diff/failure/lifecycle/GPU; OpenViking not authority for live state and no private evidence/secrets in it. Do not assume historic operational-schema gateway. Allowed: isolated candidate and private scratch, owned model lifecycle, read-only admitted media/contracts, narrowly demonstrated value-preserving mechanical corrections (paths/imports/formatting, actual host interface spelling only when demonstrably equivalent). Preserve exact full diff and local SHA/tree; one bounded mechanical commit if necessary, no push. No semantic refactoring, weakening validation or tests, force-nomination, show-specific hardcoding, new model/serving setup or downloads, CPU fallback, global config changes, source rewrites, external service eviction, media/job mutation, #9 scheduling work, unrelated repository changes, PR/merge/deployment. Never modify /workspace/pyannote or protected Nemotron checkpoint in AGENTS.md. Continue independent tasks when another phase blocks.

## A. Exact candidate admission and deterministic fault qualification (CPU only)

Use Bash semantics and a literal candidate workdir. Set TARGET_SHA, TARGET_TREE from kickoff. Read owner state before fetch.

    cd /workspace/nvidia-transcribe
    git status --short --branch
    git rev-parse HEAD
    git remote get-url origin
    git fetch origin refs/heads/issue/25
    test "$(git rev-parse FETCH_HEAD)" = "$TARGET_SHA"
    test "$(git rev-parse "$TARGET_SHA^{tree}")" = "$TARGET_TREE"
    umask 077
    PRIVATE_ROOT=$(mktemp -d /tmp/nemovox-resumable-review.XXXXXXXX)
    git worktree add --detach "$PRIVATE_ROOT/worktree" "$TARGET_SHA"
    OWNER=/workspace/nvidia-transcribe
    PY="$OWNER/.venv/bin/python"
    STUDIO_PY="$OWNER/.venv-studio/bin/python"
    cd "$PRIVATE_ROOT/worktree"
    git status --porcelain=v1
    git rev-parse HEAD HEAD^{tree}
    mkdir -p "$PRIVATE_ROOT/evidence/logs" "$PRIVATE_ROOT/inputs"

Verify both interpreter/import identities with PYTHONPATH=src:. from candidate. Do not switch/reset/clean owner. Run raw logs and actual exit codes (set -o pipefail if tee).

    PYTHONPATH=src:. "$PY" -m pytest tests/test_recording_review.py tests/test_deferred_review.py tests/test_connected_units.py tests/test_translation_review.py tests/test_model_resources.py tests/test_resource_integration.py tests/test_resource_lifecycle.py -m 'not gpu'
    PYTHONPATH=src:. "$PY" -m compileall -q src/nvidia_transcribe experiments
    PYTHONPATH=src:. "$PY" -m pytest -m 'not gpu'
    PYTHONPATH=src:. "$STUDIO_PY" -m pytest tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py -m 'not gpu'
    (cd web && npm ci && npm test && npm run build)

Prior counts 325 CPU,49 Studio,14 frontend; counts not gates. Focused fault regressions must prove: malformed duplicate/missing keyed verdict rejection, grouped recovery instead of immediate per-decision fanout, no proposal regeneration after early deferred comparison, same actual downstream snapshots and exact completed-row equality when early term becomes accepted, failed evidence refinement confined to its whole atomic unit with successful acquisition retained across fresh provider instance, no extra calls for explicit none, retryable transport does not fan out, current-version corruption rejected, scene sampling and nearby context projection, complete preserved primary for mixed pending stages, prior connected/uncertainty/caption/resource invariants. No live phase until relevant deterministic tests pass. Mechanical integration fixes only as expressly allowed; preserve raw failure and complete diff, rerun focused plus full deterministic gates on actual final local SHA. Substantive failures are Central-owned; do not implement around them.

## B. Authenticate matched inputs and runtime

Latest prior private archive (previous actual host path /tmp/nemovox-evidence-review.8j4w2LnT/nemovox-issue25-evidence-discourse-qualification-8ab0fdc2.tar.gz or exact relocated copy), 1,423,042 bytes SHA-256 476abb65c0058fb7b608261d70ff651fba8fad4e524ab5bad05ece52ef4c2298. Authenticate before extraction. Exact members:

    tar -xOf "$ARCHIVE" ./zh-integrated/translated.contract.json > "$PRIVATE_ROOT/inputs/zh.translated.json"
    tar -xOf "$ARCHIVE" ./fr-focused/bilingual/reviewed.contract.json > "$PRIVATE_ROOT/inputs/fr.translated.json"
    sha256sum "$PRIVATE_ROOT/inputs/"*.translated.json

ZH 329 events digest beb81468e62bea427f3c7fcc91c9ff7e7dca0f0be29f9242947f05a9fcd8ee47; FR 257 events digest 37d081e67adc20f97ab7c8b9ac43e5bf39a954a0b9ce08bc42315899b9a13434. These are latest private reviewed contracts with original primary/source rows; runner resets only selected targets to primary and uses full recording context. For prior performance/semantic comparisons inspect the actual archive reports/logs, including fresh vs continuation. Retained #24 recording contracts ZH ac577bdebf534a5e89b33b845616fc20b9418e6798fa97a6b43a7a5ebd19de56 and FR affe54d99cbed6c4e23870646a96eca19d723790ac7d011684867192cf38b9c3. Prior root /workspace/nvidia-transcribe/outputs/handbacks/native-promotion-49a7c13f.NFd9iM2U/review_stage/runs with zh_ep166/recording.contract.json and fr_s04e01/recording.contract.json. Authenticate actual operator-authorized media full SHA ZH a5084539d29cb3fd8a9e2190821315c5fea96290ae27d2420bd83708f457d588, FR 0e77ce3c809e498b97ffe48b53575c29e10d8b08a039b78e2150cd0c845c81b0, selected audio stream. Set absolute ZH_CONTRACT, FR_CONTRACT, ZH_MEDIA, FR_MEDIA, RESOURCE_CONFIG using verified actual paths. Missing data: precise blocker, no substitute invented baseline.

Reconfirm owned NInfer nemovox-text-review.service, loaded qwen3_8_27b_swift15abl_nvfp4full-dflash2.ninfer, ninfer-serve commit 40dec5ff312809d162cbe41dd5682e054e0bcc00, --max-context 262144 --vision --prefix-reuse, actual review/visual fingerprint 7c4ddcaf9e48cbcb1e675f1484eef3d9e08aadc9fc6b4a921c8f3f6c0a03cb2a, Riva 51e174c84e775b5a5628be1117811fc126df47af95526fa9645492bda3096210, MOSS 57826061bf43c59bcb2cbf024760a65a52c8117e9f3cabf703af6748b37dc50b. Recheck effective runtime/resource config and actual model identity; existing authenticated immutable checkpoint manifest sufficient, no huge repeated weights hash. GPU pre/post, owned service only, no unrelated eviction. No credentials/private config in archive. Balanced/natural, no glossary digest 0b845b4dad8d. Report mismatches; no false matched claim.

## C. Bounded real review qualification (GPU as needed)

Use one Mandarin workload covering the actual problematic batches (early 43–44, 66–76 supplement, 203–214, 250–258, 262–266), terminology and caption cases, plus one French control. This is a targeted workload with potentially different packed batches, not a controlled repetition of the old 57-row selection; compare stage mechanics and per-case semantics, report actual selected IDs and boundary differences, normalize useful telemetry rather than claim exact matched speedup. No extra ASR/Riva except nominated short crop. Run once each:

    PYTHONPATH=src:. "$PY" experiments/qualify_recording_review.py \
      --contract "$PRIVATE_ROOT/inputs/zh.translated.json" \
      --contract-sha256 beb81468e62bea427f3c7fcc91c9ff7e7dca0f0be29f9242947f05a9fcd8ee47 \
      --model-resources "$RESOURCE_CONFIG" --output-dir "$PRIVATE_ROOT/evidence/zh-focused" \
      --target 43:53 66:76 100 106 184:186 190:195 203:214 250:258 262:266 \
      --modes bilingual --adaptive --media "$ZH_MEDIA" --resume-check \
      --probe-visual 265 --probe-focus speech_activity \
      --probe-question "Describe the visible participants, their actions and interactions across the ordered frames, including visible mouth activity only if observable; transcribe any legible dialogue captions."
    PYTHONPATH=src:. "$PY" experiments/qualify_recording_review.py \
      --contract "$PRIVATE_ROOT/inputs/fr.translated.json" \
      --contract-sha256 37d081e67adc20f97ab7c8b9ac43e5bf39a954a0b9ce08bc42315899b9a13434 \
      --model-resources "$RESOURCE_CONFIG" --output-dir "$PRIVATE_ROOT/evidence/fr-focused" \
      --target 39:41 71 115:116 125 231 253 --modes bilingual --adaptive --media "$FR_MEDIA" --resume-check

Explicit ZH diagnostic is one real scene-understanding/denser sampling transport control AFTER ordinary review; it is NOT automatic nomination evidence and must not be injected into the ordinary prompt or checkpoint. No forced caption correction or show-specific hints. Verify actual visible observations category, description vs interpretation vs uncertainty, exact frame IDs/hashes/requested seeks, focus and actual sampling geometry, participant/action/interaction/speech-related observations when relevant and visible. Mouth activity is not measured lip alignment or speaker proof. No exhaustive mandatory inventory per request. Show actual automatic evidence receipts and whether prior nearby scene observation is supplied/reused in later relevant context; deterministic reuse regression proves mechanics even if no naturally occurring downstream use in selected cases. Do not claim live non-caption linguistic improvement unless actual case supports it. Include genuine literal captions 74 (择贤/钦承前绪) and 265 (打又打不过) where independently observed; original ASR immutable. Prior case 100 acoustic confirms wording but not proper-name sense; assess modality suitability, avoid repeated audio merely to resolve semantics. Preserve recurring 明皇/冥河/通天 interpretations only as supported; no forced spellings. Clean FR controls should not acquire unnecessary media. Compare actual initial/selected English, source, frame evidence and adverse changes; previous model verdict is not truth. Record indeterminate outcomes.

Report exact logical calls and actual native completion attempts by stage (proposal, proposal recovery, evidence refinement, initial comparison, grouped comparison recovery), malformed output reason, group width, time, tokens/cache reads, actual provider_requests, visual/acoustic attempts and elapsed, in-memory and checkpoint reuses, final unresolved atomic IDs. Distinguish receipt count from real new acquisitions. Record progress file bytes/CPU overhead if material; capture raw staged request receipts and actual output. No alternate models or repeated sample campaign. One bounded retry of a demonstrated mechanical defect allowed if appropriate, independent work continues. If focused continuation changes any complete row, reports widespread new unresolved work, fails schema/semantic invariants, or recovery volume remains disproportionate to the selected workload, STOP before full episode and return evidence; do not burn another full episode merely to reach green.

## D. One integrated episode only after focused qualification

If A/C support useful automatic evidence, scene transport, preserved discourse/semantic gains and corrected operational behavior, run ONE normal complete retained Mandarin episode (no fresh ASR, no second FR episode), existing normal default bilingual/auto, no condensation.

    PYTHONPATH=src:. "$PY" -m nvidia_transcribe.cli generate-subtitles \
      "$ZH_MEDIA" --recording-contract "$ZH_CONTRACT" \
      --output-dir "$PRIVATE_ROOT/evidence/zh-integrated" \
      --model-resources "$RESOURCE_CONFIG" --translator riva --translation-review balanced \
      --style natural --source-quality auto --fresh
    PYTHONPATH=src:. "$PY" experiments/verify_cue_lineage.py "$PRIVATE_ROOT/evidence/zh-integrated"

Capture fresh complete translated contract, raw metadata, actual SRT/ASS, current progress checkpoint before continuing; use private directory, no overwrite of prior evidence. Example Bash:

    INTEGRATED="$PRIVATE_ROOT/evidence/zh-integrated"
    mkdir -p "$PRIVATE_ROOT/evidence/zh-fresh"
    cp "$INTEGRATED/translated.contract.json" "$INTEGRATED/subtitles.metadata.json" \
       "$INTEGRATED/translation-review.checkpoint.json" "$PRIVATE_ROOT/evidence/zh-fresh/"
    cp "$INTEGRATED/"*.srt "$INTEGRATED/"*.ass "$PRIVATE_ROOT/evidence/zh-fresh/"
    sha256sum "$PRIVATE_ROOT/evidence/zh-fresh/"* > "$PRIVATE_ROOT/evidence/logs/zh_fresh.sha256"

Run same normal command once without --fresh. Then:

    PYTHONPATH=src:. "$PY" experiments/verify_review_continuation.py \
      --fresh-contract "$PRIVATE_ROOT/evidence/zh-fresh/translated.contract.json" \
      --resumed-contract "$INTEGRATED/translated.contract.json" \
      --fresh-progress "$PRIVATE_ROOT/evidence/zh-fresh/translation-review.checkpoint.json" \
      --resumed-progress "$INTEGRATED/translation-review.checkpoint.json"
    PYTHONPATH=src:. "$PY" experiments/verify_cue_lineage.py "$INTEGRATED"

Verifier requires exact completed rows/stages, unchanged canonical/primary, same original context snapshots, all prior successful acquisitions unchanged, no new proposal or out-of-unit comparison/refinement on continuation, no new pending IDs and no inference when already complete. If genuine unresolved remain after one continuation, report PARTIAL (even if invariant verifier PASS); don't chain full retries. If fresh has required-review errors rather than explicitly deferred primary-preserved stages, stop and return diagnostic evidence; no blind full retry. Verify SRT/ASS ordered complete text coverage and real source intervals, selected-unit membership, raw lineage; compare actual elapsed and stage costs with previous 2060.5s fresh +1200.09s resume and 750.3s earlier efficient reference. Expect material reduction in unnecessary recovery/continuation, not a hard speed threshold independent of useful acquisitions. Monitor actual progress and resource ownership; if clearly repeating the former prolonged failure pattern, stop cleanly at a safe owned lifecycle boundary with partial evidence rather than launch another repeated campaign. If near completion, preserve finished evidence. Do not stop unrelated services. Report actual quality/cost tradeoffs and any residual gaps. Preserve resumed contract/metadata/SRT/ASS/progress before offline --re-render, then offline re-render same formats, verify byte-identical contract/SRT/ASS and equal lineage, service inactive, zero inference. Restore original owned service/GPU and owner state; no Studio deployment.

## E. Handback and publication

No remote push, PR, merge or deployment. Four obsolete experiment files already retired; don't recreate. Current development runners and this handoff remain until Central independent inspection; Central owns later clean-forward removal and single PR. Local mechanical commit only if necessary, complete diff and actual SHA/tree; preserve candidate and exact evidence even on partial. No arbitrary cleanup.

One private self-contained tar.gz mode 0600 under collision-safe scratch, manifest relative paths/bytes/SHA-256, SHA256SUMS, concise REPORT and case table. Include raw deterministic logs/exit codes, input/runtime identity, exact candidate/owner/remote SHA/tree, complete focused contracts/reports/resume receipts and current progress checkpoints, real visual/acoustic evidence and limited matching-hash screenshots, case-level source/context adjudication, scene/discourse consumption, separate telemetry for fresh acquisitions/reuses and model stages, full integrated fresh AND resumed contract/raw lineage/SRT/ASS/progress, continuation verifier and offline receipts when executed, resource restoration, actual diffs. Current progress includes private recording dialogue; never upload to GitHub or durable retrieval. Exclude full media, weights, secrets, venvs, huge unrelated raw logs and archive itself. Return absolute path, bytes, SHA-256 and attach directly to Central ChatGPT. Conversational RESULT=PASS/PARTIAL/FAIL/BLOCKED with criterion dispositions: deterministic, keyed structure/recovery, completed-stage immutability, acquired observation reuse, actual general scene transport and any demonstrated semantic utility, retained bilingual/discourse/caption quality, integrated runtime/continuation/lineage/offline, restoration. PASS requires actual corrected continuation and proportionate full-episode behavior, not merely successful exit or retained primary fallback. No invented timing/semantic facts or repeated expensive attempts to manufacture PASS. Explain blockers, mechanical deviations and unresolved evidence.

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

Also COMMANDS_MATCH_HANDOFF=yes/no and actual deviations. Distinguish ordinary product/test failure from invocation/interface friction. Central owns final adjudication, any substantive repair, exact-head independent review, diagnostic retirement and one implementation PR.
