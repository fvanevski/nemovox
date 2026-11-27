# Evidence-informed English review: final focused qualification

## Authority and role

Repository fvanevski/nemovox, issue #25 under #23, owner /workspace/nvidia-transcribe, branch issue/25. Exact candidate head/tree and this file blob from Central kickoff. Accepted predecessor 62c864f6b215c8fdd07496d6fb109f1c9f3a59ef (tree dd422df4d60dc99b86ffca0fa5d4f5aa29738ac9), main 75d8534b3e3ae12ebf2a50dd76c5fd5e8f5b6227. Read AGENTS.md and current issue. The previous campaign qualified comprehensive bilingual dialogue for French and Mandarin, connected translation, decision-owned recovery, native lineage and real shared NInfer visual/acoustic transport. It did NOT qualify automatic evidence nomination or discourse accumulation: both optional output collections were effectively absent. Central has now staged mandatory explicit dispositions, structured literal source-caption observations, cited competing interpretations, and accepted discourse reuse. Bilingual policy is settled; English-only context remains supported for other model/language needs. The local-only qualification arm is retired. Do not repeat the six-arm comparison or redesign the translation architecture.

LOCAL_AGENT_PRIMARY=build
VALIDATION_ROLE=caller-owned
MECHANICAL_REMEDIATION=allowed
SEMANTIC_REMEDIATION=forbidden
REMOTE_UPDATE=forbidden
PR_CREATION=forbidden
DEPLOYMENT=forbidden

Use a Build-capable local harness that does not compete for the GPU during inference. Native git/gh, pytest, FFmpeg, systemd, GPU and repo-native tools are authoritative. Serena for focused semantic navigation/mechanical edits, Probe only if helpful, RTK for routine verbose output only when lossless; raw native evidence for SHAs/diffs/failures/lifecycle/GPU. OpenViking not authority for mutable state and no private evidence/secrets in it. No presumed historic operational-schema gateway. Allowed isolated worktree/private scratch, current owned lifecycle, read-only source media, bounded actual mechanical integration repairs (paths, imports, installed interface spelling, formatting, static gates) and repeat failed focused checks. No weakening validators, semantic policy changes, hard-coded episode fixes, forced nominations in normal live proof, model/engine replacement, installs/downloads, global config change, CPU fallback, source rewriting, arbitrary caption-to-speech claims, external service eviction, user media/jobs mutation, unrelated repo, #9 rendering campaign, PR/merge/deployment. Never modify /workspace/pyannote or the Nemotron checkpoint protected by AGENTS.md. Preserve actual edits/diff and local commit/tree, no remote push in this campaign. Complete independent work if a phase blocks.

## A. Candidate admission and deterministic checks (CPU only)

Use Bash semantics; literal candidate workdir. Set TARGET_SHA, TARGET_TREE from kickoff. Check owner status/origin before fetch.

    cd /workspace/nvidia-transcribe
    git status --short --branch
    git rev-parse HEAD
    git remote get-url origin
    git fetch origin refs/heads/issue/25
    test "$(git rev-parse FETCH_HEAD)" = "$TARGET_SHA"
    test "$(git rev-parse "$TARGET_SHA^{tree}")" = "$TARGET_TREE"
    umask 077
    PRIVATE_ROOT=$(mktemp -d /tmp/nemovox-evidence-review.XXXXXXXX)
    git worktree add --detach "$PRIVATE_ROOT/worktree" "$TARGET_SHA"
    OWNER=/workspace/nvidia-transcribe
    PY="$OWNER/.venv/bin/python"
    STUDIO_PY="$OWNER/.venv-studio/bin/python"
    cd "$PRIVATE_ROOT/worktree"
    git status --porcelain=v1
    git rev-parse HEAD HEAD^{tree}
    mkdir -p "$PRIVATE_ROOT/evidence/logs" "$PRIVATE_ROOT/inputs"

Verify candidate imports in both interpreters with PYTHONPATH=src:. from candidate root. No owner checkout reset/clean/switch. Record raw logs and exit codes; pipefail if tee. Run focused before GPU work and full suite on final actual candidate; no test-count gate.

    PYTHONPATH=src:. "$PY" -m pytest tests/test_recording_review.py tests/test_connected_units.py tests/test_translation_review.py tests/test_model_resources.py tests/test_resource_integration.py tests/test_resource_lifecycle.py -m 'not gpu'
    PYTHONPATH=src:. "$PY" -m compileall -q src/nvidia_transcribe experiments
    PYTHONPATH=src:. "$PY" -m pytest -m 'not gpu'
    PYTHONPATH=src:. "$STUDIO_PY" -m pytest tests/test_studio_api.py tests/test_studio_export_api.py tests/test_studio_exports.py -m 'not gpu'
    (cd web && npm ci && npm test && npm run build)

Prior counts 319 CPU,49 Studio,14 frontend. Tests cover mandatory explicit none/non-none plus keyed recovery, no extra ordinary call, actual literal caption/frame/occurrence association, wrong language/occurrence/noncaption rejection, caption-informed proposal selection/rejection, replay and rejected-term isolation, shared/distinct model lease and acoustic crop. Any substantive failure is Central-owned; do not change test requirements. Permitted mechanical fixes may be committed once locally for evidence. Run affected regression plus full final suite; continue useful independent tasks without serial handoffs.

## B. Authenticated inputs and existing model (host-bound)

Prior private evidence archive (actual prior path /tmp/nemovox-recording-review.6Np4LGSB/nemovox-issue25-recording-review-qualification-62c864f6.tar.gz, or exact relocated copy): 2,584,356 bytes, SHA-256 c9657f33adda0f9ac63585adf6d6d0f7784b73e04f4fa4dac2cdd5c17024ea98. Authenticate before extraction. Extract only these exact members, not arbitrary tar contents:

    tar -xOf "$ARCHIVE" ./zh-integrated/translated.contract.json > "$PRIVATE_ROOT/inputs/zh.translated.json"
    tar -xOf "$ARCHIVE" ./fr-context/bilingual/reviewed.contract.json > "$PRIVATE_ROOT/inputs/fr.translated.json"
    sha256sum "$PRIVATE_ROOT/inputs/"*.translated.json

ZH full 329 events: 47c431c1d8cde2ad5acccf249ca2aad957a6e666fed79f0cda9c4d7fee212112. FR 257 events: a9e4fa926aeb7ecdead42967ec95a2ba4123271f81e1f6586f721a95650911e6. FR selected previous bilingual output, unselected rows read-only; both retain original primary MT/source. Retained #24 recording contracts: ZH ac577bdebf534a5e89b33b845616fc20b9418e6798fa97a6b43a7a5ebd19de56 and FR affe54d99cbed6c4e23870646a96eca19d723790ac7d011684867192cf38b9c3. Prior root /workspace/nvidia-transcribe/outputs/handbacks/native-promotion-49a7c13f.NFd9iM2U/review_stage/runs with zh_ep166/recording.contract.json and fr_s04e01/recording.contract.json. Verify actual media from operator-authorized existing path, full SHA-256 ZH a5084539d29cb3fd8a9e2190821315c5fea96290ae27d2420bd83708f457d588, FR 0e77ce3c809e498b97ffe48b53575c29e10d8b08a039b78e2150cd0c845c81b0, selected audio stream. Assign absolute ZH_CONTRACT, FR_CONTRACT, ZH_MEDIA, FR_MEDIA, RESOURCE_CONFIG from verified actual paths. Missing input: report precise blocker, no substituted baseline.

Reconfirm same owned NInfer service (nemovox-text-review.service), loaded checkpoint qwen3_8_27b_swift15abl_nvfp4full-dflash2.ninfer, ninfer-serve commit 40dec5ff312809d162cbe41dd5682e054e0bcc00, --max-context 262144 --vision --prefix-reuse. Expected review/visual fingerprint 7c4ddcaf9e48cbcb1e675f1484eef3d9e08aadc9fc6b4a921c8f3f6c0a03cb2a; Riva 51e174c84e775b5a5628be1117811fc126df47af95526fa9645492bda3096210; MOSS 57826061bf43c59bcb2cbf024760a65a52c8117e9f3cabf703af6748b37dc50b. Verify effective runtime and actual configuration/model identity, not historical assumption; avoid repeated huge checkpoint hash if authenticated manifest matches. No separate model setup. GPU and owned service pre/post, no unrelated eviction. No secrets or private config archived. Balanced/natural, no glossary digest 0b845b4dad8d. Record any mismatch; do not call changed input/model a matched comparison.

## C. One focused live campaign (GPU as needed)

Goal: establish actual automatic evidence selection (not merely transport), useful literal captions entering the comparison, and cross-batch accepted discourse reuse. Same bilingual context, exact prior rows, no ASR/Riva recomputation except justified automatically acquired short MOSS crop. Use normal review calls, not forced nominations, modified source text, show-specific prompt injection or a forced screenshot probe as acceptance proof. The runner expands complete connected units. Run:

    PYTHONPATH=src:. "$PY" experiments/qualify_recording_review.py \
      --contract "$PRIVATE_ROOT/inputs/zh.translated.json" \
      --contract-sha256 47c431c1d8cde2ad5acccf249ca2aad957a6e666fed79f0cda9c4d7fee212112 \
      --model-resources "$RESOURCE_CONFIG" --output-dir "$PRIVATE_ROOT/evidence/zh-focused" \
      --target 75:79 82:83 89 96:100 106 128:129 184:186 190:199 203:205 212 223 239 260:267 275:277 282:284 \
      --modes bilingual --adaptive --media "$ZH_MEDIA" --resume-check
    PYTHONPATH=src:. "$PY" experiments/qualify_recording_review.py \
      --contract "$PRIVATE_ROOT/inputs/fr.translated.json" \
      --contract-sha256 a9e4fa926aeb7ecdead42967ec95a2ba4123271f81e1f6586f721a95650911e6 \
      --model-resources "$RESOURCE_CONFIG" --output-dir "$PRIVATE_ROOT/evidence/fr-focused" \
      --target 39:41 71 115:116 125 231 253 --modes bilingual --adaptive --media "$FR_MEDIA" --resume-check

The Mandarin caption case is especially important. Previous independent inspection of frame 3 from actual diagnostic visual request around event-00265 showed 打又打不过, vs MOSS 大人打不过; frame 1 reportedly 一直死心塌地... vs ASR 一只死心塌地. Original MOSS and primary MT must remain unchanged, alternative visible wording must be structured and cited, and any selected English must be evaluated against actual observed caption plus discourse (caption not definitive speech or speaker proof). The previous integrated translation "my lord can't beat him" is not a golden reference; distinguish plausible subject/object and unknowns. Do not force the model to use this particular literal, nominate this ID, or select a specific English phrase. If the new workflow nominates another genuinely useful case, assess it too. Verify real frame hashes and actual visible text independently when feasible using matching private samples; requested seeks are not measured frame PTS. Read actual media for material ambiguous interpretations when necessary. Record whether source caption, acoustic or scene observation actually changed the initial proposal and selected English; receipt must preserve initial proposal, explicit reason, observation IDs and comparative verdict. A visual request that just describes redundant scene context is transport, not proven semantic benefit. If no automatic nomination, or no useful evidence enters adjudication, preserve the negative result; one bounded optional explicit diagnostic probe allowed only if needed to distinguish structural/runtime failure (previous transport already passed). No repeated sampling campaign or episode-specific fixes.

Discourse: test recurring 明皇/明皇宫, 冥河/冥河反噬, 通天/通天河 and French Ambrosius/Mu terminology as supported by actual context; distinguish related terms and genuinely different senses, do not demand arbitrary English spelling. For each repeated concept, show accepted literal source/reference IDs, actual accepted English, later context_receipt.advisory_discourse_terms consumption, final selection, any conflicting senses or inappropriate substitutions. Include actual "backslash" prior error at event-00186, not just term counts. A ledger entry without later consumption does not prove cross-batch utility. Confirm rejected proposals do not seed accepted ledger, no fabricated reference IDs, no canonical changes. Explicit assessments for every successfully reviewed target (both none and non-none as appropriate); empty discourse list is valid where no supported repeated term. Clean controls FR 71/125/231/253 and ordinary ZH lines should not acquire unnecessary media just to increase nomination count. Capture all missing/invalid dispositions, recovery, deferred, evidence failures and initial-to-final comparisons. No self-approval-only grading.

Compare outputs against exact prior integrated/selected baseline and source/evidence, not literal old output. Include complete target-set and case table with outcome and source evidence, any material adverse changes and indeterminate cases. Count auto decision kinds, actual acquisitions by kind/status, caption IDs proposed/selected, accepted terms and later consumption, cache hits/calls/elapsed and actual native usage including visual/acoustic separately; include raw responses/receipts available in staged outputs, no secrets. Check replay equal rows and zero calls for successfully cached selections; legitimate unresolved retry is reported, not mislabeled as cache pass. If evidence-dependent path or meaningful discourse reuse still fails, mark PARTIAL/FAIL as appropriate. Do not loop beyond one bounded repair of a demonstrated mechanical defect, but complete independent checks.

## D. Proportionate normal integrated confirmation

If C establishes meaningful automatic acquisition and discourse reuse with no blocking regression, run ONE ordinary complete retained ZH episode via CLI, explicit media and accepted #24 contract, normal default bilingual and auto evidence, no condensation. No second complete FR episode or fresh ASR campaign; preceding full FR/EP167/JA and current focused French control provide cumulative evidence, not new exact-head fresh ASR qualification. If C is blocked by a semantic failure, do not spend full-episode compute merely to get green. Normal integrated command:

    PYTHONPATH=src:. "$PY" -m nvidia_transcribe.cli generate-subtitles \
      "$ZH_MEDIA" --recording-contract "$ZH_CONTRACT" \
      --output-dir "$PRIVATE_ROOT/evidence/zh-integrated" \
      --model-resources "$RESOURCE_CONFIG" --translator riva --translation-review balanced \
      --style natural --source-quality auto --fresh
    PYTHONPATH=src:. "$PY" experiments/verify_cue_lineage.py "$PRIVATE_ROOT/evidence/zh-integrated"

Verify raw complete metadata, SRT/ASS ordered text coverage, actual source intervals, selected allocations and complete occurrence IDs against retained #24 source and prior primary MT; no invented timing, omissions, duplicate speech, independent speaker merge or source rewrite. Report real caption interpretation and terminology consistency throughout complete episode. Measure runtime and usage/cost including evidence calls, compare prior 750.3s and 0 deferred as context, not exact performance gate. Repeat same normal command without --fresh; assert byte-identical selected contract and SRT/ASS when all decisions reusable, verify no unnecessary calls, separately identify any legitimately retryable unresolved observations. Offline --re-render with same formats must preserve contract, SRT/ASS, lineage with service inactive, no inference. Verify Studio child CLI path and deterministic tests, no production deployment. Record final service/GPU state, owner checkout and unrelated services unchanged.

## E. Publication, cleanup and handback

Four former obsolete files already removed at 62c864f6. Do not recreate them. Current two qualification runners and this file remain until Central review; Central will remove diagnostic-only artifacts before final PR when warranted. Local semantic changes forbidden; one bounded local mechanical commit only if needed. Do not push, PR, merge or deploy. Preserve candidate and actual SHA/tree, full diff, tests and exact head evidence, even if partial. No arbitrary cleanup.

One private self-contained tar.gz mode 0600 in collision-safe scratch. Include manifest relative path/bytes/SHA-256 for every staged file, SHA256SUMS, concise report, exact commands/exit codes and full deterministic logs, authenticated input/runtime identity, complete focused contracts/reports/resume receipts, case-level source-supported table, actual automatic decision and caption evidence receipts, limited matching-hash visual samples (no full media), accepted discourse entries and exact later consumption, cost and lifecycle telemetry, full integrated contract/raw subtitles.metadata.json/SRT/ASS and resume/offline receipts if run, exact local/remote SHA/tree and complete diffs. Exclude full media, weights, secrets, venvs, huge unrelated raw logs and archive itself. Attach archive directly to Central ChatGPT; do not upload private artifacts to GitHub. Return absolute path, bytes, SHA-256, RESULT=PASS/PARTIAL/FAIL/BLOCKED with criterion-level dispositions, material residual errors and blockers, whether meaningful auto acquisition and discourse reuse demonstrated, restoration, actual mutations/PR count. PASS requires actual evidence-supported utility, not just schema compliance/green tests or diagnostic probe. Report missing evidence honestly; no needless serial handoffs.

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

Also COMMANDS_MATCH_HANDOFF=yes/no and actual deviations. Distinguish ordinary product/test failures from invocation/interface friction; preserve both where applicable. No destructive/unrelated scope or architectural improvisation. Central owns final semantic adjudication, exact-head independent review, diagnostic retirement and one PR.
