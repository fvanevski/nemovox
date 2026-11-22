# Issue #12 / PR #13: integrated source-bound Studio output-workflow qualification

## Mission

Carry Central-authored Issue #12 candidate from the existing GitHub development branch
through **one consolidated** deterministic CPU/frontend -> isolated live Studio/browser
-> safe real French GPU -> complete source-bound evidence campaign. Do not open a
second implementation PR, redesign the application, split into micro-handoffs or
promote to production before the independent PR and merge gates.

Central has **already supplied** substantive production work:
- src/nvidia_transcribe/studio_exports.py: explicit host export allowlist,
  task-selected copied artifacts, safe collision handling, idempotent job-specific
  destination, manifest, early remote-media export;
- studio_api.py: ExportPlan persistence, export-directory discovery,
  auto-export, separate processing/export outcome, typed stage persistence,
  full-log chunk and download routes, job-history pagination, remote artifacts;
- subtitle_pipeline.py + cli.py: actual translation, English review and render
  stage events through the existing **single** CLI invocation, without extra
  GPU/model passes;
- web/src/App.tsx, api.ts, export-options.ts: destination picker, task-aware
  checkbox selection and remote-source option, stage panel, persistent/copyable
  logs, export report, paginated older jobs and advanced internal files;
- new CPU/TS regressions and docs/gui/STUDIO_OUTPUTS.md.

No host tests or live browser tests have yet been executed by Central.
This is a **draft** implementation candidate, not a prequalified deployment.

### Immutable start authority and guardrails

REPO=fvanevski/nemovox
ISSUE=12
EXISTING_PR=13 (DRAFT)
EXISTING_BRANCH=issue/12-studio-output-workflow
INITIAL_SOURCE_BRANCH_HEAD_BEFORE_THIS_HANDOFF=12d192bfb5ceeb4fffbce26eb06c3ec977d41375
BASE_MAIN=332a99697edcc0b72fcc842168d5b1f57782e203
LOCAL_PRIMARY=/workspace/nvidia-transcribe (inspect; don't reset or replace)
EXPECTED_CANDIDATE_PORT=127.0.0.1:8766 (choose another free loopback port if needed)
PRODUCTION_STUDIO=nemovox-studio.service, 127.0.0.1:8765, READ_ONLY
MODEL_UNITS=nemovox-asr.service, nemovox-text-review.service
GPU=RTX 5090 single shared resource; canonical existing lifecycle lock only
HOST_AGENT=OpenCode build
LOCAL_HARNESS_ENVIRONMENT=host Python/.venv + .venv-studio, Node, Playwright Chromium;
GPU only when established lease/service admission permits
SOURCE_REMOTE_MUTATION=ONLY bounded mechanical corrections on same branch
EVIDENCE_REMOTE_MUTATION=ONE evidence-only commit on same branch when stable
NEW_PR=forbidden
MERGE=forbidden
MAIN_WRITE=forbidden
PRODUCTION_PROMOTION=forbidden
ISSUE_CLOSE=forbidden
THIRD_PARTY_GPU_SERVICE_MUTATION=forbidden
MEDIA_CHECKPOINT_WEIGHTS_COPY_OR_DOWNLOAD=forbidden
TEST_WEAKENING_OR_SKIP_GATE=forbidden

The instruction commit itself will be an additive handoff file on this
existing branch, so fetch the actual **full 40-character remote head** from
GitHub at startup and match the kickoff's immutable SHA. Do not assume that
12d192b above is still the branch head. Any subsequent authorized mechanical
fix must be committed non-force, rebind all qualification to the new exact
source SHA and return the complete diff.

Read the current Issue #12, PR #13, repo AGENTS.md, relevant current
src/tests/web files and docs/gui/STUDIO_OUTPUTS.md. Keep Issue #9's subtitle
renderer/readability scope separate.

## Phase A. Exact-source materialization and CPU/browser-build qualification

1. Inspect authenticated origin refs / base SHA / PR head, local checkout
   status and host runtime ownership. Fetch the remote issue branch **before**
   attempting local object proof. Never force-reset, stash, clean or overwrite
   the operator's production checkout or editable environments. Create a
   collision-safe clean **separate** detached candidate worktree at the
   fetched exact source commit (or an equivalently isolated existing worktree).
   Record Git tree, changed-path inventory and Python import origins. The
   trusted candidate source must not accidentally import the production
   editable checkout through .venv.
2. Use the existing validated Python interpreters, not a fresh heavy GPU
   dependency install. Force PYTHONPATH=<candidate>/src for CPU tests
   (including CLI child when isolated candidate Studio is run). In this
   specific host, .venv editable installation historically resolves to
   /workspace/nvidia-transcribe/src, so prove imports from **candidate**.
   Use .venv-studio for FastAPI tests if that environment has the actual
   dependencies; if one environment skips the Studio tests, run those tests
   in the existing compatible environment rather than treating the skip as PASS.
3. Required deterministic execution and exit evidence:
   - python -m compileall -q src tests experiments
   - python -m pytest -q tests/test_studio_exports.py tests/test_studio_export_api.py
     tests/test_studio_api.py tests/test_resource_integration.py
     tests/test_translation.py tests/test_translation_review.py
     tests/test_subtitles.py
   - python -m pytest -q -m 'not gpu'
   - web/: npm test ; npm run build (includes TypeScript).
   Run with exact candidate source paths and normal host toolchain; use the
   existing pinned node_modules via a **temporary worktree-local symlink**
   if needed, and remove that symlink when done. Do not rewrite
   package-lock.json or change global CUDA/PyTorch/tool versions to get green.
   Capture test counts, any new skips, warnings and full bounded failure output.
4. If a check fails, independently inspect the actual production invariant,
   failure traceback, test fixture and code. You may correct **obvious,
   narrowly bounded mechanical mistakes** (imports, typo, syntax/TypeScript,
   API field name, test fixture binding, existing-env wiring, insignificant
   interface shape mismatch) on the SAME development branch, with full
   before/after diff and focused regression. Do not silently solve a
   semantic export-policy, security/path authorization, state-transition,
   model-lifecycle or architecture defect yourself: return a concrete
   reproducer and proposed change to Central. Do not bypass or weaken tests,
   hidden flags, permissions or validation to gain PASS. Avoid repeated
   stalled/failing runs with no new diagnosis.

## Phase B. Isolated Studio and real UI/API filesystem verification (CPU first)

1. Start a **single loopback-only candidate Studio** at 127.0.0.1:8766 (or
   verified free alternative), with new isolated job_root, upload_root and
   NEMOVOX_EXPORT_ROOTS pointing at bounded temporary test directories, source
   imports and child CLI Python proven to resolve to this exact candidate.
   Preserve the production Studio, jobs, user output tree and port :8765.
   Do not mutate real model units merely by starting this candidate.
2. In Chromium/Playwright through actual candidate-served web/dist verify
   tab 1 source/file/URL switching, folder roots navigation, manual nested
   output path entry, checkbox defaults/per-task choices, and URL downloaded
   media option. The target must visibly say **host folder, not browser
   Downloads**. Verify actual request export plan and persisted job.json.
   Exercise keyboard/accessible label interactions, UI error responses and
   the older job migration/default export-disabled path.
3. With synthetic/contract and controlled test data, verify only selected
   final files are copied beneath the chosen folder's collision-safe
   nemovox-<job-id> child; compare exact bytes/hashes to internal originals.
   The full internal outputs, ASR/MT source contracts and requested media
   remain. Exercise two jobs with colliding filenames, idempotent retry,
   missing/unwritable/outside-root, ../ and symlink escape, copying error,
   no hidden overwrite, and cancelled/failed job export status. Confirm
   exactly which paths are created and which are *not*.
4. Stub or explicitly simulate the yt-dlp download for a synthetic remote
   sample: preserve the original media in the job-owned remote directory,
   exercise selected media export and unselected negative case, then cause
   a downstream processing error and prove the downloaded file, export
   manifest and full log remain accessible. No need to download YouTube
   copyrighted media or expose other private files. Do not loosen existing
   URL public-address/no-playlist checks.
5. Verify three *separate* observed translation -> English review ->
   rendering UI stage transitions from one existing generate-subtitles
   subprocess and chronological structured event records; no extra
   translation/review call. Exercise disabled review, same-language bypass,
   source-contract re-use, optional condensation and failure/cancel paths
   with truthful skipped/failed states. Test a really large multiline log
   plus Unicode across chunk boundaries, visible tail, Copy visible/Copy
   full, download full, **retention after completion/reload/reopening**,
   missing job/error states, and all-runs pagination beyond eight/50.
   Stage durations/progress should come from real event receipt, not
   fabricated percentages or three separate model launches.
6. Verify no fixed wall of individual export buttons in the prominent UI;
   inspect automatic export report (success/partial/failure) and keep
   full internal job artifacts/download access under Advanced. If a
   selected file cannot be exported, processing success must not be falsely
   advertised as successful export.

## Phase C. Safe source-bound real French Studio/GPU acceptance

After deterministic + CPU/browser phases PASS, inspect the actual canonical
model lifecycle lease, GPU memory and active jobs. The already merged
Issue #10 has verified owned ASR and review unit provisioning, but do not
take ownership of external resources, disable shared lease admission,
evict existing GPU processes, or restart production Studio.

Run a **short authorized real French source** via the isolated candidate
Studio (previously established FLEURS audio if host-allowed) with explicit
French -> English, Riva CUDA, review balanced, subtitles SRT/ASS, chosen
temporary host destination and only SRT + final_english selected for
automatic export. Use a **fresh output/job/cache** so real inference is
not disguised as a warm-cache PASS. Require:
- one ASR -> Riva CUDA -> English-review -> render pipeline with recorded
  staged state transitions and each actual model endpoint/resource identity;
- source transcript unmodified, original Riva MT retained beside final
  reviewed English, genuine timed SRT/ASS and browser preview;
- only selected SRT and English TXT in the external job subfolder; all
  generated formats, JSON, logs and media still held in internal job dir;
- terminal UI displays completed stage history plus full log and exported
  path/status after reload and from job-history reopen;
- safe model unit release and canonical lock use; unrelated services,
  production Studio PID/health and GPU workloads remain unchanged.

Keep this live test bounded; no full episode, unrequested SIGKILL storm,
extra unrelated model evaluation, software upgrades or model weight copies.
Run direct CLI or cached/offline parity only where changed behavior
actually requires it; reuse Issue #10's accepted unchanged model-lifecycle
evidence rather than recertifying the entire GPU service stack.

If the interactive local-agent model itself occupies admission/VRAM or a
legitimate lease denies the GPU phase, **do not bypass the lease**.
Finish CPU/host preflight, release the model lock normally, and report
READY_FOR_FINAL_OPERATOR_RUN with **one minimal exact-source native
command/probe** that can run after the agent exits, or a clear BLOCKED
result. Do not claim live PASS.

## Phase D. Sanitize, publish and hand back one evidence packet

Publish **one** appropriately complete evidence-only commit to the same
issue/12-studio-output-workflow branch under
staging/issue12/evidence/<UTC_RUN_ID>/, with manifest.json containing
exhaustive member path, SHA-256, byte size, associated criterion and
evidence class. A prior authorized mechanical source repair commit may
precede it, but its new exact SHA becomes the **tested** source identity
and must be independently reported; do not mix source and evidence in the
evidence commit. Never force update a changed remote ref. If publication
fails, return the full safe handback without claiming a remote artifact.

Suggested sufficient inspectable public-safe members (prefer actual
stdout/exit snippets and sanitized API statuses, not unsupported claims):
- source/identity.json (full source SHA/tree, origin ref, import paths,
  changes since Central staging, final host workspace state);
- source/qualification.diff (only if source was mechanically repaired);
- validation/cpu-tests.txt, validation/web-tests.txt;
- validation/export-paths-and-collisions.md;
- validation/remote-source-retention.md;
- validation/stage-event-and-log-contract.md;
- validation/isolated-studio-browser.md;
- validation/french-gpu-studio-e2e.md (only when actually run);
- validation/service-and-production-invariance.md;
- review/acceptance-matrix.md, review/findings.md;
- COMMAND_SHAPE_FRICTION.txt.
Do not publish raw tokens, protected env files, personal audio, copyrighted
media, full private transcripts or live unrestricted process logs. Preserve
small controlled synthetic fixtures and safe event schemas when informative.
A complete GitHub packet obviates duplicate private archives. Supply one
secure minimal private archive only if a material required piece cannot
be sanitized and published.

### Required handback

LOCAL_AGENT_HANDBACK:
  RESULT=<PASS|FAIL|BLOCKED|READY_FOR_FINAL_OPERATOR_RUN>
  ISSUE=12
  PR=13
  BASE_MAIN_SHA=332a99697edcc0b72fcc842168d5b1f57782e203
  TESTED_SOURCE_SHA=<exact 40 chars>
  TESTED_TREE_SHA=<exact 40 chars>
  SOURCE_REMOTE_CHANGES=<none|bounded mechanical commits with full SHAs/diff>
  CPU_FOCUSED=<PASS|FAIL; commands, counts>
  CPU_FULL=<PASS|FAIL; commands, counts>
  FRONTEND_VITEST_BUILD=<PASS|FAIL; commands/counts>
  OUTPUT_ROOT_AUTHORIZATION=<PASS|FAIL>
  TASK_EXPORT_SELECTION=<PASS|FAIL>
  COLLISION_IDEMPOTENCE_AND_FAILURE=<PASS|FAIL>
  REMOTE_MEDIA_RETENTION=<PASS|FAIL>
  STAGE_EVENTS_AND_SINGLE_CLI=<PASS|FAIL>
  PERSISTENT_COMPLETE_FULL_LOG=<PASS|FAIL>
  HISTORY_AND_OLD_JOBS=<PASS|FAIL>
  ISOLATED_GUI_BROWSER=<PASS|FAIL|UNVERIFIED; browser process, source>
  FRENCH_RIVA_CUDA_E2E=<PASS|FAIL|BLOCKED|UNVERIFIED; exact job/unit>
  MODEL_RESOURCE_AND_PRODUCTION_ISOLATION=<PASS|FAIL|UNVERIFIED>
  EVIDENCE_GITHUB_ROOT=<branch:path @ exact evidence SHA | unavailable>
  MANIFEST_DIGESTS=<all PASS|FAIL|UNVERIFIED>
  PRIVATE_ARCHIVE_ADDS_REQUIRED_EVIDENCE=<yes|no>
  COMMAND_SHAPE_FRICTION=<canonical event block or encountered=no/events=[]>
  BLOCKING_FINDINGS=<none|precise defect+reproducer>
  REQUIRED_OPERATOR_ACTION=<none|one minimal post-exit command>

### Local-harness tool routing and stop conditions

Use Serena for bounded semantic references and diagnostics where available,
Probe only for supplemental extraction, RTK only where filtering preserves
evidence, OpenViking only for historical bounded context (never current Git
or host truth). Prefer native authenticated git/gh, actual host processes,
pytest, npm, FFprobe, systemctl --user and nvidia-smi for source/runtime
authority. Inspect before editing and audit changes afterward.

Stop/escalate for unknown mutable head or workspace drift, a substantive
API/export semantic/security defect, unowned service conflicts, unsafe
GPU capacity, a model download or incompatible runtime rewrite, or inability
to preserve the operator's production state. Fix ordinary test/setup and
narrow mechanical mistakes autonomously with evidence; do not return after
each successful micro-step.

A local PASS is qualification **not** independent PR approval, merge
authorization, production promotion or Issue #12 closure. Central owns
those decisions after reading the exact-head handback and evidence.
