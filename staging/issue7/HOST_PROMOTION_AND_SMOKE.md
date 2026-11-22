# Issue #7: promote merged Nemovox Studio and verify the running GUI

Status: **required post-merge host promotion and functional acceptance**. Issue #7 was reopened because previously accepted source/build checks did not prove the live host served the remediated application. This handoff covers ONE bounded host campaign: discover → reversible promotion → fresh-process GUI/API smoke → publish evidence. Do not open a PR or close the issue.

## Immutable authority and execution plane

\`\`\`text
REPO=fvanevski/nemovox
ISSUE=7
MERGED_PRODUCTION_SOURCE=8fee1e177db9742dbdb38c4a22bdd663c20ac80f
MERGED_PRODUCTION_TREE=2c733a3036443121cc4c1c7c2b7a09bb0d5680a5
APPROVED_PR=8
APPROVED_PR_HEAD=67a2e03b6e588f5dd520064e8ff57c3b9e99facb
EXPECTED_MAIN_AT_HANDOFF=8fee1e177db9742dbdb38c4a22bdd663c20ac80f
HANDOFF_STAGING_BRANCH=issue/7-studio-subtitle-workflow
OPENCODE_PRIMARY=build
HOST_TARGET=Garuda Linux; Nemovox Studio only
SOURCE_CODE_EDITS=forbidden
MAIN_GITHUB_WRITES=forbidden
NEW_PULL_REQUESTS=forbidden
ISSUE_STATE_WRITES=forbidden
EXTERNAL_GPU_SERVICE_MUTATIONS=forbidden
\`\`\`

This is a deployment/operational task: **Build** is authorized for narrowly reversible Nemovox-host source, installation, service and browser-test mutations, and for publishing sanitized evidence to the preexisting staging branch. Do not use strict read-only Evidence as the mutating primary. The **fresh application server and browser process** used for acceptance must be independently identified after activation; the interactive agent/session alone does not prove deployed state. If a separate Evidence-role session is genuinely necessary to inspect the final live application, do that in a bounded second phase without making this into a series of new implementation tasks.

Current source authority:
- \`AGENTS.md\`, \`web/README.md\`, \`docs/gui/ARCHITECTURE.md\`, \`docs/gui/SUBTITLE_WORKFLOW.md\` at the merged main SHA.
- \`src/nvidia_transcribe/studio_api.py\` (FastAPI \`/api/health\`, \`/api/media/inspect\`, \`/api/glossaries\`, \`/api/translation-backends\`, \`/api/jobs\` and output routes); \`web/src/App.tsx\` and \`web/src/ExplainedSelect.tsx\`.
- Two venv roles documented by the repo: \`.venv-studio\` for FastAPI and \`.venv\` as \`NEMOVOX_CLI_PYTHON\`; the package is often installed editable. Do not assume an editable venv imports the source from the current checkout.
- The documented backend is \`127.0.0.1:8765\`, one uvicorn worker; production build \`web/dist\` is served by FastAPI if present; a dev Vite frontend may separately run at \`127.0.0.1:5173\`. These are documentary defaults, **not claims about the actual host**.
- External ASR/Riva/text-model processes and shared GPU services are NOT owned by this promotion.

If \`origin/main\` has advanced beyond the exact merged SHA, do not silently deploy an unknown revision: record head and whether the issue #7 files are still identical, then stop for Central rebinding if material source/activation changes exist. If the staging branch has advanced, inspect the new handoff/evidence material and use exact CAS. Do not force, reset, clean, stash, or rebase operator work.

## Phase A: host discovery and rollback-ready promotion (Build)

1. Inspect the actual host: Nemovox repo checkout/worktrees, Git status and remote SHA, \`web/dist\` freshness, Python interpreter/package import locations, .venv/.venv-studio dependencies, node/npm, ffmpeg/ffprobe, installed browser(s), loopback listeners on the app ports, and how Studio is launched (systemd --user unit, shell, tmux, supervisor or no running process). Record current PID, executable, command, cwd, startup/config environment *keys and sanitized values only*, logs and UI URL. Do not inspect/dump secrets, tokens, entire environments or personal media. Determine whether a live job is queued/running and who owns its job root and lock. Check whether host configuration \`NEMOVOX_MODEL_RESOURCES\`, \`NEMOVOX_STUDIO_JOBS\`, \`NEMOVOX_STUDIO_UPLOADS\`, \`NEMOVOX_MEDIA_ROOTS\`, \`NEMOVOX_CLI_PYTHON\` and frontend mode are configured; do not invent paths.
2. Establish an exact-source, rollbackable promotion plan suited to the **discovered** installation. Snapshot the small required service/unit/config state and existing serving source/tree/dist hashes. Preserve ignored \`outputs/\`, all user media, host-local resource JSON, job state, secrets and venvs. Never overwrite a dirty checkout or user branch; use a clean isolated release worktree or a safe fast-forward on an authenticated clean production checkout. If a new release worktree is used, maintain stable existing job/upload paths via verified env/config, and ensure \`PROJECT\` based path resolution, static asset root, and CLI \`cwd\`/package source imports refer to the promoted release. Prefer the host's existing service manager and native restart. Do not add a detached deployment controller, new public endpoint, broad system unit or unrelated service unless necessary for the observed Nemovox setup.
3. Check for an active Studio job before changing a running instance. Do **not** cancel or interrupt user jobs to satisfy a smoke test. When idle, promote source **exactly** from \`origin/main@8fee1e177db9742dbdb38c4a22bdd663c20ac80f\`; install/rebind just the necessary Nemovox Studio Python dependency environment if needed; build the frontend using the committed lockfile (reuse existing trustworthy \`node_modules\` only when compatible). Preserve the lockfile/source. Use a stable source path for the running API and the CLI child; prove \`nvidia_transcribe.__file__\` in **both interpreters**, module/hash identity and the built dist asset fingerprint. Do not silently use a different editable checkout. Rebind or restart **only** the identified Nemovox-owned server/development frontend, with one backend worker, same loopback interface/port and service ownership. If there was no running Studio, start exactly one documented loopback Studio instance using the existing host convention, record ownership and how it will persist; do not create a LAN service.
4. Capture a deploy receipt with prior/live source and process identity, backup/rollback references, changed paths and commands, service readiness, the new effective source+environment+frontend build identity. If promotion fails, restore **only** what this campaign changed with authenticated rollback, confirm the previous service still works and return \`BLOCKED\`; do not claim deployment PASS. Keep reversible recovery material private until Central accepts.

## Phase B: fresh running-app smoke (same bounded campaign; no GPU service mutation)

Launch/refresh a **newly observed** Studio server process after promotion and a fresh browser tab/process or independently admitted browser automation session against the actual serving URL, not a Vite-only offline preview. Prove the server PID/start time, working directory, module/package file hashes and production \`web/dist\` identity correspond to merged source. If the host uses Vite dev mode for routine browsing, validate both the actual production backend-served built frontend and the active dev URL separately; report which one the operator really uses. A successful \`npm run build\` by itself is NOT a deployed GUI PASS.

Exercise these user-visible and API paths against localhost:

1. **Serving + navigation:** real browser loads the production Studio app from the activated service (HTTP SPA JS/CSS 200, no critical console errors); three tabs Media & task / Configure / Process & preview navigate; job history remains available after reload. Capture synthetic-content screenshots or DOM/console/browser-readback with URL, PID/time and source identity.
2. **Configuration parity:** UI default translator Riva, English-review profile balanced, source refinement absent/off, \`condense_high_cps=false\`, \`No glossary\` default; English-source option explains same-language bypass. All four ASR choices, English-review/localization descriptions display. Exercise keyboard Arrow/Home/End/Enter/Escape and pointer selection in a real DOM, inspect accessible \`aria-activedescendant\` and changing description/focus; do not infer screen-reader announcement from static markup. If assistive-technology output cannot be observed, mark specifically \`SCREEN_READER_AUDIO=UNVERIFIED\` while retaining DOM evidence.
3. **Real media track inspection:** use an authorized short existing non-sensitive media file with >=2 audio streams if available; otherwise generate a **few seconds** of purely synthetic video/two audio tracks in a throwaway approved media root using ffmpeg (no copyrighted source). Verify \`POST /api/media/inspect\` returns accurate **audio-only** indices, codec/language/title; UI selector offers only those streams; switching files resets invalid indices/does not submit stale source; invalid stream and an out-of-root path are rejected by API. Do not export the media bytes/screenshots containing personal episodes.
4. **Glossaries and resource readiness:** \`/api/glossaries\` lists both valid French/Mandarin examples and the UI displays their character/term notes; no automatic selection. Prove \`/api/translation-backends\` is **configuration readiness only**, not model-inference health. No fallback or model download. Malformed glossary error isolation is already qualified by premerge tests; do not corrupt host glossaries merely to repeat it.
5. **Real launched job, no fabricated outputs:** create only a few-second **synthetic English recording contract** under an authorized root, with speakers and valid timed English utterances and a synthetic source-media path when preview is needed. Submit through the running Studio API/UI with explicit \`source_language=en\`, \`target_language=en\`, \`translator=riva\`, \`translation_review=balanced\`, \`refinement=off\`, \`condense_high_cps=false\`. Verify job queued→running→complete, real CLI subprocess identity bound to merged package code, actual job.json and logs, unchanged source utterances, \`bypassed_same_language\`/\`skipped_same_language\`, resulting \`.srt\` and \`.ass\` (if requested), job list, preview, permitted download, and browser refreshing. This is a **CPU/no-inference** end-to-end exercise of the hosted worker and UI; do not label it live Riva translation. A standalone CLI run or FastAPI TestClient does not replace this active-service test.
6. **Foreign-language translation path if safe:** inspect existing Riva/ASR/text endpoint reachability and GPU occupancy read-only. If a short already-permitted, representative foreign-language recording contract and already-running/configured Riva and review model allow a tiny run **without** restarting, evicting, loading a large new checkpoint, downloading weights, or disrupting shared agents, run one real Studio translation job and confirm unchanged original ASR and Riva machine translation, selective English corrections, and final SRT/ASS. If resources are absent/unsafe/unreachable, mark \`LIVE_RIVA_TEXT_E2E=UNVERIFIED|BLOCKED_BY_ENV\` with reason; do not manufacture an inference PASS or widen the campaign. This is a limitation to be adjudicated by Central before issue closure, not permission to change external services.
7. **Video/ASS preview:** open the small synthetic playable media in the actual browser after the completed test job. Confirm synchronized cue overlay, basic playback/seek and downloads. Verify JASSUB ASS overlay when available. Distinguish actual JASSUB success from HTML fallback and unsupported codec. If browser automation is not available, collect bounded manual browser evidence; do not claim interactive acceptance from SSR/Vitest alone. Never capture/upload private viewing material.

A few bounded test jobs and host-only reversible artifacts are acceptable. Do not run entire TV episodes or full GPU benchmarks. Do not mutate other projects, Qwen/ASR/systemd inference services, shared model memory allocation, or PyAnnote. If material application defects are uncovered, record a short reproducible failure plus root-cause evidence for Central; **do not patch merged production source**, weaken tests or silently roll to a new unreviewed SHA.

## Evidence matrix (must be fulfilled or explicitly UNVERIFIED/BLOCKED)

| Acceptance criterion | Required artifact/observation | Authority / privacy |
| --- | --- | --- |
| Exact installed source and rollback | before/after Git/tree/blob identity, import paths in both venvs, service owner/command/PID/start, \`web/dist\` hash, backup/restore plan | Native Git/process/FS on host; sanitize absolute private details for GitHub |
| Actual promotion and activation | bounded deploy log, startup/readiness, process epoch, loopback listener; no active jobs interrupted; proof non-Nemovox services untouched | Native systemd/process/port evidence, host-only raw logs |
| Browser GUI behavior | fresh browser URL, DOM/control state, screenshots with synthetic content only, console outcome, keyboard/focus and tab/navigation observations | Browser test on the running host, not static SSR |
| FFprobe & selection | synthetic/approved test file FFprobe stream JSON, local inspect API JSON, browser selector parity, negative 403/422 checks | Local ffprobe/API/DOM; no private media |
| Runtime job E2E | issued request, job ID, state transitions, bounded sanitized stdout/stderr, actual SRT/ASS and contract counts/digests, content invariants, preview/download HTTP | Running FastAPI worker/CLI, synthetic evidence only |
| Translation and glossary | live Riva configuration discovery (not inference), glossary list and selected preview; separate foreign-language inference result or explicit omission | Live config, no secrets/model dumps |
| Final state/rollback | PID/config/source still active after test, health plus SPA readback, no unrelated services touched, stored private rollback info | Fresh readback; sanitized hashes/status |
| Evidence publication | manifest with path, byte size, SHA-256, sources, command exits, omitted evidence, privacy scan and comparison to GitHub members | Exact staging-branch commit, public sanitized subset |

## GitHub publication and handback

Publish **one** small, self-contained, sanitized evidence packet to the **existing** \`issue/7-studio-subtitle-workflow\` staging branch under \`staging/issue7/host-promotion/<UTC_RUN_ID>/\`. The working source for deployment remains immutable merged \`main@8fee1e177db9742dbdb38c4a22bdd663c20ac80f\`, NOT the evidence branch. Publish using the host's GitHub permissions if granted, under one evidence-only commit with exact remote-head precondition and post-write readback. No PR, no write to \`main\`, no branch reset/force-push, no source/test edits. If lacking publication permission, report exact blocker and retain the sanitized packet locally for explicit Central resolution rather than falsely claiming published evidence.

Required public packet files (small text/JSON; optional synthetic-only screenshots where safe):
- \`manifest.json\`: UTC run, source SHA/tree, staged commit, artifact digests/bytes, collection method, redactions, missingness, manifest external hash.
- \`source/installed_identity.json\`, \`promotion/deploy_receipt.md\`, \`promotion/rollback_receipt.md\`
- \`validation/production_http.json\`, \`validation/browser_gui.md\`, \`validation/media_tracks.json\`, \`validation/job_e2e.md\`, \`validation/backend_and_glossaries.json\`, \`validation/final_live_readback.md\`
- \`review/acceptance_matrix.md\`, \`review/findings.md\`, \`COMMAND_SHAPE_FRICTION.txt\`.

The packet must distinguish \`HOST_ATTESTED\` from independently retrievable GitHub evidence and preserve command lines/exit codes or observable request/response identities, not just checkboxes. Exclude credentials, HTTP auth headers, raw environment, hostnames other than permitted localhost, private paths, personal media, copyrighted video screenshots/transcripts, model weights and full logs. Keep private backup materials and full raw receipts on host. A sanitized \`.tar.gz\` can exist as an optional local backup, **but do not request an operator upload** if GitHub already contains all required reviewable evidence. If the host has indispensable evidence that cannot safely be published, specify the exact delta and whether a private upload is required.

Return one compact handback:

\`\`\`text
RESULT=PASS|PARTIAL|BLOCKED|FAIL
ISSUE=7
REQUIRED_SOURCE_SHA=8fee1e177db9742dbdb38c4a22bdd663c20ac80f
DEPLOYED_SOURCE_SHA=<exact value or UNVERIFIED>
DEPLOYED_TREE_SHA=<exact value or UNVERIFIED>
PREVIOUS_HOST_VERSION=<identity>
HOST_PROMOTION=PASS|BLOCKED|ROLLED_BACK
ROLLBACK_AVAILABLE=yes|no
BACKEND_PROCESS_IDENTITY=PASS|FAIL|UNVERIFIED
STUDIO_CLI_IMPORT_IDENTITY=PASS|FAIL|UNVERIFIED
FRONTEND_PRODUCTION_ASSET_IDENTITY=PASS|FAIL|UNVERIFIED
FRESH_BROWSER_GUI=PASS|FAIL|UNVERIFIED
INTERACTIVE_COMBOBOX_DOM=PASS|FAIL|UNVERIFIED
SCREEN_READER_AUDIO=PASS|FAIL|UNVERIFIED
MEDIA_FFPROBE_TWO_TRACKS=PASS|FAIL|UNVERIFIED
LIVE_GLOSSARY_PREVIEW=PASS|FAIL|UNVERIFIED
LIVE_JOB_ENGLISH_CONTRACT_E2E=PASS|FAIL|UNVERIFIED
LIVE_RIVA_TEXT_E2E=PASS|FAIL|UNVERIFIED|BLOCKED_BY_ENV
VIDEO_PREVIEW=PASS|FALLBACK|FAIL|UNVERIFIED
JASSUB_ASS_PREVIEW=PASS|FALLBACK|FAIL|UNVERIFIED
FINAL_LIVE_READBACK=PASS|FAIL|UNVERIFIED
UNRELATED_GPU_SERVICE_MUTATIONS=none|<facts>
BLOCKING_FINDINGS=[...]
UNVERIFIED_ITEMS=[...]
HOST_SOURCE_EDITS=no
GITHUB_EVIDENCE_STATUS=COMPLETE|INCOMPLETE|BLOCKED
GITHUB_EVIDENCE_ROOT=staging/issue7/host-promotion/<UTC_RUN_ID>
GITHUB_EVIDENCE_COMMIT=<exact SHA or none>
PRIVATE_ARCHIVE_EXISTS=yes|no
PRIVATE_ARCHIVE_ADDS_REQUIRED_EVIDENCE=yes|no
PRIVATE_ARCHIVE_ADDITIONAL_CONTENT=<specific items or none>
PRIVATE_ARCHIVE_UPLOAD_REQUIRED=yes|no
REQUIRED_OPERATOR_ACTION=NONE|UPLOAD_ARCHIVE
COMMAND_SHAPE_FRICTION=<requested/effective/why; none if absent>
\`\`\`

**STOP conditions:** can't identify the actual host-controlled Studio process/source; dirty checkout without safe alternate release; active user jobs would be interrupted; no safe rollback; conflicting production source; need to touch external GPU services or protected paths; API requires LAN exposure; unrecoverable live GUI failure; or branch/head drift at evidence publication. Return diagnostics and the narrowest necessary escalation. Do not close issue #7: Central will review acceptance and perform the governed close only after the live gate passes.
