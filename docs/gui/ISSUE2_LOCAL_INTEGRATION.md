# Issue #2: consolidated Studio remediation integration and evidence handoff

## Task and authority
Repository: fvanevski/nemovox
Issue: https://github.com/fvanevski/nemovox/issues/2
Working branch: integration/gui-foundation-20261121
Start SHA: **read the exact current 40-character remote head and confirm the handoff commit** before integration.
Baseline source: main@9b4c8b539c67f2f6266fef135a259e92f5477b19 at staging time, not an evergreen main pin.
Host project: /workspace/nvidia-transcribe

Central ChatGPT has authored the B1 process-group lifecycle repair, B2 imported-contract transcription, I1 active-source UI correction and focused regressions. Your assignment is to integrate and qualify this **complete consolidated issue #2 slice** on the existing working branch, not to independently redesign any of these fixes. Read AGENTS.md, docs/gui/ARCHITECTURE.md and the issue acceptance criteria before execution.

### Authority
- Host integrator: existing Antigravity CLI workflow (or explicitly selected OpenCode primary build).
- Scope: existing Studio API, frontend, focused tests, related docs and exact-host verification only.
- You may fix demonstrated in-scope implementation defects, compatibility failures, fixtures or integration gaps autonomously. Preserve architecture, production behavior, safety and source/transcript fidelity. When a correction changes semantic behavior substantially, document root cause and full diff for Central review; escalate only material architecture/safety decisions or protected/out-of-scope changes.
- No new GUI PR, merge, rebase, force push, changes to main, or branch proliferation. Work on the existing integration branch. Update it by ordinary non-force push only after current remote-head proof, tests and diff review. If a conflicting remote head appears, stop instead of overwriting.
- The evidence-only branch must be based on the final implementation commit; it must never be merged into main.

## A. Host checkout and scoped verification

1. Authenticate exact GitHub repository and current heads of main, working branch and prior evidence branch. Read current files, and preserve user workspace modifications without reset/clean/stash. A fresh separate worktree or clean checkout may be used.
2. Confirm files added or changed by Central: src/nvidia_transcribe/studio_api.py, tests/test_studio_api.py, web/src/App.tsx, web/src/source-selection.ts, web/src/source-selection.test.ts, this handoff.
3. Inspect B1 process-group cleanup for a leader that exits before descendants, TERM-resistant child, normal-completion cleanup, concurrent cancellation and graceful FastAPI lifespan shutdown. Check that group targets remain Nemovox-owned; never signal external GPU services. If Linux /proc or host execution semantics expose a concrete flaw, repair it within Studio and add an exact regression.
4. Exercise B2 contract-backed transcription with populated, untimed, empty and invalid synthetic contracts. Confirm a completed job owns downloadable asr/recording.contract.json and asr/transcript.txt; preview shows original source utterances. Missing/invalid contracts must not produce a false complete state. Preserve existing contract-backed subtitle/translation flow.
5. Exercise I1 Host/Upload/URL changes including late upload completion and request payload. Confirm only the *visible, active method's* source may be submitted and no prior-mode media is sent. Include browser-level interaction and network-request assertion when feasible.

## B. Required evidence-driven checks

- Use the existing .venv-studio for API and .venv for CLI, no dependency installation into /workspace/pyannote or model checkpoint paths.
- Run focused tests (including new cases) before and after any local corrections.
- Run CPU suites in both environments, e.g. .venv-studio/bin/pytest -q -m "not gpu" and .venv/bin/pytest -q -m "not gpu". If one environment lacks Studio extras, report and use the intended Studio environment rather than distorting environment authority.
- In web/: npm test and npm run build, with clean dependency resolution using existing package-lock.json. Test active-mode source switches, stale response, refresh/history and accessible buttons.
- Run one localhost HTTP/API/CLI synthetic contract flow and bounded real subprocess cancellation. Include status, logs, contract+transcription preview/download, subtitle creation where local mock endpoints suffice, and no owned active descendant left after cancellation/shutdown.
- Browser smoke in Chromium plus FireDragon if available, using synthetic media only. Confirm JASSUB ASS if supported and honest fallback where not. Do not reprocess full commercial episodes or expose server to LAN.
- Live vLLM/SwiftU inference only if already available and safe. Do not stop/restart unrelated GPU services. Record not exercised as UNVERIFIED, not as PASS.
- Audit any necessary startup instructions and minimal end-user usage guidance (localhost, CLI Python env, model services preconditions). Keep documentation functional.

## C. Publication and handback

- Produce one final implementation commit series on the *existing* working branch; prefer one bounded integration commit if fixes are needed. If all Central changes validate, no unnecessary additional source commit.
- Create/update the separate evidence branch from the **final exact implementation SHA**, publishing a sanitized self-contained tar.gz (target <=25 MiB) plus standalone HANDback.md, test-results.md, browser-review.md, manifest.json under staging/agent-evidence/issue-2-studio/. Include source comparison/diff, exact SHAs, commands and exit codes, logs, regression outputs, process-group/cleanup observations, bounded synthetic screenshots and checksums.
- Preserve old evidence branch data when advancing: use a dedicated new folder; never rewrite public evidence history or publish raw user media, actual TV frames, transcript payloads, credentials, model weights, private host addresses or other sensitive logs.
- Verify archive integrity and GitHub branch readback before reporting PASS. Return links to each artifact and exact full working/evidence branch SHA, tests, current limitations, out-of-scope defects if any, changed-path list, and what remains for Central independent PR review.
- Stop after the cohesive issue #2 host integration/evidence objective. Do not open a PR until Central has reviewed the corrected exact head.

## D. Stop/escalate conditions

Escalate only if a defensible fix requires altering ASR/diarization/CTC/translation semantics, dependency architecture, security boundaries, external service lifecycle, large host destructive actions, or protected GitHub state. Resolve bounded Studio bugs autonomously within this campaign. An unavoidable inability to prove forced crash cleanup must be described as unverified: do not claim unclean SIGKILL recovery or group ownership across a server restart without authenticated evidence.

## Required handback: command-shape friction

Return the following block even when there was no friction; each event is bounded and sanitized:

COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events:
    - scenario=<short repository/task scenario key>
      requested_shape=<submitted command/invocation>
      effective_shape=<post-hook executed/rejected command|same-as-requested|unknown>
      rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
      event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
      execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
      correction_or_annotation=<bounded correction or decisive error>
      replacement_shape=<successful corrected shape|none>
      recipe_reference=<GHDEV-10 recipe key|none>
      resource_reference=<resource used to recover|none>
      contract_conflict=<yes|no|uncertain>
      durable_gap_candidate=<yes|no|uncertain>
  omitted_sensitive_values=<yes|no>

For no friction, use encountered=no, events=[], omitted_sensitive_values=no. Report substantive result independently as PASS / FAIL / BLOCKED and do not conceal recovered execution friction.
