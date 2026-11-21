# Issue #2 Codex findings: final frontend regression qualification

## Authority
Repository `fvanevski/nemovox`, PR #3, issue #2.
This handoff applies to the one Central-authored frontend commit directly following `49fac0f4ae9c607078cbc71e34bc448c5aab318f`. Resolve the *current* exact GitHub head before host qualification. Do not substitute a historical SHA or test working-tree changes as remote authority.

Relevant Codex threads:
- https://github.com/fvanevski/nemovox/pull/3#discussion_r4654337880
- https://github.com/fvanevski/nemovox/pull/3#discussion_r4654337882

## Scope

Central authored the complete code correction in `web/src/App.tsx`, `web/src/job-display.ts`, and `web/src/job-display.test.ts`.
1. The `preview` and `artifacts` query keys change from `working` to `final` when job status becomes `complete`, causing a fresh final-result request even if both slow result polls missed the completion boundary.
2. A remote URL job does not render/load `/api/jobs/{id}/media` while its source path is absent. The pending placeholder becomes a video player after the backend stores `source_path`. Failed/no-source jobs show unavailable status. ASS/JASSUB startup is gated on actual video render readiness.
3. A new Vitest module checks the cache-key transition and synthetic QueryClient fetch behavior, plus remote-media state transitions.

This is a tightly scoped post-Codex GUI remediation. It must not change the Python pipeline, model-serving architecture, job scheduler, browser stack, GPU services, or unrelated workflow.

## Host integration and verification (single campaign)

- Inspect the current PR, implementation branch, exact incoming Central commit, and `AGENTS.md`; preserve unrelated local changes. Use a clean worktree if needed.
- Run `cd web && npm test && npm run build`. Run relevant Python CPU/API suite only if needed to confirm unchanged backend contracts; do not rerun GPU ASR/SwiftU.
- With the existing lightweight localhost Studio and synthetic media, run one targeted browser/HTTP qualification that **controls polling order**:
  - Submit or open a synthetic job whose preview/artifacts initially respond with pending/empty results; change status to `complete` between the faster status poll and slower result polls; verify **both** endpoints refetch after completion and the final cue text and artifact download are visible without a manual refresh.
  - For remote URL, delay the source-path update so that the job remains queued/running for multiple status polls. Verify that the video element is not requested during download and that the pending message appears. Then supply a valid local synthetic video path and verify the video element/media request appears and works without reloading the page. Test the terminal failed/no-source branch and existing local-media preview.
  - If practical, confirm ASS/JASSUB does not attempt to attach before video availability and attaches once rendered, with honest fallback if unsupported.
- Prefer Chromium for deterministic browser/network interception. FireDragon need not be repeated if its unchanged capabilities are already evidenced; explain source equivalence. Use synthetic WebM/contract data only.
- Fix only demonstrable integration/fixture defects if any, with minimal diffs and regression proof. Do not use new PRs or write protected `main`.
- Publish one small sanitized final-head evidence archive and plaintext handback/test/browser/manifest under a new evidence-only branch, based directly on the final implementation head. Reuse the existing evidence branches for earlier issue #2 acceptance rather than duplicating large screenshots.
- Return exact full source/evidence SHAs, changed-file list, commands/exits, request-order/network captures, screenshot(s) if useful, results, limitations and a complete `COMMAND_SHAPE_FRICTION` block in the existing project format.
- Do not merge, close issue #2, mark Codex threads resolved, or request a second Codex review. Stop once host qualification and evidence publishing are complete.

## Stop/escalate conditions

Stop for repository-head mismatch, unauthenticated source, unsafe GPU/service actions, or a required architecture change. Ordinary fixable browser harness issues are within scope. A true regression should be repaired, not waived or masked.
