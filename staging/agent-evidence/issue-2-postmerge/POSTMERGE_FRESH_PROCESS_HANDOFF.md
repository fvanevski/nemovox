# Nemovox Studio issue #2: merged-main fresh-process acceptance

## Mission and immutable inputs

- Repository: `fvanevski/nemovox`; issue: https://github.com/fvanevski/nemovox/issues/2
- Merged PR: https://github.com/fvanevski/nemovox/pull/3 (squash, approved)
- **Required source commit and `main` head:** `242c418f55cd2afb796ee2c4fa742a10ff04f7a3`
- **Required source tree:** `122f8466db22098ba9427373af7e87ceb8421076`
- Approved pre-squash head: `bc5f067340d8bc84b7129bbdea7589ea401222c8` (same tree).
- Evidence branch: `evidence/issue-2-postmerge-20261121`, based directly on merged `main`.
- Project checkout normally `/workspace/nvidia-transcribe`.
- This file is a **validation contract**, not production-code authority; current `AGENTS.md`, merged source, and existing GitHub issue #2 govern.

## Authority and scope

One bounded local Build/host-supervisor campaign is authorized to authenticate the merged source, create a **fresh isolated test checkout/worktree and scratch roots**, install/reuse dependencies as needed to run existing tests, build the frontend if absent, launch and stop **only its own** new loopback Studio processes, create synthetic media/contracts and test jobs, inspect process groups, run focused browser/API probes, and publish one sanitized evidence-only commit to the existing evidence branch. This is post-merge validation, not an implementation campaign. `SOURCE_REMOTE_UPDATE=forbidden`, `EVIDENCE_REMOTE_UPDATE=allowed` only on the dedicated evidence branch, `SEMANTIC_REMEDIATION=forbidden`, `MECHANICAL_REMEDIATION=forbidden`. No new PR, source edits, forced git mutation, or issue #2 closure. On a real source defect, stop with `FAIL` and return reproduction evidence for Central.

No GPU ASR/SwiftU inference is required or implied. Do not start/restart/unload vLLM, SwiftU, checkpoints, system services, or existing operator Studio instances. Do not touch `/workspace/pyannote`, its venv or media. Do not stop an already bound port's owner or tamper with the preexisting job root. Use an unused loopback port and a brand-new job root. No real TV/anime media, no external URL download.

## Source authentication and execution bindings

1. Read `AGENTS.md`, `web/README.md`, this handoff, and issue #2. Re-read live remote `main` and issue states. If `main` no longer names the required SHA, return `STALE` without silently testing another revision. The previous merge is not authorization for a changed commit.
2. Run `git fetch origin refs/heads/main` **before** local object proof; show exact `FETCH_HEAD` SHA, `git cat-file -t 242c418f55cd2afb796ee2c4fa742a10ff04f7a3`, and `git rev-parse 242c418f55cd2afb796ee2c4fa742a10ff04f7a3^{tree}`. Expect `commit` and tree `122f8466db22098ba9427373af7e87ceb8421076`. Preserve a dirty normal workspace: use a fresh detached worktree or verified clean checkout, no `reset --hard`, `git clean`, forced branch switch or destructive overwrite.
3. Host runtime source identity is **not** implied by worktree HEAD. Ensure the actual Python process imports `nvidia_transcribe.studio_api` from that authenticated worktree, and records its `__file__` and `PROJECT`. If reusing `/workspace/nvidia-transcribe/.venv-studio/bin/python` from a detached worktree, bind `PYTHONPATH=<authenticated worktree>/src` to both Studio and CLI subprocesses and verify those imports. Set `NEMOVOX_CLI_PYTHON` to the existing dedicated CLI venv executable at `/workspace/nvidia-transcribe/.venv/bin/python` if present; no model service activation.
4. Build `web/dist` from the authenticated worktree (`cd web && npm test && npm run build`) if absent/stale. Do not reuse `dist` from an unrelated source head. Frontend source had eight Vitest tests on the prior equal-tree head. Existing full CPU reports (85 in each Python environment) remain provenance; optional rerun is inexpensive but **fresh-process behavior cannot be substituted** by old test reports.

## Independent fresh Studio process, isolated roots

- Create collision-safe private scratch under a caller-owned writable filesystem (e.g., unique `/tmp/nemovox-postmerge-...`); preserve it for evidence until review. Prepare `media/`, `jobs/`, `uploads/`, `reports/`. Create them **before** process launch.
- Set `NEMOVOX_STUDIO_JOBS=<scratch>/jobs`, `NEMOVOX_STUDIO_UPLOADS=<scratch>/uploads`, `NEMOVOX_MEDIA_ROOTS=<scratch>/media`, `NEMOVOX_CLI_PYTHON=<existing CLI Python>`, and `PYTHONPATH=<authenticated worktree>/src`. Use an unused TCP port and `127.0.0.1` only.
- Start a **new** `.venv-studio/bin/python -m uvicorn nvidia_transcribe.studio_api:app --host 127.0.0.1 --port <port> --workers 1` process with authenticated `cwd`, env and captured PID/start timestamp. This is a fresh **Studio process**, not merely refreshing an old web page. Keep a bounded private startup log. Capture process PID, executable, cmdline/cwd, listening port, actual imported source/module, `GET /api/health` 200 and a real frontend document response from `/`. If an unrelated service occupies the port, select another; do not kill it.
- Attempt a second Studio process using the **same isolated job root** but a different unused loopback port. Its startup must reject exclusive ownership, while the first remains healthy and its job state unchanged. Terminate only owned test children. If a test cannot be safely launched owing to harness permission, mark BLOCKED rather than substitute process simulations.
- After completing the synthetic job, gracefully terminate **only** the owned first Studio PID, verify exit, then launch a second **new** Studio PID with the same isolated roots. Confirm completed job history, preview and artifacts survive restart. Optionally stage a synthetic queued record while the owner is stopped and verify it transitions to `interrupted` on next owned startup (do not assert unknown crash descendants were killed).

## End-to-end API and bounded browser acceptance

**Actual CPU-only queued job:** Create an original synthetic `recording.contract.json` under scratch media, with at least one timed and one untimed utterance containing distinct source-language Unicode and speaker IDs. A small locally generated synthetic WebM source path may be embedded in its `source.path` for video preview (FFmpeg only if present; never use real media). Submit `POST /api/jobs` with `source.kind=contract` and `task=transcription`. Poll the actual live API through `queued/running -> complete` with a bounded deadline, preserving intermediate states if observable. A very short job can legitimately complete before intermediate polling. Assert `GET /api/jobs/{id}` reports complete and job-owned `asr/recording.contract.json` and `asr/transcript.txt` exist. Verify imported source text/speaker/timed-vs-untimed fidelity, `GET /preview` cues, `GET /artifacts` download links, and exact HTTP content. Capture access control negatives: outside-root media path yields 403, invalid contract/unsupported media is rejected, private/localhost remote URL yields 422 without fetching. Avoid file paths in published evidence unless sanitized.

**Synthetic render-only artifact check:** In that *isolated test job root only*, write minimal valid `.srt` and `.ass` artifacts with a filename such as `Episode #1? 50% 日本語.en.{srt,ass}`. Clearly label these as **injected fixtures**, not generated by ASR/translation. Assert the API returns percent-encoded URLs retaining directory separators; both HTTP download responses are 200 and match original bytes/SHA-256. With a synthetic browser-compatible video, use one bounded Chromium/Playwright smoke at the served UI to verify the job URL opens, dialogue and download links appear, and JASSUB fetches the encoded `.ass` via HTTP 200 and reaches `ASS · libass` if the synthetic video/codec and WASM environment support it. Otherwise report exact browser/codec limitation, demonstrate HTML fallback, and retain prior browser evidence separately. No need to repeat the prior multi-scenario browser campaign.

**Lifecycle/cancellation:** Run existing focused Studio API regressions from the merged source, including the real SIGTERM-resistant subprocess descendant case, terminal cancellation serialization, and two-server owner exclusion, e.g. `.venv-studio/bin/pytest -v tests/test_studio_api.py`. Preserve real command output/exit and check no owned test process remains. Do not simulate a clean SIGKILL or claim guaranteed orphan cleanup after a forced crash.

**No hidden expansion:** Do not run full live Nemotron inference, transcript correction, forced alignment benchmarks, external media ingestion, or broad architecture refactoring. Do not introduce additional PRs or repeated campaigns for unrelated concerns.

## Criterion-to-evidence matrix

| Criterion | Required evidence artifact | Assertion |
|---|---|---|
| Merged-main source | `source-identity.txt` | `main` SHA, worktree tree, `studio_api.__file__`, interpreter, `web/dist` build identity |
| Fresh startup & exclusivity | `fresh-process.json`, bounded `startup.log` | Two genuine new PID/start observations; localhost health/UI; second concurrent owner denied |
| Processing & fidelity | `api-job.json`, `api-requests.txt` | Actual postmerge job completes; source Unicode/timing/speaker fidelity; download returns original bytes |
| URL and browser | `browser-review.md`, optional sanitized screenshots | Correct encoded request paths, SRT/ASS byte hashes, browser fallback or JASSUB HTTP success |
| Persistence & cleanup | `restart-cleanup.md` | Graceful owned-server exit, new PID, same job retained, no test-owned descendants |
| Regression suite | `test-results.md`, bounded raw exit logs | Studio API tests and frontend build/test outcome; failures preserved |
| Privacy/provenance | `manifest.json`, `HANDback.md`, archive/checksum | All required evidence, explicit omissions and limitations, exact source/evidence SHAs |

## Publication, handback and closure boundary

1. Produce one **sanitized, self-contained, operator-attachable `.tar.gz`** with the named evidence, actual producer exits, hashes, bounded process/API/browser observations, and member inventory. Do not include secrets, personal host paths, actual media, screenshots of private content, model logs, or full raw shell history. Scan every archive member, verify member names/digests and `gzip -t`; compute final archive SHA-256 outside the manifest. Keep full private raw logs local if needed, but publish only bounded sanitized excerpts.
2. Publish at most **one evidence-only commit** under `staging/agent-evidence/issue-2-postmerge/` on existing `evidence/issue-2-postmerge-20261121`; preserve this handoff and leave `src/`, `tests/`, `web/` and production docs identical to merged `main`. Exact-read the evidence branch head, fast-forward/no-force push, remote readback. No PR.
3. Report `PASS | FAIL | BLOCKED | STALE`, exact merged-main SHA, actual executable/module/PIDs/start times, command exits, source/evidence branch SHAs, archive path/size/SHA-256, per-criterion verdict, omissions and `COMMAND_SHAPE_FRICTION`. Specifically include `EVIDENCE_ARCHIVE_REQUIRED=yes`, `EVIDENCE_ARCHIVE_STATUS`, `EVIDENCE_ARCHIVE_PATH`, `EVIDENCE_ARCHIVE_SHA256`, `EVIDENCE_ARCHIVE_BYTES`, `EVIDENCE_MANIFEST`, `EVIDENCE_REQUIRED_ARTIFACTS`, `EVIDENCE_INCLUDED_ARTIFACTS`, `EVIDENCE_MISSING_ARTIFACTS`, `EVIDENCE_PRIVACY_CHECK`, `EVIDENCE_INTEGRITY_CHECK`, `CENTRAL_REVIEW_ATTACHMENT_REQUIRED=yes`.
4. End handback with `REQUIRED OPERATOR ACTION: Attach the verified evidence archive to the next Central ChatGPT message together with this handback.`
5. **Stop.** Issue #2 stays OPEN until Central reviews merged-main host evidence and separately authorizes/completes closure. No source edits, issue closure, merge, review writes, external-service intervention, or additional GitHub PR.

Mandatory command-shape ledger:

```text
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events:
    - scenario=<short-key>
      requested_shape=<requested command>
      effective_shape=<executed/rejected command|same-as-requested|unknown>
      rewrite_provenance=<none|rtk_opencode_hook|shell_or_alias|other_harness_hook|unknown>
      event_kind=<pre_execution_rejection|execution_failure|successful_recovery|guard_annotation|harness_rewrite|runtime_misbinding|interface_discovery|other>
      execution_effect=<not_executed|executed_failed|executed_succeeded|recovered_after_retry|unknown>
      correction_or_annotation=<bounded finding>
      replacement_shape=<successful corrected shape|none>
      recipe_reference=<GHDEV-10 key|none>
      resource_reference=<resource|none>
      contract_conflict=<yes|no|uncertain>
      durable_gap_candidate=<yes|no|uncertain>
  omitted_sensitive_values=<yes|no>
```

If no friction: `encountered=no; events=[]; omitted_sensitive_values=no`. Recovered failures still count if material.
