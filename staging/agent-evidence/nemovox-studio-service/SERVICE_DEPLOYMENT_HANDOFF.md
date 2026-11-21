# Host deployment contract: Nemovox Studio persistent systemd user service

## Exact authority and verified prerequisites

- Repository: `fvanevski/nemovox`.
- Source `main` at creation: `242c418f55cd2afb796ee2c4fa742a10ff04f7a3`; Git tree `122f8466db22098ba9427373af7e87ceb8421076`.
- Prior GUI acceptance: PR #3 MERGED, issue #2 CLOSED, merged-main fresh-process evidence `evidence/issue-2-postmerge-20261121@7ee376c07fcac0d3125231dc16426564094a3332`.
- Target host: user's Garuda Linux system, project normally `/workspace/nvidia-transcribe`, `.venv-studio` for FastAPI/uvicorn, `.venv` for CLI.
- Host operator readback immediately before staging: no matching running Studio process; zero matching user/system unit files; `loginctl show-user "$USER" -p Linger` returned `Linger=yes`. **These are historical observations; recheck before any write.**
- Centrally authored unit: `staging/agent-evidence/nemovox-studio-service/nemovox-studio.service`; optional environment example `studio.env.example`; this handoff in the same directory.
- Evidence/deployment branch `evidence/nemovox-studio-service-20261121` contains **only deployment artifacts**, not an implementation PR. The unit is intentionally not merged into `main` under this narrow host-deployment request.

## Execution boundary

`LOCAL_AGENT_PRIMARY=build`. This is a bounded **host configuration** mutation, already requested by the operator: deploy one persistent loopback-only Nemovox Studio user service, enable/start it through `systemctl --user`, verify restart and boot-enable configuration, and return inspectable host evidence. Agent owns host-bound installation, environment inspection, and verification, not architecture redesign.

Allowed:
- Fetch exact authenticated GitHub refs and materialize only the staged unit/example/handoff.
- Inspect host filesystem, ports, current service state and available venv binaries; build existing merged GUI frontend as needed; use `uv pip`/existing lockfile only to restore already-declared Studio dependencies when actually missing.
- With existing project worktree **clean and on the intended source commit**, fast-forward an ordinary local `main` checkout to the authenticated remote `main`. If tracked local changes would be overwritten, never reset/stash/clean them: use a separate persistent clean worktree at an available location and mechanically substitute only its confirmed root in `WorkingDirectory`, `PYTHONPATH`, and `ExecStartPre`; keep `NEMOVOX_CLI_PYTHON` bound to the existing primary repo venv. Record chosen source-root identity and the diff from the staged unit.
- Choose a different unused loopback port only if 8765 is already owned, and report effective URL. Do not stop another process to free the port.
- Install the unit into `~/.config/systemd/user/nemovox-studio.service`, optionally create `~/.config/nemovox/studio.env` for verified environment overrides, reload the user manager, `enable --now`, restart once when idle, and retain the service running.
- Publish one sanitized evidence-only commit to the existing deployment evidence branch and a corresponding small attachable archive.

Not allowed:
- No sudo/system-wide unit installation, no change to `Linger` (already yes), no reboot required, no destructive workspace operations, no model checkpoint/weights changes, no modification of `/workspace/pyannote`, no service stop/restart outside the newly owned `nemovox-studio.service`, and no new PR or GitHub issue close.
- Do not auto-start/stop vLLM, Nemotron, SwiftU, embeddings or other GPU services. No full episode inference. Do not import personal media into test evidence.
- No modifications to Python source, tests, lockfiles, model or GUI semantics. Unexpected semantic/runtime defects: return FAIL with reproducible evidence for Central.

## Phase A: admission and dependency/source proof

1. Read `AGENTS.md`, `web/README.md`, this handoff and the staged unit. Re-read GitHub `origin/main` SHA and unit source branch SHA. Fetch remote selectors **before** local `git cat-file`, `git show` or worktree materialization. Verify main commit and Git tree. If `main` changed since `242c418f...`, stop with STALE pending Central readjudication.
2. Authenticate current active processes and user/system unit files again with `pgrep -af 'uvicorn.*nvidia_transcribe.studio_api'`, `systemctl --user list-unit-files '*nemovox*' '*studio*'`, and `systemctl list-unit-files '*nemovox*' '*studio*'`. Verify effective `Linger=yes`. Inspect `ss -ltnp` for localhost port 8765. If an existing installed unit, override, drop-in or unknown server is found, **inspect before replacement** and stop on unexplained drift; do not clobber host-managed state.
3. Authenticate `/workspace/nvidia-transcribe` and real user home, interpreter paths, `ffmpeg`, `ffprobe`, `yt-dlp`, and `node/npm` if a frontend build is required. Do not print secrets or export full environment. Assert the actual interpreter imports `nvidia_transcribe.studio_api` from the authenticated source root (unit `PYTHONPATH` is deliberately explicit), and check the CLI Python executable exists. Confirm `web/dist/index.html` belongs to a fresh build from this exact source tree: run `cd web && npm test && npm run build` using existing deps/lockfile (if dependencies absent, `npm ci` with the committed lockfile; no dependency upgrades). Capture build commands, versions and exits.
4. Preserve existing `outputs/studio/jobs` and `outputs/studio/uploads`. Create missing directories with private permissions but never delete historical records. Confirm no already active queue owner. The existing API's own startup lock controls exclusivity.

## Phase B: install and activate the single user unit

1. Copy the staged `nemovox-studio.service` verbatim to a temporary same-directory file under `~/.config/systemd/user/`, after checking path ownership and absent/exact prior bytes. When necessary, apply **bounded, documented host-only substitutions** to authenticated worktree root, path used for required executable/CLI and port. Preserve core invariants: loopback bind; one Uvicorn worker; existing Studio/CLI venvs; persistent job/upload roots; `EnvironmentFile=-...` optional; `Restart=on-failure`; `KillMode=control-group`; `TimeoutStopSec=45s`; `UMask=0077`; `WantedBy=default.target`. Do not infer real ASR/SwiftU base URLs or model names.
2. Validate the **staged candidate** with `systemd-analyze verify` (on-host where available); make the candidate file's basename end in `.service` for correct parsing. If valid, promote the candidate to `~/.config/systemd/user/nemovox-studio.service` via atomic same-filesystem rename. **Only after promotion**, run `systemctl --user daemon-reload` and `systemctl --user cat nemovox-studio.service`, and verify the loaded unit bytes/digest against the candidate. Do not overwrite a conflicting unit. If a host override file is needed, write permissions `0600`; do not store credentials or private media paths in public evidence.
3. Run `systemctl --user enable --now nemovox-studio.service`. Confirm `is-enabled=enabled`, `is-active=active`, nonzero `MainPID`, actual `FragmentPath`, `ExecStart`, bind port and process executable; `curl -fsS http://127.0.0.1:<PORT>/api/health` returns status `ok`; `curl` GUI root returns HTML with nonempty asset references and bundled JS/CSS reachable. Check `journalctl --user -u nemovox-studio.service --no-pager` for startup exceptions.
4. With an idle job queue, `systemctl --user restart nemovox-studio.service`; demonstrate a **new** `MainPID` and active service, reused persistent job directory and reachable health/GUI. When feasible, execute one CPU-only synthetic contract transcription through the actual service and confirm complete/preview/artifacts/history on both sides of restart, without media leakage. If the original persistent job root contains active work, do not interrupt it merely to run the test; use nonmutating health/UI checks and report deferred restart as a limitation.
5. Confirm effective `Environment` does not accidentally invoke local model services; readback environment property only with redaction. From service context check `/api/health` flags for ffmpeg/yt-dlp and accurately report unavailable tools. Confirm no unexpected child process remains after controlled restart.

## Boot and rollback evidence

`Linger=yes` and an enabled `WantedBy=default.target` user unit establish **configured startup at boot even before login**. This does *not* prove a physical reboot was performed. Record separate fields:
- `USER_SYSTEMD_ENABLEMENT=enabled|not-enabled`
- `USER_SYSTEMD_ACTIVE=active|failed|inactive`
- `LINGER=yes|no`
- `BOOT_STARTUP_CONFIG=PASS|FAIL|UNVERIFIED`
- `POST_REBOOT_OBSERVED=yes|no` (expect no unless an operator-approved reboot actually occurs)
- `GUI_URL=http://127.0.0.1:<PORT>/`
- `ACTUAL_MODULE_PATH=`, `SOURCE_SHA=`, `SERVICE_PID_BEFORE/AFTER=`.

Rollback **only if installation/activation fails**, and only for this task's installed unit: stop/disable the newly installed service, restore a verified prior unit only if one was identified/backed up, `daemon-reload`, verify there are no newly owned running processes. Never remove jobs, uploads, original source worktrees, or toggle user lingering. If removing or changing unknown existing service state would be required, BLOCK and escalate; no broad `killall`.

## Evidence publication and stop

Produce:
- `HANDback.md`: PASS/FAIL/BLOCKED/STALE, exact main SHA, staged service SHA + effective installed-unit SHA and adaptations, current URL, enablement/active/Linger/boot-config vs reboot distinction, tests, limitations, and rollback state.
- `service-identity.txt`: systemctl status/show (sanitized), MainPID/bind, import path and interpreter, unit hash, job-root path class, daemon-reload/start/restart command exit codes.
- `service-health.json`: bounded health and GUI/JS/CSS responses (no secrets).
- `test-results.md`: source/front-end build and optional synthetic job, restart proof, failures or skipped cases.
- `manifest.json`: SHA256 for every archived member, exact source/branch/installed unit identity; sanitized journal snippets and command/output excerpt.
- One small **operator-attachable** `tar.gz`, with member inventory and hash, excluding personal media, credentials, environment dumps, full logs, or other nonessential host data. Verify `gzip -t` and member digests; include `COMMAND_SHAPE_FRICTION` ledger. Hash and read back GitHub evidence branch commit/archive.

Publish at most one evidence-only commit under `staging/agent-evidence/nemovox-studio-service/` on the *existing* evidence branch, without altering staged unit or production source. If an effective host substitution is needed, publish a sanitized copy of the **effective installed unit** and its hash in the evidence branch (no private secrets). No PR, merge, issue mutations, or new Codex request. Leave service ACTIVE/ENABLED if PASS. Return evidence links and explicit `REQUIRED OPERATOR ACTION: Attach the verified evidence archive for Central review.`

Mandatory canonical ledger even if empty:
```text
COMMAND_SHAPE_FRICTION:
  encountered=<yes|no>
  events: []
  omitted_sensitive_values=<yes|no>
```
When friction occurs, expand events with `scenario`, `requested_shape`, `effective_shape`, `rewrite_provenance`, `event_kind`, `execution_effect`, `correction_or_annotation`, `replacement_shape`, `recipe_reference`, `resource_reference`, `contract_conflict`, `durable_gap_candidate` as prescribed by GHDEV-02/10. Never hide host state or claim boot-reboot validation without an observed reboot.
