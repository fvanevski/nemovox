# Restart Persistence, Exclusive Ownership & Descendant Cleanup

## 1. Concurrent Secondary Server Exclusion

- **Primary Studio process (PID #1):** PID `4003808` started `Sat Nov 21 11:33:18 2026` on `127.0.0.1:18765` owning `NEMOVOX_STUDIO_JOBS=/tmp/nemovox-postmerge-scratch-20261121/jobs`.
- **Concurrent secondary Studio process (PID #2):** PID `4003880` launched on `127.0.0.1:18769` against the same `NEMOVOX_STUDIO_JOBS` failed startup with exit code `3` and `RuntimeError: Studio job directory is already owned by another running server.` (via `fcntl.flock(LOCK_EX | LOCK_NB)` on `.studio-owner.lock`), leaving PID #1 healthy and unaffected.

## 2. Graceful Shutdown & Fresh Restart Survival (PID #1 -> PID #3)

- Gracefully terminated owned PID `4003808` (`SIGTERM`, exit `-15`); Uvicorn logged clean FastAPI lifespan shutdown (`Application shutdown complete.`).
- While stopped, staged an interrupted orphan job `eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee` with `state="queued"`.
- Launched fresh Studio **PID #3** (`4004565`, started `Sat Nov 21 11:33:23 2026` on `127.0.0.1:18765`) against the same isolated roots:
  - Previously completed job `c46f34300e0f4e1c8334edaf96334344` persisted cleanly as `complete`, retaining `GET /preview` cues (`1` cue) and all `4` downloadable artifacts.
  - Staged orphan `eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee` transitioned on authorized startup to `state="interrupted"`.
- Gracefully terminated PID `4004565` (`SIGTERM`, exit `-15`).
- Process scan (`ps -ef` filtered for `nemovox-postmerge`) confirmed **0** leftover test-owned processes (`OWNED_REMAINING: 0`).

## 3. Focused Lifecycle & Cancellation Suite

- `.venv-studio/bin/pytest -v tests/test_studio_api.py` executed from the isolated `main` worktree passed all `14` tests (`exit 0`), including:
  - `test_cancel_cleans_descendant_after_group_leader_exits` (`SIGTERM`-resistant child after exited session leader terminated cleanly via Linux `/proc` session scan + `SIGTERM`/`SIGKILL`).
  - `test_app_creation_does_not_reconcile_another_running_server`
  - `test_cancel_and_completion_have_serialized_terminal_state`
  - `test_cancel_process_group_and_env_forwarding`
