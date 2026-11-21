# Live Host Promotion (`/workspace/nvidia-transcribe`)

- **Pre-Promotion Inspection:**
  - Live root `/workspace/nvidia-transcribe` on `main...origin/main` at `242c418f55cd2afb796ee2c4fa742a10ff04f7a3` (tree `122f8466db22098ba9427373af7e87ceb8421076`), clean tracked & untracked worktree (`git status --short --untracked-files=all` empty).
  - Checked all `outputs/studio/jobs/*/job.json` (`1` job: `9be92af3ba144e6eb984ee43af0807de`, `state="complete"`). Zero `queued` or `running` jobs.
  - Private rollback snapshot of `nemovox-studio.service` and prior commit identity retained in local private scratch directory outside Git/public packet.
- **Controlled Stop & Fast-Forward:**
  - Re-fetched `refs/heads/main` and verified `FETCH_HEAD == 7b4112cc31784953abbb041fd0b5d7997851841d`.
  - Stopped `nemovox-studio.service` once while idle (`MainPID=4157910` -> `ActiveState=inactive`, old PID verified gone).
  - Executed `git -C /workspace/nvidia-transcribe merge --ff-only FETCH_HEAD`.
  - Verified post-promotion `HEAD == 7b4112cc31784953abbb041fd0b5d7997851841d`, `HEAD^{tree} == 3d2cc429f9b49da3079df7d1fc178f6fd09d7987`, clean worktree.
  - Verified `.venv/bin/python` and `.venv-studio/bin/python` both resolve `nvidia_transcribe` to `/workspace/nvidia-transcribe/src/nvidia_transcribe/__init__.py` without modifying packages or venvs.
  - Verified `src/nvidia_transcribe/_qwen_align_worker.py`, `outputs/modular-eval/.venv/bin/python`, and `/workspace/models/Qwen3-ForcedAligner-0.6B` present.
- **Preserved Artifacts:** `.venv`, `.venv-studio`, `outputs/`, `web/dist/`, and `nemovox-studio.service` unit file preserved intact.
