# Nemovox Studio Systemd User Service Deployment Handback

**Verdict:** `PASS`

## Authoritative Source, Deployment & Service Identity

- **Repository:** `https://github.com/fvanevski/nemovox`
- **Source `origin/main` SHA:** `242c418f55cd2afb796ee2c4fa742a10ff04f7a3` (Git tree `122f8466db22098ba9427373af7e87ceb8421076`)
- **Deployment handoff base (`evidence/nemovox-studio-service-20261121`):** `e739cd604088e65a1a2d674ee563fe3a83d5dc73`
- **Staged unit SHA-256:** `118c256bc94768b52c2f286c8b3cbdc9cf28423718413bdd6e9e3ba2107342d3`
- **Installed unit (`<USER_HOME>/.config/systemd/user/nemovox-studio.service`) SHA-256:** `118c256bc94768b52c2f286c8b3cbdc9cf28423718413bdd6e9e3ba2107342d3` *(copied verbatim; zero host substitutions required)*
- **Optional environment override (`<USER_HOME>/.config/nemovox/studio.env`):** not created (default unit environment cleanly exposes `PATH`, `PYTHONPATH`, `NEMOVOX_CLI_PYTHON`, `NEMOVOX_STUDIO_JOBS`, `NEMOVOX_STUDIO_UPLOADS` with zero accidental model-endpoint overrides)
- **Effective GUI URL:** `http://127.0.0.1:8765/`
- **Actual imported Python module:** `/workspace/nvidia-transcribe/src/nvidia_transcribe/studio_api.py` via `/workspace/nvidia-transcribe/.venv-studio/bin/python`
- **Service PIDs:** `4154778` (initial `enable --now`) -> `4157910` (post-restart, active and running)

## State & Boot Configuration

```text
USER_SYSTEMD_ENABLEMENT=enabled
USER_SYSTEMD_ACTIVE=active
LINGER=yes
BOOT_STARTUP_CONFIG=PASS
POST_REBOOT_OBSERVED=no
GUI_URL=http://127.0.0.1:8765/
ACTUAL_MODULE_PATH=/workspace/nvidia-transcribe/src/nvidia_transcribe/studio_api.py
SOURCE_SHA=242c418f55cd2afb796ee2c4fa742a10ff04f7a3
SERVICE_PID_BEFORE=4154778
SERVICE_PID_AFTER=4157910
```

- **Boot config vs actual reboot:** `Linger=yes` and `WantedBy=default.target` symlink (`default.target.wants/nemovox-studio.service`) establish configured startup at boot before login (`BOOT_STARTUP_CONFIG=PASS`). An actual physical machine reboot was **not** executed (`POST_REBOOT_OBSERVED=no`).
- **Rollback state:** Activation succeeded cleanly on the first attempt; no rollback was needed. The service remains **enabled and running** (`PID 4157910`).

## Limitations & Safeguards

- Live GPU Nemotron ASR (`127.0.0.1:8190`) and SwiftU (`127.0.0.1:8001`) services were not started or modified and remain **UNVERIFIED**.
- `/workspace/pyannote` and `/workspace/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` were untouched.
- Synthetic CPU contract transcription job `9be92af3ba144e6eb984ee43af0807de` was verified before and after `systemctl --user restart nemovox-studio.service`.

## Command-Shape Friction

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events: []
  omitted_sensitive_values=no
```

REQUIRED OPERATOR ACTION: Attach the verified evidence archive for Central review.
