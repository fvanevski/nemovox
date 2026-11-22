# Unit Rollback & Post-Test Host Verification (`20261122T054544Z`)

## Rollback Preimage & Procedure
- Prior to provisioning, neither `nemovox-asr.service` nor `nemovox-text-review.service` existed under `/home/filip/.config/systemd/user/` (recorded as `ABSENT=nemovox-asr.service`, `ABSENT=nemovox-text-review.service`).
- Complete rollback command if reverting host unit provisioning:
  ```bash
  systemctl --user stop nemovox-asr.service nemovox-text-review.service || true
  rm -f ~/.config/systemd/user/nemovox-asr.service ~/.config/systemd/user/nemovox-text-review.service
  systemctl --user daemon-reload
  ```
- Under this pre-merge verification campaign, both static units are left stopped (`inactive/dead`), un-enabled (`UnitFileState=static`), and production `nemovox-studio.service` (`127.0.0.1:8765`, `MainPID=2922624`) was never modified or restarted.

## Verified Post-Test Host State
- Candidate Studio (`127.0.0.1:8767`): stopped; port `8767` closed.
- `nemovox-asr.service`: `ActiveState=inactive`, `UnitFileState=static`, `MainPID=0`.
- `nemovox-text-review.service`: `ActiveState=inactive`, `UnitFileState=static`, `MainPID=0`.
- `nemovox-studio.service`: `ActiveState=active`, `UnitFileState=enabled`, `MainPID=2922624`, HTTP `GET http://127.0.0.1:8765/api/health` -> `200`.
- GPU (`nvidia-smi`): `3049 MiB used / 29078 MiB free`, `0 %` utilization; `/workspace/nvidia-transcribe` main checkout clean.
