# Post-Test State & Rollback Verification (`20261122T063100Z`)

## Verified Clean Post-Test Host State
- Candidate Studio (`127.0.0.1:8767`): stopped; port `8767` closed.
- `nemovox-asr.service`: `ActiveState=inactive`, `UnitFileState=static`, `MainPID=0`.
- `nemovox-text-review.service`: `ActiveState=inactive`, `UnitFileState=static`, `MainPID=0`.
- `nemovox-studio.service`: `ActiveState=active`, `UnitFileState=enabled`, `MainPID=2922624`, HTTP `GET http://127.0.0.1:8765/api/health` -> `200`.
- GPU (`nvidia-smi`): `3048 MiB used / 29079 MiB free`, `0 %` utilization; `/workspace/nvidia-transcribe` main checkout clean.

## Rollback Reference
If reverting host user unit provisioning is ever desired:
```bash
systemctl --user stop nemovox-asr.service nemovox-text-review.service || true
rm -f ~/.config/systemd/user/nemovox-asr.service ~/.config/systemd/user/nemovox-text-review.service
systemctl --user daemon-reload
```
