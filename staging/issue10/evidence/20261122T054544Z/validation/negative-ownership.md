# Phase C: Negative Ownership & Marker-Isolation Probes (`20261122T054544Z`)

Executed read-only against live host `systemd --user` (`ModelLifecycle.inspect()` and `ModelLifecycle.endpoint()` failure paths), without starting, stopping, or altering any unowned unit.

## Probe Results (`tests`: direct Python calls against isolated temp configs)
1. **Unmarked Nemovox unit (`nemovox-studio.service` mapped to `asr_local`):**
   - `ModelLifecycle.inspect(asr_local)` returned:
     `{"state": "managed_unit_not_configured", "ready": false, "managed": false, "startable": false}`
   - `ModelLifecycle.endpoint(asr_local)` raised `ResourceLifecycleError`:
     `status="managed_unit_not_configured"`, `error="asr_local: no verified Nemovox-owned service for unavailable endpoint"`
   - Confirmed production `nemovox-studio.service` (`127.0.0.1:8765`, `MainPID=2922624`) remained untouched and `active`.

2. **Non-Nemovox external unit (`litellm-proxy.service` mapped to `asr_local`):**
   - Rejected during `_service_config` parsing before any systemd lookup:
     `ResourceLifecycleError(status="invalid_lifecycle_config", detail="Invalid unit, timeout or release policy for asr_local")`

3. **Mismatched identity markers (`nemovox-asr.service` mapped to `text_local`):**
   - `nemovox-asr.service` declares `NEMOVOX_RESOURCE_NAME=asr_local`, `NEMOVOX_RESOURCE_MODEL=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`, `NEMOVOX_RESOURCE_ENDPOINT=http://127.0.0.1:8190/v1`.
   - When inspected/acquired under `text_local` (`chat`, `http://127.0.0.1:8001/v1`):
     - `ModelLifecycle.inspect(text_local)` returned `{"state": "managed_unit_not_configured", "ready": false, "managed": false, "startable": false}`
     - `ModelLifecycle.endpoint(text_local)` raised `ResourceLifecycleError(status="managed_unit_not_configured", detail="text_local: no verified Nemovox-owned service for unavailable endpoint")`
