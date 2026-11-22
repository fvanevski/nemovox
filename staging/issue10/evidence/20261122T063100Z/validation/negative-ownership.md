# Phase C: Negative Ownership & Marker-Isolation Probes (`20261122T063100Z`)

Executed read-only against live `systemd --user` on canonical `~/.local/state/nemovox/model-lifecycle.lock`, without altering any unowned service.

1. **Unmarked Nemovox unit (`nemovox-studio.service` mapped to `asr_local`):**
   - `ModelLifecycle.inspect(asr_local)` -> `{"state": "managed_unit_not_configured", "ready": false, "managed": false, "startable": false}`
   - `ModelLifecycle.endpoint(asr_local)` -> `ResourceLifecycleError(status="managed_unit_not_configured", detail="asr_local: no verified Nemovox-owned service for unavailable endpoint")`
   - Production `nemovox-studio.service` (`MainPID=2922624`, `127.0.0.1:8765`) remained untouched and `active`.
2. **Non-Nemovox external unit (`litellm-proxy.service` mapped to `asr_local`):**
   - Rejected during `_service_config` parsing -> `ResourceLifecycleError(status="invalid_lifecycle_config", detail="Invalid unit, timeout or release policy for asr_local")`.
3. **Mismatched identity markers (`nemovox-asr.service` mapped to `text_local`):**
   - `ModelLifecycle.inspect(text_local)` -> `{"state": "managed_unit_not_configured", "ready": false, "managed": false, "startable": false}`
   - `ModelLifecycle.endpoint(text_local)` -> `ResourceLifecycleError(status="managed_unit_not_configured", detail="text_local: no verified Nemovox-owned service for unavailable endpoint")`.
