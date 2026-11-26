# Resource lifecycle

MOSS and Riva are isolated local GPU workers, admitted under the cooperative model-stage lock and actual free-memory checks. They exit when the stage finishes. Never stop an unrelated service, silently downgrade device or implicitly download weights. The optional FireRed AED observation worker explicitly uses CPU.

Text inference may use a healthy unowned endpoint without modifying it, or acquire a verified Nemovox-owned service using the configured unit, model, endpoint and resource markers. Stop only the exact owned invocation when safe; never preempt live external clients or active cooperative leases. Startup/readiness, timeouts, signal handling and idle recovery remain subject to resource_lifecycle.py. The example lifecycle config now contains only the owned text-review unit.

NInfer Messages enforces the supported thinking contract independently of final response capacity; do not silently fall back to unbounded thinking. Offline rerender does not acquire inference resources. Studio status is read-only; an installed local worker is startable, not an already-running HTTP endpoint.

Migration: prepare private resource/lifecycle config without retired ASR service mappings, verify in an isolated Studio instance, and leave live host services/checkpoints untouched until separately authorized deployment.
