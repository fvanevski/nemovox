# Phase A: Riva Automatic-Device Admission Trace (20261122T170806Z)

- **Source SHA:** `2bce0d8b3730c7e8784eef9b10cc616c273e86d2`
- **Target code:** `src/nvidia_transcribe/translation.py` (`translate_utterances`)
- **Provenance class:** Deterministic CPU execution trace with mocked `subprocess.run` and `TracedLifecycle` (`tests/test_translation.py` + direct candidate trace verification).

## Verified P1 Invariants
1. **Uncached `riva_device="auto"`:**
   - Events: `[["lease_enter", "riva_local"], ["worker_spawn", "auto", 1], ["lease_exit", "riva_local"]]`
   - Result: `status="ok"`, `device_used="cuda:0"` — lease acquired **before** worker invocation, retained across completion, released after.
2. **Uncached explicit `riva_device="cuda:0"`:**
   - Events: `[["lease_enter", "riva_local"], ["worker_spawn", "cuda:0", 1], ["lease_exit", "riva_local"]]`
   - Result: `status="ok"`, `device_used="cuda:0"`.
3. **Uncached explicit `riva_device="cpu"`:**
   - Events: `[["worker_spawn", "cpu", 1]]`
   - Result: `status="ok"`, `device_used="cpu"` — preserves explicit CPU route without claiming the GPU lease.
4. **Denied admission (`ResourceLifecycleError("resource_lease_timeout")`):**
   - Events: `[["lease_enter", "riva_local"]]`
   - Error: `ResourceLifecycleError:resource_lease_timeout` — worker subprocess is **never** spawned.
5. **Failed worker launch unwinds lease:**
   - Events: `[["lease_enter", "riva_local"], ["worker_spawn", "auto", 1], ["lease_exit", "riva_local"]]`
   - Error: `RuntimeError:boom` — lease cleanly released in context manager exit.
6. **English-to-English bypass (`source_language="en"`, `riva_device="auto"`):**
   - Events: `[]`
   - Result: `status="bypassed_same_language"`, `device_used=null` — no lease entered, no worker spawned.
7. **Cached translation hit (`resume=True`, `riva_device="auto"`):**
   - Events: `[]`
   - Result: `status="ok"`, `device_used=null` — no lease entered, no worker spawned.
