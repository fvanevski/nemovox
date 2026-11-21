# Per-Codex-Finding Remediation & Adversarial Verification (`1702608f82808fca30fb3eee1b62954230e6da32`)

Raw standalone adversarial verification trace: [`logs/adversarial_codex_checks.json`](logs/adversarial_codex_checks.json).

## Finding 1 (P1, `#discussion_r4655089438`): Exact cooperative ownership & crash-recovery authorization

- **Implementation (`src/nvidia_transcribe/gpu_handoff.py`):**
  - `_has_exact_cooperative_markers(environment, lease_path)` parses systemd `Environment=` assignments with `shlex.split`, failing closed on `ValueError` or tokens lacking `"="`, and requires `observed["NEMOVOX_MANAGED_ASR"] == ["1"]` and `observed["NEMOVOX_ASR_LEASE_PATH"] == [str(lease_path)]`.
  - Called in both `_verified_owner(cfg)` and `recover_pending_managed_asr(cfg)` before any process/systemctl action.
- **Verified adversarial cases:**
  - Exact unquoted assignment (`NEMOVOX_MANAGED_ASR=1 NEMOVOX_ASR_LEASE_PATH=.../asr.lock`) -> `True`.
  - Exact quoted assignment with spaces (`"NEMOVOX_ASR_LEASE_PATH=.../asr space.lock"`) -> `True`.
  - Suffix/prefix path (`{lease}.other`) -> rejected (`False`).
  - Duplicate assignment (`NEMOVOX_ASR_LEASE_PATH={lease} NEMOVOX_ASR_LEASE_PATH={lease}`) -> rejected (`False`).
  - Unterminated quote (`"NEMOVOX_ASR_LEASE_PATH=unterminated`) -> rejected (`False`).
  - Crash recovery (`recover_pending_managed_asr`) with `{lease}.other` -> raises `ManagedASRBlocked("asr_not_cooperatively_owned")`, `_run_systemctl("start", ...)` never called (`start_called=[]`), journal stays `"pending_restore"`.
- **Verdict:** **`PASS`**

## Finding 2 (P2, `#discussion_r4655089439`): Partially timed utterances with untimed leading/trailing dialogue

- **Implementation (`src/nvidia_transcribe/selective_alignment.py`):**
  - When `valid_in_span` is non-empty and either `not span[0].get("valid_timing")` (`untimed_prefix`) or `not span[-1].get("valid_timing")` (`untimed_suffix`), `plan_selective_recovery` requires `len(grouped_diar) == 1`, `untimed_spans_by_speaker.get(speaker, 0) == 1`, and overlap between `grouped_diar[0]` and `[native_start - 0.35, native_end + 0.35]`.
  - When satisfied, expands the candidate crop to include the diarized untimed edge (`min(diar_start, native_start) - 0.35` / `max(diar_end, native_end) + 0.35`), still bounded by chunk bounds and `prior_timed` / `following_timed` native neighbors. Otherwise skips with `skipped_partial_span_unanchored_edge`.
- **Verified adversarial cases (native anchor at `7.0..8.0s`):**
  - Long untimed leading edge (`6` untimed units, diarization `[0.6, 8.0]`, native anchor `[7.0, 8.0]`) -> admitted with full window **`[0.25, 8.35]`**.
  - Long untimed trailing edge (native anchor `[7.0, 8.0]`, `6` untimed units, diarization `[7.0, 16.5]`) -> admitted with full window **`[6.65, 16.85]`**.
  - Neighbor-bounded leading edge (`spk_1` `[0.4, 1.2]` before, `spk_1` `[8.3, 9.0]` after) -> clipped cleanly to **`[1.15, 8.35]`** without invading prior speaker's native timing.
  - Absent diarization (`[]`) and split/ambiguous diarization (`[0.6, 3.0]` + `[5.5, 8.0]`) -> `0` candidates, reason-coded **`skipped_partial_span_unanchored_edge`**, zero invented timings.
- **Note / Conservative behavior observed in GPU replay:** Under `zh_multichunk_synthetic` `--qwen-policy fine`, `utt-0002` (`span_range [44, 66]`, untimed leading `'脚'`) is skipped (`skipped_partial_span_unanchored_edge`) because speaker `1` has 2 separated diarization intervals in chunk 0 (`[22.41, 27.06]` and `[28.06, 30.98]`, so `len(grouped_diar) == 2 != 1`). This fails closed safely per design rather than cropping off `'脚'`.
- **Verdict:** **`PASS`**

## Finding 3 (P2, `#discussion_r4655089443`): Fatal propagation of post-stop ASR lifecycle/restoration failures

- **Implementation (`src/nvidia_transcribe/gpu_handoff.py`, `src/nvidia_transcribe/recording.py`):**
  - `ManagedASRLifecycleError(ManagedASRBlocked)` distinguishes post-stop lifecycle/restoration errors from pre-stop admission refusals.
  - Inside `managed_asr_handoff`, once `stopped = True`, any `ManagedASRBlocked` (including `systemctl_unavailable` on `stop`) raises `ManagedASRLifecycleError("asr_handoff_unverified", exc.status)`, and any failure inside the restoration `finally` block (including `systemctl_unavailable` on `start`, unit wait timeout, health failure, or journal write error) raises `ManagedASRLifecycleError("asr_restore_unverified", ...)`, retaining `"phase": "pending_restore"`.
  - `process_media` catches `ManagedASRLifecycleError` before `ManagedASRBlocked` and raises `RuntimeError("Managed ASR lifecycle could not be verified: ...")`.
- **Verified adversarial cases:**
  - Synthetic post-stop `start` raising `ManagedASRBlocked("systemctl_unavailable", "TimeoutExpired")` -> raises `ManagedASRLifecycleError("asr_restore_unverified")` (`__cause__.status == "systemctl_unavailable"`), journal remains `"pending_restore"`.
  - Synthetic `stop` raising `ManagedASRBlocked("systemctl_unavailable", "TimeoutExpired")` -> raises `ManagedASRLifecycleError("asr_handoff_unverified")` (`__cause__.status == "systemctl_unavailable"`), never downgraded to deferral.
  - `process_media` with post-stop `ManagedASRLifecycleError` -> raises `RuntimeError("Managed ASR lifecycle could not be verified: ...")`.
  - Pre-stop `ManagedASRBlocked("asr_has_live_connections")` -> still defers cleanly (`qwen_recovery.status == "asr_has_live_connections"`), Qwen aligner never loaded, no service touched.
- **Verdict:** **`PASS`**
