# Nemovox PR #4 Codex Remediation Qualification Handback

- **Overall Disposition:** **`PASS`** (with **`MANAGED_GPU_BLOCKED`** continuation for live `--qwen-gpu-mode managed` ASR takeover due to absent cooperative ASR unit/lease on host).
- **Repository:** `fvanevski/nemovox` (Issue #1, PR #4, branch `issue/1`)
- **Base `main` SHA:** `242c418f55cd2afb796ee2c4fa742a10ff04f7a3`
- **Previous Qualified `issue/1` SHA:** `d5700b47a4e2f1abb60ebe99f53750c254de2696`
- **Controlling Implementation SHA:** `5e56304ec848f43807cf9ad91d61350090fd7525`
- **Final `issue/1` Source SHA:** `1702608f82808fca30fb3eee1b62954230e6da32` (tree `3d2cc429f9b49da3079df7d1fc178f6fd09d7987`)
- **Evidence Branch:** `evidence/issue-1-codex-20261121` (branched directly from `1702608f82808fca30fb3eee1b62954230e6da32`)
- **Evidence Directory:** `staging/agent-evidence/issue-1-codex/`

## Per-Finding Qualification Verdicts

1. **Codex P1 (`#discussion_r4655089438`) — Exact cooperative ownership values & crash-recovery authorization: `PASS`**
   - `_has_exact_cooperative_markers` in `src/nvidia_transcribe/gpu_handoff.py` parses systemd `Environment=` assignments via `shlex.split`, requiring exact `NEMOVOX_MANAGED_ASR == ["1"]` and `NEMOVOX_ASR_LEASE_PATH == [str(lease_path)]`, rejecting prefixes/suffixes (e.g., `/tmp/a.lock.other`), duplicates, `NEMOVOX_MANAGED_ASR=10`, and malformed/unterminated quotes while accepting valid quoted assignments containing spaces.
   - Enforced in **both** `_verified_owner` (ordinary admission) and `recover_pending_managed_asr` (crash recovery refuses to start a wrong-lease unit and keeps `phase == "pending_restore"`).
   - Verified by all newly added focused regressions (`6/6` parametrized/exact tests) and standalone adversarial checks ([`logs/adversarial_codex_checks.json`](logs/adversarial_codex_checks.json)).

2. **Codex P2 (`#discussion_r4655089439`) — Full localization of partially timed utterances with long untimed leading/trailing dialogue: `PASS` (with documented conservative multi-interval limitation)**
   - `plan_selective_recovery` in `src/nvidia_transcribe/selective_alignment.py` detects `untimed_prefix` / `untimed_suffix` on partially timed spans and expands the crop to `min(diar_start, native_start) - 0.35` / `max(diar_end, native_end) + 0.35` (bounded by chunk limits and `prior_timed` / `following_timed` native neighbors) when `len(grouped_diar) == 1`, `untimed_spans_by_speaker[speaker] == 1`, and the grouped diarization overlaps native timing.
   - Absent or split/ambiguous diarization fails closed with `skipped_partial_span_unanchored_edge` (no invented timings or cropped speech).
   - **Observed delta vs `d5700b4`:** On `zh_multichunk_synthetic` (`--qwen-policy fine`), `utt-0002` (`span_range [44, 66]`, speaker `1`, untimed leading `'脚'`) is now skipped (`skipped_partial_span_unanchored_edge`) because speaker `1` has two separated diarization intervals in chunk 0 (`[22.41, 27.06]` and `[28.06, 30.98]`, `len(grouped_diar) == 2 != 1`). See [`codex-remediation.md`](codex-remediation.md) and [`gpu-replay.md`](gpu-replay.md).

3. **Codex P2 (`#discussion_r4655089443`) — Fatal propagation of post-stop ASR restoration failures including `systemctl_unavailable`: `PASS`**
   - `ManagedASRLifecycleError(ManagedASRBlocked)` is raised for any post-stop lifecycle or restoration verification failure (including `systemctl_unavailable` on `stop` or `start`), preserving `phase == "pending_restore"` when restoration is unverified.
   - `process_media` in `src/nvidia_transcribe/recording.py` catches `ManagedASRLifecycleError` *before* `ManagedASRBlocked` and raises `RuntimeError("Managed ASR lifecycle could not be verified: ...")`, while pre-stop `ManagedASRBlocked` continues to defer cleanly without touching services.

## Summary of Verification Receipts

- **Source delta scope (`d5700b4..1702608`):** Confined strictly to `docs/alignment/ISSUE1_CODEX_REMEDIATION_HANDOFF.md`, `docs/alignment/ISSUE1_QWEN_DESIGN.md`, `src/nvidia_transcribe/gpu_handoff.py`, `src/nvidia_transcribe/recording.py`, `src/nvidia_transcribe/selective_alignment.py`, and `tests/test_qwen_session_and_policy.py` (no local source corrections needed; correction diff is empty).
- **Package compilation:** `python -m compileall -q src/nvidia_transcribe` -> exit `0` ([`logs/compileall.txt`](logs/compileall.txt)).
- **Focused Qwen/ownership/admission tests:** `43 passed, 1 warning in 4.66s` (exit `0`) ([`logs/focused_pytest.txt`](logs/focused_pytest.txt)).
- **Full CPU suite (`-m "not gpu"`):**
  - `.venv` (Python 3.13.15): **`120 passed, 1 warning in 8.65s`** (exit `0`) ([`logs/pytest_venv_cpu.txt`](logs/pytest_venv_cpu.txt)).
  - `.venv-studio` (Python 3.13): **`120 passed in 8.64s`** (exit `0`) ([`logs/pytest_venv_studio_cpu.txt`](logs/pytest_venv_studio_cpu.txt)).
- **Bounded RTX 5090 CUDA BF16 recorded ASR replay (`available` mode):**
  - `two_span_oov`: `off` (`0/42`, `failed: 2`) -> `qwen subtitle` (`42/42`, `full: 2`, `model_load_count=1`, `align_count=2`, `cuda:0` `torch.bfloat16`, span #1 `1.3548s` -> warm span #2 `0.0346s`, peak `1,900,171,776 B`).
  - `zh_multichunk_synthetic`: `off` (`176/198`, `full: 6, partial: 1, failed: 1`) -> `qwen subtitle` (`197/198`, `full: 7, partial: 1`, `'钾'` `[12.24, 12.48]`) -> `qwen fine` (`197/198`, `full: 7, partial: 1`; `[44, 66]` conservatively skipped as `skipped_partial_span_unanchored_edge`).
  - Pre/post RTX 5090 VRAM: `3052 MiB -> 3052 MiB` used (`29076 MiB` free), zero worker leaks.
- **Host services & isolation:** `nemovox-studio.service` and `local_embedding.service` remained `active` and untouched; port `8190` closed; `/workspace/nvidia-transcribe` untouched; `main` unchanged at `242c418f55cd2afb796ee2c4fa742a10ff04f7a3`; PR #4 points to `1702608f82808fca30fb3eee1b62954230e6da32`.

## Remaining Limitations

- **Live managed ASR handoff (`MANAGED_GPU_BLOCKED`):** No cooperatively owned `nemovox-*.service` ASR unit, private lease, or live `8190` ASR server exists on host; live stop/VRAM-release/restore remains unexercised outside isolated mocks.
- **Multi-interval chunk conservatism for partial spans:** `selective_alignment.py` checks `len(grouped_diar) != 1` across the entire chunk when a partially timed span has an untimed prefix/suffix, so partially timed utterances in multi-turn same-speaker chunks (e.g., `zh_multichunk_synthetic` `utt-0002` under `--qwen-policy fine`) skip conservatively rather than matching the unique overlapping interval for that utterance.

## `COMMAND_SHAPE_FRICTION`

```text
COMMAND_SHAPE_FRICTION:
  encountered=no
  events=[]
  omitted_sensitive_values=no
```
