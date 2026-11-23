# Nemovox Issue #14 Bounded Source-Recovery & Lineage Evidence Packet (`20261123T013938Z`)

- **Repository:** `fvanevski/nemovox` | **Issue:** `#14` (`OPEN`) | **PR:** `#15` (`OPEN`, `DRAFT`)
- **Branch:** `issue/14-longform-timing-boundaries` | **Base `main`:** `8f6c03cbc70e84bf1970ba364207a02c9fe22b44`
- **Central Source Commit:** `a46490266f5f161057eeb47daf2fee3a07bb2084` (`tree=960fc7abf24a8ad57db275e179e019bd23d6863e`)
- **Admitted PR Head:** `ae10f9b260e6bbba987ff546e23eccfdeba0c9a7` (`tree=6f44ca73dc0610478fd15c0c777d1d04d2050335`, zero source/test drift vs Central source)

## Summary of Results

1. **CPU & Structural Gates (`validation/cpu-gates.json`):** `compileall` exit `0`; focused Issue #14 recovery/renderer suite `41 passed` (`0.39s`); full non-GPU suite `337 passed` (`10.18s`); `git diff --check` exit `0`; `inspect_forced_alignment` re-verified across all `13` (`9` batch + `4` bounded) GPU alignment cases (`text_verified=False`, `timestamp_is_acoustic_truth=False`).
2. **Offline Source Replay & Recovery Planning (`validation/source-recovery-plan.json`):** Authenticated `fr_orig16` (strict PASS), `fr_full35` (strict rejected `c28`, forensic PASS), and `zh_ep166` (strict rejected `c14`, forensic PASS). Demonstrated joint-VAD blind spots (`W04c`, `H2` opening) yield `0` selective requests without `--auditioned-misses` and are scheduled by `--coverage-end-s` (`63` FR windows, `60` ZH windows), with `fr_full35` `--max-requests 128` hitting `partial_budget` (`128` selected / `11` pending in `unplanned_due_to_budget`).
3. **Bounded Qwen3-ASR-1.7B Pilot & Sweeps (`validation/cross-model-coverage.md`, `validation/runtime-and-resources.json`):** Under `ModelLifecycle.worker` admission (`4480.79 MiB` peak VRAM, zero service/GPU mutation), executed `8/8` pilot clips (`5.44s`), `63/63` FR coverage-floor windows (`255.36s`), and `60/60` ZH EP166 coverage-floor windows (`59.35s`). Sweeps recovered `W04c` + narration (`win[57]`), `H2` opening (`win[16]`), `c28` (`win[50..52]`), `S05` (`win[61]`), `ZH_ordinary` (`win[04]`), and `ZH_overlap` (`win[23..24]`), while documenting singing/SFX false positives and `P2_W04c_captain` short-crop hallucination.
4. **Render Preservation & Lineage (`validation/source-turn-lineage.json`):** Re-rendered `fr_orig16` (`308 -> 309` cues, `+1` conservative retained duplicate-looking cue) and `zh_ep166` (`307/304/3` with `ass_max_lanes=2`, `307/307/0` with `ass_max_lanes=3`), and validated end-to-end `source_turn_id` / `source_revision_id` preservation and v1 checkpoint/resume parity.

## Privacy & Non-Authoritative Scope

- Raw WAV crops, full transcripts, raw model responses, and private source-evidence ledgers remain under `chmod 700` host scratch outside Git. `WER/CER=UNVERIFIED`.
