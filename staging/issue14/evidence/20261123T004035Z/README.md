# Nemovox Issue #14 Source-Event Integration & Replay Packet (`20261123T004035Z`)

- **Repository:** `fvanevski/nemovox` | **Issue:** `#14` (`OPEN`) | **PR:** `#15` (`OPEN`, `DRAFT`)
- **Branch:** `issue/14-longform-timing-boundaries` | **Base `main`:** `8f6c03cbc70e84bf1970ba364207a02c9fe22b44`
- **Evaluated Head:** `469c78f341eac9d1b9b14b8733700db0d6ed2c4c` (`tree=15a46364c50757b688cd5c412a8f01927ccd310f`)
- **Central Source Commit:** `84713a13fd521b3d99b91f09863aff95153c8fff` (`tree=6580e4fc96372df4dc4e2ee217d8034c4770c187`, zero source/test remediation required)
- **Host Execution Mode:** CPU-only offline verification, cached ASR replay, and read-only short-clip CPU detector checks (`gpu_inference_run=false`, zero production/media/pyannote mutations).

## Summary of Results

1. **Phase 1 (CPU & Integration Gates — PASS):**
   - `compileall -q src/nvidia_transcribe`: exit `0`
   - Focused `source_events` + `source_event_replay` tests: `14 passed` (`0.34s`, exit `0`)
   - Full non-GPU suite (`-m "not gpu"`): `308 passed` (`10.15s`, exit `0`)
   - Frontend (`web/`): `6` test files / `14` Vitest tests passed + `tsc --noEmit && vite build` (`2040` modules), exit `0`
   - `git diff --check`: exit `0`
2. **Phase 2 (Offline Cached Replay — PASS with expected forensic partials):**
   - `orig16` (16-chunk FR): strict replay PASS (`16` observations, `2415` lexical observations, `30` correspondences, `0` dropped, repeat-run IDs 100% identical).
   - `full35` (35-chunk FR): strict replay rejected chunk `28` (`generation_limit_exhausted`); `--forensic-incomplete` PASS (`34` observations, `1` rejected chunk gap `c28`, `2263` lexical observations, `30` correspondences, `0` dropped).
   - `zh_ep166` (15-chunk ZH): strict replay rejected chunk `14` (`generation_limit_exhausted`); `--forensic-incomplete` PASS (`14` observations, `1` rejected chunk gap `c14`, `3200` lexical observations, `42` correspondences, `1965` `unresolved` timing without subinterval/model_path, `0` dropped).
3. **Phase 3 (Readiness & Acoustic Detector Comparison):**
   - Measured read-only CPU Silero VAD v6 (`silero_vad_v6.onnx`) across `20` short clips and `pyannote/speaker-diarization-community-1` (`HF_HUB_OFFLINE=1` CPU) across `17` clips; verified local `Qwen3-ForcedAligner-0.6B` & `Qwen3-ASR-1.7B` weights without running GPU inference.

## Limitations & Privacy Scope

- Offline replay uses native CTC parsing only; it cannot reproduce prior subinterval recovery, forced alignment, or TitaNet global speaker embeddings.
- Raw `source-evidence.v1.json` ledgers, source WAVs, raw model responses, and proprietary episode text are withheld in private chmod-700 host storage and excluded from GitHub.
