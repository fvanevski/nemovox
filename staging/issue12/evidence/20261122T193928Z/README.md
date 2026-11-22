
# Issue #12 / PR #13 — Byte-Identical Contract Archive & Export Qualification (`20261122T193928Z`)

- **Repository:** `fvanevski/nemovox`
- **Issue / PR:** `#12` / `#13` (draft)
- **Branch:** `issue/12-studio-output-workflow`
- **Tested Source SHA:** `51f59f65249ff4fb9705bc1f67638abb521d9062`
- **Tested Tree SHA:** `cb11b80b8ca4b900c44f83964fa5516f0443dc4d`
- **Prior Evidence Carrier (historical):** `125d0e8f60c3e09b58e27110dcb99e74538a760b`
- **Source Edits by Local Agent:** `none`
- **Result:** `PASS`

## Summary
Verified `0dc2243` (`_read_transcription_contract_snapshot` + `_atomic_bytes`) and `51f59f6` (`transcription` + `translation` + `subtitles` raw-byte preservation assertions) at `51f59f65249ff4fb9705bc1f67638abb521d9062`:
- `compileall -q src tests experiments`: `exit=0`
- `.venv-studio` import-task regressions: `3 passed in 0.33s`
- `.venv-studio` full Studio suite (`test_studio_export_api.py`, `test_studio_api.py`, `test_studio_exports.py`): `48 passed in 3.47s`
- `.venv` full non-GPU suite (`pytest -q -m 'not gpu'`): `250 passed, 0 skipped, 1 warning in 9.26s`
- Live isolated Studio API (`127.0.0.1:8769`) across all 3 tasks (`transcription`, `translation`, `subtitles`): compact `316`-byte input (`sha256=9bc27859f8198d00f3c684846eac0dc3e0d327b39e598e651856434534f72289`) matched archived `asr/recording.contract.json` and exported `recording.contract.json` byte-for-byte in every task, while `translation` & `subtitles` passed the original contract path to `generate-subtitles` and preserved untimed chunk-1 ordering (`290-610s`, `estimated_order_s=450.0 > 150.0`).
- Earlier frontend (`14` Vitest + Vite build) and French CUDA E2E packets remain historical corroboration for unchanged frontend/GPU paths.
