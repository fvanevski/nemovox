# PR #6 Exact-Head Evidence Findings (`f8f443750575273a76e9e863dce32083cceea009`)

- **Blocking findings:** None.
- **Source edits:** None (`SOURCE_EDITS=forbidden`).
- **Adapter validation:** Incompatible adapter fields, invalid device strings, and non-CPU TitaNet devices are rejected with `ResourceConfigError`; valid `configs/model-resources.example.json` and explicit valid bindings resolve cleanly.
- **Documentation (`docs/model-resources/CONFIGURATION.md`):** Accurately matches `_ADAPTER_FIELDS` and device validation in `src/nvidia_transcribe/model_resources.py`.
- **Live inference endpoints:** Not started or mutated (`TEXT_LIVE_SMOKE=UNVERIFIED`, `ASR_LIVE_SMOKE=UNVERIFIED`, `GPU_SERVICE_MUTATION=no`).
