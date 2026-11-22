# PR #6 Codex P2 Remediation Evidence Findings (`45705ab72faaa8df0a2817305e73b6558d584aa2`)

- **Blocking findings:** None.
- **Source changes:** None (`SOURCE_EDITS=forbidden`).
- **CTC checkpoint & provenance:** Verified alternative ASR resources without a checkpoint, `ASR_MODEL`, `asr_local.model`, and `--model` overrides do not inherit the shipped Nemotron checkpoint; `process_media` stores `model_path: null` and `ctc_vocabulary_status: not_configured` without fabricating vocabulary-OOV diagnostics; default Nemotron checkpoint loads CTC vocabulary (`ctc_vocabulary_status: loaded`); `_config_signature` includes `ctc_metadata_checkpoint`.
- **Documentation (`docs/model-resources/CONFIGURATION.md`):** Accurately documents optional ASR `checkpoint_path`, `model_path: null`, `ctc_vocabulary_status: not_configured`, model-change clearing vs endpoint-only retention, and cache identity.
- **Unavailable services:** Live ASR (`127.0.0.1:8190`) and text (`127.0.0.1:8001`) services remain `UNVERIFIED` (`GPU_SERVICE_MUTATION=forbidden`).
