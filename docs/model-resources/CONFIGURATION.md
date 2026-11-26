# Model resources

Use `configs/model-resources.example.json` as the template for an ignored local JSON file (select via `NEMOVOX_MODEL_RESOURCES` or CLI `--model-resources`). Default local paths require actual provisioning. Registry roles bind named resources; inspect actual checkpoint, runtime and device before inference.

- `asr`: `moss_events`, model identifier, checkpoint_path, worker_python, device cuda:0. No endpoint or CTC overrides. Select a configured worker using --asr-resource.
- `riva_translation`: `riva_worker`, local checkpoint_path, worker_python, device cuda:0.
- `source_refinement`, `text_translation`, `translation_review`, `subtitle_condensation`: named `openai_chat` endpoint/model resources; optional api_key_env. Source refinement is only an explicit imported-contract operation, not native recognition.
- Optional `speech_activity`: `firered_aed`, model identifier, local checkpoint_path, worker_python, explicit cpu. No default binding.

Text environment overrides TEXT_BASE_URL/TEXT_MODEL apply to the default text binding. Explicit CLI text overrides are last. Never store credentials in config. Unsupported fields or adapters fail closed. Remove retired ASR endpoint env settings and old resource definitions from a private migrated config before using the promoted version.

Optional detector example, add to resources and map roles.speech_activity to its name:

```json
{"adapter":"firered_aed","model":"FireRedTeam/FireRedVAD","checkpoint_path":"/local/admitted/AED","worker_python":"/local/isolated/bin/python","device":"cpu"}
```

Use `assess-source ORIGINAL_MEDIA --recording-contract PATH --output-dir DIR --activity` or explicit `generate-subtitles --source-quality on` with original media. Assessment is non-destructive; no automatic challenge or correction.
