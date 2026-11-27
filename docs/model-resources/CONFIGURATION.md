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

## Adaptive review resources

`visual_inspection` is a separate `openai_chat` registry role using the NInfer Messages image-capable transport. If omitted it inherits the selected `translation_review` resource, including an explicit `--translation-review-resource` override. To select a different provisioned Messages-capable vision endpoint, add a named resource and bind `roles.visual_inspection`, or use `--visual-inspection-resource NAME`. Shared resources reuse the active review lease. Distinct resources release the review lease before acquiring the configured visual endpoint; the next text request reacquires review. No unowned service is stopped. `--visual-max-frames` (3..8), `--visual-thinking-budget` (1024..4096), `--review-evidence-per-batch` (0..12; zero disables acquisition without disabling text review), and `--review-evidence auto|off` control acquisition. Existing occurrence IDs are required before durable proposal admission. Unavailable resources and fixed out-of-budget acoustic geometry are terminal evidence limitations, while transient acquisition failures and exhausted positive per-batch budgets can be retried. Invalid references enter bounded proposal recovery; a retained invalid nomination fails explicitly rather than repeatedly acquiring, and `--fresh-translation` regenerates affected review state. Source media must be explicitly supplied and content-bound. Audio re-observation uses the admitted `asr` resource and selected recording audio stream. Existing Studio jobs with original media use the same CLI pipeline and role registry; imported contracts without admitted video do not follow arbitrary embedded paths. No implicit model downloads or endpoint substitution.

Adaptive review requires explicit per-target evidence dispositions. Visual inspection returns frame-linked literal text separately from scene descriptions; only source-language dialogue captions associated with the occurrence can be cited as alternative wording evidence. The configured resource retains the existing NInfer Messages image transport and xhigh reasoning effort. Caption evidence does not independently authenticate speech, speaker identity or timing.

The native reviewer uses the current `native-review-progress/v1` checkpoint schema with per-stage request receipts and frozen actual advisory context for continuation. Successful media acquisition is reusable independently of subsequent semantic refinement. General visual observations carry categories, frame identities, visible descriptions, tentative interpretations and uncertainty. Task focus selects distributed scene or concentrated speech/activity/text sampling within the configured frame budget. Previously observed nearby scene facts are advisory; they do not establish speech, identities or scene continuity. Runtime summaries distinguish real acquisitions from retained/reused observations and initial versus recovery comparison requests.
