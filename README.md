# NemoVox

Local, recording-level multilingual dialogue recognition and English subtitle generation. MOSS preserves native timed speaker-attributed events. Riva provides initial English translation, with optional recording-wide contextual review and SRT/ASS rendering. With managed native contextual review enabled, bounded connected Riva alternatives are adjudicated against the original occurrence boundaries and independent MT; selected English allocations retain source IDs and timing. No implicit model downloads, forced word alignment, or cross-window speaker stitching.

## Configure

Use [model resources](docs/model-resources/CONFIGURATION.md) and [lifecycle](docs/model-resources/LIFECYCLE.md). Install the admitted MOSS checkpoint and isolated worker runtime, and configure actual paths. The example paths are installation targets, not claims about installed assets. FFmpeg/ffprobe must be available.

From the repository root with the project interpreter:

```bash
PYTHONPATH=src:. python -m nvidia_transcribe.cli process-media episode.mkv --source-language fr --output-dir outputs/recording
PYTHONPATH=src:. python -m nvidia_transcribe.cli generate-subtitles --recording-contract outputs/recording/recording.contract.json --output-dir outputs/subtitles
PYTHONPATH=src:. python -m nvidia_transcribe.cli assess-source --recording-contract outputs/recording/recording.contract.json --output-dir outputs/assessment
```

Select the actual spoken language for new media. Imported recording contracts can supply it. Use the selected audio-stream ordinal when media has multiple audio tracks. Native source assessment is geometry-only by default; explicit acoustic event assessment requires an admitted CPU detector and the exact original media. No detector changes source wording, event times or speakers.

For an already translated job, rerender with `generate-subtitles --re-render --output-dir outputs/subtitles`, without inference. See [Studio](web/README.md) for the browser workflow and [source assessment](docs/source-reconstruction/NATIVE_DIALOGUE.md).

Development: follow [AGENTS.md](AGENTS.md); run `PYTHONPATH=src:. python -m pytest -m "not gpu"`. Qualify live workers separately.

### Recording-wide review and adaptive evidence

The managed native English reviewer sees compact complete bilingual dialogue by default, with detailed target observations and read-only primary English. `--review-context english` retains complete English and localized original source if that is more suitable for the model/language. `--review-context-bytes` bounds the global UTF-8 representation; oversized recordings use an explicitly recorded subset. This is not an exact tokenizer measurement. The reviewer may nominate a material visual or acoustic ambiguity, then reconsider only the relevant complete linguistic unit using independently recorded observations. Supply original admitted media alongside a retained recording contract to enable evidence acquisition; contract-only jobs continue text review and report unavailable media. `--review-evidence off` disables supplementary acquisition. `visual_inspection` inherits the selected translation review resource unless independently configured. No additional ordinary-path condensation or media inspection pass. A targeted same-MOSS acoustic crop is an alternative observation, not independent recognizer confirmation or a source rewrite.

The native reviewer explicitly records whether each target needs supplementary evidence and whether source-cited recurring terminology is supported. No evidence is acquired for an explicit `none` decision. Visual observations include literal frame-linked visible text separately from scene descriptions; dialogue captions may support competing source interpretations, but never overwrite the canonical ASR, speaker attribution or timing. Accepted recurring terms are advisory context in later batches, not global replacements. Comprehensive bilingual is the qualified default for French and Mandarin; English-only global context remains selectable.

Native contextual review checkpoints retain the actual admitted recording/advisory snapshot, initial proposals, successful media observations, evidence-informed unit proposals and exact keyed comparative decisions. Continuation retries unfinished stages without reopening completed downstream allocations merely because a prior discourse interpretation changed. Older batch-cache checkpoints are regenerated; no legacy migration. Deferred units retain complete primary English explicitly until resolved. Visual inspection supports task-sensitive general scene observations (participants, actions, interactions, temporal progression, speech-related visual activity, affect and environment), separate direct description, interpretation and uncertainty, plus literal visible text; focus controls bounded sampling, not lip-reading or speech attribution. Nearby frame-bound observations can inform later review, without automatic source rewrites or a continuous visual-analysis pass.

Evidence-assisted review is best effort: useful scene or caption evidence may be missed, and the model may still select an incorrect interpretation even when evidence is available. Recurring terms and nearby scene observations are advisory, not guaranteed corrections. Live qualification demonstrated useful automatic acquisition and discourse reuse; semantic benefit from cross-batch scene reuse has not yet been established. The retained source observations and review receipts remain available for inspection.
