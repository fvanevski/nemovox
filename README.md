# NemoVox

Local, recording-level multilingual dialogue recognition and English subtitle generation. MOSS preserves native timed speaker-attributed events. Riva provides initial English translation, with optional bounded contextual review and SRT/ASS rendering. No implicit model downloads, forced word alignment, or cross-window speaker stitching.

## Configure

Use [model resources](docs/model-resources/CONFIGURATION.md) and [lifecycle](docs/model-resources/LIFECYCLE.md). Install the admitted MOSS checkpoint and isolated worker runtime, and configure actual paths. The example paths are installation targets, not claims about installed assets. FFmpeg/ffprobe must be available.

From the repository root with the project interpreter:

```bash
PYTHONPATH=src:. python -m nvidia_transcribe.cli process-media episode.mkv --source-language fr --output-dir outputs/recording
PYTHONPATH=src:. python -m nvidia_transcribe.cli generate-subtitles --recording-contract outputs/recording/recording.contract.json --output-dir outputs/subtitles
PYTHONPATH=src:. python -m nvidia_transcribe.cli assess-source --recording-contract outputs/recording/recording.contract.json --output-dir outputs/assessment
```

Select the actual spoken language for new media. Imported recording contracts can supply it. Use the selected audio-stream ordinal when media has multiple audio tracks. Native source assessment is geometry-only by default; explicit acoustic event assessment requires an admitted CPU detector and the exact original media. No detector changes source wording, event times or speakers.

For an already translated job, rerender with `generate-subtitles --re-render --output-dir outputs/subtitles`, without inference. See [Studio](web/README.md) for the browser workflow.

Development: follow [AGENTS.md](AGENTS.md); run `PYTHONPATH=src:. python -m pytest -m "not gpu"`. Qualify live workers separately.
