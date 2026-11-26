# Native dialogue and non-destructive source assessment

MOSS is the primary recording recognizer, selected after the comparative campaign and episode-scale Mandarin and French qualification at da39c293711b624118bed704c9c0bd4ac80c7886. Those runs are host-specific evidence, not an accuracy guarantee or a test-count target. Retain native event text, model-observed segment intervals and anonymous recording-local speaker labels. The input hash, selected audio ordinal, decoded audio hash, model/runtime and raw generation bind the observation. No CTC lexical-reconstruction path remains supported.

The ordinary assessment is geometry-only and inexpensive: union valid intervals (including simultaneous observations), complement against complete decoded duration, retain untimed/clamped IDs and report speaker label counts. It neither verifies speech recall nor validates character identity.

Optional explicit CPU FireRed AED assessment uses the exact same decoded audio. It records speech, singing and music independently, intersects speech activity with the uncovered intervals, and preserves raw observations. Overlap with singing/music is reported, not used to silently discard potential speech. Nominations are unresolved investigation candidates, never established omissions or speaker labels. There are no automatically scheduled challenges or applied corrections. Actual model/worker identity and raw observations bind resume. Unsupported device, mismatched audio or invalid intervals fail visibly.

```bash
PYTHONPATH=src:. python -m nvidia_transcribe.cli assess-source --recording-contract PATH --output-dir DIR
PYTHONPATH=src:. python -m nvidia_transcribe.cli assess-source ORIGINAL_MEDIA --recording-contract PATH --output-dir DIR --activity --model-resources PRIVATE_CONFIG
```

The second command requires a configured speech_activity resource; it performs no translation. Default recognition does not use the detector. Explicit `generate-subtitles --source-quality on` requests the same assessment; `auto` computes geometry only and `off` skips additional assessment, without erasing the recording's existing report.

Evaluate actual material failures rather than every gap or background sound. Investigate coherent scenes only when a candidate warrants it. Independent recognition, alignment and diarization are not permanent dependencies without demonstrated benefit. Contextual interpretation and subtitle translation quality belong to the subsequent translation work; do not repair source misrecognition by inventing supported speech or timestamps.
