# Subtitle task configuration: behavior and host boundaries

## Processing sequence

1. MOSS records native dialogue events with observed segment boundaries and anonymous speaker labels. Select actual spoken language for new media.
2. Geometry assessment reports covered and uncovered intervals, untimed events and speaker-label distribution. Uncovered time is not proof of missing speech. Optional CPU acoustic activity is separately requested through CLI, not mandatory.
3. Riva receives preserved source dialogue. Missing or invalid worker results fail visibly.
4. Balanced contextual English review compares primary translations against source and nearby context, retaining original MT and consequential meaning. NInfer Messages enforces the bounded thinking contract.
5. Render SRT/ASS with automatic measured ASS placement by default; no additional condensation pass.

No routine acoustic re-recognition, forced lexical alignment, or speaker stitching. Independent speaker, timing or lexical evidence may be investigated for material defects without automatically modifying source events. Existing contracts are a downstream translation/render input; fresh recognition uses the new worker.

## Audio streams

The media picker does not invent four stream slots. For an authorized local
file or uploaded media, \`POST /api/media/inspect\` reads the actual streams
with FFprobe, returns **audio-only ordinal indices** and optional codec,
language, title, channels and sample-rate metadata, and the GUI offers only
these tracks. The backend revalidates the submitted audio index. A host
cannot inspect a remote URL source before it downloads; it uses the first
audio stream in that case and does not present guessed metadata. An already
transcribed \`recording.contract.json\` requires no audio-track choice.

No inspection endpoint can read arbitrary host files outside authorized media
roots. This is still a **local single-user** tool, not a public/LAN service.

## Glossaries

Two example files are packaged for known series:
\`configs/glossaries/mcog_fr.yaml\` (French) and
\`configs/glossaries/a_will_eternal_zh.yaml\` (Mandarin). These include
character names, canonical terminology, source-language variants and
background notes. They are optional **prompt hints**, not a new ASR model,
automatic identification of a selected episode, deterministic replacement
dictionary, or evidence that every possible homophone should be changed.

**No glossary** is the default for unrelated media. Selecting one previews
the full relevant entries and notes. The glossary provides optional contextual hints to the **English translation reviewer** and participates in its cache identity. Legacy CLI-only source refinement may use it when explicitly selected.

## Host requirements and limits

Configure MOSS and Riva local checkpoints and isolated worker interpreters as described in [model resources](../model-resources/CONFIGURATION.md). The default device is explicit cuda:0. A locally configured endpoint supports contextual English review. Studio readiness is not proof of load success or quality. No implicit downloads, unrelated service eviction or automatic device downgrade. See [lifecycle](../model-resources/LIFECYCLE.md).

## English-to-English

Explicit English media or imported English recordings bypass MT and English translation review, preserving source text, IDs, speakers and timing. No Riva worker is required for that route. Optional glossaries remain independently discoverable; malformed entries do not disable valid entries.

## Automatic ASS presentation

Native libass measures rendered text, wrapping, outline and shadow at the presentation coordinate scale. Bounded grayscale samples from the explicitly supplied, admitted non-attached video stream estimate stable picture bars and bright-stroke caption candidates. Candidate regions are not confirmed captions, language identification or evidence of absence. A dark or unsampled frame does not prove absence. A stable bar is preferred when the full measured block fits; otherwise available vertical space is used with a conservative lower reserve for unknown captions. Overlapping events retain fixed chronological positions; discretionary padding may be curtailed to fit, while real concurrency and selected English remain distinct from presentation constraints. Overflow and conflicts are reported. Only positively known scene boundaries can constrain scene padding. `standard` and `raised_main` are explicit fixed overrides. Missing native libass is reported with a fixed raised fallback, not a successful measurement.

`presentation.observations.json` binds to source fingerprint, full content digest where available, and duration. It preserves sampled observations, requested seek times (not exact PTS), candidate regions and optional independently supported interval-bound caption-region evidence and scene boundaries. The latter fields are intended for admitted evidence producers; this sampler does not manufacture confirmation. Offline `--re-render` reuses the retained document without opening media if none is supplied, and never rewrites selected English. Exact output reproducibility assumes the same font and libass environment. ASS positions and measured bounds are present in cue lineage. SRT remains independently scheduled.
