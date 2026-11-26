# Subtitle task configuration: behavior and host boundaries

The shared CLI/Studio workflow was exercised in a fresh merged-main Studio job
with current-source quality, CUDA challenger, Riva, balanced English review,
exact-input continuation and offline rerender. See the
[deployed acceptance record](https://github.com/fvanevski/nemovox/issues/19#issuecomment-7178926236).
This is bounded host-specific qualification, not a guarantee for every recording.

## Processing sequence

1. **ASR:** Preserve the original audio-bound ASR observations and timing.
2. **Current-source quality:** Generate recording-context hypotheses, validate their literal evidence references,
   triage material ambiguity and selectively acquire separate audio contexts.
   An explicitly configured `whisper_ct2` CUDA worker supplies independent acoustic
   observations; visual observations may ground relevant visible literals, not spoken words. Only positively corroborated source events can change canonical source; competing occurrences remain unresolved rather than being silently deleted.
3. **Dedicated translation:** Riva receives canonical source text. Missing or invalid worker results fail visibly.
4. **Contextual English review:** Review primary translations against canonical source and nearby context; preserve original MT, important meaning and polarity. Use bounded NInfer Messages in current-source quality jobs.
5. **Render SRT/ASS:** Existing rendering; optional additional condensation is off by default.

Current-media Riva jobs use source quality automatically. Imported recording
contracts without original media can be translated but cannot be acoustically
reviewed. Current job evidence includes original ASR, canonical source and
source-quality decisions; model corroboration is not verified word timing or
speaker identity. Unresolved cases and budget deferrals remain explicit.

## ASR transcription options

| Option | Actual prompt behavior | Use |
| --- | --- | --- |
| Speaker-attributed (\`speaker\`) | Spoken words/fillers, punctuation, speaker tags | Default subtitle dialogue; subsequent cross-chunk reconciliation |
| Overlapping speakers (\`multispeaker_sot\`) | Stable speaker-turn tags, including possible overlap; **no punctuation** | Overlap-prone media; does not guarantee separation |
| Verbatim (\`verbatim\`) | Literal words, fillers/repetitions, punctuation; does **not** request speaker tags | Text-focused transcription rather than diarization |
| Rich (\`rich\`) | Spoken content with \`<ev:*>\` event and \`<sty:*>\` style tokens | Experimental event/style capture, not a guarantee of accurate tags |

These settings are ASR *instructions*, not different guaranteed audio
separation algorithms or interchangeable transcript contracts.

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

Default Riva resource: local
\`/workspace/models/Riva-Translate-4B-Instruct-v2\` checkpoint and the
configured Python interpreter for its worker. The host should set appropriate
paths in ignored \`configs/model-resources.local.json\`. The shipped Riva
device is explicit \`cuda:0\`: no automatic CPU downgrade; an operator may
explicitly configure CPU/auto if desired. An NInfer Messages-compatible text model is
required for current-media source quality and contextual English review.
Legacy text-only source refinement is not part of the normal Studio route. Extra
condensation is off unless explicitly enabled.

The Studio readiness UI checks that Riva checkpoint/worker paths exist; that
is **not proof** of load success, language support, speed, quality or available
VRAM. No model is installed, downloaded, started, stopped or evicted by
selecting options. At execution time the CLI automatically acquires and
releases only explicitly owned model resources under the [lifecycle contract](../model-resources/LIFECYCLE.md). Test live Riva, text and ASR only when host ownership allows,
and report unavailable inference as \`UNVERIFIED\`.

Check interfaces and smoke prerequisites on the actual host before claiming
successful interactive use.


## English-to-English subtitle tasks

When the ASR recording contract identifies English (including auto-detected
English) and the requested output is English, no machine translation is
required. The pipeline renders the original ASR text with its original IDs,
speaker labels and timing, marking translation as
`bypassed_same_language` and English translation review as
`skipped_same_language`. This avoids a spurious unsupported Riva `en→en`
language pair and avoids rewriting English source dialogue. Explicitly selected
English sources do not need a configured Riva worker. Unknown/auto-language
media still require Riva readiness for potentially non-English speech.

Glossary discovery skips malformed optional YAML/JSON entries independently,
so one invalid file does not prevent other glossary examples appearing.

Optional speaker embeddings default to off. TitaNet is an explicit local-resource choice; missing weights are not a reason to download or silently substitute a model. The single-command subtitle CLI supports both `--audio-stream` and `--speaker-embeddings`.
