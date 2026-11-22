# Subtitle task configuration: behavior and host boundaries

This is the current **issue #7 implementation target**, subject to exact-head
CPU/frontend qualification. It is not an assertion that every model is
installed or running on the operator's host.

## Processing sequence

1. **ASR:** Nemotron transcribes the original audio. Preserve the resulting source-language text, speaker IDs and timing without attempting to rewrite the source using a text model.
2. **Dedicated translation:** Riva-Translate receives that **unchanged source transcript**, producing one initial English machine translation for each cue. Missing/empty or duplicate worker results fail visibly; no implicit model fallback.
3. **Selective contextual English review:** A single configured OpenAI-compatible text model receives the original machine translations for a batch of target cues, their neighboring original English translations, unchanged ASR source strings, speaker/timing context, and an optional glossary. It examines each cue for concrete semantic mistranslations, names, references, omissions and contextual idioms. **No change is the default when evidence is insufficient.** For changes, output per-utterance correction rationale and preserve original text/IDs/timing and `machine_translated_text`. Localization style guides *warranted changes*, not mass rewriting.
4. **Render SRT/ASS:** Use accepted English translations without a further model editing stage. The optional extra high-CPS condensation pass remains a deliberate opt-in for special cases and is **off by default in Studio**.

There is **no source-language LLM correction in the normal pipeline**. For contract compatibility the disabled refinement path performs a fast noninferential copy (`refined_text=source_text`) and reports `profile: off`; it does not contact a model. All existing `raw_source_text` and `source_text` fields remain unchanged.

CLI source correction remains a **legacy explicit opt-in** with `--refinement conservative|balanced|expressive`. The CLI and Studio default to `--refinement off`, and new Studio job submissions reject requests enabling source correction. Legacy direct text-model primary translation (`--translator openai_chat|swiftu`) remains an explicit CLI option; Studio defaults to `riva` and English review `balanced`. The CLI retains its existing primary-backend default for backward compatibility, but source correction is now off unless explicitly requested.

The reviewer operates on **original machine-translated English**, not a previously corrected English variant. Its source-context input is taken from `source_text`, never from a legacy `refined_text` hypothesis. The review output is keyed by utterance ID; failures preserve machine translation and record errors rather than fabricating repaired content. This constrains edits but does not establish objective semantic accuracy without a reliable source reference or listening to audio.

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
explicitly configure CPU/auto if desired. An OpenAI-compatible text model is
required for English translation review. Optional legacy source refinement and extra condensation require a compatible text endpoint only when explicitly enabled.

The Studio readiness UI checks that Riva checkpoint/worker paths exist; that
is **not proof** of load success, language support, speed, quality or available
VRAM. No model is installed, downloaded, started, stopped or evicted by
selecting options. Test live Riva, text and ASR only when host ownership allows,
and report unavailable inference as \`UNVERIFIED\`.

Check interfaces and smoke prerequisites on the actual host before claiming
successful interactive use.
