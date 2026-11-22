# Subtitle task configuration: behavior and host boundaries

This is the current **issue #7 implementation target**, subject to exact-head
CPU/frontend qualification. It is not an assertion that every model is
installed or running on the operator's host.

## Processing sequence

1. **ASR:** Nemotron transcribes the original audio, retaining speaker IDs and
   native CTC timing. Source media audio selection uses its audio-only track
   ordinal (\`0:a:0\`, \`0:a:1\`, etc.).
2. **Source transcript refinement (optional):** an OpenAI-compatible text model
   corrects well-supported source-language recognition mistakes (not English
   translation). Conservative, balanced, and expressive profiles control
   correction aggressiveness.
3. **Primary translation:** an explicitly configured **Riva-Translate**
   checkpoint/worker creates the initial English machine translation from
   refined source-language utterances. A failure, missing/empty utterance, or
   mismatched device is an error, never a silent text-model fallback.
4. **English translation review (optional):** a separately resolved text model
   sees the original source, initial Riva English, speaker/timing context,
   nearby utterances and glossary. It can correct supported mistranslations
   or localization issues without retranslating from scratch. Profiles:
   off (keep Riva), conservative, balanced (GUI default), expressive. The
   requested localization style affects this stage rather than Riva's direct
   worker output.
5. **Condensation (optional):** shorten high-CPS English cues, then render
   timed SRT/ASS. Existing source/acoustic identifiers must remain unchanged.

Each stage has its own cache identity; post-edit cache keys include resource
identity, machine translation, source context, style, profile and glossary.
The review preserves \`machine_translated_text\`, writes a separate
\`translation_reviewed_text\`/status/reason, and only changes the final
\`translated_text\` for validated review edits. Invalid output retains Riva
translation and reports an error. "Supported" here means constrained by
source/provenance/context validation and operator review, **not** an objective
guarantee that the LLM's proposed English is semantically correct.

Existing CLI \`--translator openai_chat|swiftu\` paths remain **explicit
legacy primary translation**, and CLI \`--translation-review\` defaults to off
for compatibility; the Studio GUI no longer exposes those as a default
primary translation choice.

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
the full relevant entries and notes. Configured glossary content may affect
source refinement and English review, and participates in their cache identity.

## Host requirements and limits

Default Riva resource: local
\`/workspace/models/Riva-Translate-4B-Instruct-v2\` checkpoint and the
configured Python interpreter for its worker. The host should set appropriate
paths in ignored \`configs/model-resources.local.json\`. The shipped Riva
device is explicit \`cuda:0\`: no automatic CPU downgrade; an operator may
explicitly configure CPU/auto if desired. An OpenAI-compatible text model is
separately required for non-off refinement, translation review and condensation.

The Studio readiness UI checks that Riva checkpoint/worker paths exist; that
is **not proof** of load success, language support, speed, quality or available
VRAM. No model is installed, downloaded, started, stopped or evicted by
selecting options. Test live Riva, text and ASR only when host ownership allows,
and report unavailable inference as \`UNVERIFIED\`.

Check interfaces and smoke prerequisites on the actual host before claiming
successful interactive use.
