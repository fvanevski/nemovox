# NemoVox Studio

Browser workflow: choose original media or an existing recording contract, select actual spoken language for new media and an audio track, select transcription/translation/subtitles, and choose output/export settings. MOSS recognizes recording-level timed dialogue and anonymous speakers. Ordinary operation does not configure chunk windows, forced alignment or speaker embedding reconciliation.

The backend is the repository's FastAPI Studio service (nvidia_transcribe.studio_api), using the same host resource registry as CLI. Configure actual media/export roots and isolated model workers; see repository README and model-resource documentation. From web/, install the locked dependencies with npm ci, run npm test and npm run build. Use npm run dev for local frontend development; do not expose private media to untrusted networks. Existing job history is retained. No source-quality detector runs implicitly.

The normal Riva translation and bounded English review settings, glossary selection, preview, export and ASS presentation controls remain available.

Studio's managed native reviewer receives recording-wide context and may acquire targeted visual/acoustic observations from the admitted job media. Visual inspection inherits the translation review resource unless the host registry binds `visual_inspection` separately. Contract-only imports remain text-based unless original media is explicitly admitted by the workflow. See the model-resource configuration documentation.

The shared native workflow records explicit evidence-need decisions and advisory recurring terminology, and can use source-language captions as competing wording observations without altering the original dialogue record.

Automatic ASS layout is the default. Standard and raised layouts are manual overrides. No extra condensation inference is offered. Studio exposes presentation diagnostics, an optional observations export, and emitted ASS intervals when JASSUB is active; the SRT fallback has its own timing. Unknown captions and missing native measurement are explicitly distinguished from measured placement.
