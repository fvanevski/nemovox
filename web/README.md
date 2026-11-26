# NemoVox Studio

Browser workflow: choose original media or an existing recording contract, select actual spoken language for new media and an audio track, select transcription/translation/subtitles, and choose output/export settings. MOSS recognizes recording-level timed dialogue and anonymous speakers. Ordinary operation does not configure chunk windows, forced alignment or speaker embedding reconciliation.

The backend is the repository's FastAPI Studio service (nvidia_transcribe.studio_api), using the same host resource registry as CLI. Configure actual media/export roots and isolated model workers; see repository README and model-resource documentation. From web/, install the locked dependencies with npm ci, run npm test and npm run build. Use npm run dev for local frontend development; do not expose private media to untrusted networks. Existing job history is retained. No source-quality detector runs implicitly.

The normal Riva translation and bounded English review settings, glossary selection, preview, export and ASS presentation controls remain available.
