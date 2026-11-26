"""NemoVox: local speaker-attributed recording, translation and subtitles."""

from nvidia_transcribe.contract import build_recording_contract, to_pyannote_compatible_segments
from nvidia_transcribe.media import MediaInfo, MediaInspectionError, probe_media
from nvidia_transcribe.moss_recording import process_moss_media
from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
from nvidia_transcribe.subtitles import SubtitleConfig, render_subtitles
from nvidia_transcribe.translation import TranslationConfig, translate_utterances

__all__ = [
    "MediaInfo", "MediaInspectionError", "probe_media", "build_recording_contract",
    "to_pyannote_compatible_segments", "process_moss_media", "SubtitlePipelineConfig",
    "generate_subtitles", "SubtitleConfig", "render_subtitles", "TranslationConfig",
    "translate_utterances",
]
