"""Standalone evaluation and chunked transcription core for NVIDIA Nemotron 3.5 Transcribe NVFP4."""

from nvidia_transcribe.alignment import (
    ExternalForcedAligner,
    find_ctc_oov_characters,
    load_ctc_vocabulary_chars,
    recover_chunk_alignment,
)
from nvidia_transcribe.checkpoint import (
    DEFAULT_MODEL_PATH,
    CheckpointValidationError,
    inspect_checkpoint,
    validate_checkpoint,
)
from nvidia_transcribe.chunking import (
    ChunkConfig,
    ChunkSpec,
    plan_chunks,
    reconcile_chunk_boundaries,
)
from nvidia_transcribe.contract import (
    build_recording_contract,
    build_transcript_contract,
    to_pyannote_compatible_segments,
)
from nvidia_transcribe.media import MediaInfo, MediaInspectionError, probe_media
from nvidia_transcribe.parser import (
    CTCMetadataError,
    apply_source_offset,
    parse_ctc_timestamps,
    parse_speaker_turns,
    split_lexical_units,
)
from nvidia_transcribe.prompts import (
    build_instruction,
    cleanup_generation_loops,
)
from nvidia_transcribe.qwen_aligner import QwenAlignerConfig, QwenForcedAlignerAdapter
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media
from nvidia_transcribe.refinement import GlossaryConfig, RefinementConfig, load_glossary, refine_utterances_with_text_model
from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
from nvidia_transcribe.subtitles import SubtitleConfig, SubtitleCue, assign_subtitle_lanes, build_display_cues, render_subtitles
from nvidia_transcribe.translation import TranslationConfig, translate_utterances
from nvidia_transcribe.speakers import (
    SpeakerEmbedder,
    TitaNetSpeakerEmbedder,
    normalize_chunk_speakers,
    reconcile_speakers,
)
from nvidia_transcribe.utterances import (
    join_multilingual_units,
    reconstruct_speaker_utterances,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "CheckpointValidationError",
    "ChunkConfig",
    "ChunkSpec",
    "CTCMetadataError",
    "ExternalForcedAligner",
    "MediaInfo",
    "MediaInspectionError",
    "RecordingPipelineConfig",
    "SpeakerEmbedder",
    "TitaNetSpeakerEmbedder",
    "apply_source_offset",
    "build_instruction",
    "build_recording_contract",
    "build_transcript_contract",
    "cleanup_generation_loops",
    "find_ctc_oov_characters",
    "inspect_checkpoint",
    "join_multilingual_units",
    "load_ctc_vocabulary_chars",
    "normalize_chunk_speakers",
    "parse_ctc_timestamps",
    "parse_speaker_turns",
    "plan_chunks",
    "probe_media",
    "QwenAlignerConfig",
    "QwenForcedAlignerAdapter",
    "process_media",
    "reconcile_chunk_boundaries",
    "reconcile_speakers",
    "reconstruct_speaker_utterances",
    "recover_chunk_alignment",
    "split_lexical_units",
    "to_pyannote_compatible_segments",
    "validate_checkpoint",
]
