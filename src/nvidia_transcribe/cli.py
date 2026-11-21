"""Command-line interface for Nemotron 3.5 Transcribe evaluation and chunked recording processing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nvidia_transcribe.checkpoint import DEFAULT_MODEL_PATH, CheckpointValidationError, validate_checkpoint
from nvidia_transcribe.client import InferenceRequestConfig, NemotronVLLMClient
from nvidia_transcribe.media import MediaInspectionError
from nvidia_transcribe.parser import CTCMetadataError
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media
from nvidia_transcribe.refinement import RefinementConfig
from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
from nvidia_transcribe.subtitles import SubtitleConfig
from nvidia_transcribe.translation import TranslationConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nvidia-transcribe",
        description="Evaluate or transcribe local audio/video using Nemotron 3.5 Transcribe via vLLM.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate-checkpoint", help="Validate local checkpoint files and quantization config.")
    v.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))

    t = sub.add_parser("transcribe", help="Run single-clip transcription/translation against a running vLLM server.")
    t.add_argument("audio", help="Path to local audio file.")
    t.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    t.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    t.add_argument(
        "--mode",
        default="verbatim",
        choices=["verbatim", "readable", "speaker", "multispeaker_sot", "rich", "fleurs", "ascend", "translate", "lid", "contextual"],
    )
    t.add_argument("--source-language", default=None, help="Optional explicit language or 'auto'.")
    t.add_argument("--target-language", default="English")
    t.add_argument("--context-terms", default=None)
    t.add_argument("--source-offset", type=float, default=0.0, help="Clip start offset in seconds.")
    t.add_argument("--no-timestamps", action="store_true", help="Disable capture_ctc_timestamps.")
    t.add_argument("--require-timestamps", action="store_true", help="Fail if ctc_timestamps has error or is absent.")
    t.add_argument("--out-json", default=None, help="Write normalized contract JSON to file.")
    t.add_argument("--out-raw", default=None, help="Write raw vLLM JSON response to file.")

    pm = sub.add_parser("process-media", help="Process a full audio/video recording into a chunked canonical transcript.")
    pm.add_argument("media", help="Path to local audio or video media file.")
    pm.add_argument("--output-dir", required=True, help="Directory to store chunks, manifest, transcript, and contract JSON.")
    pm.add_argument("--base-url", default="http://127.0.0.1:8190/v1")
    pm.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4")
    pm.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    pm.add_argument("--source-language", default=None, help="Explicit source language (e.g. 'fr', 'zh') or omit for auto.")
    pm.add_argument("--mode", default="speaker", choices=["speaker", "multispeaker_sot", "rich", "verbatim"])
    pm.add_argument("--chunk-seconds", type=float, default=90.0)
    pm.add_argument("--context-seconds", type=float, default=10.0)
    pm.add_argument("--min-tail-seconds", type=float, default=15.0)
    pm.add_argument("--audio-stream", type=int, default=0)
    pm.add_argument("--speaker-embeddings", default="titanet", choices=["titanet", "none"])
    pm.add_argument("--embedding-threshold", type=float, default=0.65)
    pm.add_argument("--no-subinterval-recovery", action="store_true")
    pm.add_argument("--alignment-fallback", default="off", choices=["off", "qwen"])
    pm.add_argument("--fresh", action="store_true", help="Ignore previous manifest checkpoints and recompute all chunks.")

    gs = sub.add_parser(
        "generate-subtitles",
        help="Run or resume ASR -> SwiftU contextual refinement -> English translation -> SRT/ASS rendering.",
    )
    gs.add_argument("media", nargs="?", default=None, help="Path to local media file (omit when using --recording-contract).")
    gs.add_argument("--recording-contract", default=None, help="Existing recording.contract.json to start from (skips ASR).")
    gs.add_argument("--output-dir", required=True, help="Directory for refined/translated contracts and SRT/ASS outputs.")
    gs.add_argument("--source-language", default=None, help="Source language code (e.g. 'fr', 'zh').")
    gs.add_argument("--target-language", default="en", help="Target subtitle language (default: en).")
    gs.add_argument("--refinement", default="balanced", choices=["off", "conservative", "balanced", "expressive"])
    gs.add_argument("--translator", default="swiftu", choices=["swiftu", "riva", "none"])
    gs.add_argument("--style", default="natural", choices=["natural", "literal", "expressive"])
    gs.add_argument("--glossary", default=None, help="Path to character/term glossary (.yaml/.json) or comma-separated terms.")
    gs.add_argument("--formats", default="srt,ass", help="Comma-separated subtitle formats (default: srt,ass).")
    gs.add_argument("--episode-stem", default=None, help="Override subtitle output filename stem.")
    gs.add_argument("--sidecar-dir", default=None, help="Optional directory for EpisodeName.<lang>.srt/.ass sidecars.")
    gs.add_argument("--overwrite-sidecar", action="store_true", help="Allow overwriting existing sidecar files.")
    gs.add_argument("--base-url", default="http://127.0.0.1:8190/v1", help="Nemotron ASR vLLM base URL.")
    gs.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4", help="Nemotron ASR served model.")
    gs.add_argument("--alignment-fallback", default="off", choices=["off", "qwen"])
    gs.add_argument("--chunk-seconds", type=float, default=90.0)
    gs.add_argument("--context-seconds", type=float, default=10.0)
    gs.add_argument("--min-tail-seconds", type=float, default=15.0)
    gs.add_argument("--swiftu-url", default="http://127.0.0.1:8001/v1", help="SwiftU/NInfer OpenAI-compatible base URL.")
    gs.add_argument("--swiftu-model", default="chat", help="SwiftU served model ID.")
    gs.add_argument("--max-chars", type=int, default=42)
    gs.add_argument("--max-lines", type=int, default=2)
    gs.add_argument("--min-display", type=float, default=1.2)
    gs.add_argument("--max-display", type=float, default=6.5)
    gs.add_argument("--overlap-threshold", type=float, default=0.4)
    gs.add_argument("--ass-layout", default="standard", choices=["standard", "raised_main"], help="ASS positioning layout ('raised_main' clears bottom Chinese hardsubs).")
    gs.add_argument("--ass-margin-v", type=int, default=150, help="Bottom margin (px) for raised_main ASS layout.")
    gs.add_argument("--ass-opaque-box", action="store_true", help="Draw semi-opaque dark box behind ASS cues.")
    gs.add_argument("--ass-max-lanes", type=int, default=3, choices=[2, 3], help="Maximum simultaneous ASS speaker lanes.")
    gs.add_argument("--condense-high-cps", action="store_true", help="Condense overly fast (>threshold CPS) English cues with SwiftU.")
    gs.add_argument("--condense-threshold", type=float, default=25.0, help="CPS threshold triggering selective SwiftU condensation.")
    gs.add_argument("--fresh", action="store_true", help="Recompute ASR, refinement, and translation without reusing caches.")
    gs.add_argument("--fresh-refinement", action="store_true", help="Ignore refinement checkpoint cache.")
    gs.add_argument("--fresh-translation", action="store_true", help="Ignore translation checkpoint cache.")
    gs.add_argument("--re-render", action="store_true", help="Re-render SRT/ASS directly from translated.contract.json without model calls.")
    gs.add_argument("--validation-overlay", default=None, help="Optional viewing-validation JSON overlay for coarse timing & polish.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-checkpoint":
        try:
            summary = validate_checkpoint(args.model_path)
        except CheckpointValidationError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2), file=sys.stderr)
            return 2
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    if args.command == "transcribe":
        cfg = InferenceRequestConfig(
            mode=args.mode,
            source_language=args.source_language,
            target_language=args.target_language,
            context_terms=args.context_terms,
            capture_ctc_timestamps=not args.no_timestamps,
            require_ctc_timestamps=args.require_timestamps,
            source_offset_s=args.source_offset,
        )
        client = NemotronVLLMClient(base_url=args.base_url, model=args.model)
        try:
            result = client.transcribe(args.audio, cfg)
        except (FileNotFoundError, CTCMetadataError, RuntimeError) as exc:
            print(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2), file=sys.stderr)
            return 1

        if args.out_raw:
            Path(args.out_raw).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_raw).write_text(json.dumps(result["raw_response"], ensure_ascii=False, indent=2), encoding="utf-8")
        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_json).write_text(json.dumps(result["contract"], ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(json.dumps(result["contract"], ensure_ascii=False, indent=2))
        return 0

    if args.command == "process-media":
        client = NemotronVLLMClient(base_url=args.base_url, model=args.model, timeout_s=900.0)
        rcfg = RecordingPipelineConfig(
            source_language=args.source_language,
            mode=args.mode,
            chunk_seconds=args.chunk_seconds,
            context_seconds=args.context_seconds,
            min_tail_seconds=args.min_tail_seconds,
            audio_stream_index=args.audio_stream,
            speaker_embeddings=args.speaker_embeddings,
            embedding_threshold=args.embedding_threshold,
            enable_subinterval_recovery=not args.no_subinterval_recovery,
            alignment_fallback=args.alignment_fallback,
            resume=not args.fresh,
            model_path=args.model_path,
        )
        try:
            out = process_media(args.media, output_dir=args.output_dir, client=client, config=rcfg)
        except (FileNotFoundError, MediaInspectionError, ValueError, RuntimeError) as exc:
            print(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2), file=sys.stderr)
            return 1

        c = out["contract"]
        summary = {
            "status": "OK",
            "contract_path": out["contract_path"],
            "transcript_path": out["transcript_path"],
            "manifest_path": out["manifest_path"],
            "source_language": c["source_language"],
            "language_provenance": c["language_provenance"],
            "duration_s": c["source"]["duration_s"],
            "chunk_count": c["processing"]["chunk_count"],
            "utterance_count": len(c["utterances"]),
            "speaker_count": len(c["speakers"]),
            "boundary_deduplicated_units": c["processing"]["boundary_reconciliation"]["deduplicated_units"],
            "boundary_recovered_units": c["processing"]["boundary_reconciliation"]["recovered_boundary_units"],
            "total_wall_time_s": out["total_wall_time_s"],
            "warnings": c["processing_warnings"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "generate-subtitles":
        formats = tuple(x.strip().lower() for x in str(args.formats).split(",") if x.strip())
        ref_cfg = RefinementConfig(
            backend="off" if args.refinement == "off" else "swiftu",
            profile=args.refinement,
            base_url=args.swiftu_url,
            model=args.swiftu_model,
        )
        trans_cfg = TranslationConfig(
            backend=args.translator,
            style=args.style,
            target_language=args.target_language,
            base_url=args.swiftu_url,
            model=args.swiftu_model,
        )
        sub_cfg = SubtitleConfig(
            max_chars=args.max_chars,
            max_lines=args.max_lines,
            min_display_s=args.min_display,
            max_display_s=args.max_display,
            overlap_threshold_s=args.overlap_threshold,
            ass_layout=args.ass_layout,
            raised_main_margin_v=args.ass_margin_v,
            ass_opaque_box=args.ass_opaque_box,
            ass_max_lanes=args.ass_max_lanes,
        )
        scfg = SubtitlePipelineConfig(
            source_language=args.source_language,
            target_language=args.target_language,
            chunk_seconds=args.chunk_seconds,
            context_seconds=args.context_seconds,
            min_tail_seconds=args.min_tail_seconds,
            alignment_fallback=args.alignment_fallback,
            asr_base_url=args.base_url,
            asr_model=args.model,
            refinement=ref_cfg,
            translation=trans_cfg,
            subtitles=sub_cfg,
            formats=formats,
            episode_stem=args.episode_stem,
            sidecar_dir=args.sidecar_dir,
            overwrite_sidecar=args.overwrite_sidecar,
            resume=not args.fresh,
            resume_refinement=not (args.fresh or args.fresh_refinement),
            resume_translation=not (args.fresh or args.fresh_translation),
            resume_condensation=not (args.fresh or args.fresh_translation),
            re_render_only=args.re_render,
            condense_high_cps=args.condense_high_cps,
            condense_cps_threshold=args.condense_threshold,
            validation_overlay_path=args.validation_overlay,
        )
        try:
            res = generate_subtitles(
                args.media,
                recording_contract_path=args.recording_contract,
                output_dir=args.output_dir,
                config=scfg,
                glossary=args.glossary,
                progress_callback=lambda msg: print(f"[nvidia-transcribe] {msg}", file=sys.stderr, flush=True),
            )
        except (FileExistsError, FileNotFoundError, MediaInspectionError, ValueError, RuntimeError) as exc:
            print(json.dumps({"status": "ERROR", "error": str(exc)}, indent=2), file=sys.stderr)
            return 1
        print(json.dumps({"status": "OK", **res}, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
