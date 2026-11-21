"""Command-line interface for Nemotron 3.5 Transcribe evaluation and chunked recording processing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nvidia_transcribe.checkpoint import DEFAULT_MODEL_PATH, CheckpointValidationError, validate_checkpoint
from nvidia_transcribe.client import InferenceRequestConfig, NemotronVLLMClient
from nvidia_transcribe.gpu_handoff import ManagedASRBlocked, ManagedASRConfig, recover_pending_managed_asr
from nvidia_transcribe.media import MediaInspectionError
from nvidia_transcribe.model_resources import ResourceConfigError, load_model_resources
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
    t.add_argument("--base-url", default=None, help="Explicit ASR endpoint (otherwise use configured role).")
    t.add_argument("--model", default=None, help="Explicit served ASR model.")
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
    pm.add_argument("--base-url", default=None, help="Explicit ASR endpoint override.")
    pm.add_argument("--model", default=None, help="Explicit served ASR model override.")
    pm.add_argument("--model-path", default=None, help="Explicit ASR checkpoint path override.")
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
        help="Run or resume ASR -> text refinement -> translation -> SRT/ASS rendering.",
    )
    gs.add_argument("media", nargs="?", default=None, help="Path to local media file (omit when using --recording-contract).")
    gs.add_argument("--recording-contract", default=None, help="Existing recording.contract.json to start from (skips ASR).")
    gs.add_argument("--output-dir", required=True, help="Directory for refined/translated contracts and SRT/ASS outputs.")
    gs.add_argument("--source-language", default=None, help="Source language code (e.g. 'fr', 'zh').")
    gs.add_argument("--target-language", default="en", help="Target subtitle language (default: en).")
    gs.add_argument("--refinement", default="balanced", choices=["off", "conservative", "balanced", "expressive"])
    gs.add_argument("--translator", default="openai_chat", choices=["openai_chat", "swiftu", "riva", "none"],
                    help="OpenAI chat text model, Riva worker, or none; swiftu is a compatibility alias.")
    gs.add_argument("--style", default="natural", choices=["natural", "literal", "expressive"])
    gs.add_argument("--glossary", default=None, help="Path to character/term glossary (.yaml/.json) or comma-separated terms.")
    gs.add_argument("--formats", default="srt,ass", help="Comma-separated subtitle formats (default: srt,ass).")
    gs.add_argument("--episode-stem", default=None, help="Override subtitle output filename stem.")
    gs.add_argument("--sidecar-dir", default=None, help="Optional directory for EpisodeName.<lang>.srt/.ass sidecars.")
    gs.add_argument("--overwrite-sidecar", action="store_true", help="Allow overwriting existing sidecar files.")
    gs.add_argument("--base-url", default=None, help="Explicit ASR endpoint override.")
    gs.add_argument("--model", default=None, help="Explicit served ASR model override.")
    gs.add_argument("--alignment-fallback", default="off", choices=["off", "qwen"])
    gs.add_argument("--chunk-seconds", type=float, default=90.0)
    gs.add_argument("--context-seconds", type=float, default=10.0)
    gs.add_argument("--min-tail-seconds", type=float, default=15.0)
    gs.add_argument("--text-url", "--swiftu-url", dest="text_url", default=None,
                    help="Explicit endpoint override for all text roles; --swiftu-url is legacy.")
    gs.add_argument("--text-model", "--swiftu-model", dest="text_model", default=None,
                    help="Explicit served-model override for all text roles; --swiftu-model is legacy.")
    gs.add_argument("--refinement-resource", default=None, help="Named text resource for source refinement.")
    gs.add_argument("--translation-resource", default=None, help="Named text resource for translation.")
    gs.add_argument("--condensation-resource", default=None, help="Named text resource for condensation.")
    gs.add_argument("--max-chars", type=int, default=42)
    gs.add_argument("--max-lines", type=int, default=2)
    gs.add_argument("--min-display", type=float, default=1.2)
    gs.add_argument("--max-display", type=float, default=6.5)
    gs.add_argument("--overlap-threshold", type=float, default=0.4)
    gs.add_argument("--ass-layout", default="standard", choices=["standard", "raised_main"], help="ASS positioning layout ('raised_main' clears bottom Chinese hardsubs).")
    gs.add_argument("--ass-margin-v", type=int, default=150, help="Bottom margin (px) for raised_main ASS layout.")
    gs.add_argument("--ass-opaque-box", action="store_true", help="Draw semi-opaque dark box behind ASS cues.")
    gs.add_argument("--ass-max-lanes", type=int, default=3, choices=[2, 3], help="Maximum simultaneous ASS speaker lanes.")
    gs.add_argument("--condense-high-cps", action="store_true", help="Condense overly fast (>threshold CPS) English cues with the configured text model.")
    gs.add_argument("--condense-threshold", type=float, default=25.0, help="CPS threshold triggering text-model condensation.")
    gs.add_argument("--fresh", action="store_true", help="Recompute ASR, refinement, and translation without reusing caches.")
    gs.add_argument("--fresh-refinement", action="store_true", help="Ignore refinement checkpoint cache.")
    gs.add_argument("--fresh-translation", action="store_true", help="Ignore translation checkpoint cache.")
    gs.add_argument("--re-render", action="store_true", help="Re-render SRT/ASS directly from translated.contract.json without model calls.")
    gs.add_argument("--validation-overlay", default=None, help="Optional viewing-validation JSON overlay for coarse timing & polish.")
    for parser in (t, pm, gs):
        parser.add_argument("--model-resources", default=None,
                            help="Host model-resource JSON path (or NEMOVOX_MODEL_RESOURCES).")
        parser.add_argument("--asr-resource", default=None, help="Named ASR/CTC model resource.")
    for parser in (pm, gs):
        parser.add_argument("--qwen-policy", default="subtitle", choices=["subtitle", "fine"],
                            help="Admit useful untimed subtitle windows by default; fine admits smaller gaps.")
        parser.add_argument("--qwen-device", default=None, choices=["cuda:0", "cpu"],
                            help="CUDA BF16 required unless CPU is explicitly selected.")
        parser.add_argument("--qwen-gpu-mode", default="available", choices=["available", "managed"],
                            help="Default never alters other services; managed requires a cooperative ASR lease.")
        parser.add_argument("--qwen-owned-asr-unit", default=None,
                            help="Explicit Nemovox-owned ASR systemd user unit, managed mode only.")
        parser.add_argument("--qwen-asr-lease-path", default=None,
                            help="Cooperative ASR ownership lease file, managed mode only.")

    recover = sub.add_parser(
        "recover-qwen-asr", help="Explicitly verify and restore a previously owned interrupted ASR handoff."
    )
    recover.add_argument("--asr-unit", required=True)
    recover.add_argument("--base-url", required=True)
    recover.add_argument("--model", required=True)
    recover.add_argument("--lease-path", required=True)
    recover.add_argument("--journal", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "recover-qwen-asr":
        try:
            report = recover_pending_managed_asr(ManagedASRConfig(
                unit=args.asr_unit, lease_path=Path(args.lease_path),
                base_url=args.base_url, served_model=args.model,
                journal_path=Path(args.journal),
            ))
        except (ManagedASRBlocked, ValueError, RuntimeError) as exc:
            print(json.dumps({"status": "BLOCKED", "reason": getattr(exc, "status", type(exc).__name__)}),
                  file=sys.stderr)
            return 2
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    if args.command == "validate-checkpoint":
        try:
            summary = validate_checkpoint(args.model_path)
        except CheckpointValidationError as exc:
            print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2), file=sys.stderr)
            return 2
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    if args.command in ("transcribe", "process-media", "generate-subtitles"):
        try:
            registry = load_model_resources(args.model_resources)
            asr_resource = registry.resolve(
                "asr", resource_name=args.asr_resource, endpoint=args.base_url, model=args.model)
            if args.command in ("process-media", "generate-subtitles"):
                qwen_resource = registry.resolve(
                    "forced_alignment", require_local_paths=(args.alignment_fallback == "qwen" and
                                         (args.command == "process-media" or
                                          (not args.recording_contract and not args.re_render))))
                speaker_resource = registry.resolve("speaker_embedding")
                qwen_device = args.qwen_device or qwen_resource.device or "cuda:0"
            if args.command == "generate-subtitles":
                text_overrides = {"endpoint": args.text_url, "model": args.text_model}
                refinement_resource = registry.resolve(
                    "source_refinement", resource_name=args.refinement_resource, **text_overrides)
                translation_resource = registry.resolve(
                    "text_translation", resource_name=args.translation_resource, **text_overrides)
                condensation_resource = registry.resolve(
                    "subtitle_condensation", resource_name=args.condensation_resource, **text_overrides)
                riva_resource = registry.resolve(
                    "riva_translation", require_local_paths=args.translator == "riva" and not args.re_render)
            asr_key = asr_resource.credential() if (
                args.command != "generate-subtitles" or
                (args.recording_contract is None and not args.re_render)) else None
            if args.command == "generate-subtitles":
                refinement_key = (refinement_resource.credential()
                                  if args.refinement != "off" and not args.re_render else None)
                translation_key = (translation_resource.credential()
                                   if args.translator in ("openai_chat", "swiftu") and not args.re_render else None)
                condensation_key = (condensation_resource.credential()
                                    if args.condense_high_cps and not args.re_render else None)
        except ResourceConfigError as exc:
            print(json.dumps({"status": "CONFIG_ERROR", "error": str(exc)}), file=sys.stderr)
            return 2

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
        client = NemotronVLLMClient(base_url=asr_resource.endpoint, model=asr_resource.model,
                                    api_key=asr_key)
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
        client = NemotronVLLMClient(base_url=asr_resource.endpoint, model=asr_resource.model,
                                    api_key=asr_key, timeout_s=900.0)
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
            qwen_policy=args.qwen_policy,
            qwen_device=qwen_device,
            qwen_model_path=qwen_resource.checkpoint_path,
            qwen_worker_python=qwen_resource.worker_python,
            qwen_gpu_mode=args.qwen_gpu_mode,
            qwen_owned_asr_unit=args.qwen_owned_asr_unit,
            qwen_asr_lease_path=args.qwen_asr_lease_path,
            resume=not args.fresh,
            model_path=args.model_path or asr_resource.checkpoint_path or str(DEFAULT_MODEL_PATH),
            asr_resource_fingerprint=asr_resource.fingerprint(),
            speaker_embedding_model=speaker_resource.model or "titanet_small",
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
            backend="off" if args.refinement == "off" else "openai_chat",
            profile=args.refinement,
            base_url=refinement_resource.endpoint,
            model=refinement_resource.model,
            resource_fingerprint=refinement_resource.fingerprint(),
            api_key=refinement_key,
        )
        trans_cfg = TranslationConfig(
            backend=args.translator,
            style=args.style,
            target_language=args.target_language,
            base_url=translation_resource.endpoint,
            model=translation_resource.model,
            resource_fingerprint=(riva_resource if args.translator == "riva" else translation_resource).fingerprint(),
            api_key=translation_key,
            riva_model_path=riva_resource.checkpoint_path,
            riva_python=riva_resource.worker_python,
            riva_device=riva_resource.device or "auto",
        )
        cond_cfg = TranslationConfig(
            backend="openai_chat", style=args.style, target_language=args.target_language,
            base_url=condensation_resource.endpoint, model=condensation_resource.model,
            resource_fingerprint=condensation_resource.fingerprint(), api_key=condensation_key,
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
            qwen_policy=args.qwen_policy,
            qwen_device=qwen_device,
            qwen_model_path=qwen_resource.checkpoint_path,
            qwen_worker_python=qwen_resource.worker_python,
            qwen_gpu_mode=args.qwen_gpu_mode,
            qwen_owned_asr_unit=args.qwen_owned_asr_unit,
            qwen_asr_lease_path=args.qwen_asr_lease_path,
            asr_base_url=asr_resource.endpoint,
            asr_model=asr_resource.model,
            asr_api_key=asr_key,
            asr_model_path=asr_resource.checkpoint_path,
            asr_resource_fingerprint=asr_resource.fingerprint(),
            speaker_embedding_model=speaker_resource.model or "titanet_small",
            refinement=ref_cfg,
            translation=trans_cfg,
            condensation=cond_cfg,
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
