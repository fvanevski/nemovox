"""NemoVox command-line recording, translation and rendering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nvidia_transcribe.media import MediaInspectionError
from nvidia_transcribe.model_resources import ResourceConfigError, load_model_resources
from nvidia_transcribe.moss_recording import process_moss_media
from nvidia_transcribe.resource_lifecycle import ModelLifecycle, ResourceLifecycleError
from nvidia_transcribe.refinement import RefinementConfig
from nvidia_transcribe.subtitle_pipeline import SubtitlePipelineConfig, generate_subtitles
from nvidia_transcribe.subtitles import SubtitleConfig
from nvidia_transcribe.translation import TranslationConfig, is_english_source_to_english
from nvidia_transcribe.translation_review import TranslationReviewConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nvidia-transcribe",
        description="Recognize and translate local audiovisual dialogue using native speaker-attributed events.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pm = sub.add_parser("process-media", help="Recognize recording-level speaker-attributed dialogue.")
    pm.add_argument("media", help="Path to local audio or video media file.")
    pm.add_argument("--output-dir", required=True, help="Directory for transcript, raw generation, manifest and recording contract.")
    pm.add_argument("--source-language", required=True, help="Actual source language (e.g. fr, zh).")
    pm.add_argument("--audio-stream", type=int, default=0)
    pm.add_argument("--fresh", action="store_true", help="Recompute native dialogue instead of resuming.")

    gs = sub.add_parser(
        "generate-subtitles",
        help="Run or resume ASR -> dedicated translation -> optional English review -> SRT/ASS rendering.",
    )
    gs.add_argument("media", nargs="?", default=None, help="Path to local media file (omit when using --recording-contract).")
    gs.add_argument("--recording-contract", default=None, help="Existing recording.contract.json (skips ASR); original media is needed only for optional acoustic assessment.")
    gs.add_argument("--source-quality", choices=["auto", "on", "off"], default="auto",
                    help="auto: native interval assessment; on: explicit configured speech-activity assessment; off: skip new assessment.")
    gs.add_argument("--output-dir", required=True, help="Directory for refined/translated contracts and SRT/ASS outputs.")
    gs.add_argument("--source-language", default=None, help="Source language code (e.g. 'fr', 'zh').")
    gs.add_argument("--target-language", default="en", help="Target subtitle language (default: en).")
    gs.add_argument("--refinement", default="off",
                    choices=["off", "conservative", "balanced", "expressive"],
                    help="Source-language text correction is OFF by default. Non-off profiles are explicit legacy opt-ins.")
    gs.add_argument("--translator", default="riva", choices=["openai_chat", "riva", "none"],
                    help="OpenAI chat text model, Riva worker, or none.")
    gs.add_argument("--style", default="natural", choices=["natural", "literal", "expressive"])
    gs.add_argument("--translation-review", default=None,
                    choices=["off", "conservative", "balanced", "expressive"],
                    help="Contextual English post-edit; defaults to balanced with Riva, otherwise off.")
    gs.add_argument("--glossary", default=None, help="Path to character/term glossary (.yaml/.json) or comma-separated terms.")
    gs.add_argument("--formats", default="srt,ass", help="Comma-separated subtitle formats (default: srt,ass).")
    gs.add_argument("--episode-stem", default=None, help="Override subtitle output filename stem.")
    gs.add_argument("--sidecar-dir", default=None, help="Optional directory for EpisodeName.<lang>.srt/.ass sidecars.")
    gs.add_argument("--overwrite-sidecar", action="store_true", help="Allow overwriting existing sidecar files.")
    gs.add_argument("--audio-stream", type=int, default=0)
    gs.add_argument("--text-url", default=None,
                    help="Explicit endpoint override for all text roles.")
    gs.add_argument("--text-model", default=None,
                    help="Explicit served-model override for all text roles.")
    gs.add_argument("--refinement-resource", default=None, help="Named text resource for source refinement.")
    gs.add_argument("--translation-resource", default=None,
                    help="Named text resource for legacy general-text-model primary translation.")
    gs.add_argument("--translation-review-resource", default=None,
                    help="Named OpenAI-compatible text resource for English post-edit.")
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
    assess = sub.add_parser("assess-source", help="Assess native dialogue intervals without translation or automatic correction.")
    assess.add_argument("media", nargs="?", help="Original recording (required with --activity).")
    assess.add_argument("--recording-contract", required=True)
    assess.add_argument("--output-dir", required=True)
    assess.add_argument("--activity", action="store_true", help="Explicitly run the configured CPU acoustic event detector.")
    assess.add_argument("--fresh", action="store_true")
    assess.add_argument("--model-resources", default=None)
    for parser in (pm, gs):
        parser.add_argument("--model-resources", default=None,
                            help="Host model-resource JSON path (or NEMOVOX_MODEL_RESOURCES).")
        parser.add_argument("--asr-resource", default=None, help="Named MOSS recording resource.")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "assess-source":
        from nvidia_transcribe.source_assessment import assess_native_dialogue
        from nvidia_transcribe.speech_activity import assess_recording_activity
        from nvidia_transcribe.subtitle_pipeline import _atomic_write
        try:
            contract = json.loads(Path(args.recording_contract).read_text(encoding="utf-8"))
            if args.activity:
                if not args.media:
                    raise ValueError("--activity requires the original media")
                resource = load_model_resources(args.model_resources).resolve("speech_activity")
                report, _ = assess_recording_activity(contract, media_path=args.media, resource=resource,
                    output_dir=Path(args.output_dir), resume=not args.fresh)
            else:
                report = assess_native_dialogue(contract)
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            _atomic_write(out / "source-quality.json", json.dumps(report, ensure_ascii=False, indent=2))
            print(json.dumps({"status": "OK", "report": str(out / "source-quality.json")}))
            return 0
        except (ResourceConfigError, OSError, ValueError, RuntimeError) as exc:
            print(json.dumps({"status": "ERROR", "error": str(exc)}), file=sys.stderr)
            return 1
    if args.command == "generate-subtitles" and args.translation_review is None:
        args.translation_review = "balanced" if args.translator == "riva" else "off"

    try:
        registry = load_model_resources(args.model_resources)
        asr_resource = registry.resolve("asr", resource_name=args.asr_resource)
        if args.command == "process-media":
            out = process_moss_media(args.media, output_dir=args.output_dir, resource=asr_resource,
                lifecycle=ModelLifecycle(), source_language=args.source_language,
                audio_stream_index=args.audio_stream, resume=not args.fresh)
            c = out["contract"]
            print(json.dumps({"status": "OK", **{k:v for k,v in out.items() if k != "contract"},
                              "source_language": c["source_language"],
                              "language_provenance": c["language_provenance"],
                              "duration_s": c["source"]["duration_s"],
                              "utterance_count": len(c["utterances"]), "speaker_count": len(c["speakers"]),
                              "source_quality": c["source_quality"],
                              "warnings": c["processing_warnings"]}, ensure_ascii=False, indent=2))
            return 0
        source_for_preflight = args.source_language
        if not source_for_preflight and args.recording_contract and not args.re_render:
            try:
                contract = json.loads(Path(args.recording_contract).read_text(encoding="utf-8"))
                if isinstance(contract, dict):
                    source_for_preflight = contract.get("source_language")
            except (OSError, ValueError, TypeError):
                pass  # Pipeline admission owns invalid-input errors.
        same_language = is_english_source_to_english(source_for_preflight, args.target_language)
        text_overrides = {"endpoint": args.text_url, "model": args.text_model}
        refinement_resource = registry.resolve("source_refinement", resource_name=args.refinement_resource, **text_overrides)
        translation_resource = registry.resolve("text_translation", resource_name=args.translation_resource, **text_overrides)
        condensation_resource = registry.resolve("subtitle_condensation", resource_name=args.condensation_resource, **text_overrides)
        review_resource = registry.resolve("translation_review", resource_name=args.translation_review_resource, **text_overrides)
        activity_resource = registry.resolve("speech_activity") if "speech_activity" in registry.roles else None
        riva_resource = registry.resolve("riva_translation", require_local_paths=(
            args.translator == "riva" and not args.re_render and not same_language))
        refinement_key = refinement_resource.credential() if args.refinement != "off" and not args.re_render else None
        translation_key = translation_resource.credential() if args.translator == "openai_chat" and not args.re_render else None
        condensation_key = (condensation_resource.credential()
                            if args.condense_high_cps and args.translator != "none" and not args.re_render else None)
        review_key = (review_resource.credential() if args.translation_review != "off" and args.translator == "riva"
                      and not args.re_render and not same_language else None)
    except ResourceConfigError as exc:
        print(json.dumps({"status": "CONFIG_ERROR", "error": str(exc)}), file=sys.stderr)
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "RESOURCE_BLOCKED" if isinstance(exc, ResourceLifecycleError) else "ERROR",
                          "reason": exc.status if isinstance(exc, ResourceLifecycleError) else None,
                          "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    if args.command == "generate-subtitles":
        if args.translation_review != "off" and args.translator != "riva":
            print(json.dumps({"status": "CONFIG_ERROR",
                              "error": "English translation review requires --translator riva"}), file=sys.stderr)
            return 2
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
            riva_device=riva_resource.device or "cuda:0",
        )
        review_cfg = TranslationReviewConfig(
            profile=args.translation_review, style=args.style,
            target_language=args.target_language,
            base_url=review_resource.endpoint, model=review_resource.model,
            resource_fingerprint=review_resource.fingerprint(), api_key=review_key,
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
            audio_stream_index=args.audio_stream,
            refinement=ref_cfg,
            translation=trans_cfg,
            translation_review=review_cfg,
            condensation=cond_cfg,
            subtitles=sub_cfg,
            formats=formats,
            episode_stem=args.episode_stem,
            sidecar_dir=args.sidecar_dir,
            overwrite_sidecar=args.overwrite_sidecar,
            resume=not args.fresh,
            resume_refinement=not (args.fresh or args.fresh_refinement),
            resume_translation=not (args.fresh or args.fresh_translation),
            resume_translation_review=not (args.fresh or args.fresh_translation),
            resume_condensation=not (args.fresh or args.fresh_translation),
            re_render_only=args.re_render,
            condense_high_cps=args.condense_high_cps,
            condense_cps_threshold=args.condense_threshold,
            validation_overlay_path=args.validation_overlay,
            source_quality=args.source_quality,
        )
        try:
            res = generate_subtitles(
                args.media,
                recording_contract_path=args.recording_contract,
                output_dir=args.output_dir,
                config=scfg,
                glossary=args.glossary,
                progress_callback=lambda msg: print(
                    msg if msg.startswith("NEMOVOX_EVENT ") else f"[nvidia-transcribe] {msg}",
                    file=sys.stderr, flush=True),
                lifecycle=ModelLifecycle(),
                lifecycle_resources={
                    "asr": asr_resource, "source_refinement": refinement_resource,
                    "text_translation": translation_resource, "riva_translation": riva_resource,
                    "translation_review": review_resource, "subtitle_condensation": condensation_resource,
                    **({"speech_activity": activity_resource} if activity_resource else {}),
                },
            )
        except (FileExistsError, FileNotFoundError, MediaInspectionError, ValueError, RuntimeError) as exc:
            print(json.dumps({"status": "RESOURCE_BLOCKED" if isinstance(exc, ResourceLifecycleError) else "ERROR",
                              "reason": exc.status if isinstance(exc, ResourceLifecycleError) else None,
                              "error": str(exc)}, indent=2), file=sys.stderr)
            return 1
        print(json.dumps({"status": "OK", **res}, ensure_ascii=False, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
