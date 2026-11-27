"""End-to-end multilingual subtitle generation pipeline and re-render orchestrator."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from nvidia_transcribe.model_resources import ModelResource
from nvidia_transcribe.moss_recording import is_moss_contract, process_moss_media, validate_moss_contract
from nvidia_transcribe.resource_lifecycle import ModelLifecycle, ResourceLifecycleError, managed_text_client
from nvidia_transcribe.source_assessment import assess_native_dialogue
from nvidia_transcribe.speech_activity import assess_recording_activity
from nvidia_transcribe.refinement import (
    GlossaryConfig,
    RefinementConfig,
    load_glossary,
    refine_utterances_with_text_model,
)
from nvidia_transcribe.subtitles import SubtitleConfig, render_subtitles
from nvidia_transcribe.translation import (
    TranslationConfig,
    condense_high_cps_cues_with_text_model,
    is_english_source_to_english,
    translate_utterances,
)
from nvidia_transcribe.translation_review import TranslationReviewConfig, review_translated_utterances
from nvidia_transcribe.ninfer_messages_client import QUALITY_COMPLETION_TOKENS, QUALITY_TIMEOUT_S
from nvidia_transcribe.utterances import apply_validation_overlay


@dataclass(frozen=True)
class SubtitlePipelineConfig:
    source_language: str | None = None
    target_language: str = "en"
    audio_stream_index: int = 0
    refinement: RefinementConfig = RefinementConfig(backend="off", profile="off")
    translation: TranslationConfig = TranslationConfig()
    translation_review: TranslationReviewConfig = TranslationReviewConfig()
    condensation: TranslationConfig | None = None
    subtitles: SubtitleConfig = SubtitleConfig()
    formats: tuple[str, ...] = ("srt", "ass")
    episode_stem: str | None = None
    sidecar_dir: str | None = None
    overwrite_sidecar: bool = False
    resume: bool = True
    resume_refinement: bool = True
    resume_translation: bool = True
    resume_translation_review: bool = True
    resume_condensation: bool = True
    re_render_only: bool = False
    condense_high_cps: bool = False
    condense_cps_threshold: float = 25.0
    source_quality: str = "auto"  # auto: native geometry; on: explicit acoustic activity assessment
    validation_overlay_path: str | Path | None = None
    review_evidence: str = "auto"
    visual_max_frames: int = 6
    visual_thinking_budget: int = 1024


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _format_utterance_transcript_lines(utterances: Sequence[dict[str, Any]], key: str, fallback_key: str | None = None) -> str:
    lines: list[str] = []
    for i, u in enumerate(utterances):
        if key == "translated_text" and (
            u.get("merged_into")
            or u.get("fragment_suppressed")
            or u.get("overlay_suppressed")
            or u.get("suppressed_duplicate_of")
        ):
            continue
        uid = u.get("utterance_id") or u.get("id") or f"utt-{i:04d}"
        spk = u.get("speaker_id") or u.get("global_speaker_id") or "SPK_UNK"
        s = u.get("source_start")
        e = u.get("source_end")
        ts = f"{s:7.2f}s - {e:7.2f}s" if s is not None and e is not None else "   untimed        "
        txt = u.get(key)
        if (txt is None or not str(txt).strip()) and fallback_key:
            txt = u.get(fallback_key)
        lines.append(f"[{uid}] [{ts}] {spk}: {str(txt or '').strip()}")
    return "\n".join(lines) + ("\n" if lines else "")


def validate_deferred_review(rows, review, translation):
    """A deferred stage may render only complete preserved primary allocations."""
    if review.get("status") not in ("completed_with_deferred_comparisons","completed_with_deferred_evidence"):
        return
    comparison=set(review.get("deferred_comparison_ids") or [])
    evidence={x["target_id"] for x in review.get("supplementary_proposal_errors",[])}
    def uid(row):
        return str(row.get("utterance_id") or row.get("id"))
    observed_comparison={uid(r) for r in rows if r.get("translation_review_status")==
                         "review_deferred_comparative_preserved_machine_translation"}
    observed_evidence={uid(r) for r in rows if r.get("translation_review_status")==
                       "review_deferred_evidence_preserved_machine_translation"}
    pending=comparison|evidence
    if (not pending or comparison&evidence or comparison!=observed_comparison
            or evidence!=observed_evidence or translation.get("errors")
            or review.get("reviewed",0)+len(pending)!=len(rows)
            or any(not str(r.get("machine_translated_text") or "").strip()
                   or (uid(r) in pending and r.get("translated_text")!=r.get("machine_translated_text"))
                   for r in rows)):
        raise RuntimeError("Deferred comparative review lacks complete preserved primary MT")


def generate_subtitles(
    media_path: str | Path | None = None,
    *,
    recording_contract_path: str | Path | None = None,
    output_dir: str | Path,
    config: SubtitlePipelineConfig | None = None,
    glossary: GlossaryConfig | str | Path | None = None,
    text_client: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    lifecycle: ModelLifecycle | None = None,
    lifecycle_resources: Mapping[str, ModelResource] | None = None,
) -> dict[str, Any]:
    """Run staged ASR -> primary translation -> optional English review -> subtitle rendering.

    Can start from raw ``media_path`` OR resume directly from ``recording_contract_path``
    without loading the recognition model model. Explicit CPU-only re-rendering requires
    an existing ``translated.contract.json``; missing input never starts inference.
    """
    t0 = time.perf_counter()
    cfg = config or SubtitlePipelineConfig()
    out_dir = Path(output_dir)
    translated_path = out_dir / "translated.contract.json"
    if cfg.re_render_only and not translated_path.is_file():
        raise FileNotFoundError(f"Offline re-render requires an existing translated contract: {translated_path}")
    if cfg.review_evidence not in ("auto", "off"):
        raise ValueError("Unsupported adaptive review evidence policy")
    if cfg.source_quality not in ("auto", "on", "off"):
        raise ValueError("Unsupported source quality mode")
    if cfg.translation_review.profile != "off" and cfg.translation.backend != "riva":
        raise ValueError("Contextual English post-edit requires dedicated Riva primary translation")
    if (cfg.condense_high_cps and cfg.translation.backend == "riva"
            and cfg.condensation is None and not cfg.re_render_only):
        raise ValueError(
            "Riva translation with subtitle condensation requires an explicit "
            "OpenAI-compatible condensation resource config."
        )
    asr_resource = (lifecycle_resources or {}).get("asr")
    retained_contract = None
    if recording_contract_path is not None and not cfg.re_render_only:
        rc_input = Path(recording_contract_path)
        if not rc_input.is_file():
            raise FileNotFoundError(f"Recording contract not found: {rc_input}")
        retained_contract = json.loads(rc_input.read_text(encoding="utf-8"))
        if not isinstance(retained_contract, dict) or not isinstance(retained_contract.get("utterances"), list):
            raise ValueError("Recording contract must contain utterances")
    native_source = not cfg.re_render_only and (
        retained_contract is None or is_moss_contract(retained_contract))
    retained_canonical = False
    if native_source:
        if cfg.refinement.profile != "off":
            raise ValueError("Native dialogue preserves observed source wording; text-only source rewriting is unsupported")
        if retained_contract is not None:
            validate_moss_contract(retained_contract, media_path=media_path, source_language=cfg.source_language)
        elif (asr_resource is None or getattr(asr_resource, "adapter", None) != "moss_events"
              or lifecycle is None):
            raise ValueError("Fresh recording requires a managed MOSS local worker resource")
        if cfg.source_quality == "on" and (media_path is None
                or not lifecycle_resources or "speech_activity" not in lifecycle_resources):
            raise ValueError("Explicit acoustic assessment requires original media and a configured speech_activity resource")
    elif not cfg.re_render_only:
        if cfg.source_quality == "on":
            raise ValueError("Acoustic assessment requires a native dialogue recording")
        # Imported contracts remain a downstream entry point, never a reason to
        # re-acquire old lexical observations or resurrect a retired backend.
        quality = retained_contract.get("source_quality") or {}
        if quality.get("schema") == "nemovox-current-source-quality/v1":
            if (quality.get("status") not in ("complete", "partial")
                    or not quality.get("media_fingerprint_sha256")
                    or quality["media_fingerprint_sha256"] != retained_contract["source"].get("fingerprint_sha256")
                    or quality.get("source_language") != retained_contract.get("source_language")):
                raise ValueError("Retained canonical source quality identity mismatch")
            retained_canonical = True
    if (cfg.translation_review.profile != "off" and (native_source or retained_canonical)
            and lifecycle is not None and lifecycle_resources and "translation_review" in lifecycle_resources):
        cfg = replace(cfg, translation_review=replace(cfg.translation_review,
            wire_protocol="ninfer_messages", thinking_budget=2048, reasoning_effort="xhigh",
            max_tokens=max(cfg.translation_review.max_tokens,QUALITY_COMPLETION_TOKENS),
            timeout_s=max(cfg.translation_review.timeout_s,QUALITY_TIMEOUT_S)))
    out_dir.mkdir(parents=True, exist_ok=True)
    gl = glossary if isinstance(glossary, GlossaryConfig) else load_glossary(glossary)

    def _log(msg: str) -> None:
        if progress_callback is not None:
            progress_callback(msg)

    def _event(stage: str, status: str, detail: str = "") -> None:
        if progress_callback is not None:
            progress_callback("NEMOVOX_EVENT " + json.dumps({
                "schema": "nemovox-progress/v1", "stage": stage,
                "status": status, "detail": detail,
            }, ensure_ascii=False))

    def _text_stage(role: str):
        if lifecycle is None or text_client is not None or not lifecycle_resources:
            return nullcontext(text_client)
        resource = lifecycle_resources.get(role)
        if resource is None:
            return nullcontext(text_client)
        stage_config = {
            "source_refinement": cfg.refinement,
            "text_translation": cfg.translation,
            "translation_review": cfg.translation_review,
            "subtitle_condensation": cfg.condensation or cfg.translation,
        }[role]
        return managed_text_client(
            lifecycle, resource,
            timeout_s=stage_config.timeout_s,
            reasoning_effort=stage_config.reasoning_effort,
            **({"wire_protocol": stage_config.wire_protocol}
               if role == "translation_review" and stage_config.wire_protocol != "openai_chat" else {}),
        )

    refined_path = out_dir / "refined.contract.json"
    quality_resume="not_requested"
    salvage_review_error = False

    if cfg.re_render_only:
        _log("Stage 1-3: Reusing existing translated.contract.json (--re-render)")
        for skipped in ("source_quality", "translation", "review", "condensation"):
            _event(skipped, "skipped", "Offline re-render uses retained contract")
        retained_contract_bytes = translated_path.read_bytes()
        translated_contract = json.loads(retained_contract_bytes)
        ref_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        trans_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        condense_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        review_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        retained_review = translated_contract.get("translation_review") or {}
        if retained_review.get("status") in ("completed_with_deferred_comparisons","completed_with_deferred_evidence"):
            validate_deferred_review(translated_contract["utterances"],retained_review,translated_contract.get("translation") or {})
            review_summary["retained_review_status"] = retained_review["status"]
            review_summary["deferred_comparison_ids"] = retained_review.get("deferred_comparison_ids",[])
            review_summary["supplementary_proposal_errors"] = retained_review.get("supplementary_proposal_errors",[])
        elif retained_review.get("status") == "completed_with_errors":
            salvage_review_error = True
            if cfg.validation_overlay_path is not None:
                raise ValueError("Failed-review salvage must preserve the translated contract; overlay not allowed")
            review_summary.update(
                retained_review_status="completed_with_errors",
                errors=retained_review.get("errors", []),
                resource_blocked_reason=retained_review.get("resource_blocked_reason"),
            )
            # A failed or interrupted new render must not inherit an earlier
            # successful salvage receipt. Existing subtitle files are retained.
            (out_dir / "subtitles.metadata.json").unlink(missing_ok=True)
            _log("Offline salvage: required English review remains failed; rendering retained translations")
            _event("review", "failed", "Retained required review failed; offline salvage only")
    else:
        # Stage 1: Load existing recording contract or run recording-level ASR.
        if recording_contract_path is not None:
            rc_path = Path(recording_contract_path)
            if not rc_path.is_file():
                raise FileNotFoundError(f"Recording contract not found: {rc_path}")
            _log(f"Stage 1/4: Loading existing recording contract {rc_path}")
            contract = retained_contract
            # Retained recordings preserve their actual dialogue IDs.
            default_stem = Path(str((contract.get("source") or {}).get("path") or rc_path)).stem
        else:
            if media_path is None:
                raise ValueError("Must supply either media_path or --recording-contract.")
            m_path = Path(media_path)
            if not m_path.is_file():
                raise FileNotFoundError(f"Media file not found: {m_path}")
            _event("asr", "running", "Recognizing recording-level dialogue events")
            asr_res = process_moss_media(m_path, output_dir=out_dir / "asr", resource=asr_resource,
                lifecycle=lifecycle, source_language=cfg.source_language,
                audio_stream_index=cfg.audio_stream_index, resume=cfg.resume)
            _event("asr", "complete")
            contract = asr_res["contract"]
            default_stem = m_path.stem

        if native_source and cfg.source_quality != "off":
            _event("source_quality", "running", "Assessing observed dialogue coverage")
            if cfg.source_quality == "on":
                report, quality_resume = assess_recording_activity(contract, media_path=media_path,
                    resource=lifecycle_resources["speech_activity"], output_dir=out_dir,
                    resume=cfg.resume)
            else:
                report = assess_native_dialogue(contract)
                quality_resume = "computed_geometry"
            contract = {**contract, "source_quality": report}
            _atomic_write(out_dir / "source-quality.json", json.dumps(report, ensure_ascii=False, indent=2))
            _atomic_write(out_dir / "canonical-source.contract.json", json.dumps(contract, ensure_ascii=False, indent=2))
            _event("source_quality", "complete", report["scope"] + "; no automatic source corrections")
        else:
            _event("source_quality", "skipped", "Explicit skip or imported recording evidence")
        src_lang = contract.get("source_language") if native_source else cfg.source_language or contract.get("source_language")
        raw_utterances = list(contract.get("utterances") or [])

        # The ordinary path preserves the ASR transcript unchanged. The
        # compatibility contract still includes refined_text=source_text,
        # but this fast copy requires no text-model invocation.
        if cfg.refinement.profile == "off" or cfg.refinement.backend == "off":
            _log("Stage 2/4: Preserving raw ASR transcript (source correction disabled)")
        else:
            _log(f"Stage 2/4: Explicit legacy source correction (profile={cfg.refinement.profile})")
        with _text_stage("source_refinement") as stage_client:
            refined_utts, ref_summary = refine_utterances_with_text_model(
                raw_utterances,
                source_language=src_lang,
                config=cfg.refinement,
                glossary=gl,
                text_client=stage_client,
                checkpoint_path=out_dir / "refinement.checkpoint.json",
                resume=cfg.resume_refinement,
            )
        refined_contract = {
            **contract,
            "utterances": refined_utts,
            "refinement": ref_summary,
        }
        _atomic_write(refined_path, json.dumps(refined_contract, ensure_ascii=False, indent=2))

        # Stage 3: Dedicated machine translation of unchanged ASR by default
        _event("translation", "running")
        _log(f"Stage 3/4: Translation ({src_lang} -> {cfg.target_language}, backend={cfg.translation.backend}, style={cfg.translation.style})")
        # Nomination sees source-bound uncertainty before requesting any
        # alternative. No extra Riva requests for review-off or English bypass.
        translation_input = refined_utts
        if native_source and cfg.translation.backend == "riva":
            from nvidia_transcribe.review_uncertainty import bind_review_uncertainty
            translation_input = bind_review_uncertainty(refined_utts, contract.get("source_quality"))
        primary_config = replace(cfg.translation, connected_hypotheses=(
            native_source and cfg.translation.backend == "riva"
            and cfg.translation_review.profile != "off"
            and cfg.translation_review.wire_protocol == "ninfer_messages"))
        with _text_stage("text_translation") as stage_client:
            translated_utts, trans_summary = translate_utterances(
                translation_input,
                source_language=src_lang,
                config=primary_config,
                glossary=gl,
                text_client=stage_client,
                lifecycle=lifecycle,
                riva_resource_name=(lifecycle_resources["riva_translation"].name
                                    if lifecycle_resources and "riva_translation" in lifecycle_resources else "riva_local"),
                checkpoint_path=out_dir / "translation.checkpoint.json",
                resume=cfg.resume_translation,
            )
        # Legacy text backends can retain source text after request/ID
        # failures without aborting rendering. Preserve that recovery, but
        # do not present degraded primary translation as a successful stage.
        translation_errors = trans_summary.get("errors") or []
        translation_skipped = (
            cfg.translation.backend == "none" or
            (cfg.translation.backend == "riva" and
             is_english_source_to_english(src_lang, cfg.target_language))
        )
        _event("translation",
               "skipped" if translation_skipped else
               "failed" if translation_errors else "complete",
               (f"completed_with_errors: {len(translation_errors)} translation error(s); "
                "fallback source text retained") if translation_errors else
               str(trans_summary.get("status", "")))
        # Retain the first-stage MT independently of all text-model edits.
        if (cfg.translation.backend == "riva"
                and is_english_source_to_english(src_lang, cfg.target_language)):
            # Already-English speech must not enter a text-based source correction.
            for u in translated_utts:
                u["translation_review_status"] = "skipped_same_language"
            review_summary = {
                "status": "skipped_same_language", "profile": cfg.translation_review.profile,
                "reviewed": 0, "changed": 0,
            }
            _event("review", "skipped", "Same-language translation bypass")
        elif cfg.translation.backend == "riva":
            if cfg.translation_review.profile != "off":
                _event("review", "running")
            _log(f"Stage 3b/4: Contextual English post-edit (profile={cfg.translation_review.profile})")
            from nvidia_transcribe.review_uncertainty import bind_review_uncertainty
            translated_utts=bind_review_uncertainty(translated_utts,contract.get("source_quality"))
            with _text_stage("translation_review") as stage_client:
                evidence_provider = None
                if (native_source and cfg.review_evidence == "auto" and cfg.translation_review.profile != "off"
                        and cfg.translation_review.max_evidence_requests_per_batch > 0
                        and cfg.translation_review.wire_protocol == "ninfer_messages" and lifecycle is not None
                        and lifecycle_resources and "translation_review" in lifecycle_resources and text_client is None):
                    from nvidia_transcribe.review_evidence import ReviewEvidenceProvider
                    review_resource = lifecycle_resources["translation_review"]
                    evidence_provider = ReviewEvidenceProvider(media_path=media_path, contract=contract,
                        lifecycle=lifecycle, review_resource=review_resource,
                        visual_resource=lifecycle_resources.get("visual_inspection", review_resource),
                        acoustic_resource=asr_resource, review_client=stage_client,
                        max_frames=cfg.visual_max_frames, visual_thinking_budget=cfg.visual_thinking_budget)
                translated_utts, review_summary = review_translated_utterances(
                    translated_utts, source_language=src_lang, config=cfg.translation_review,
                    glossary=gl, text_client=stage_client,
                    checkpoint_path=out_dir / "translation-review.checkpoint.json",
                    resume=cfg.resume_translation_review,
                    **({"evidence_provider": evidence_provider} if evidence_provider is not None else {}),
                )
            validate_deferred_review(translated_utts,review_summary,trans_summary)
            _event("review",
                   "skipped" if review_summary.get("status") == "disabled" else
                   "failed" if review_summary.get("status") == "completed_with_errors" else "complete",
                   str(review_summary.get("status", "")))
        else:
            review_summary = {"status": "not_applicable_legacy_translation"}
            _event("review", "skipped", "Not a dedicated Riva review path")
        if cfg.condense_high_cps and cfg.translation.backend != "none":
            _event("condensation", "running")
            _log(f"Stage 3c/4: Selective condensation (> {cfg.condense_cps_threshold} CPS)")
            with _text_stage("subtitle_condensation") as stage_client:
                translated_utts, condense_summary = condense_high_cps_cues_with_text_model(
                    translated_utts,
                    source_language=src_lang,
                    config=cfg.condensation or cfg.translation,
                    cps_threshold=cfg.condense_cps_threshold,
                    target_cps=cfg.subtitles.target_cps,
                    text_client=stage_client,
                    checkpoint_path=out_dir / "condensation.checkpoint.json",
                    resume=cfg.resume_condensation,
                )
            _event("condensation", "complete", str(condense_summary.get("status", "")))
        else:
            condense_summary = {"status": "disabled", "candidate_count": 0, "condensed_count": 0}
            _event("condensation", "skipped", "Optional condensation disabled")
        translated_contract = {
            **refined_contract,
            "target_language": cfg.target_language,
            "utterances": translated_utts,
            "translation": trans_summary,
            "translation_review": review_summary,
            "condensation": condense_summary,
        }
        review_failed = (cfg.translation_review.profile != "off"
                         and review_summary.get("status") == "completed_with_errors")
        if review_failed:
            # Invalidate an older successful render receipt before publishing
            # the failed contract, even if the new failure repeats its bytes.
            (out_dir / "subtitles.metadata.json").unlink(missing_ok=True)
        _atomic_write(translated_path, json.dumps(translated_contract, ensure_ascii=False, indent=2))
        if review_failed:
            if review_summary.get("resource_blocked_reason"):
                raise ResourceLifecycleError(
                    review_summary["resource_blocked_reason"],
                    "Required English review model unavailable; "
                    "machine translations preserved in translated.contract.json",
                )
            raise RuntimeError("Required contextual English review did not complete; "
                               "machine translations were preserved in translated.contract.json")
        _atomic_write(out_dir / "raw_source_transcript.txt", _format_utterance_transcript_lines(translated_utts, "raw_source_text", "source_text"))
        _atomic_write(out_dir / "refined_source_transcript.txt", _format_utterance_transcript_lines(translated_utts, "refined_text", "source_text"))
        _atomic_write(out_dir / "final_english_subtitles.txt", _format_utterance_transcript_lines(translated_utts, "translated_text"))

    # Determine episode stem cleanly
    src_path_meta = (translated_contract.get("source") or {}).get("path")
    ep_stem = cfg.episode_stem or (Path(media_path).stem if media_path else (Path(str(src_path_meta)).stem if src_path_meta else "subtitles"))

    overlay_summary: dict[str, Any] = {"status": "none"}
    if cfg.validation_overlay_path is not None:
        from nvidia_transcribe import connected_units
        connected_units.validate_selected_units(translated_contract.get("utterances") or [],
            require_complete=(translated_contract.get("translation_review") or {}).get("translation_unit_schema") == connected_units.SCHEMA)
        overlay_doc = json.loads(Path(cfg.validation_overlay_path).read_text(encoding="utf-8"))
        translated_contract, overlay_summary = apply_validation_overlay(translated_contract, overlay_doc)
        _atomic_write(translated_path, json.dumps(translated_contract, ensure_ascii=False, indent=2))
        overlaid_utts = translated_contract.get("utterances") or []
        _atomic_write(out_dir / "raw_source_transcript.txt", _format_utterance_transcript_lines(overlaid_utts, "raw_source_text", "source_text"))
        _atomic_write(out_dir / "refined_source_transcript.txt", _format_utterance_transcript_lines(overlaid_utts, "refined_text", "source_text"))
        _atomic_write(out_dir / "final_english_subtitles.txt", _format_utterance_transcript_lines(overlaid_utts, "translated_text"))

    # Stage 4: Render English SRT & ASS subtitles
    _event("render", "running")
    _log(f"Stage 4/4: Rendering subtitles ({','.join(cfg.formats)}) -> {ep_stem}.{cfg.target_language}.*")
    render_res = render_subtitles(
        translated_contract,
        output_dir=out_dir,
        episode_stem=ep_stem,
        target_language=cfg.target_language,
        formats=cfg.formats,
        config=cfg.subtitles,
        sidecar_dir=Path(cfg.sidecar_dir) if cfg.sidecar_dir else None,
        overwrite_sidecar=cfg.overwrite_sidecar,
    )

    elapsed_s = round(time.perf_counter() - t0, 3)
    metadata = {
        "episode_stem": ep_stem,
        "source_language": cfg.source_language or translated_contract.get("source_language"),
        "target_language": cfg.target_language,
        "files": {
            "raw_source_transcript": str(out_dir / "raw_source_transcript.txt"),
            "refined_source_transcript": str(out_dir / "refined_source_transcript.txt"),
            "final_english_subtitles": str(out_dir / "final_english_subtitles.txt"),
            "refined_contract": str(refined_path),
            "translated_contract": str(translated_path),
            **render_res["outputs"],
        },
        "sidecars": render_res["sidecars"],
        "source_quality": translated_contract.get("source_quality", {"status": "not_run"}),
        "source_quality_resume":quality_resume,
        "refinement": ref_summary,
        "translation": trans_summary,
        "translation_review": review_summary,
        "condensation": condense_summary,
        "validation_overlay": overlay_summary,
        "subtitles": render_res["stats"],
        "subtitle_conflicts": render_res["conflicts"][:200],
        "cue_lineage": render_res.get("cue_lineage", []),
        "total_wall_time_s": elapsed_s,
    }
    if render_res.get("no_renderable_dialogue"):
        # Retain the local diagnostic files and the actual conflict report,
        # but never publish a successful job or export empty sidecars.
        metadata["render_status"]="failed_no_renderable_dialogue"
        _atomic_write(out_dir / "subtitles.metadata.json",json.dumps(metadata,ensure_ascii=False,indent=2))
        _event("render","failed","Recognized dialogue has no usable subtitle timing")
        raise RuntimeError("Recognized dialogue has no usable subtitle timing; diagnostic artifacts preserved")
    if salvage_review_error:
        # Successful rendering is not successful required review. Bind this
        # receipt to actual retained contract bytes and local subtitle outputs.
        # Basenames make the receipt verifiable after moving a whole job.
        if translated_path.read_bytes() != retained_contract_bytes:
            raise RuntimeError("Retained translated contract changed during offline salvage")
        outputs = {}
        for fmt, filename in render_res["outputs"].items():
            path = Path(filename)
            if path.resolve().parent != out_dir.resolve():
                raise ValueError("Salvage render output must be inside the job directory")
            outputs[fmt] = {"name": path.name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if not outputs:
            raise ValueError("Salvage render requires at least one subtitle output")
        metadata["render_status"] = "rendered_with_review_errors"
        metadata["salvage_render_receipt"] = {
            "schema": "nemovox-failed-review-render/v1",
            "translated_contract_sha256": hashlib.sha256(retained_contract_bytes).hexdigest(),
            "outputs": outputs,
        }
    _atomic_write(out_dir / "subtitles.metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    _event("render", "complete", "Offline salvage; review remains failed" if salvage_review_error else "")
    return metadata
