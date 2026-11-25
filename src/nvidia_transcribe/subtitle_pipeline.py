"""End-to-end multilingual subtitle generation pipeline and re-render orchestrator."""

from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.model_resources import ModelResource
from nvidia_transcribe.resource_lifecycle import ModelLifecycle, ResourceLifecycleError, managed_asr_client, managed_text_client
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media
from nvidia_transcribe.source_quality import SourceQualityPolicy, review_current_source
from nvidia_transcribe.source_quality_checkpoint import identity as quality_checkpoint_identity, run_checkpoint
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
    asr_mode: str = "speaker"
    chunk_seconds: float = 90.0
    context_seconds: float = 10.0
    chunk_policy: str = "fixed"  # opt in to pause_aware after qualification
    min_tail_seconds: float = 15.0
    audio_stream_index: int = 0
    alignment_fallback: str = "off"
    qwen_policy: str = "subtitle"
    qwen_device: str = "cuda:0"
    qwen_model_path: str | None = None
    qwen_worker_python: str | None = None
    qwen_gpu_mode: str = "available"
    qwen_owned_asr_unit: str | None = None
    qwen_asr_lease_path: str | None = None
    speaker_embeddings: str = "none"
    speaker_embedding_model: str = "titanet_small"
    asr_base_url: str = "http://127.0.0.1:8190/v1"
    asr_model: str = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
    asr_api_key: str | None = None
    asr_resource_fingerprint: str = ""
    asr_model_path: str | None = None
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
    source_quality: str = "auto"  # auto: Riva subtitle jobs with original media and managed resources
    source_quality_policy: SourceQualityPolicy = SourceQualityPolicy()
    validation_overlay_path: str | Path | None = None


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


def generate_subtitles(
    media_path: str | Path | None = None,
    *,
    recording_contract_path: str | Path | None = None,
    output_dir: str | Path,
    config: SubtitlePipelineConfig | None = None,
    glossary: GlossaryConfig | str | Path | None = None,
    asr_client: NemotronVLLMClient | None = None,
    text_client: Any = None,
    progress_callback: Callable[[str], None] | None = None,
    lifecycle: ModelLifecycle | None = None,
    lifecycle_resources: Mapping[str, ModelResource] | None = None,
) -> dict[str, Any]:
    """Run staged ASR -> primary translation -> optional English review -> subtitle rendering.

    Can start from raw ``media_path`` OR resume directly from ``recording_contract_path``
    without loading the 30B ASR model. Explicit CPU-only re-rendering requires
    an existing ``translated.contract.json``; missing input never starts inference.
    """
    t0 = time.perf_counter()
    cfg = config or SubtitlePipelineConfig()
    out_dir = Path(output_dir)
    translated_path = out_dir / "translated.contract.json"
    if cfg.re_render_only and not translated_path.is_file():
        raise FileNotFoundError(f"Offline re-render requires an existing translated contract: {translated_path}")
    if cfg.source_quality not in ("auto", "on", "off"):
        raise ValueError("Unsupported source quality mode")
    quality_enabled = (not cfg.re_render_only and media_path is not None and (
        cfg.source_quality == "on" or (cfg.source_quality == "auto"
        and cfg.translation.backend == "riva" and lifecycle is not None
        and lifecycle_resources is not None)))
    if cfg.source_quality == "on" and not cfg.re_render_only and not media_path:
        raise ValueError("Source quality requires the original media")
    if quality_enabled:
        if lifecycle is None or not lifecycle_resources or not {"asr", "translation_review"} <= lifecycle_resources.keys():
            raise ValueError("Source quality requires managed ASR and NInfer review resources")
        if cfg.refinement.profile != "off":
            raise ValueError("Current source quality cannot be combined with text-only source rewriting")
    if cfg.translation_review.profile != "off" and cfg.translation.backend != "riva":
        raise ValueError("Contextual English post-edit requires dedicated Riva primary translation")
    if (cfg.condense_high_cps and cfg.translation.backend == "riva"
            and cfg.condensation is None and not cfg.re_render_only):
        raise ValueError(
            "Riva translation with subtitle condensation requires an explicit "
            "OpenAI-compatible condensation resource config."
        )
    # An explicitly supplied canonical contract is a supported downstream
    # entry point, not an implicit cache hit. Keep its accepted source rows
    # and quality evidence instead of re-auditing their order or re-running ASR.
    retained_contract = None
    retained_canonical = False
    if recording_contract_path is not None and not cfg.re_render_only:
        rc_input = Path(recording_contract_path)
        if not rc_input.is_file():
            raise FileNotFoundError(f"Recording contract not found: {rc_input}")
        retained_contract = json.loads(rc_input.read_text(encoding="utf-8"))
        if not quality_enabled:
            quality = retained_contract.get("source_quality") or {}
            if quality.get("schema") == "nemovox-current-source-quality/v1":
                if (quality.get("status") not in ("complete", "partial")
                        or not quality.get("media_fingerprint_sha256")
                        or quality["media_fingerprint_sha256"] != retained_contract["source"].get("fingerprint_sha256")
                        or quality.get("source_language") != retained_contract.get("source_language")):
                    raise ValueError("Retained canonical source quality identity mismatch")
                retained_canonical = True
    if (cfg.translation_review.profile != "off"
            and (quality_enabled or (retained_canonical and lifecycle is not None
                 and lifecycle_resources and "translation_review" in lifecycle_resources))):
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

    if cfg.re_render_only:
        _log("Stage 1-3: Reusing existing translated.contract.json (--re-render)")
        for skipped in ("source_quality", "translation", "review", "condensation"):
            _event(skipped, "skipped", "Offline re-render uses retained contract")
        translated_contract = json.loads(translated_path.read_text(encoding="utf-8"))
        ref_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        trans_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        condense_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        review_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        retained_review = translated_contract.get("translation_review") or {}
        if retained_review.get("status") == "completed_with_deferred_comparisons":
            review_summary["retained_review_status"] = retained_review["status"]
            review_summary["deferred_comparison_ids"] = retained_review["deferred_comparison_ids"]
    else:
        # Stage 1: Load existing recording contract or run chunked ASR.
        if recording_contract_path is not None:
            rc_path = Path(recording_contract_path)
            if not rc_path.is_file():
                raise FileNotFoundError(f"Recording contract not found: {rc_path}")
            _log(f"Stage 1/4: Loading existing recording contract {rc_path}")
            contract = retained_contract
            # Retained and fresh recordings enter source quality with the same
            # immutable IDs. No positional renumbering or heuristic duplicate label.
            default_stem = Path(str((contract.get("source") or {}).get("path") or rc_path)).stem
        else:
            if media_path is None:
                raise ValueError("Must supply either media_path or --recording-contract.")
            m_path = Path(media_path)
            if not m_path.is_file():
                raise FileNotFoundError(f"Media file not found: {m_path}")
            _log(f"Stage 1/4: Transcribing media via Nemotron chunked core: {m_path.name}")
            client = asr_client or NemotronVLLMClient(base_url=cfg.asr_base_url, model=cfg.asr_model,
                                                   api_key=cfg.asr_api_key, timeout_s=900.0)
            with (managed_asr_client(lifecycle, lifecycle_resources["asr"], client)
                  if lifecycle and lifecycle_resources else nullcontext(client)) as stage_client:
                asr_res = process_media(
                    m_path,
                    output_dir=out_dir / "asr",
                    client=stage_client,
                    config=RecordingPipelineConfig(
                        source_language=cfg.source_language,
                        mode=cfg.asr_mode,
                        chunk_seconds=cfg.chunk_seconds,
                        context_seconds=cfg.context_seconds,
                        chunk_policy=cfg.chunk_policy,
                        min_tail_seconds=cfg.min_tail_seconds,
                        audio_stream_index=cfg.audio_stream_index,
                        alignment_fallback=cfg.alignment_fallback,
                        qwen_policy=cfg.qwen_policy,
                        qwen_device=cfg.qwen_device,
                        qwen_model_path=cfg.qwen_model_path,
                        qwen_worker_python=cfg.qwen_worker_python,
                        qwen_gpu_mode=cfg.qwen_gpu_mode,
                        qwen_owned_asr_unit=cfg.qwen_owned_asr_unit,
                        qwen_asr_lease_path=cfg.qwen_asr_lease_path,
                        speaker_embeddings=cfg.speaker_embeddings,
                        speaker_embedding_model=cfg.speaker_embedding_model,
                        asr_resource_fingerprint=cfg.asr_resource_fingerprint,
                        model_path=cfg.asr_model_path,
                        resume=cfg.resume,
                    ),
                )
            contract = asr_res["contract"]
            rc_path = Path(asr_res["contract_path"]) if quality_enabled else None
            default_stem = m_path.stem

        if quality_enabled:
            _event("source_quality", "running", "Reviewing consequential source ambiguity")
            challenge_client = asr_client or NemotronVLLMClient(
                base_url=cfg.asr_base_url, model=cfg.asr_model,
                api_key=lifecycle_resources["asr"].credential(), timeout_s=900.0)
            def compute_quality():
                return review_current_source(
                contract, recording_path=rc_path, media_path=media_path, output_dir=out_dir,
                text_scope=lambda: managed_text_client(lifecycle,
                    lifecycle_resources["translation_review"], wire_protocol="ninfer_messages",
                    timeout_s=QUALITY_TIMEOUT_S),
                asr_scope=lambda: managed_asr_client(lifecycle,
                    lifecycle_resources["asr"], challenge_client),
                policy=cfg.source_quality_policy,
                challenger_resource=lifecycle_resources.get("source_challenger"),
                lifecycle=lifecycle)
            if asr_client is None and text_client is None:
                fingerprints={role:{"name":lifecycle_resources[role].name,
                                    "fingerprint":lifecycle_resources[role].fingerprint()}
                    for role in ("asr","translation_review","source_challenger")
                    if lifecycle_resources.get(role) is not None}
                fingerprints["owned_services"]={name:asdict(service)
                    for name,service in lifecycle.services.items()}
                key=quality_checkpoint_identity(contract,rc_path,media_path,cfg.source_quality_policy,fingerprints)
                contract,quality_report,quality_resume=run_checkpoint(
                    out_dir/"source-quality.checkpoint.json",key,contract,resume=cfg.resume,run=compute_quality)
            else:
                contract,quality_report=compute_quality()
                quality_resume="computed_injected_client"
            _atomic_write(out_dir / "source-quality.json", json.dumps(quality_report, ensure_ascii=False, indent=2))
            _atomic_write(out_dir / "canonical-source.contract.json", json.dumps(contract, ensure_ascii=False, indent=2))
            _event("source_quality", "complete", quality_report["status"])
        else:
            _event("source_quality", "skipped",
                   "Using explicitly supplied canonical source: " + contract["source_quality"]["status"]
                   if retained_canonical else "No current-media quality review requested")
        src_lang = cfg.source_language or contract.get("source_language")
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
        with _text_stage("text_translation") as stage_client:
            translated_utts, trans_summary = translate_utterances(
                refined_utts,
                source_language=src_lang,
                config=cfg.translation,
                glossary=gl,
                text_client=stage_client,
                lifecycle=lifecycle,
                riva_resource_name=(lifecycle_resources["riva_translation"].name
                                    if lifecycle_resources else "riva_local"),
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
            with _text_stage("translation_review") as stage_client:
                translated_utts, review_summary = review_translated_utterances(
                    translated_utts, source_language=src_lang, config=cfg.translation_review,
                    glossary=gl, text_client=stage_client,
                    checkpoint_path=out_dir / "translation-review.checkpoint.json",
                    resume=cfg.resume_translation_review,
                )
            if review_summary.get("status") == "completed_with_deferred_comparisons":
                deferred = set(review_summary.get("deferred_comparison_ids") or [])
                observed = {str(u.get("utterance_id") or u.get("id")) for u in translated_utts
                            if u.get("translation_review_status") ==
                            "review_deferred_comparative_preserved_machine_translation"}
                if (not deferred or deferred != observed or trans_summary.get("errors")
                        or review_summary.get("reviewed", 0) + len(deferred) != len(translated_utts)
                        or any(not str(u.get("machine_translated_text") or "").strip()
                               or (str(u.get("utterance_id") or u.get("id")) in deferred
                                   and u.get("translated_text") != u.get("machine_translated_text"))
                               for u in translated_utts)):
                    raise RuntimeError("Deferred comparative review lacks complete preserved primary MT")
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
        _atomic_write(translated_path, json.dumps(translated_contract, ensure_ascii=False, indent=2))
        if cfg.translation_review.profile != "off" and review_summary.get("status") == "completed_with_errors":
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
        "total_wall_time_s": elapsed_s,
    }
    _atomic_write(out_dir / "subtitles.metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    _event("render", "complete")
    return metadata
