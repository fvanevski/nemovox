"""End-to-end multilingual subtitle generation pipeline and re-render orchestrator."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media
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
    translate_utterances,
)
from nvidia_transcribe.translation_review import TranslationReviewConfig, review_translated_utterances
from nvidia_transcribe.utterances import apply_validation_overlay, audit_and_reconstruct_contract


@dataclass(frozen=True)
class SubtitlePipelineConfig:
    source_language: str | None = None
    target_language: str = "en"
    asr_mode: str = "speaker"
    chunk_seconds: float = 90.0
    context_seconds: float = 10.0
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
    speaker_embeddings: str = "titanet"
    speaker_embedding_model: str = "titanet_small"
    asr_base_url: str = "http://127.0.0.1:8190/v1"
    asr_model: str = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"
    asr_api_key: str | None = None
    asr_resource_fingerprint: str = ""
    asr_model_path: str | None = None
    refinement: RefinementConfig = RefinementConfig()
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
    reconstruct_untimed_order: bool = True
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
) -> dict[str, Any]:
    """Run staged ASR -> text refinement -> translation -> subtitle rendering.

    Can start from raw ``media_path`` OR resume directly from ``recording_contract_path``
    without loading the 30B ASR model. Supports fast CPU-only re-rendering when
    ``re_render_only=True`` or cached ``translated.contract.json`` exists.
    """
    t0 = time.perf_counter()
    cfg = config or SubtitlePipelineConfig()
    if cfg.translation_review.profile != "off" and cfg.translation.backend != "riva":
        raise ValueError("Contextual English post-edit requires dedicated Riva primary translation")
    if (cfg.condense_high_cps and cfg.translation.backend == "riva"
            and cfg.condensation is None and not cfg.re_render_only):
        raise ValueError(
            "Riva translation with subtitle condensation requires an explicit "
            "OpenAI-compatible condensation resource config."
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gl = glossary if isinstance(glossary, GlossaryConfig) else load_glossary(glossary)

    def _log(msg: str) -> None:
        if progress_callback is not None:
            progress_callback(msg)

    translated_path = out_dir / "translated.contract.json"
    refined_path = out_dir / "refined.contract.json"

    if cfg.re_render_only and translated_path.is_file():
        _log("Stage 1-3: Reusing existing translated.contract.json (--re-render)")
        translated_contract = json.loads(translated_path.read_text(encoding="utf-8"))
        ref_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        trans_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        condense_summary: dict[str, Any] = {"status": "reused_translated_contract"}
        review_summary: dict[str, Any] = {"status": "reused_translated_contract"}
    else:
        # Stage 1: Load existing recording contract or run chunked ASR.
        if recording_contract_path is not None:
            rc_path = Path(recording_contract_path)
            if not rc_path.is_file():
                raise FileNotFoundError(f"Recording contract not found: {rc_path}")
            _log(f"Stage 1/4: Loading existing recording contract {rc_path}")
            contract = json.loads(rc_path.read_text(encoding="utf-8"))
            if cfg.reconstruct_untimed_order and any(u.get("source_start") is None for u in (contract.get("utterances") or [])):
                m_path = rc_path.parent / "manifest.json"
                manifest_chunks = None
                if m_path.is_file():
                    try:
                        manifest_chunks = json.loads(m_path.read_text(encoding="utf-8")).get("chunks")
                    except Exception:
                        manifest_chunks = None
                chunks_dir = rc_path.parent / "chunks"
                contract, _audit = audit_and_reconstruct_contract(
                    contract,
                    chunk_raw_dir=chunks_dir if chunks_dir.is_dir() else None,
                    manifest_chunks=manifest_chunks,
                )
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
            asr_res = process_media(
                m_path,
                output_dir=out_dir / "asr",
                client=client,
                config=RecordingPipelineConfig(
                    source_language=cfg.source_language,
                    mode=cfg.asr_mode,
                    chunk_seconds=cfg.chunk_seconds,
                    context_seconds=cfg.context_seconds,
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
            default_stem = m_path.stem

        src_lang = cfg.source_language or contract.get("source_language")
        raw_utterances = list(contract.get("utterances") or [])

        # Stage 2: Contextual source refinement (without mandatory audio reverification)
        _log(f"Stage 2/4: Source refinement (backend={cfg.refinement.backend}, profile={cfg.refinement.profile})")
        refined_utts, ref_summary = refine_utterances_with_text_model(
            raw_utterances,
            source_language=src_lang,
            config=cfg.refinement,
            glossary=gl,
            text_client=text_client,
            checkpoint_path=out_dir / "refinement.checkpoint.json",
            resume=cfg.resume_refinement,
        )
        refined_contract = {
            **contract,
            "utterances": refined_utts,
            "refinement": ref_summary,
        }
        _atomic_write(refined_path, json.dumps(refined_contract, ensure_ascii=False, indent=2))

        # Stage 3: English translation/localization
        _log(f"Stage 3/4: Translation ({src_lang} -> {cfg.target_language}, backend={cfg.translation.backend}, style={cfg.translation.style})")
        translated_utts, trans_summary = translate_utterances(
            refined_utts,
            source_language=src_lang,
            config=cfg.translation,
            glossary=gl,
            text_client=text_client,
            checkpoint_path=out_dir / "translation.checkpoint.json",
            resume=cfg.resume_translation,
        )
        # Retain the first-stage MT independently of all text-model edits.
        if cfg.translation.backend == "riva":
            _log(f"Stage 3b/4: Contextual English post-edit (profile={cfg.translation_review.profile})")
            translated_utts, review_summary = review_translated_utterances(
                translated_utts, source_language=src_lang, config=cfg.translation_review,
                glossary=gl, text_client=text_client,
                checkpoint_path=out_dir / "translation-review.checkpoint.json",
                resume=cfg.resume_translation_review,
            )
        else:
            review_summary = {"status": "not_applicable_legacy_translation"}
        if cfg.condense_high_cps and cfg.translation.backend != "none":
            _log(f"Stage 3c/4: Selective condensation (> {cfg.condense_cps_threshold} CPS)")
            translated_utts, condense_summary = condense_high_cps_cues_with_text_model(
                translated_utts,
                source_language=src_lang,
                config=cfg.condensation or cfg.translation,
                cps_threshold=cfg.condense_cps_threshold,
                target_cps=cfg.subtitles.target_cps,
                text_client=text_client,
                checkpoint_path=out_dir / "condensation.checkpoint.json",
                resume=cfg.resume_condensation,
            )
        else:
            condense_summary = {"status": "disabled", "candidate_count": 0, "condensed_count": 0}
        translated_contract = {
            **refined_contract,
            "target_language": cfg.target_language,
            "utterances": translated_utts,
            "translation": trans_summary,
            "translation_review": review_summary,
            "condensation": condense_summary,
        }
        _atomic_write(translated_path, json.dumps(translated_contract, ensure_ascii=False, indent=2))
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
    return metadata
