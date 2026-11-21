"""Recording-level chunked Nemotron transcription orchestrator with resume & reconciliation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nvidia_transcribe.alignment import ExternalForcedAligner, recover_chunk_alignment
from nvidia_transcribe.checkpoint import DEFAULT_MODEL_PATH
from nvidia_transcribe.chunking import ChunkConfig, ChunkSpec, plan_chunks, reconcile_chunk_boundaries
from nvidia_transcribe.client import InferenceRequestConfig, NemotronVLLMClient
from nvidia_transcribe.contract import (
    build_recording_contract,
    detect_script_language_hint,
    extract_events_from_raw_text,
    normalize_language_code,
)
from nvidia_transcribe.media import MediaInfo, compute_file_sha256, extract_wav_segment, probe_media, validate_wav_file
from nvidia_transcribe.prompts import cleanup_generation_loops
from nvidia_transcribe.qwen_aligner import (QwenAlignerConfig, QwenForcedAlignerAdapter, map_language_for_qwen_aligner)
from nvidia_transcribe.selective_alignment import AdmissionPolicy, apply_selective_recovery, plan_selective_recovery
from nvidia_transcribe.gpu_handoff import (ManagedASRBlocked, ManagedASRLifecycleError, ManagedASRConfig, managed_asr_handoff)
from nvidia_transcribe.speakers import (
    SpeakerEmbedder,
    TitaNetSpeakerEmbedder,
    normalize_chunk_speakers,
    reconcile_speakers,
)
from nvidia_transcribe.utterances import reconstruct_speaker_utterances


@dataclass(frozen=True)
class RecordingPipelineConfig:
    source_language: str | None = None
    mode: str = "speaker"
    chunk_seconds: float = 90.0
    context_seconds: float = 10.0
    min_tail_seconds: float = 15.0
    audio_stream_index: int = 0
    speaker_embeddings: str = "titanet"  # "titanet" | "none"
    embedding_threshold: float = 0.65
    enable_subinterval_recovery: bool = True
    alignment_fallback: str = "off"  # "off" | "qwen"
    qwen_policy: str = "subtitle"     # "subtitle" | "fine"
    qwen_device: str = "cuda:0"      # "cuda:0" | "cpu" (explicit only)
    qwen_model_path: str | None = None
    qwen_worker_python: str | None = None
    qwen_gpu_mode: str = "available"  # "available" | "managed"
    qwen_owned_asr_unit: str | None = None
    qwen_asr_lease_path: str | None = None
    resume: bool = True
    model_path: str = str(DEFAULT_MODEL_PATH)
    asr_resource_fingerprint: str = ""
    speaker_embedding_model: str = "titanet_small"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _extraction_signature(*, media: MediaInfo, spec: ChunkSpec, audio_stream_index: int, sample_rate_hz: int = 16000) -> str:
    payload = {
        "media_path": media.path,
        "media_fingerprint": media.fingerprint_sha256,
        "media_size_bytes": media.size_bytes,
        "audio_stream": audio_stream_index,
        "infer_start_s": spec.infer_start_s,
        "infer_duration_s": spec.infer_duration_s,
        "sample_rate_hz": sample_rate_hz,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True))


def _config_signature(
    *,
    media: MediaInfo,
    model_name: str,
    cfg: RecordingPipelineConfig,
) -> str:
    payload = {
        "media_path": media.path,
        "media_fingerprint": media.fingerprint_sha256,
        "media_size_bytes": media.size_bytes,
        "media_duration_s": media.duration_s,
        "audio_stream": media.selected_audio_stream,
        "model_name": model_name,
        "asr_resource_fingerprint": cfg.asr_resource_fingerprint,
        "mode": cfg.mode,
        "source_language": normalize_language_code(cfg.source_language) or "auto",
        "chunk_seconds": cfg.chunk_seconds,
        "context_seconds": cfg.context_seconds,
        "min_tail_seconds": cfg.min_tail_seconds,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True))


def _resolve_recording_language(
    *,
    explicit_language: str | None,
    client: NemotronVLLMClient,
    chunk_wavs: list[Path],
    raw_texts: list[str],
) -> tuple[str, str, dict[str, Any]]:
    """Resolve recording source language without leaving resolved outputs labeled 'auto'."""
    norm_explicit = normalize_language_code(explicit_language)
    if norm_explicit is not None:
        return norm_explicit, "explicit", {"explicit_input": explicit_language}

    # Sample up to 3 evenly spaced chunks for model LID + inspect Unicode script hints.
    votes: list[str] = []
    if chunk_wavs:
        indices = sorted({0, len(chunk_wavs) // 2, len(chunk_wavs) - 1})
        for idx in indices:
            wav = chunk_wavs[idx]
            if not wav.is_file():
                continue
            try:
                res = client.transcribe(
                    wav,
                    InferenceRequestConfig(mode="lid", capture_ctc_timestamps=False),
                )
                raw_lid = normalize_language_code((res["contract"].get("raw_text") or "").strip())
                if raw_lid:
                    votes.append(raw_lid)
            except Exception:
                continue

    script_hints = [h for txt in raw_texts if (h := detect_script_language_hint(txt)) is not None]
    vote_counts = Counter(votes)
    details = {"lid_votes": votes, "script_hints": script_hints}

    if vote_counts:
        top_lang, top_count = vote_counts.most_common(1)[0]
        if len(vote_counts) == 1 or top_count > len(votes) / 2.0:
            # Cross-check CJK script contradiction (e.g. Han script with non-zh LID vote)
            if "zh" in script_hints and top_lang not in ("zh", "ja"):
                return "ambiguous", "ambiguous", details
            return top_lang, "model_detected", details
        return "ambiguous", "ambiguous", details

    # Only fall back to unambiguous non-Latin script if LID produced zero votes.
    cjk_votes = [s for s in script_hints if s in ("zh", "ja", "ko")]
    if cjk_votes and len(set(cjk_votes)) == 1:
        return cjk_votes[0], "script_detected", details
    return "und", "ambiguous", details


def _format_transcript_txt(contract: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        f"# Source: {contract['source']['path']} | Language: {contract['source_language']} "
        f"({contract['language_provenance']}) | Utterances: {len(contract['utterances'])}"
    )
    lines.append("")
    for utt in contract["utterances"]:
        s = utt.get("source_start")
        e = utt.get("source_end")
        ts = f"[{s:07.2f} -> {e:07.2f}]" if s is not None and e is not None else "[untimed -> untimed]"
        spk = utt.get("speaker_id") or "unknown"
        status = utt.get("alignment_status") or "unknown"
        lines.append(f"{ts} {spk} ({status}): {utt['source_text']}")
    return "\n".join(lines) + "\n"


def process_media(
    media_path: str | Path,
    *,
    output_dir: str | Path,
    client: NemotronVLLMClient,
    config: RecordingPipelineConfig | None = None,
    embedder: SpeakerEmbedder | None = None,
    external_aligner: ExternalForcedAligner | None = None,
) -> dict[str, Any]:
    """Process a complete audio or video recording through bounded Nemotron chunks."""
    t_start = time.perf_counter()
    cfg = config or RecordingPipelineConfig()
    if cfg.alignment_fallback not in ("off", "qwen"):
        raise ValueError(f"Unsupported alignment_fallback {cfg.alignment_fallback!r}; expected 'off' or 'qwen'.")
    if cfg.qwen_policy not in ("subtitle", "fine"):
        raise ValueError("qwen_policy must be subtitle or fine")
    if cfg.qwen_device not in ("cuda:0", "cpu") or cfg.qwen_gpu_mode not in ("available", "managed"):
        raise ValueError("Unsupported Qwen device/GPU resource mode")
    if cfg.qwen_gpu_mode == "managed" and cfg.qwen_device != "cuda:0":
        raise ValueError("Managed GPU mode requires CUDA and is not a CPU selection")
    out_dir = Path(output_dir)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    media = probe_media(media_path, audio_stream_index=cfg.audio_stream_index)
    chunk_cfg = ChunkConfig(
        chunk_seconds=cfg.chunk_seconds,
        context_seconds=cfg.context_seconds,
        min_tail_seconds=cfg.min_tail_seconds,
    )
    chunks: list[ChunkSpec] = plan_chunks(media.duration_s, chunk_cfg)
    sig = _config_signature(media=media, model_name=client.model, cfg=cfg)

    existing_manifest: dict[str, Any] = {}
    if cfg.resume and manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if loaded.get("config_signature") == sig:
                existing_manifest = loaded
        except Exception:
            existing_manifest = {}

    if not cfg.resume or not existing_manifest:
        # Invalidate any stale chunk WAVs / raw JSONs from an earlier or changed run
        for stale in chunks_dir.glob("chunk_*"):
            if stale.is_file():
                stale.unlink(missing_ok=True)

    completed_by_idx: dict[int, dict[str, Any]] = {}
    for entry in existing_manifest.get("chunks", []):
        if entry.get("status") == "completed":
            raw_p = Path(entry.get("raw_response_path", ""))
            if raw_p.is_file() and _sha256_text(raw_p.read_text(encoding="utf-8")) == entry.get("raw_response_sha256"):
                completed_by_idx[int(entry["chunk_index"])] = entry

    chunk_wavs: list[Path] = []
    chunk_raws: list[dict[str, Any]] = []
    chunk_manifest_entries: list[dict[str, Any]] = []
    chunk_latencies: list[float] = []
    resumed_count = 0

    norm_explicit_lang = normalize_language_code(cfg.source_language)
    lang_hint_for_prompt = cfg.source_language if norm_explicit_lang else None

    for spec in chunks:
        wav_path = chunks_dir / f"chunk_{spec.chunk_index:03d}.wav"
        raw_path = chunks_dir / f"chunk_{spec.chunk_index:03d}.raw.json"
        cur_ext_sig = _extraction_signature(media=media, spec=spec, audio_stream_index=cfg.audio_stream_index)

        prev_entry = completed_by_idx.get(spec.chunk_index)
        wav_valid_for_reuse = False
        actual_dur = 0.0
        wav_sha256 = ""
        if (
            cfg.resume
            and prev_entry is not None
            and prev_entry.get("extraction_signature") == cur_ext_sig
            and prev_entry.get("wav_sha256")
            and wav_path.is_file()
        ):
            try:
                actual_dur, wav_sha256 = validate_wav_file(wav_path, expected_sample_rate_hz=16000)
                if wav_sha256 == prev_entry.get("wav_sha256"):
                    wav_valid_for_reuse = True
            except Exception:
                wav_valid_for_reuse = False

        if not wav_valid_for_reuse:
            # Stale or missing WAV: invalidate dependent raw response and sub-intervals, then re-extract atomically.
            if raw_path.is_file():
                raw_path.unlink(missing_ok=True)
            for sub_stale in chunks_dir.glob(f"chunk_{spec.chunk_index:03d}_sub_*.wav"):
                sub_stale.unlink(missing_ok=True)
            actual_dur = extract_wav_segment(
                media.path,
                start_s=spec.infer_start_s,
                duration_s=spec.infer_duration_s,
                out_wav=wav_path,
                audio_stream_index=cfg.audio_stream_index,
            )
            wav_sha256 = compute_file_sha256(wav_path)

        chunk_wavs.append(wav_path)

        if wav_valid_for_reuse and prev_entry is not None and raw_path.is_file():
            raw_resp = json.loads(raw_path.read_text(encoding="utf-8"))
            entry = prev_entry
            resumed_count += 1
        else:
            req_cfg = InferenceRequestConfig(
                mode=cfg.mode,
                source_language=lang_hint_for_prompt,
                capture_ctc_timestamps=True,
                require_ctc_timestamps=False,
                source_offset_s=spec.infer_start_s,
            )
            res = client.transcribe(wav_path, req_cfg)
            raw_resp = res["raw_response"]
            raw_text_dump = json.dumps(raw_resp, ensure_ascii=False, indent=2)
            _atomic_write_text(raw_path, raw_text_dump)
            chunk_latencies.append(float(res["latency_s"]))
            entry = {
                **spec.to_dict(),
                "status": "completed",
                "extraction_signature": cur_ext_sig,
                "wav_path": str(wav_path),
                "wav_sha256": wav_sha256,
                "actual_wav_duration_s": actual_dur,
                "raw_response_path": str(raw_path),
                "raw_response_sha256": _sha256_text(raw_text_dump),
                "latency_s": res["latency_s"],
                "finish_reason": ((raw_resp.get("choices") or [{}])[0]).get("finish_reason"),
                "usage": raw_resp.get("usage"),
            }

        chunk_raws.append(raw_resp)
        chunk_manifest_entries.append(entry)

        # Persist incremental manifest atomically after every chunk so interruption can safely resume.
        manifest_state = {
            "config_signature": sig,
            "media": media.to_dict(),
            "model": client.model,
            "chunk_config": chunk_cfg.to_dict(),
            "chunks": chunk_manifest_entries,
        }
        _atomic_write_text(manifest_path, json.dumps(manifest_state, ensure_ascii=False, indent=2))

    raw_texts = [
        cleanup_generation_loops((((r.get("choices") or [{}])[0]).get("message") or {}).get("content") or "")
        for r in chunk_raws
    ]

    resolved_lang, lang_prov, lang_details = _resolve_recording_language(
        explicit_language=cfg.source_language,
        client=client,
        chunk_wavs=chunk_wavs,
        raw_texts=raw_texts,
    )

    # Alignment validation & bounded recovery per chunk.
    alignment_diagnostics: list[dict[str, Any]] = []
    chunk_units_norm: list[list[dict[str, Any]]] = []
    chunk_profiles: list[dict[str, dict[str, Any]]] = []
    raw_events: list[dict[str, Any]] = []
    warnings: list[str] = []

    if lang_prov == "ambiguous":
        warnings.append(f"Source language detection was ambiguous: {lang_details}")

    # Phase 1: finish *every* ASR/LID and native CTC/subinterval request
    # before considering CUDA Qwen or a managed ASR vLLM release.
    native_records: list[dict[str, Any]] = []

    def _extract_sub_wav(index: int, sub_start: float, sub_end: float) -> Path:
        duration = max(0.2, sub_end - sub_start)
        path = chunks_dir / (
            f"chunk_{index:03d}_sub_{int(round(sub_start * 100))}_{int(round(sub_end * 100))}.wav"
        )
        extract_wav_segment(
            media.path, start_s=sub_start, duration_s=duration, out_wav=path,
            audio_stream_index=cfg.audio_stream_index,
        )
        return path

    for idx, (spec, raw_resp, raw_text, wav_path) in enumerate(zip(chunks, chunk_raws, raw_texts, chunk_wavs)):
        def _sub_wav_extract(sub_start: float, sub_end: float, _idx: int = idx) -> Path:
            return _extract_sub_wav(_idx, sub_start, sub_end)

        def _subinterval_runner(sub_start: float, sub_end: float, _idx: int = idx) -> tuple[str, Any]:
            sub_wav = _extract_sub_wav(_idx, sub_start, sub_end)
            sub_res = client.transcribe(
                sub_wav,
                InferenceRequestConfig(
                    mode=cfg.mode, source_language=lang_hint_for_prompt,
                    capture_ctc_timestamps=True, require_ctc_timestamps=False,
                    source_offset_s=sub_start,
                ),
            )
            return sub_res["contract"]["raw_text"], sub_res["raw_response"].get("ctc_timestamps")

        rec = recover_chunk_alignment(
            raw_text=raw_text,
            ctc_payload=raw_resp.get("ctc_timestamps"),
            source_offset_s=spec.infer_start_s,
            chunk_index=idx,
            model_path=cfg.model_path,
            subinterval_runner=_subinterval_runner if cfg.enable_subinterval_recovery else None,
            # Caller-injected test aligners retain their existing API. Production
            # Qwen no longer runs in this native-ASR phase.
            external_aligner=external_aligner,
            chunk_wav_path=wav_path,
            source_language=resolved_lang,
            subinterval_wav_extractor=_sub_wav_extract,
        )
        native_records.append(rec)
        raw_events.extend(
            extract_events_from_raw_text(
                raw_text, chunk_index=idx, chunk_start_s=spec.infer_start_s,
                chunk_end_s=spec.infer_end_s,
            )
        )

    # Keep an inspectable native timing checkpoint before any managed GPU
    # transition. A failed/blocked optional aligner never discards native work.
    qwen_metrics: dict[str, Any] = {"status": "disabled"}
    if cfg.alignment_fallback == "qwen" and external_aligner is None:
        _atomic_write_text(
            out_dir / "native-alignment.checkpoint.json",
            json.dumps({
                "config_signature": sig,
                "chunk_raw_sha256": [e.get("raw_response_sha256") for e in chunk_manifest_entries],
                "native_records": native_records,
            }, ensure_ascii=False, indent=2),
        )
        policy = AdmissionPolicy(mode=cfg.qwen_policy)
        plans = [
            plan_selective_recovery(
                native=rec, raw_text=raw_texts[idx],
                chunk_start_s=spec.infer_start_s,
                chunk_end_s=spec.infer_end_s, policy=policy,
            )
            for idx, (rec, spec) in enumerate(zip(native_records, chunks))
        ]
        admitted = sum(len(cands) for cands, _ in plans)
        qwen_metrics = {
            "status": "no_eligible_spans", "policy": cfg.qwen_policy,
            "eligible_candidates": admitted,
            "skipped_candidates": sum(
                sum(str(d.get("status") or "").startswith("skipped") for d in decisions)
                for _, decisions in plans
            ),
            "mode": cfg.qwen_gpu_mode,
        }
        if not map_language_for_qwen_aligner(resolved_lang):
            qwen_metrics["status"] = "unsupported_language"
        elif admitted:
            from contextlib import nullcontext
            managed_scope: Any = nullcontext({"mode": "available", "released": False})
            if cfg.qwen_gpu_mode == "managed":
                if not cfg.qwen_owned_asr_unit or not cfg.qwen_asr_lease_path:
                    qwen_metrics["status"] = "blocked_missing_owned_asr_configuration"
                    managed_scope = None
                else:
                    managed_scope = managed_asr_handoff(ManagedASRConfig(
                        unit=cfg.qwen_owned_asr_unit,
                        lease_path=Path(cfg.qwen_asr_lease_path),
                        base_url=getattr(client, "base_url", "http://127.0.0.1:8190/v1"),
                        served_model=client.model,
                        journal_path=out_dir / "qwen-asr-handoff.journal.json",
                    ))
            if managed_scope is not None:
                try:
                    with managed_scope as lease_report:
                        with QwenForcedAlignerAdapter(
                            QwenAlignerConfig(**{
                                "device": cfg.qwen_device,
                                **({"model_path": cfg.qwen_model_path} if cfg.qwen_model_path else {}),
                                **({"worker_python": cfg.qwen_worker_python} if cfg.qwen_worker_python else {}),
                            })
                        ) as adapter:
                            for idx, (spec, rec) in enumerate(zip(chunks, native_records)):
                                if not plans[idx][0]:
                                    continue
                                apply_selective_recovery(
                                    native=rec, raw_text=raw_texts[idx],
                                    chunk_start_s=spec.infer_start_s,
                                    chunk_end_s=spec.infer_end_s,
                                    chunk_index=idx, source_language=resolved_lang,
                                    aligner=adapter, policy=policy,
                                    extract_wav=lambda start, end, _idx=idx: _extract_sub_wav(_idx, start, end),
                                )
                            qwen_metrics.update(adapter.metrics)
                            attempted_statuses = [
                                d.get("aligner_status") for rec in native_records
                                for d in (rec.get("diagnostic", {}).get("external_aligner") or {}).get("localized_spans", [])
                                if d.get("aligner_status")
                            ]
                            blocked = next(
                                (v for v in attempted_statuses
                                 if v.startswith("blocked_") or v in {
                                     "worker_timeout", "worker_exited", "worker_failed",
                                     "unavailable", "device_mismatch", "worker_error",
                                 }), None,
                            )
                            qwen_metrics["status"] = (
                                blocked or ("executed" if adapter.metrics["align_count"]
                                else "deferred_no_successful_worker_requests")
                            )
                    # Snapshot the handoff report *after* context exit so
                    # restored=True reflects verified ASR health, not a
                    # premature pre-restoration observation.
                    qwen_metrics["resource_handoff"] = dict(lease_report)
                except ManagedASRLifecycleError as exc:
                    # A stop was attempted or restoration could not be proved:
                    # never convert this into a successful optional deferral.
                    raise RuntimeError(
                        f"Managed ASR lifecycle could not be verified: {exc.status}"
                    ) from exc
                except ManagedASRBlocked as exc:
                    # Only failures before a stop attempt may defer alignment.
                    qwen_metrics["status"] = exc.status
                    warnings.append(f"Optional managed Qwen deferred: {exc.status}")
                except Exception:
                    # Do not mask an unexpected lifecycle or processing error.
                    raise
        else:
            # Explicitly prove that zero eligible spans does not allocate a Qwen
            # worker or touch a GPU service.
            qwen_metrics["model_load_count"] = 0
        for idx, (rec, plan) in enumerate(zip(native_records, plans)):
            if "external_aligner" not in rec["diagnostic"]:
                rec["diagnostic"]["external_aligner"] = {
                    "backend": "qwen3_forced_aligner",
                    "status": qwen_metrics["status"],
                    "policy": cfg.qwen_policy,
                    "candidate_count": len(plan[0]),
                    "localized_spans": plan[1],
                }

    # Phase 2: normalize speakers and build final utterances only after the
    # Qwen session has been torn down and ASR restoration verified.
    for idx, rec in enumerate(native_records):
        alignment_diagnostics.append(rec["diagnostic"])
        if rec["alignment_status"] not in (
            "ctc_aligned",
            "subinterval_ctc_recovered",
            "external_recovered",
            "hybrid_ctc_and_external_recovered",
        ):
            warnings.append(
                f"Chunk {idx} alignment degraded ({rec['alignment_status']}): "
                f"cause={rec['diagnostic'].get('failure_cause')}, "
                f"oov={rec['diagnostic'].get('oov_characters')}"
            )
        norm_units, profs = normalize_chunk_speakers(
            chunk_index=idx, units=rec["units"],
            diarization=rec["diarization"],
            speaker_mapping=rec["speaker_mapping"],
        )
        chunk_units_norm.append(norm_units)
        chunk_profiles.append(profs)

    # Cross-chunk speaker reconciliation.
    chosen_embedder = embedder
    if chosen_embedder is None and cfg.speaker_embeddings == "titanet":
        chosen_embedder = TitaNetSpeakerEmbedder(model_name=cfg.speaker_embedding_model)

    spk_res = reconcile_speakers(
        chunks=chunks,
        chunk_profiles=chunk_profiles,
        media_path=Path(media.path),
        audio_stream_index=cfg.audio_stream_index,
        embedder=chosen_embedder,
        embedding_threshold=cfg.embedding_threshold,
    )
    local_to_global = spk_res["local_to_global"]

    # Propagate global speaker IDs onto units & events before boundary reconciliation.
    for units in chunk_units_norm:
        for u in units:
            u["global_speaker_id"] = local_to_global.get(u.get("local_speaker_id", ""), "SPK_UNK")
    for ev in raw_events:
        if ev.get("local_speaker_id") in local_to_global:
            ev["global_speaker_id"] = local_to_global[ev["local_speaker_id"]]

    # Boundary deduplication & recovery.
    merged_units, boundary_diag = reconcile_chunk_boundaries(chunk_units_norm, chunks)

    # Independent per-speaker utterance reconstruction anchored to true chunk bounds.
    chunk_windows = {c.chunk_index: (c.infer_start_s, c.infer_end_s) for c in chunks}
    utterances = reconstruct_speaker_utterances(
        merged_units,
        source_language=resolved_lang,
        chunk_windows=chunk_windows,
    )

    # Summarize chunk records cleanly.
    chunk_records = [
        {
            **entry,
            "alignment_status": alignment_diagnostics[i]["status"],
            "raw_text_preview": raw_texts[i][:240],
        }
        for i, entry in enumerate(chunk_manifest_entries)
    ]

    total_elapsed = round(time.perf_counter() - t_start, 4)
    contract = build_recording_contract(
        source=media.to_dict(),
        source_language=resolved_lang,
        language_provenance=lang_prov,
        model_provenance={
            "served_model": client.model,
            "model_path": cfg.model_path,
            "request_mode": cfg.mode,
            "alignment_fallback": cfg.alignment_fallback,
            "qwen_recovery": qwen_metrics,
            "language_detection": lang_details,
            "total_wall_time_s": total_elapsed,
            "resumed_chunks": resumed_count,
            "chunk_request_latencies_s": chunk_latencies,
        },
        chunk_config=chunk_cfg.to_dict(),
        chunk_records=chunk_records,
        speakers=spk_res["speakers"],
        utterances=utterances,
        events=raw_events,
        alignment_diagnostics=alignment_diagnostics,
        speaker_reconciliation=spk_res["reconciliation"],
        boundary_reconciliation=boundary_diag,
        processing_warnings=warnings,
    )

    contract_path = out_dir / "recording.contract.json"
    transcript_path = out_dir / "transcript.txt"
    _atomic_write_text(contract_path, json.dumps(contract, ensure_ascii=False, indent=2))
    _atomic_write_text(transcript_path, _format_transcript_txt(contract))

    return {
        "contract": contract,
        "contract_path": str(contract_path),
        "transcript_path": str(transcript_path),
        "manifest_path": str(manifest_path),
        "total_wall_time_s": total_elapsed,
    }
