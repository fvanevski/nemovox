"""Recording representation and language normalization, independent of recognition backend."""
from __future__ import annotations

from typing import Any

LANGUAGE_ALIAS_MAP: dict[str, str] = {
    "fr": "fr",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "français": "fr",
    "zh": "zh",
    "zho": "zh",
    "chi": "zh",
    "cmn": "zh",
    "cmn_hans_cn": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "en": "en",
    "eng": "en",
    "english": "en",
    "es": "es",
    "spa": "es",
    "spanish": "es",
    "de": "de",
    "deu": "de",
    "german": "de",
    "ja": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "ko": "ko",
    "kor": "ko",
    "korean": "ko",
}


def normalize_language_code(lang: str | None) -> str | None:
    """Normalize explicit or model-detected language labels to canonical ISO-639-1 codes."""
    if lang is None:
        return None
    cleaned = lang.strip().lower()
    if not cleaned or cleaned in ("auto", "none", "unknown", "und"):
        return None
    if cleaned in LANGUAGE_ALIAS_MAP:
        return LANGUAGE_ALIAS_MAP[cleaned]
    if "_" in cleaned:
        prefix = cleaned.split("_", 1)[0]
        if prefix in LANGUAGE_ALIAS_MAP:
            return LANGUAGE_ALIAS_MAP[prefix]
    if len(cleaned) == 2 and cleaned.isalpha():
        return cleaned
    return cleaned


def build_recording_contract(
    *,
    source: dict[str, Any],
    source_language: str,
    language_provenance: str,
    model_provenance: dict[str, Any],
    chunk_config: dict[str, Any],
    chunk_records: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    events: list[dict[str, Any]],
    alignment_diagnostics: list[dict[str, Any]],
    speaker_reconciliation: dict[str, Any],
    boundary_reconciliation: dict[str, Any],
    processing_warnings: list[str],
) -> dict[str, Any]:
    """Assemble the canonical recording-level JSON contract (nemovox-recording-contract/v1)."""
    # Strip internal boundary helper keys if present while retaining full unit provenance.
    clean_utts: list[dict[str, Any]] = []
    for utt in utterances:
        clean_utts.append(
            {
                "id": utt["id"],
                "source_unit_ids": utt.get("source_unit_ids", []),
                "recognized_turn_ids": utt.get("recognized_turn_ids", []),
                "occurrence_span_ids": utt.get("occurrence_span_ids", []),
                "source_text": utt["source_text"],
                "source_language": utt.get("source_language") or source_language,
                "speaker_id": utt.get("global_speaker_id") or utt.get("speaker_id"),
                "local_speaker_id": utt.get("local_speaker_id"),
                "raw_speaker_tag": utt.get("raw_speaker_tag"),
                "source_start": utt.get("source_start"),
                "source_end": utt.get("source_end"),
                "source_words": utt.get("source_words", []),
                "source_characters": utt.get("source_characters", []),
                "unit_count": utt.get("unit_count", len(utt.get("source_words", [])) + len(utt.get("source_characters", []))),
                "timed_unit_count": utt.get("timed_unit_count", utt.get("valid_timed_unit_count", 0)),
                "untimed_unit_count": utt.get("untimed_unit_count", 0),
                "timing_gaps": utt.get("timing_gaps", []),
                "alignment_status": utt.get("alignment_status"),
                "timestamp_provenance": utt.get("timestamp_provenance", {}),
                "chunk_indices": utt.get("chunk_indices", []),
                "event_refs": utt.get("event_refs", []),
            }
        )

    return {
        "schema_version": "nemovox-recording-contract/v1",
        "source": dict(source),
        "source_language": source_language,
        "language_provenance": language_provenance,
        "model_provenance": dict(model_provenance),
        "processing": {
            "chunk_config": dict(chunk_config),
            "chunk_count": len(chunk_records),
            "chunks": chunk_records,
            "boundary_reconciliation": boundary_reconciliation,
        },
        "speakers": speakers,
        "utterances": clean_utts,
        "events": events,
        "alignment_diagnostics": alignment_diagnostics,
        "speaker_reconciliation": speaker_reconciliation,
        "processing_warnings": list(processing_warnings),
    }


def format_recording_transcript(contract: dict[str, Any]) -> str:
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


def to_pyannote_compatible_segments(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt a normalized contract into WhisperX/Pyannote-style ``segments`` for read-only subtitle testing."""
    segments: list[dict[str, Any]] = []
    for utt in contract.get("utterances", []):
        all_units = list(utt.get("source_words") or []) + list(utt.get("source_characters") or [])
        all_units.sort(key=lambda x: (x["start"] if x.get("start") is not None else float("inf")))
        words_out = []
        for u in all_units:
            if not u.get("valid_timing"):
                continue
            words_out.append(
                {
                    "word": u["text"],
                    "start": u["start"],
                    "end": u["end"],
                    "speaker": u.get("global_speaker_id") or u.get("speaker") or utt.get("speaker_id") or "unknown",
                }
            )
        if not words_out and (utt.get("source_start") is None or utt.get("source_end") is None):
            continue
        segments.append(
            {
                "start": utt.get("source_start"),
                "end": utt.get("source_end"),
                "text": utt.get("source_text") or "",
                "speaker": utt.get("speaker_id") or "unknown",
                "words": words_out,
            }
        )
    return segments
