"""Canonical transcript & recording contracts and pyannote compatibility adapter."""

from __future__ import annotations

import unicodedata
from typing import Any

from nvidia_transcribe.parser import (
    EVENT_TAG_RE,
    STYLE_TAG_RE,
    ParsedCTCAlignment,
    SpeakerTurn,
    TimedUnit,
    parse_ctc_timestamps,
    parse_speaker_turns,
    split_lexical_units,
)
from nvidia_transcribe.utterances import join_multilingual_units, reconstruct_speaker_utterances

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


def detect_script_language_hint(text: str) -> str | None:
    """Detect unambiguous script hints from recognized Unicode text (e.g. Han -> zh)."""
    cjk_count = 0
    kana_count = 0
    hangul_count = 0
    latin_count = 0
    for ch in text or "":
        if ch.isspace():
            continue
        name = unicodedata.name(ch, "")
        if name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
            cjk_count += 1
        elif name.startswith(("HIRAGANA", "KATAKANA")):
            kana_count += 1
        elif name.startswith("HANGUL"):
            hangul_count += 1
        elif name.startswith("LATIN"):
            latin_count += 1
    if cjk_count >= 4 and kana_count == 0 and hangul_count == 0:
        return "zh"
    if kana_count >= 2:
        return "ja"
    if hangul_count >= 2:
        return "ko"
    if latin_count >= 4 and cjk_count == 0:
        return "latin"
    return None


def extract_events_from_raw_text(
    raw_text: str,
    *,
    chunk_index: int = 0,
    speaker_lookup: dict[str, str] | None = None,
    chunk_start_s: float | None = None,
    chunk_end_s: float | None = None,
) -> list[dict[str, Any]]:
    """Extract nonlexical <ev:*> and <sty:*> tags separately from spoken lexical text."""
    events: list[dict[str, Any]] = []
    for turn in parse_speaker_turns(raw_text):
        raw_tag = turn.speaker_tag if turn.speaker_tag.startswith("<spk:") else None
        local_spk = (
            f"chunk_{chunk_index:03d}:spk_{raw_tag.removeprefix('<spk:').removesuffix('>')}"
            if raw_tag
            else f"chunk_{chunk_index:03d}:unknown"
        )
        global_spk = (speaker_lookup or {}).get(local_spk)
        lexical_only = turn.text.strip()
        for ev in turn.events:
            events.append(
                {
                    "kind": "event",
                    "tag": f"<ev:{ev}>",
                    "name": ev,
                    "chunk_index": chunk_index,
                    "local_speaker_id": local_spk,
                    "global_speaker_id": global_spk,
                    "has_inline_speech": bool(lexical_only),
                    "context_window_start": chunk_start_s,
                    "context_window_end": chunk_end_s,
                    "timing_status": "chunk_window_only",
                }
            )
        for sty in turn.styles:
            events.append(
                {
                    "kind": "style",
                    "tag": f"<sty:{sty}>",
                    "name": sty,
                    "chunk_index": chunk_index,
                    "local_speaker_id": local_spk,
                    "global_speaker_id": global_spk,
                    "has_inline_speech": bool(lexical_only),
                    "context_window_start": chunk_start_s,
                    "context_window_end": chunk_end_s,
                    "timing_status": "chunk_window_only",
                }
            )
    return events


def _join_units_text(units: list[TimedUnit]) -> str:
    """Reconstruct readable text from word or CJK character units."""
    return join_multilingual_units([{"text": u.text, "unit_type": u.unit_type} for u in units])


def _group_timed_units_by_speaker_runs(units: list[TimedUnit]) -> list[tuple[str, list[TimedUnit]]]:
    """Legacy contiguous run grouping helper retained for backward API compatibility."""
    runs: list[tuple[str, list[TimedUnit]]] = []
    cur_spk: str | None = None
    cur_items: list[TimedUnit] = []
    for u in units:
        spk = u.speaker or "unknown"
        if cur_spk is None:
            cur_spk = spk
            cur_items = [u]
        elif spk == cur_spk:
            cur_items.append(u)
        else:
            runs.append((cur_spk, cur_items))
            cur_spk = spk
            cur_items = [u]
    if cur_items and cur_spk is not None:
        runs.append((cur_spk, cur_items))
    return runs


def build_transcript_contract(
    *,
    raw_text: str,
    source_language: str | None,
    ctc_payload: Any = None,
    source_offset_s: float = 0.0,
    model_provenance: dict[str, Any] | None = None,
    english_translation: str | None = None,
    uncertainty: dict[str, Any] | None = None,
    language_provenance: str | None = None,
) -> dict[str, Any]:
    """Build the normalized multilingual transcript contract without inventing missing fields."""
    turns: list[SpeakerTurn] = parse_speaker_turns(raw_text)
    parsed_ctc: ParsedCTCAlignment = parse_ctc_timestamps(
        ctc_payload, source_offset_s=source_offset_s, strict=False
    )
    events = extract_events_from_raw_text(raw_text, chunk_index=0)

    resolved_lang = normalize_language_code(source_language)
    lang_prov = language_provenance or ("explicit" if resolved_lang else "unspecified")

    utterances: list[dict[str, Any]] = []
    units = list(parsed_ctc.units)

    if units and parsed_ctc.error is None:
        unit_dicts = [u.to_dict() for u in units]
        reconstructed = reconstruct_speaker_utterances(unit_dicts, source_language=resolved_lang or source_language)
        for u in reconstructed:
            u["timestamp_provenance"]["source_offset_s"] = source_offset_s
        utterances = reconstructed
    else:
        for idx, turn in enumerate(turns):
            split_units = split_lexical_units(turn.text)
            spk_id = turn.speaker_tag if turn.speaker_tag != "speaker_0" else "unknown"
            if turn.speaker_tag == "speaker_0" and "<spk:" not in raw_text:
                spk_id = "unknown"
            word_items = [
                {
                    "text": u["text"],
                    "start": None,
                    "end": None,
                    "speaker": spk_id if spk_id != "unknown" else None,
                    "unit_type": u["unit_type"],
                    "valid_timing": False,
                    "timing_error": parsed_ctc.error or "no_ctc_timestamps",
                    "raw_start": None,
                    "raw_end": None,
                }
                for u in split_units
                if u["unit_type"] != "character"
            ]
            char_items = [
                {
                    "text": u["text"],
                    "start": None,
                    "end": None,
                    "speaker": spk_id if spk_id != "unknown" else None,
                    "unit_type": "character",
                    "valid_timing": False,
                    "timing_error": parsed_ctc.error or "no_ctc_timestamps",
                    "raw_start": None,
                    "raw_end": None,
                }
                for u in split_units
                if u["unit_type"] == "character"
            ]
            utterances.append(
                {
                    "id": f"utt-{idx:04d}",
                    "source_language": resolved_lang or source_language,
                    "source_text": turn.text,
                    "speaker_id": spk_id,
                    "source_start": None,
                    "source_end": None,
                    "source_words": word_items,
                    "source_characters": char_items,
                    "timestamp_provenance": {
                        "backend": "none",
                        "source_offset_s": source_offset_s,
                        "timing_status": "unavailable",
                        "error": parsed_ctc.error or "no_ctc_timestamps",
                    },
                }
            )

    clean_full_text = " ".join(t.text for t in turns if t.text).strip()
    return {
        "schema_version": "nemotron-transcript-contract/v1",
        "source_language": resolved_lang or source_language,
        "language_provenance": lang_prov,
        "source_text": clean_full_text,
        "raw_text": raw_text,
        "english_translation": english_translation,
        "utterances": utterances,
        "events": events,
        "diarization_intervals": [d.to_dict() for d in parsed_ctc.diarization],
        "speaker_mapping": dict(parsed_ctc.speaker_mapping),
        "timestamp_provenance": {
            "backend": "nemotron_ctc" if ctc_payload is not None else "none",
            "error": parsed_ctc.error,
            "invalid_timing_count": parsed_ctc.invalid_timing_count,
            "source_offset_s": source_offset_s,
            "unit_count": len(parsed_ctc.units),
        },
        "model_provenance": dict(model_provenance or {}),
        "uncertainty": dict(uncertainty) if uncertainty is not None else None,
    }


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
    """Assemble the canonical recording-level JSON contract (nemotron-recording-contract/v1)."""
    # Strip internal boundary helper keys if present while retaining full unit provenance.
    clean_utts: list[dict[str, Any]] = []
    for utt in utterances:
        clean_utts.append(
            {
                "id": utt["id"],
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
        "schema_version": "nemotron-recording-contract/v1",
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
