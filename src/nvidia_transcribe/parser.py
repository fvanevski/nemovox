"""Parsers for Nemotron speaker-tagged transcripts, CTC timestamps, and Unicode lexical units."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

SPEAKER_TAG_RE = re.compile(r"<spk:(\d+)>")
EVENT_TAG_RE = re.compile(r"<ev:([^<>\s]+)>")
STYLE_TAG_RE = re.compile(r"<sty:([^<>\s]+)>")
SPECIAL_TAG_RE = re.compile(r"<[^<>\s]+>")


class CTCMetadataError(RuntimeError):
    """Raised when CTC timestamp metadata is missing, malformed, or returned an alignment error."""


@dataclass(frozen=True)
class SpeakerTurn:
    speaker_tag: str
    text: str
    raw_text: str
    turn_index: int
    events: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimedUnit:
    text: str
    start: float | None
    end: float | None
    speaker: str | None
    unit_type: str  # "word" | "character" | "event"
    valid_timing: bool
    timing_error: str | None = None
    raw_start: float | None = None
    raw_end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "unit_type": self.unit_type,
            "valid_timing": self.valid_timing,
            "timing_error": self.timing_error,
            "raw_start": self.raw_start,
            "raw_end": self.raw_end,
        }


@dataclass(frozen=True)
class DiarizationInterval:
    speaker: str
    start: float | None
    end: float | None
    valid_timing: bool
    raw_start: float | None = None
    raw_end: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "valid_timing": self.valid_timing,
            "raw_start": self.raw_start,
            "raw_end": self.raw_end,
        }


@dataclass(frozen=True)
class ParsedCTCAlignment:
    units: tuple[TimedUnit, ...]
    diarization: tuple[DiarizationInterval, ...]
    speaker_mapping: dict[str, Any]
    error: str | None
    invalid_timing_count: int
    source_offset_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "units": [u.to_dict() for u in self.units],
            "diarization": [d.to_dict() for d in self.diarization],
            "speaker_mapping": dict(self.speaker_mapping),
            "error": self.error,
            "invalid_timing_count": self.invalid_timing_count,
            "source_offset_s": self.source_offset_s,
            "metadata": dict(self.metadata),
        }


def _finite_nonneg(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val) or val < 0.0:
        return None
    return round(val, 6)


def _is_cjk_char(ch: str) -> bool:
    if not ch or ch.isspace():
        return False
    name = unicodedata.name(ch, "")
    return name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))


def classify_lexical_unit(token: str) -> str:
    """Classify a single aligned token as event, character (CJK), or word."""
    if EVENT_TAG_RE.fullmatch(token):
        return "event"
    stripped = token.strip()
    if len(stripped) == 1 and _is_cjk_char(stripped):
        return "character"
    return "word"


def split_lexical_units(text: str) -> list[dict[str, str]]:
    """Split text into lexical units, splitting CJK ideographs individually without whitespace assumption."""
    cleaned = SPECIAL_TAG_RE.sub(" ", text or "")
    units: list[dict[str, str]] = []
    for chunk in cleaned.split():
        buf: list[str] = []
        for ch in chunk:
            if _is_cjk_char(ch):
                if buf:
                    units.append({"text": "".join(buf), "unit_type": "word"})
                    buf.clear()
                units.append({"text": ch, "unit_type": "character"})
            else:
                buf.append(ch)
        if buf:
            units.append({"text": "".join(buf), "unit_type": "word"})
    return units


def parse_speaker_turns(raw_text: str) -> list[SpeakerTurn]:
    """Parse ``<spk:N>`` turns while preserving raw text and extracting event/style tags."""
    text = (raw_text or "").strip()
    if not text:
        return []
    parts = SPEAKER_TAG_RE.split(text)
    if len(parts) == 1:
        events = tuple(EVENT_TAG_RE.findall(text))
        styles = tuple(STYLE_TAG_RE.findall(text))
        clean = " ".join(SPECIAL_TAG_RE.sub(" ", text).split())
        return [SpeakerTurn(speaker_tag="speaker_0", text=clean, raw_text=text, turn_index=0, events=events, styles=styles)]

    turns: list[SpeakerTurn] = []
    prefix = parts[0].strip()
    if prefix:
        events = tuple(EVENT_TAG_RE.findall(prefix))
        styles = tuple(STYLE_TAG_RE.findall(prefix))
        clean = " ".join(SPECIAL_TAG_RE.sub(" ", prefix).split())
        turns.append(
            SpeakerTurn(speaker_tag="unattributed", text=clean, raw_text=prefix, turn_index=0, events=events, styles=styles)
        )

    for idx in range(1, len(parts), 2):
        spk_id = parts[idx]
        segment = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
        events = tuple(EVENT_TAG_RE.findall(segment))
        styles = tuple(STYLE_TAG_RE.findall(segment))
        clean = " ".join(SPECIAL_TAG_RE.sub(" ", segment).split())
        turns.append(
            SpeakerTurn(
                speaker_tag=f"<spk:{spk_id}>",
                text=clean,
                raw_text=segment,
                turn_index=len(turns),
                events=events,
                styles=styles,
            )
        )
    return turns


def apply_source_offset(value: float | None, source_offset_s: float) -> float | None:
    """Convert excerpt-relative timestamp to original-media timestamp."""
    if value is None:
        return None
    if not math.isfinite(source_offset_s) or source_offset_s < 0.0:
        raise ValueError(f"source_offset_s must be finite and >= 0, got {source_offset_s}")
    return round(value + source_offset_s, 6)


def parse_ctc_timestamps(
    ctc_payload: Any,
    *,
    source_offset_s: float = 0.0,
    strict: bool = False,
) -> ParsedCTCAlignment:
    """Validate and normalize a vLLM ``ctc_timestamps`` payload without fabricating timing.

    Args:
        ctc_payload: Raw ``ctc_timestamps`` dict from vLLM response (or ``None``).
        source_offset_s: Offset in seconds to add to excerpt-relative timestamps.
        strict: If True, raise ``CTCMetadataError`` when metadata is missing or carries an error.
    """
    if ctc_payload is None or not isinstance(ctc_payload, dict):
        msg = "ctc_timestamps metadata is missing or not a mapping"
        if strict:
            raise CTCMetadataError(msg)
        return ParsedCTCAlignment(
            units=(),
            diarization=(),
            speaker_mapping={},
            error="missing_ctc_metadata",
            invalid_timing_count=0,
            source_offset_s=source_offset_s,
        )

    error = ctc_payload.get("error")
    if strict and error is not None:
        raise CTCMetadataError(f"CTC alignment returned error: {error!r}")

    raw_words = ctc_payload.get("words")
    if raw_words is None:
        raw_words = []
    if not isinstance(raw_words, list):
        raise CTCMetadataError("ctc_timestamps['words'] must be a list")

    units: list[TimedUnit] = []
    invalid_count = 0
    for item in raw_words:
        if not isinstance(item, dict):
            invalid_count += 1
            continue
        token = str(item.get("word") or "")
        raw_s = _finite_nonneg(item.get("start"))
        raw_e = _finite_nonneg(item.get("end"))
        timing_err: str | None = None
        if raw_s is None or raw_e is None:
            timing_err = "missing_or_nonfinite_timestamp"
            valid = False
            invalid_count += 1
            abs_s = None
            abs_e = None
        elif raw_e < raw_s:
            timing_err = "inverted_interval"
            valid = False
            invalid_count += 1
            abs_s = None
            abs_e = None
        else:
            valid = True
            abs_s = apply_source_offset(raw_s, source_offset_s)
            abs_e = apply_source_offset(raw_e, source_offset_s)

        spk_raw = item.get("speaker")
        speaker = str(spk_raw) if spk_raw is not None and str(spk_raw).strip() != "" else None
        units.append(
            TimedUnit(
                text=token,
                start=abs_s,
                end=abs_e,
                speaker=speaker,
                unit_type=classify_lexical_unit(token),
                valid_timing=valid,
                timing_error=timing_err,
                raw_start=raw_s,
                raw_end=raw_e,
            )
        )

    raw_diar = ctc_payload.get("diarization") or []
    diar_intervals: list[DiarizationInterval] = []
    if isinstance(raw_diar, list):
        for seg in raw_diar:
            if not isinstance(seg, dict):
                continue
            raw_s = _finite_nonneg(seg.get("start"))
            raw_e = _finite_nonneg(seg.get("end"))
            valid = raw_s is not None and raw_e is not None and raw_e >= raw_s
            if not valid:
                invalid_count += 1
            diar_intervals.append(
                DiarizationInterval(
                    speaker=str(seg.get("speaker")),
                    start=apply_source_offset(raw_s, source_offset_s) if valid else None,
                    end=apply_source_offset(raw_e, source_offset_s) if valid else None,
                    valid_timing=valid,
                    raw_start=raw_s,
                    raw_end=raw_e,
                )
            )

    raw_map = ctc_payload.get("speaker_tag_to_diarization_speaker") or {}
    speaker_mapping = {str(k): v for k, v in raw_map.items()} if isinstance(raw_map, dict) else {}

    extra_meta = {
        k: v
        for k, v in ctc_payload.items()
        if k not in ("words", "diarization", "speaker_tag_to_diarization_speaker", "error")
    }
    return ParsedCTCAlignment(
        units=tuple(units),
        diarization=tuple(diar_intervals),
        speaker_mapping=speaker_mapping,
        error=str(error) if error is not None else None,
        invalid_timing_count=invalid_count,
        source_offset_s=source_offset_s,
        metadata=extra_meta,
    )
