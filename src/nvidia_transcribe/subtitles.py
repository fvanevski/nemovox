"""Standalone English SRT and ASS subtitle compositor with speaker-aware lanes."""

from __future__ import annotations

import os
import math
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from nvidia_transcribe import connected_units

_SENTENCE_END_CHARS = frozenset(".?!。？！")
_CLAUSE_END_CHARS = frozenset(",;:—–、，；：")
_PHRASE_BREAK_WORDS = frozenset(
    {"and", "but", "or", "so", "yet", "because", "although", "while", "when", "where", "if", "that", "which", "who", "to", "for", "of", "in", "on", "at", "with", "before", "after"}
)


@dataclass(frozen=True)
class SubtitleConfig:
    max_chars: int = 42
    max_lines: int = 2
    min_display_s: float = 1.2
    max_display_s: float = 6.5
    target_cps: float = 18.0
    overlap_threshold_s: float = 0.4
    play_res: tuple[int, int] = (1920, 1080)
    font_name: str = "DejaVu Sans"
    font_size: int = 52
    margin_h: int = 54
    margin_v: int = 42
    raised_main_margin_v: int = 150
    ass_layout: str = "auto"  # "auto" | "standard" | "raised_main"
    ass_opaque_box: bool = False
    ass_max_lanes: int = 3
    outline: float = 2.5
    shadow: float = 1.2
    speaker_prefixes_in_srt_overlap: bool = False


@dataclass
class SubtitleCue:
    """Display cue retaining immutable source utterance reference and acoustic bounds."""

    utterance_id: str
    start: float
    end: float
    text: str
    speaker: str
    raw_start: float | None = None
    raw_end: float | None = None
    lane: int = 0
    timing_status: str = "full"
    unrenderable: bool = False
    overflow: bool = False
    grouped_utterance_ids: list[str] = field(default_factory=list)
    source_turn_ids: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    translation_unit_id: str | None = None
    unit_occurrence_ids: list[str] = field(default_factory=list)
    display_occurrence_ids: list[str] = field(default_factory=list)
    source_intervals: list[tuple[float, float]] = field(default_factory=list)
    display_order: int = 0
    ass_position: tuple[int, int] | None = None
    ass_bounds: tuple[int, int, int, int] | None = None


def _finite_nonneg(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")) or f < 0.0:
        return None
    return f


def _display_width(text: str) -> int:
    """Approximate visual display width (wide CJK characters count slightly wider for wrap safety)."""
    total = 0
    for ch in text:
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def wrap_subtitle_lines(text: str, max_chars: int = 42, max_lines: int = 2) -> list[str]:
    """Wrap subtitle text cleanly favoring phrase/punctuation breaks and balanced line widths."""
    cleaned = " ".join((text or "").replace("\n", " ").split()).strip()
    if not cleaned:
        return [""]
    if len(cleaned) <= max_chars and _display_width(cleaned) <= max_chars + 4:
        return [cleaned]

    words = cleaned.split(" ")
    if len(words) == 1:
        return [cleaned]

    # When target fits cleanly in 2 lines, search all split indices and pick highest-quality balanced split.
    if max_lines == 2 and len(cleaned) <= max_chars * 2 + 4:
        best_score = float("-inf")
        best_pair: tuple[str, str] | None = None
        for split in range(1, len(words)):
            l1 = " ".join(words[:split])
            l2 = " ".join(words[split:])
            if len(l1) > max_chars or len(l2) > max_chars:
                continue
            balance_penalty = abs(len(l1) - len(l2)) * 0.45
            break_bonus = 0.0
            last_char = l1.rstrip("\"'»)]}”’")[-1:]
            if last_char in _SENTENCE_END_CHARS:
                break_bonus += 16.0
            elif last_char in _CLAUSE_END_CHARS:
                break_bonus += 10.0
            elif words[split].lower() in _PHRASE_BREAK_WORDS:
                break_bonus += 4.5
            # Slight bottom-heavy preference (professional subtitling pyramid style)
            pyramid_bonus = 1.5 if len(l2) >= len(l1) else 0.0
            score = break_bonus + pyramid_bonus - balance_penalty
            if score > best_score:
                best_score = score
                best_pair = (l1, l2)
        if best_pair is not None:
            return [best_pair[0], best_pair[1]]

    # Greedy phrase-aware fallback for >2 lines or overly dense unbroken tokens
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        cand = " ".join(cur + [w])
        if cur and len(cand) > max_chars:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def _punctuation_chunks(text: str, budget: int, max_chars: int, max_lines: int) -> list[str]:
    """Split oversized subtitle text across sub-cues along sentence/clause/phrase boundaries."""
    words = [w for w in text.split(" ") if w]
    if not words:
        return [text]
    # Dynamic partition: first minimize fragment count, then avoid tiny
    # trailing fragments while rewarding grammatical break opportunities.
    # The character budget bounds the inner loop; a single oversized token
    # is retained and subsequently diagnosed, never discarded.
    n = len(words)
    best = [(0, 0.0, n)] * (n + 1)
    for i in range(n - 1, -1, -1):
        choices = []
        for j in range(i + 1, n + 1):
            candidate = " ".join(words[i:j])
            wrapped = wrap_subtitle_lines(candidate, max_chars, max_lines)
            fits = (len(candidate) <= budget and len(wrapped) <= max_lines
                    and all(len(line) <= max_chars for line in wrapped))
            if not fits and j > i + 1:
                break
            tail_count, tail_cost, _ = best[j]
            last = candidate.rstrip("\"'»)]}”’")[-1:]
            bonus = (100 if last in _SENTENCE_END_CHARS else
                     50 if last in _CLAUSE_END_CHARS else
                     20 if j < n and words[j].lower() in _PHRASE_BREAK_WORDS else 0)
            choices.append((1 + tail_count, len(candidate) ** 2 + tail_cost - bonus, j))
            if not fits:
                break
        best[i] = min(choices)
    chunks = []
    i = 0
    while i < n:
        j = best[i][2]
        chunks.append(" ".join(words[i:j]))
        i = j
    return chunks


def _select_utterance_display_text(utt: dict[str, Any]) -> str:
    """Select English/target subtitle text if available, else refined source, else raw source."""
    for key in ("translated_text", "english_text", "refined_text", "source_text"):
        val = utt.get(key)
        if val is not None and str(val).strip():
            return " ".join(str(val).strip().split())
    return ""


def _resolve_utterance_timing(utt: dict[str, Any]) -> tuple[float | None, float | None, str]:
    """Extract credible acoustic or bounded speaker activity interval without fabricating timing."""
    prov = utt.get("timestamp_provenance") or {}
    t_status = str(prov.get("timing_status") or utt.get("timing_status") or "unknown")
    s = _finite_nonneg(utt.get("source_start") if utt.get("source_start") is not None else utt.get("start"))
    e = _finite_nonneg(utt.get("source_end") if utt.get("source_end") is not None else utt.get("end"))
    if s is not None and e is not None and e > s:
        return s, e, t_status
    return None, None, "unavailable"


def build_display_cues(
    utterances: Sequence[dict[str, Any]],
    *,
    config: SubtitleConfig | None = None,
    scene_boundaries_s: Sequence[float] = (),
) -> list[SubtitleCue]:
    """Build display-ready SubtitleCue objects from translated/refined utterances."""
    cfg = config or SubtitleConfig()
    budget = max(1, cfg.max_chars * cfg.max_lines)
    cues: list[SubtitleCue] = []

    for idx, utt in enumerate(utterances):
        if (
            utt.get("merged_into")
            or utt.get("fragment_suppressed")
            or utt.get("overlay_suppressed")
            or utt.get("suppressed_duplicate_of")
        ):
            continue
        uid = str(utt.get("utterance_id") or utt.get("id") or f"utt-{idx:04d}")
        grouped = [str(g) for g in (utt.get("grouped_utterance_ids") or [uid])]
        src_refs = [str(x) for x in (utt.get("source_turn_ids") or
                                     ([utt["source_turn_id"]] if utt.get("source_turn_id") else []))]
        spk = str(utt.get("speaker_id") or utt.get("global_speaker_id") or "SPK_UNK")
        text = _select_utterance_display_text(utt).translate(str.maketrans({"‘": "\'", "’": "\'", "“": '"', "”": '"'}))
        if not text:
            continue

        selected = utt.get("translation_unit") or {}
        unit_id = selected.get("unit_id")
        unit_members = [r["id"] for r in selected.get("constituents", [])]
        raw_s, raw_e, t_status = _resolve_utterance_timing(utt)
        if raw_s is None or raw_e is None:
            cues.append(
                SubtitleCue(
                    utterance_id=uid,
                    start=0.0,
                    end=0.0,
                    text=text,
                    speaker=spk,
                    raw_start=None,
                    raw_end=None,
                    timing_status="unavailable",
                    unrenderable=True,
                    grouped_utterance_ids=grouped,
                    source_turn_ids=src_refs,
                    conflicts=["unrenderable_no_usable_timing"],
                    translation_unit_id=unit_id,
                    unit_occurrence_ids=unit_members,
                )
            )
            continue

        wrapped_full = wrap_subtitle_lines(text, max_chars=cfg.max_chars, max_lines=cfg.max_lines)
        fits_single = (
            len(text) <= budget
            and len(wrapped_full) <= cfg.max_lines
            and all(len(ln) <= cfg.max_chars for ln in wrapped_full)
        )
        chunks = [text] if fits_single else _punctuation_chunks(text, budget, cfg.max_chars, cfg.max_lines)

        # Allocate ONLY the actual source span. A display floor is not acoustic
        # evidence. Give each fragment an equal feasible dwell floor and distribute
        # the remaining span by text burden.
        span = raw_e - raw_s
        weights = [max(1, len(ch)) for ch in chunks]
        floor_s = min(cfg.min_display_s, span / len(chunks))
        remaining = max(0.0, span - floor_s * len(chunks))
        allocs = [floor_s + remaining * w / sum(weights) for w in weights]

        cursor = raw_s
        for j, ch_text in enumerate(chunks):
            sub_s = cursor
            sub_e = raw_e if j == len(chunks) - 1 else min(raw_e, cursor + allocs[j])
            cursor = sub_e

            disp_s, disp_e = sub_s, sub_e
            conflicts: list[str] = []
            if t_status in ("partial", "failed", "diarization_bounded_only", "approximate", "coarse_recovered", "relocated_reviewed"):
                conflicts.append(f"source_timing_{t_status}")

            dur = disp_e - disp_s
            if dur > cfg.max_display_s:
                disp_e = disp_s + cfg.max_display_s
                conflicts.append("max_display_clamped")
            elif dur < cfg.min_display_s:
                # Extend comfortably without exceeding max_display_s; sequential lane scheduling will clamp before next cue
                ideal_dur = max(cfg.min_display_s, min(cfg.max_display_s, len(ch_text) / max(10.0, cfg.target_cps)))
                disp_e = disp_s + ideal_dur
                if j + 1 < len(chunks):
                    disp_e = min(disp_e, max(sub_e, cursor))
            elif len(ch_text) / max(0.2, dur) > cfg.target_cps:
                # Slightly extend rapid single-chunk cues into trailing silence if next cue permits
                ideal_dur = min(cfg.max_display_s, len(ch_text) / max(10.0, cfg.target_cps))
                if ideal_dur > dur:
                    disp_e = disp_s + ideal_dur

            wrapped = wrap_subtitle_lines(ch_text, max_chars=cfg.max_chars, max_lines=cfg.max_lines)
            if len(wrapped) > cfg.max_lines or any(len(ln) > cfg.max_chars for ln in wrapped):
                conflicts.append("line_wrap_exceeded")

            cues.append(
                SubtitleCue(
                    utterance_id=uid,
                    start=round(disp_s, 3),
                    end=round(max(disp_s + 0.1, disp_e), 3),
                    text=ch_text,
                    speaker=spk,
                    raw_start=sub_s,
                    raw_end=sub_e,
                    source_intervals=[(raw_s, raw_e)],
                    timing_status=t_status,
                    unrenderable=False,
                    grouped_utterance_ids=grouped,
                    source_turn_ids=src_refs,
                    conflicts=conflicts,
                    translation_unit_id=unit_id,
                    unit_occurrence_ids=unit_members,
                )
            )
    return _group_display_continuations(cues, utterances, cfg, scene_boundaries_s)


def _group_display_continuations(cues, utterances, cfg, scene_boundaries_s=()):
    """Only co-display a proven joint allocation's grammatical continuation.

    Never alter selected English or treat same-speaker proximity as membership.
    Any intervening, overlapping or uncertain occurrence defeats grouping.
    """
    by_id = {str(row.get("utterance_id") or row.get("id") or f"utt-{i:04d}"):
             (i, row) for i, row in enumerate(utterances)}
    counts = {}
    for cue in cues:
        counts[cue.utterance_id] = counts.get(cue.utterance_id, 0) + 1
    result = []
    for cue in cues:
        if not result:
            result.append(cue)
            continue
        prev = result[-1]
        a, b = by_id[prev.utterance_id], by_id[cue.utterance_id]
        text = prev.text + " " + cue.text
        wrapped = wrap_subtitle_lines(text, cfg.max_chars, cfg.max_lines)
        selected = a[1].get("translation_unit") or {}
        start, end = prev.raw_start, cue.raw_end
        if (not prev.unrenderable and not cue.unrenderable
                and not prev.display_occurrence_ids
                and counts[prev.utterance_id] == counts[cue.utterance_id] == 1
                and b[0] == a[0] + 1
                and prev.translation_unit_id is not None
                and prev.translation_unit_id == cue.translation_unit_id
                and selected.get("selection") == "jointly_adjudicated"
                and selected == b[1].get("translation_unit")
                and prev.utterance_id in prev.unit_occurrence_ids
                and cue.utterance_id in prev.unit_occurrence_ids
                and a[1].get("speaker_id") and a[1].get("speaker_id") == b[1].get("speaker_id")
                and prev.speaker != "SPK_UNK"
                and not any(row.get(key) for row in (a[1], b[1]) for key in
                            ("review_source_uncertainty", "source_quality_status", "speaker_uncertainty"))
                and start is not None and end is not None
                and prev.raw_end is not None and cue.raw_start is not None
                and 0 <= cue.raw_start - prev.raw_end <= 0.25
                and not prev.text.rstrip("\"'").endswith((".", "!", "?", "…"))
                and cue.text[0].islower()
                and end - start <= cfg.max_display_s
                and not any(start < cut < end for cut in scene_boundaries_s)
                and len(wrapped) <= cfg.max_lines
                and all(len(line) <= cfg.max_chars for line in wrapped)
                and cue.raw_start - start <= cfg.min_display_s
                and min(prev.raw_end-start, end-cue.raw_start) < cfg.min_display_s
                and not any(other.utterance_id not in (prev.utterance_id, cue.utterance_id)
                            and other.raw_start is not None and other.raw_end is not None
                            and other.raw_start < end and other.raw_end > start for other in cues)):
            result[-1] = replace(
                prev, text=text, raw_end=end,
                end=min(start + cfg.max_display_s,
                        max(end, start + cfg.min_display_s, start + len(text) / max(10.0, cfg.target_cps))),
                grouped_utterance_ids=list(dict.fromkeys(prev.grouped_utterance_ids + cue.grouped_utterance_ids)),
                source_turn_ids=list(dict.fromkeys(prev.source_turn_ids + cue.source_turn_ids)),
                display_occurrence_ids=[prev.utterance_id, cue.utterance_id],
                source_intervals=prev.source_intervals + cue.source_intervals,
                conflicts=list(dict.fromkeys(prev.conflicts + cue.conflicts + ["display_grouped_continuation"])),
            )
        else:
            result.append(cue)
    return result


def _acoustic_bounds(cue: SubtitleCue) -> tuple[float, float]:
    if cue.raw_start is not None and cue.raw_end is not None:
        return cue.raw_start, cue.raw_end
    return cue.start, cue.end


def _acoustically_concurrent(a: SubtitleCue, b: SubtitleCue, overlap_threshold_s: float) -> bool:
    """Real overlap of a substantial fraction of a short voice is concurrency."""
    for as0, ae0 in a.source_intervals or [_acoustic_bounds(a)]:
        for bs0, be0 in b.source_intervals or [_acoustic_bounds(b)]:
            overlap = min(ae0, be0) - max(as0, bs0)
            shorter = min(ae0-as0, be0-bs0)
            if overlap > 0 and shorter > 0 and overlap >= min(
                    max(0., overlap_threshold_s), .5 * shorter):
                return True
    return False


def assign_subtitle_lanes(
    cues: Sequence[SubtitleCue],
    *,
    utterances: Sequence[dict[str, Any]] = (),
    overlap_threshold_s: float = 0.4,
    max_lanes: int = 2,
    target_cps: float = 18.0,
    max_display_s: float = 6.5,
    min_display_s: float = 1.2,
    allow_sequential_stacking: bool = True,
) -> tuple[list[SubtitleCue], list[SubtitleCue], list[SubtitleCue]]:
    """Schedule actual display occupancy at emitted centisecond precision.

    Prefer the bottom lane and trailing slack. Retain real concurrency.
    Only stack successive turns when truncation creates real reading pressure
    and the aggregate simultaneous burden is bounded. Never move an onset
    later to clear a lane. Saturation is explicit, not a hidden collision.
    SRT uses the same policy without discretionary sequential stacking.
    """
    num_lanes = max(1, min(3, int(max_lanes)))
    displayed, overflow, unrenderable = [], [], []
    active = {}
    seen = set()
    # Preserve input order on equal onset; no retroactive lane swaps.
    ordered = sorted(enumerate(cues), key=lambda item: (_acoustic_bounds(item[1])[0], item[0]))
    for order, (_, source) in enumerate(ordered):
        cue = replace(source, conflicts=list(source.conflicts),
                      grouped_utterance_ids=list(source.grouped_utterance_ids),
                      source_turn_ids=list(source.source_turn_ids),
                      unit_occurrence_ids=list(source.unit_occurrence_ids),
                      display_occurrence_ids=list(source.display_occurrence_ids),
                      source_intervals=list(source.source_intervals), display_order=order)
        if cue.unrenderable:
            unrenderable.append(cue)
            continue
        identity = (cue.utterance_id, cue.raw_start, cue.raw_end, cue.text,
                    tuple(cue.display_occurrence_ids), tuple(cue.source_turn_ids))
        if identity in seen:
            cue.unrenderable = True
            cue.conflicts.append("duplicate_display_replay")
            unrenderable.append(cue)
            continue
        seen.add(identity)
        cue.start = round(cue.start * 100) / 100
        cue.end = round(min(cue.end, cue.start + max_display_s) * 100) / 100
        if cue.end <= cue.start:
            cue.unrenderable = True
            cue.conflicts.append("unrenderable_at_display_precision")
            unrenderable.append(cue)
            continue
        active = {lane: prev for lane, prev in active.items() if prev.end > cue.start}

        def same_occurrence(prev):
            return bool(set(prev.display_occurrence_ids or [prev.utterance_id])
                        & set(cue.display_occurrence_ids or [cue.utterance_id]))

        def concurrent(prev):
            return not same_occurrence(prev) and _acoustically_concurrent(
                prev, cue, overlap_threshold_s)

        def trim_safe(prev):
            return not concurrent(prev) and prev.start < cue.start

        def clip(prev):
            prev.end = cue.start
            prev.conflicts.append("display_clamped_before_next_cue")

        def burden_ok(others):
            pool = [cue, *others]
            return (len(pool) <= 2 and sum(len(c.text) for c in pool) <= 100
                    and sum(len(c.text) / max(0.01, c.end-c.start) for c in pool)
                    <= target_cps * 1.8)

        # Split fragments must not be co-displayed as independent speakers.
        for lane, prev in list(active.items()):
            if same_occurrence(prev) and trim_safe(prev):
                clip(prev)
                del active[lane]

        # If a fragment is too short for a distinct emitted onset, retain
        # explicit pressure rather than overlapping/reordering that occurrence.
        if any(same_occurrence(prev) for prev in active.values()):
            cue.overflow = True
            cue.conflicts.append("split_fragment_display_saturation")
            overflow.append(cue)
            continue

        preferred = active.get(0)
        free = [lane for lane in range(num_lanes) if lane not in active]
        chosen = None
        if preferred is None:
            chosen = 0
        elif trim_safe(preferred):
            clipped_dwell = cue.start - preferred.start
            pressure = (clipped_dwell < min_display_s
                        or len(preferred.text) / clipped_dwell > target_cps)
            if (allow_sequential_stacking and free and pressure
                    and not concurrent(preferred) and burden_ok(list(active.values()))):
                chosen = free[0]
            else:
                clip(preferred)
                del active[0]
                chosen = 0
        elif free:
            chosen = free[0]
        else:
            for lane, prev in sorted(active.items()):
                if trim_safe(prev):
                    clip(prev)
                    del active[lane]
                    chosen = lane
                    break

        if chosen is None:
            cue.overflow = True
            cue.conflicts.append("display_lane_saturation")
            overflow.append(cue)
            continue

        # Existing padding on other lanes must not silently accumulate.
        # Actual simultaneous speech is retained even when reading load is
        # high; discretionary overlap is the first thing to sacrifice.
        for lane, prev in list(active.items()):
            if (trim_safe(prev)
                    and (not allow_sequential_stacking or not burden_ok(list(active.values())))):
                clip(prev)
                del active[lane]
        if any(not concurrent(prev) for prev in active.values()):
            cue.conflicts.append("sequential_display_overlap")
        if active and not burden_ok(list(active.values())):
            cue.conflicts.append("simultaneous_display_pressure")
        cue.lane = chosen
        active[chosen] = cue
        displayed.append(cue)

    for cue in displayed + overflow:
        cps = len(cue.text) / max(0.01, cue.end - cue.start)
        if cps > target_cps + 4:
            cue.conflicts.append(f"high_reading_speed:{cps:.1f}cps")
    displayed.sort(key=lambda c: (c.start, c.display_order))
    overflow.sort(key=lambda c: (c.start, c.display_order))
    return displayed, overflow, unrenderable


def _srt_timestamp(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    ms = int(round((s - int(s)) * 1000))
    if ms >= 1000:
        ms -= 1000
        sec += 1
    if sec >= 60:
        sec -= 60
        m += 1
    if m >= 60:
        m -= 60
        h += 1
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _ass_timestamp(seconds: float) -> str:
    s = max(0.0, float(seconds))
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    rem = s % 60
    # Avoid rounding 59.999 to 60.00 in centiseconds
    cs = int(round(rem * 100))
    sec = cs // 100
    crem = cs % 100
    if sec >= 60:
        sec -= 60
        m += 1
    if m >= 60:
        m -= 60
        h += 1
    return f"{h}:{m:02d}:{sec:02d}.{crem:02d}"


def _ass_escape(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return escaped.replace("\n", "\\N")


def format_srt(cues: Iterable[SubtitleCue], *, config: SubtitleConfig | None = None) -> str:
    """Format all renderable cues (including overflow) into UTF-8 SRT text."""
    cfg = config or SubtitleConfig()
    ordered = sorted((c for c in cues if not c.unrenderable), key=lambda c: (c.start, c.display_order))
    blocks: list[str] = []
    for i, cue in enumerate(ordered, start=1):
        wrapped = "\n".join(wrap_subtitle_lines(cue.text, max_chars=cfg.max_chars, max_lines=cfg.max_lines))
        blocks.append(f"{i}\n{_srt_timestamp(cue.start)} --> {_srt_timestamp(cue.end)}\n{wrapped}\n")
    return "\n".join(blocks)


def format_ass(lane_cues: Iterable[SubtitleCue], *, config: SubtitleConfig | None = None, title: str = "English Subtitles") -> str:
    """Format multi-lane cues into readable ASS v4+ script (Main, Top, optional TopAlt)."""
    cfg = config or SubtitleConfig()
    pw, ph = cfg.play_res
    main_margin_v = int(cfg.raised_main_margin_v) if cfg.ass_layout == "raised_main" else int(cfg.margin_v)
    top_alt_margin_v = int(cfg.margin_v) + int(cfg.font_size) * 2 + 18
    border_style = 4 if cfg.ass_opaque_box else 1
    back_colour = "&H96000000" if cfg.ass_opaque_box else "&H64000000"
    outline_px = 1.5 if cfg.ass_opaque_box else cfg.outline
    shadow_px = 0.0 if cfg.ass_opaque_box else cfg.shadow
    lines = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        f"PlayResX: {pw}",
        f"PlayResY: {ph}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Main,{cfg.font_name},{cfg.font_size},&H00FFFFFF,&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline_px},{shadow_px},2,{cfg.margin_h},{cfg.margin_h},{main_margin_v},1",
        f"Style: Top,{cfg.font_name},{cfg.font_size},&H00EAF6FF,&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline_px},{shadow_px},8,{cfg.margin_h},{cfg.margin_h},{cfg.margin_v},1",
        f"Style: TopAlt,{cfg.font_name},{cfg.font_size},&H00D8F3DC,&H000000FF,&H00000000,{back_colour},0,0,0,0,100,100,0,0,{border_style},{outline_px},{shadow_px},8,{cfg.margin_h},{cfg.margin_h},{top_alt_margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    ordered = sorted((c for c in lane_cues if not c.unrenderable), key=lambda c: (c.start, c.display_order))
    for cue in ordered:
        style = "Main" if cue.lane == 0 else ("Top" if cue.lane == 1 else "TopAlt")
        wrapped = "\n".join(wrap_subtitle_lines(cue.text, max_chars=cfg.max_chars, max_lines=cfg.max_lines))
        lines.append(
            f"Dialogue: {cue.lane},{_ass_timestamp(cue.start)},{_ass_timestamp(cue.end)},{style},{cue.speaker},0,0,0,,"
            + (r"{\an5\pos(" + f"{cue.ass_position[0]},{cue.ass_position[1]}" + ")}" if cue.ass_position else "")
            + _ass_escape(wrapped)
        )
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def render_subtitles(
    contract: dict[str, Any],
    *,
    output_dir: Path,
    episode_stem: str,
    target_language: str = "en",
    formats: Sequence[str] = ("srt", "ass"),
    config: SubtitleConfig | None = None,
    sidecar_dir: Path | None = None,
    overwrite_sidecar: bool = False,
    presentation_observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render SRT and/or ASS subtitles directly from a translated or refined contract."""
    cfg = config or SubtitleConfig()
    utterances = contract.get("utterances") or []
    connected_units.validate_selected_units(utterances, require_complete=(
        (contract.get("translation_review") or {}).get("translation_unit_schema") == connected_units.SCHEMA))
    from nvidia_transcribe.presentation import admit_observations, adaptive_config, constrain_scene_padding, place_ass
    observation = admit_observations(presentation_observations, contract.get("source") or {})
    if cfg.ass_layout not in ("auto", "standard", "raised_main"):
        raise ValueError("Unknown ASS layout")
    if cfg.ass_layout == "auto":
        cfg = adaptive_config(cfg, observation)
    cues = build_display_cues(utterances, config=cfg,
                              scene_boundaries_s=(observation or {}).get("known_scene_boundaries_s", []))
    constrain_scene_padding(cues, observation)
    displayed, overflow, unrenderable = assign_subtitle_lanes(
        cues,
        utterances=utterances,
        overlap_threshold_s=cfg.overlap_threshold_s,
        max_lanes=cfg.ass_max_lanes,
        target_cps=cfg.target_cps,
        max_display_s=cfg.max_display_s,
        min_display_s=cfg.min_display_s,
    )
    srt_displayed, srt_overflow, srt_unrenderable = assign_subtitle_lanes(
        cues, utterances=utterances, overlap_threshold_s=cfg.overlap_threshold_s,
        max_lanes=cfg.ass_max_lanes, target_cps=cfg.target_cps,
        max_display_s=cfg.max_display_s, min_display_s=cfg.min_display_s,
        allow_sequential_stacking=False,
    )

    # Limit final display times after all extension/scheduling. Floor the
    # media boundary to the common ASS/SRT time precision so serialization
    # cannot round a cue past the actual file end. Raw bounds stay unchanged.
    duration = _finite_nonneg((contract.get("source") or {}).get("duration_s"))
    if duration is not None:
        limit = math.floor(duration * 100 + 1e-9) / 100
        for bucket, unavailable in ((displayed, unrenderable), (overflow, unrenderable),
                                    (srt_displayed, srt_unrenderable), (srt_overflow, srt_unrenderable)):
            kept = []
            for cue in bucket:
                if cue.end > limit:
                    cue.end = limit
                    cue.conflicts.append("display_clamped_at_media_end")
                if cue.end <= cue.start or round(cue.end * 100) <= round(cue.start * 100):
                    cue.unrenderable = True
                    cue.conflicts.append("unrenderable_outside_media_duration")
                    unavailable.append(cue)
                    continue
                # A final clamp can create a new reading-speed conflict.
                cue.conflicts = [c for c in cue.conflicts if not c.startswith("high_reading_speed:")]
                cps = len(cue.text) / (cue.end - cue.start)
                if cps > cfg.target_cps + 4:
                    cue.conflicts.append(f"high_reading_speed:{cps:.1f}cps")
                kept.append(cue)
            bucket[:] = kept
    presentation = {"layout": cfg.ass_layout, "status": "manual_override" if cfg.ass_layout != "auto" else "ass_not_requested"}
    if cfg.ass_layout == "auto" and "ass" in {f.strip().lower() for f in formats}:
        displayed, placement_overflow, presentation = place_ass(displayed, cfg, observation)
        overflow.extend(placement_overflow)
        if presentation["status"] == "measurement_unavailable":
            cfg = replace(cfg, ass_layout="raised_main")
            presentation["effective_layout"] = "raised_main"
    all_renderable = list(srt_displayed) + list(srt_overflow)
    lang_tag = (target_language or "en").strip().lower()
    outputs: dict[str, str] = {}
    sidecars: dict[str, str] = {}
    fmts = {f.strip().lower() for f in formats if f.strip()}

    no_renderable_dialogue = bool(cues) and not (("srt" in fmts and all_renderable) or
                                                 ("ass" in fmts and displayed))
    if "srt" in fmts:
        srt_text = format_srt(all_renderable, config=cfg)
        srt_out = output_dir / f"{episode_stem}.{lang_tag}.srt"
        _atomic_write(srt_out, srt_text)
        outputs["srt"] = str(srt_out)
        if sidecar_dir is not None and (all_renderable or not cues):
            sc_srt = Path(sidecar_dir) / f"{episode_stem}.{lang_tag}.srt"
            if sc_srt.exists() and not overwrite_sidecar:
                raise FileExistsError(f"Refusing to overwrite existing sidecar without overwrite_sidecar=True: {sc_srt}")
            _atomic_write(sc_srt, srt_text)
            sidecars["srt"] = str(sc_srt)

    if "ass" in fmts:
        ass_text = format_ass(displayed, config=cfg, title=f"{episode_stem} ({lang_tag})")
        ass_out = output_dir / f"{episode_stem}.{lang_tag}.ass"
        _atomic_write(ass_out, ass_text)
        outputs["ass"] = str(ass_out)
        if sidecar_dir is not None and (displayed or not cues):
            sc_ass = Path(sidecar_dir) / f"{episode_stem}.{lang_tag}.ass"
            if sc_ass.exists() and not overwrite_sidecar:
                raise FileExistsError(f"Refusing to overwrite existing sidecar without overwrite_sidecar=True: {sc_ass}")
            _atomic_write(sc_ass, ass_text)
            sidecars["ass"] = str(sc_ass)

    conflicts_summary = [
        {
            "utterance_id": c.utterance_id,
            "speaker": c.speaker,
            "lane": c.lane,
            "start": c.start,
            "end": c.end,
            "text": c.text,
            "conflicts": c.conflicts,
        }
        for c in displayed + overflow + unrenderable
        if c.conflicts
    ]
    # Actual event bounds are distinct from allocated display sub-cue bounds.
    by_id = {str(u.get("utterance_id") or u.get("id") or f"utt-{i:04d}"): u
             for i, u in enumerate(utterances)}
    def lineage_for(placed, excess, unavailable, *, srt=False):
        lineage = []
        for disposition, bucket in (("displayed", placed), ("overflow", excess),
                                    ("unrenderable", unavailable)):
            for cue in bucket:
                ids = cue.display_occurrence_ids or [cue.utterance_id]
                for uid in ids:
                    row = by_id.get(uid)
                    if row is None:
                        continue
                    actual = (connected_units.interval({
                        "source_start": row.get("raw_source_start"), "source_end": row.get("raw_source_end")})
                        if (row.get("timestamp_provenance") or {}).get("timing_source") == "viewing_validation_overlay"
                        else connected_units.interval(row))
                    lineage.append({"occurrence_id": uid,
                        "display_occurrence_ids": ids,
                        "display_event_id": cue.display_order,
                        "translation_unit_id": cue.translation_unit_id,
                        "unit_occurrence_ids": cue.unit_occurrence_ids or [uid],
                        "actual_event_interval": actual,
                        "effective_event_interval": connected_units.interval(row),
                        "timing_status": cue.timing_status,
                        "display_allocation_interval": [cue.raw_start, cue.raw_end],
                        "display_interval": ([cue.start, cue.end]
                                             if disposition == "displayed" or (srt and disposition == "overflow")
                                             else None),
                        "disposition": ("displayed" if srt and disposition == "overflow" else disposition),
                        "lane": None if srt else cue.lane, "text": cue.text,
                        "ass_position": None if srt else cue.ass_position,
                        "ass_bounds": None if srt else cue.ass_bounds,
                        "conflicts": list(cue.conflicts)})
        return lineage

    lineage = lineage_for(displayed, overflow, unrenderable)
    srt_lineage = lineage_for(srt_displayed, srt_overflow, srt_unrenderable, srt=True)
    return {
        "outputs": outputs,
        "sidecars": sidecars,
        "cue_lineage": lineage,
        "srt_cue_lineage": srt_lineage,
        "presentation": presentation,
        "no_renderable_dialogue":no_renderable_dialogue,
        "stats": {
            "total_input_utterances": len(utterances),
            "renderable_cues": len(all_renderable),
            "srt_displayed_cues": len(all_renderable),
            "srt_unrenderable_cues": len(srt_unrenderable),
            "ass_sequential_overlap_cues": sum("sequential_display_overlap" in c.conflicts for c in displayed),
            "ass_displayed_cues": len(displayed),
            "ass_lane_1_top_cues": sum(1 for c in displayed if c.lane == 1),
            "ass_overflow_cues": len(overflow),
            "unrenderable_cues": len(unrenderable),
            "conflict_cues": len(conflicts_summary),
            "ass_placement_conflict_cues": sum(any(x.startswith("placement_") for x in c.conflicts) for c in displayed + overflow),
        },
        "conflicts": conflicts_summary,
    }
