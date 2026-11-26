"""Standalone English SRT and ASS subtitle compositor with speaker-aware lanes."""

from __future__ import annotations

import os
import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    ass_layout: str = "standard"  # "standard" | "raised_main"
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
    chunks: list[str] = []
    idx = 0
    min_fill = max(1, int(budget * 0.35))
    while idx < len(words):
        cur: list[str] = []
        last_sent: int | None = None
        last_clause: int | None = None
        last_phrase: int | None = None
        while idx < len(words):
            w = words[idx]
            cand = " ".join(cur + [w])
            wrapped = wrap_subtitle_lines(cand, max_chars=max_chars, max_lines=max_lines)
            fits = len(cand) <= budget and len(wrapped) <= max_lines and all(len(ln) <= max_chars for ln in wrapped)
            if fits:
                cur.append(w)
                idx += 1
                stripped = w.rstrip("\"'»)]}”’")
                if stripped and len(cand) >= min_fill:
                    if stripped[-1] in _SENTENCE_END_CHARS:
                        last_sent = len(cur)
                    elif stripped[-1] in _CLAUSE_END_CHARS:
                        last_clause = len(cur)
                if len(cand) >= min_fill and idx < len(words) and words[idx].lower() in _PHRASE_BREAK_WORDS:
                    last_phrase = len(cur)
                continue
            if not cur:
                cur.append(w)
                idx += 1
            break
        if idx < len(words):
            split_at = last_sent or last_clause or last_phrase
            if split_at is not None and 0 < split_at < len(cur):
                idx -= len(cur) - split_at
                cur = cur[:split_at]
        chunks.append(" ".join(cur))
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
        text = _select_utterance_display_text(utt)
        if not text:
            continue

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

        # Allocate duration across sub-chunks with a floor so partial/short turns don't starve slivers
        span = max(0.25, raw_e - raw_s)
        raw_weights = [max(10.0, float(len(ch))) for ch in chunks]
        tot_w = sum(raw_weights)
        allocs = [span * (w / tot_w) for w in raw_weights]
        if len(chunks) > 1:
            floor_s = min(0.85, span / float(len(chunks)))
            capped_need = sum(max(0.0, floor_s - a) for a in allocs)
            if capped_need > 0:
                surplus = [max(0.0, a - floor_s) for a in allocs]
                tot_surp = sum(surplus)
                if tot_surp > 1e-6:
                    allocs = [
                        (floor_s if a < floor_s else a - capped_need * (s / tot_surp))
                        for a, s in zip(allocs, surplus)
                    ]

        cursor = raw_s
        for j, ch_text in enumerate(chunks):
            sub_s = cursor
            sub_e = raw_e if j == len(chunks) - 1 else min(raw_e, cursor + allocs[j])
            if sub_e <= sub_s:
                sub_e = round(sub_s + max(0.25, span / float(len(chunks))), 3)
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
                    raw_start=round(sub_s, 3),
                    raw_end=round(sub_e, 3),
                    timing_status=t_status,
                    unrenderable=False,
                    grouped_utterance_ids=grouped,
                    source_turn_ids=src_refs,
                    conflicts=conflicts,
                )
            )
    return cues


def _speaker_talk_ranking(utterances: Sequence[dict[str, Any]]) -> list[str]:
    totals: dict[str, float] = {}
    first_idx: dict[str, int] = {}
    for i, utt in enumerate(utterances):
        spk = str(utt.get("speaker_id") or utt.get("global_speaker_id") or "SPK_UNK")
        if spk not in first_idx:
            first_idx[spk] = i
            totals[spk] = 0.0
        s, e, _ = _resolve_utterance_timing(utt)
        if s is not None and e is not None:
            totals[spk] += max(0.0, e - s)
    return sorted(totals, key=lambda s: (-totals[s], first_idx[s]))


def _acoustic_bounds(cue: SubtitleCue) -> tuple[float, float]:
    if cue.raw_start is not None and cue.raw_end is not None:
        return cue.raw_start, cue.raw_end
    return cue.start, cue.end


def _acoustically_concurrent(a: SubtitleCue, b: SubtitleCue, overlap_threshold_s: float) -> bool:
    """Real overlap of a substantial fraction of a short voice is concurrency."""
    as0, ae0 = _acoustic_bounds(a)
    bs0, be0 = _acoustic_bounds(b)
    overlap = min(ae0, be0) - max(as0, bs0)
    shorter = min(ae0-as0, be0-bs0)
    return overlap > 0 and shorter > 0 and overlap >= min(
        max(0., overlap_threshold_s), .5 * shorter)


def _solo_fraction(cue: SubtitleCue, pool: Sequence[SubtitleCue], overlap_threshold_s: float) -> float:
    cs, ce = _acoustic_bounds(cue)
    tot = ce - cs
    if tot <= 0:
        return 1.0
    ov = 0.0
    for other in pool:
        if other is cue or other.speaker == cue.speaker:
            continue
        os0, oe0 = _acoustic_bounds(other)
        span = min(ce, oe0) - max(cs, os0)
        if _acoustically_concurrent(cue, other, overlap_threshold_s):
            ov += span
    return 1.0 - min(1.0, ov / tot)


def _norm_dup_cue_text(text: str) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _deduplicate_concurrent_echo_cues(
    cues: Sequence[SubtitleCue],
    *,
    prio: dict[str, int],
) -> tuple[list[SubtitleCue], list[SubtitleCue]]:
    """Remove only replayed render copies of the SAME source event.

    Similar English, approximate timing and even identical spoken sentences
    cannot establish a shared acoustic event: two real speakers can say the
    same phrase. Grouped source-turn lineage wins over legacy utterance IDs.
    """
    kept: list[SubtitleCue] = []
    suppressed: list[SubtitleCue] = []
    ordered = sorted(cues, key=lambda c: (_acoustic_bounds(c)[0], prio.get(c.speaker, len(prio))))
    for cue in ordered:
        current_refs = frozenset(cue.source_turn_ids or cue.grouped_utterance_ids or [cue.utterance_id])
        cs, ce = _acoustic_bounds(cue)
        duplicated = False
        if current_refs:
            for prev in kept:
                previous_refs = frozenset(prev.source_turn_ids or prev.grouped_utterance_ids or [prev.utterance_id])
                if current_refs != previous_refs or (cue.source_turn_ids and not prev.source_turn_ids) or (prev.source_turn_ids and not cue.source_turn_ids):
                    continue
                ps, pe = _acoustic_bounds(prev)
                if (abs(cs - ps) <= 0.1 and abs(ce - pe) <= 0.1
                        and _norm_dup_cue_text(cue.text) == _norm_dup_cue_text(prev.text)):
                    duplicated = True
                    break
        if duplicated:
            suppressed.append(cue)
        else:
            kept.append(cue)
    return kept, suppressed


def _schedule_single_lane(
    lane_cues: list[SubtitleCue],
    *,
    target_cps: float = 18.0,
    max_display_s: float = 6.5,
    min_gap_s: float = 0.04,
) -> None:
    """Resolve display overlaps and expand high-CPS cues into trailing free gaps safely."""
    lane_cues.sort(key=lambda c: (c.start, _acoustic_bounds(c)[0], c.end))
    min_positive_s = 0.15

    # Pass 1: resolve sequential display overlaps strictly without ever inverting start/end
    for i in range(len(lane_cues) - 1):
        prev = lane_cues[i]
        curr = lane_cues[i + 1]
        if prev.end > curr.start:
            _, prev_raw_end = _acoustic_bounds(prev)
            if curr.start >= prev_raw_end or (curr.start - prev.start) >= 0.35:
                new_end = round(max(prev.start + min_positive_s, curr.start), 3)
                if new_end > curr.start and curr.end > new_end + min_positive_s:
                    curr.start = new_end
                prev.end = new_end
                if curr.start < prev_raw_end:
                    prev.conflicts.append("display_clamped_before_next_cue")
            else:
                target = min(prev.end, max(prev.start + 0.35, prev_raw_end), curr.end - 0.25)
                if target >= prev.start + min_positive_s and (curr.end - target) >= 0.25:
                    prev.end = round(target, 3)
                    curr.start = round(target, 3)
                    prev.conflicts.append("display_split_subthreshold_overlap")
                    curr.conflicts.append("display_delayed_subthreshold_overlap")
                elif curr.start >= prev.start + min_positive_s:
                    prev.end = round(curr.start, 3)
                    prev.conflicts.append("display_clamped_before_next_cue")
                else:
                    # Both share virtually identical start (e.g. sub-threshold fragment on same speaker)
                    split_pt = round(min(curr.end - min_positive_s, prev.start + max(min_positive_s, 0.5 * (curr.end - prev.start))), 3)
                    if split_pt >= prev.start + min_positive_s and curr.end >= split_pt + min_positive_s:
                        prev.end = split_pt
                        curr.start = split_pt
                        prev.conflicts.append("display_split_subthreshold_overlap")
                        curr.conflicts.append("display_delayed_subthreshold_overlap")
                    else:
                        prev.conflicts.append("same_lane_collision_unresolved")

    # Pass 2: use remaining trailing free space safely to lower high reading speed without colliding
    for i, cue in enumerate(lane_cues):
        cur_dur = max(0.05, cue.end - cue.start)
        cur_cps = len(cue.text) / cur_dur
        if cur_cps > target_cps and cur_dur < max_display_s:
            desired_dur = min(max_display_s, len(cue.text) / max(10.0, target_cps))
            desired_end = cue.start + desired_dur
            if i + 1 < len(lane_cues):
                max_safe_end = max(cue.end, lane_cues[i + 1].start - min_gap_s)
                cue.end = round(min(desired_end, max_safe_end), 3)
            else:
                cue.end = round(desired_end, 3)

    # Final invariant: every cue must have strictly positive rounded duration (centisecond-safe >= 0.05s)
    for cue in lane_cues:
        cue.start = round(max(0.0, cue.start), 3)
        cue.end = round(max(cue.start + 0.05, cue.end), 3)


def assign_subtitle_lanes(
    cues: Sequence[SubtitleCue],
    *,
    utterances: Sequence[dict[str, Any]] = (),
    overlap_threshold_s: float = 0.4,
    max_lanes: int = 2,
    target_cps: float = 18.0,
    max_display_s: float = 6.5,
) -> tuple[list[SubtitleCue], list[SubtitleCue], list[SubtitleCue]]:
    """Assign simultaneous cues to bottom/top/(optional 3rd) ASS lanes; keep overflow & unrenderable."""
    unrenderable = [c for c in cues if c.unrenderable]
    raw_renderable = [c for c in cues if not c.unrenderable]
    ranking = _speaker_talk_ranking(utterances) if utterances else []
    prio = {spk: i for i, spk in enumerate(ranking)}

    renderable, _suppressed_dups = _deduplicate_concurrent_echo_cues(raw_renderable, prio=prio)

    def _rank(c: SubtitleCue) -> int:
        return prio.get(c.speaker, len(prio))

    num_lanes = max(2, min(3, int(max_lanes)))
    order = sorted(renderable, key=lambda c: (_acoustic_bounds(c)[0], _rank(c)))
    lane_last: dict[int, SubtitleCue | None] = {lid: None for lid in range(num_lanes)}
    displayed: list[SubtitleCue] = []
    overflow: list[SubtitleCue] = []

    for c in order:
        lc = SubtitleCue(
            utterance_id=c.utterance_id,
            start=c.start,
            end=c.end,
            text=c.text,
            speaker=c.speaker,
            raw_start=c.raw_start,
            raw_end=c.raw_end,
            lane=0,
            timing_status=c.timing_status,
            unrenderable=False,
            overflow=False,
            grouped_utterance_ids=list(c.grouped_utterance_ids),
            source_turn_ids=list(c.source_turn_ids),
            conflicts=list(c.conflicts),
        )

        def _free(lid: int) -> bool:
            # A later short cue on a lane cannot erase an earlier active cue.
            return not any(occ.lane == lid and occ.speaker != lc.speaker
                           and _acoustically_concurrent(occ, lc, overlap_threshold_s)
                           for occ in displayed)

        chosen_lane: int | None = None
        for lid in range(num_lanes):
            if _free(lid):
                chosen_lane = lid
                break

        if chosen_lane is None:
            lc.lane = 0
            lc.overflow = True
            lc.conflicts.append("three_way_overlap_ass_overflow")
            overflow.append(lc)
            continue

        lc.lane = chosen_lane
        if lc.lane == 1:
            occ0 = lane_last[0]
            if occ0 is not None and _acoustically_concurrent(occ0, lc, overlap_threshold_s):
                lc_solo = _solo_fraction(lc, renderable, overlap_threshold_s)
                occ0_solo = _solo_fraction(occ0, renderable, overlap_threshold_s)
                swap_safe = not any(
                    (other.lane == 1 and other.speaker != occ0.speaker
                     and _acoustically_concurrent(other, occ0, overlap_threshold_s))
                    or (other.lane == 0 and other is not occ0 and other.speaker != lc.speaker
                        and _acoustically_concurrent(other, lc, overlap_threshold_s))
                    for other in displayed)
                if swap_safe and (lc_solo > occ0_solo + 1e-6 or
                    (abs(lc_solo - occ0_solo) <= 1e-6 and _rank(lc) < _rank(occ0))):
                    occ0.lane = 1
                    lc.lane = 0
                    lane_last[0] = lc
                    lane_last[1] = occ0
                    displayed.append(lc)
                    continue

        lane_last[lc.lane] = lc
        displayed.append(lc)

    for lid in range(num_lanes):
        bucket = [c for c in displayed if c.lane == lid]
        _schedule_single_lane(bucket, target_cps=target_cps, max_display_s=max_display_s)

    # Refresh CPS conflict flags against final post-scheduled display durations
    for cue in displayed + overflow:
        final_cps = len(cue.text) / max(0.2, cue.end - cue.start)
        if final_cps > target_cps + 4.0:
            cue.conflicts.append(f"high_reading_speed:{final_cps:.1f}cps")

    displayed.sort(key=lambda c: (c.start, c.lane, _acoustic_bounds(c)[0]))
    overflow.sort(key=lambda c: (c.start, c.speaker))
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
    ordered = sorted((c for c in cues if not c.unrenderable), key=lambda c: (c.start, c.lane, c.end))
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
    ordered = sorted((c for c in lane_cues if not c.unrenderable), key=lambda c: (c.start, c.lane, c.end))
    for cue in ordered:
        style = "Main" if cue.lane == 0 else ("Top" if cue.lane == 1 else "TopAlt")
        wrapped = "\n".join(wrap_subtitle_lines(cue.text, max_chars=cfg.max_chars, max_lines=cfg.max_lines))
        lines.append(
            f"Dialogue: {cue.lane},{_ass_timestamp(cue.start)},{_ass_timestamp(cue.end)},{style},{cue.speaker},0,0,0,,{_ass_escape(wrapped)}"
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
) -> dict[str, Any]:
    """Render SRT and/or ASS subtitles directly from a translated or refined contract."""
    cfg = config or SubtitleConfig()
    utterances = contract.get("utterances") or []
    cues = build_display_cues(utterances, config=cfg)
    displayed, overflow, unrenderable = assign_subtitle_lanes(
        cues,
        utterances=utterances,
        overlap_threshold_s=cfg.overlap_threshold_s,
        max_lanes=cfg.ass_max_lanes,
        target_cps=cfg.target_cps,
        max_display_s=cfg.max_display_s,
    )

    # Limit final display times after all extension/scheduling. Floor the
    # media boundary to the common ASS/SRT time precision so serialization
    # cannot round a cue past the actual file end. Raw bounds stay unchanged.
    duration = _finite_nonneg((contract.get("source") or {}).get("duration_s"))
    if duration is not None:
        limit = math.floor(duration * 100 + 1e-9) / 100
        for bucket in (displayed, overflow):
            kept = []
            for cue in bucket:
                if cue.end > limit:
                    cue.end = limit
                    cue.conflicts.append("display_clamped_at_media_end")
                if cue.end <= cue.start or round(cue.end * 100) <= round(cue.start * 100):
                    cue.unrenderable = True
                    cue.conflicts.append("unrenderable_outside_media_duration")
                    unrenderable.append(cue)
                    continue
                # A final clamp can create a new reading-speed conflict.
                cue.conflicts = [c for c in cue.conflicts if not c.startswith("high_reading_speed:")]
                cps = len(cue.text) / (cue.end - cue.start)
                if cps > cfg.target_cps + 4:
                    cue.conflicts.append(f"high_reading_speed:{cps:.1f}cps")
                kept.append(cue)
            bucket[:] = kept
    all_renderable = list(displayed) + list(overflow)
    lang_tag = (target_language or "en").strip().lower()
    outputs: dict[str, str] = {}
    sidecars: dict[str, str] = {}
    fmts = {f.strip().lower() for f in formats if f.strip()}

    no_renderable_dialogue=bool(cues) and not all_renderable
    if "srt" in fmts:
        srt_text = format_srt(all_renderable, config=cfg)
        srt_out = output_dir / f"{episode_stem}.{lang_tag}.srt"
        _atomic_write(srt_out, srt_text)
        outputs["srt"] = str(srt_out)
        if sidecar_dir is not None and not no_renderable_dialogue:
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
        if sidecar_dir is not None and not no_renderable_dialogue:
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
        for c in all_renderable + unrenderable
        if c.conflicts
    ]
    return {
        "outputs": outputs,
        "sidecars": sidecars,
        "no_renderable_dialogue":no_renderable_dialogue,
        "stats": {
            "total_input_utterances": len(utterances),
            "renderable_cues": len(all_renderable),
            "ass_displayed_cues": len(displayed),
            "ass_lane_1_top_cues": sum(1 for c in displayed if c.lane == 1),
            "ass_overflow_cues": len(overflow),
            "unrenderable_cues": len(unrenderable),
            "conflict_cues": len(conflicts_summary),
        },
        "conflicts": conflicts_summary,
    }
