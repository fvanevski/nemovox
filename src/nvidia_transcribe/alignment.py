"""CTC observation diagnostics, OOV inspection, and selective external alignment."""

from __future__ import annotations

import difflib
import functools
import json
import unicodedata
from pathlib import Path
from typing import Any, Callable, Protocol

from nvidia_transcribe.checkpoint import DEFAULT_MODEL_PATH
from nvidia_transcribe.source_occurrences import associate_native_ctc
from nvidia_transcribe.parser import ParsedCTCAlignment, parse_ctc_timestamps, parse_speaker_turns, split_lexical_units


class ExternalForcedAligner(Protocol):
    """Pluggable interface for an external forced aligner fallback (future integration point)."""

    def align(
        self,
        *,
        audio_path: Path,
        text: str,
        language: str | None,
        source_offset_s: float,
    ) -> list[dict[str, Any]]:
        """Return aligned unit dicts with valid start/end or empty list on failure."""


@functools.lru_cache(maxsize=4)
def load_ctc_vocabulary_chars(model_path: str = str(DEFAULT_MODEL_PATH)) -> frozenset[str] | None:
    """Load the bundled CTC SentencePiece vocabulary character set from checkpoint config.json."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        vocab = ((cfg.get("ctc_timestamps") or {}).get("decoder_config") or {}).get("vocabulary") or []
        chars: set[str] = set()
        for tok in vocab:
            if not isinstance(tok, str):
                continue
            if tok.startswith("<") and tok.endswith(">"):
                continue
            for ch in tok.replace("▁", " "):
                chars.add(ch)
        return frozenset(chars) if chars else None
    except Exception:
        return None


def find_ctc_oov_characters(text: str, vocab_chars: frozenset[str] | None) -> list[str]:
    """Identify characters in recognized text absent from the CTC head's vocabulary."""
    if not vocab_chars or not text:
        return []
    turns = parse_speaker_turns(text)
    clean = "".join(t.text for t in turns)
    missing: list[str] = []
    seen: set[str] = set()
    for ch in clean:
        if ch.isspace() or ch in ".,!?;:'\"-—–()[]{}«»“”‘’、，。！？；：（）《》【】":
            continue
        if ch not in vocab_chars and ch not in seen:
            seen.add(ch)
            missing.append(ch)
    return missing


def group_diarization_windows(
    diarization: list[dict[str, Any]],
    *,
    max_gap_s: float = 0.6,
    pad_s: float = 0.35,
    max_duration_s: float = 20.0,
) -> list[tuple[float, float]]:
    """Merge Sortformer diarization intervals into bounded speech sub-windows for localized retry."""
    valid = [
        (float(d["start"]), float(d["end"]))
        for d in diarization
        if d.get("valid_timing") and d.get("start") is not None and d.get("end") is not None
    ]
    if not valid:
        return []
    valid.sort()
    merged: list[list[float]] = []
    for s, e in valid:
        if not merged:
            merged.append([max(0.0, s - pad_s), e + pad_s])
        else:
            prev = merged[-1]
            if s - prev[1] <= max_gap_s and (e + pad_s - prev[0]) <= max_duration_s:
                prev[1] = max(prev[1], e + pad_s)
            else:
                merged.append([max(0.0, s - pad_s), e + pad_s])
    return [(round(a, 4), round(b, 4)) for a, b in merged]


def _norm_align_tok(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "").lower()
    return "".join(ch for ch in cleaned if not ch.isspace() and unicodedata.category(ch)[0] not in ("P", "S"))


def _build_original_unit_skeleton(
    raw_text: str,
    *,
    chunk_index: int,
    timing_error: str | None,
    has_diarization: bool,
) -> list[dict[str, Any]]:
    """Build an immutable ordered lexical skeleton from original recognized raw_text."""
    turns = parse_speaker_turns(raw_text)
    skeleton: list[dict[str, Any]] = []
    for turn in turns:
        raw_tag = turn.speaker_tag if turn.speaker_tag.startswith("<spk:") else None
        spk_val = raw_tag.removeprefix("<spk:").removesuffix(">") if raw_tag else None
        clean_turn_text = turn.text or ""
        cursor = 0
        first_in_turn = True
        for token_position, lex in enumerate(split_lexical_units(clean_turn_text)):
            tok = lex["text"]
            found = clean_turn_text.find(tok, cursor)
            join_no_space = False
            if found >= 0:
                gap = clean_turn_text[cursor:found]
                join_no_space = (not first_in_turn) and (gap == "")
                cursor = found + len(tok)
            first_in_turn = False
            # Attach pure trailing punctuation directly onto preceding unit so CJK sentences like "好。"
            # stay unified with CTC word tokens and do not create spurious untimed punctuation gaps.
            if skeleton and _norm_align_tok(tok) == "" and all(ch in ".,!?;:、，。！？；：）］】》」』”’'\"" for ch in tok):
                if skeleton[-1]["recognized_turn_index"] == turn.turn_index:
                    skeleton[-1]["text"] += tok
                    skeleton[-1]["recognized_token_positions"].append(token_position)
                    continue
            skeleton.append(
                {
                    "text": tok,
                    "start": None,
                    "end": None,
                    "speaker": spk_val,
                    "raw_speaker_tag": raw_tag,
                    "unit_type": lex["unit_type"],
                    "valid_timing": False,
                    "timing_error": timing_error or "ctc_alignment_unavailable",
                    "raw_start": None,
                    "raw_end": None,
                    "chunk_index": chunk_index,
                    "source_unit_index": len(skeleton),
                    "recognized_turn_index": turn.turn_index,
                    "recognized_token_positions": [token_position],
                    "join_no_space_prev": join_no_space,
                    "alignment_status": "diarization_bounded_only" if has_diarization else "untimed",
                }
            )
    return skeleton


def annotate_units_with_source_spacing(units: list[dict[str, Any]], raw_text: str) -> list[dict[str, Any]]:
    """Attach ``join_no_space_prev`` and ``source_unit_index`` using recognized ``raw_text``."""
    turns = parse_speaker_turns(raw_text)
    clean_full = " ".join(t.text for t in turns if t.text)
    # Match per-speaker cursor when multiple speakers are interleaved or single stream
    by_spk_cursor: dict[str | None, tuple[str, int]] = {}
    if len(turns) > 1:
        for spk_key in {
            (t.speaker_tag.removeprefix("<spk:").removesuffix(">") if t.speaker_tag.startswith("<spk:") else None)
            for t in turns
        }:
            joined = " ".join(
                t.text
                for t in turns
                if (t.speaker_tag.removeprefix("<spk:").removesuffix(">") if t.speaker_tag.startswith("<spk:") else None)
                == spk_key
                and t.text
            )
            by_spk_cursor[spk_key] = (joined, 0)
    by_spk_cursor[None] = (clean_full, 0)

    out: list[dict[str, Any]] = []
    for idx, u in enumerate(units):
        nu = {**u}
        nu.setdefault("source_unit_index", idx)
        if "join_no_space_prev" not in nu:
            tok = str(nu.get("text") or "").strip()
            spk = nu.get("speaker")
            key = spk if spk in by_spk_cursor else None
            haystack, cur = by_spk_cursor[key]
            found = haystack.find(tok, cur) if tok else -1
            if found >= 0:
                gap = haystack[cur:found]
                nu["join_no_space_prev"] = (cur > 0) and (gap == "")
                by_spk_cursor[key] = (haystack, found + len(tok))
        out.append(nu)
    return out


def _is_sentence_end_tok(tok: str) -> bool:
    t = (tok or "").strip()
    return bool(t) and t[-1] in ".?!。？！"


def _join_units_for_aligner(units: list[dict[str, Any]]) -> str:
    """Reconstruct natural source span text using join_no_space_prev provenance."""
    parts: list[str] = []
    for u in units:
        tok = str(u.get("text") or "").strip()
        if not tok:
            continue
        if not parts:
            parts.append(tok)
        elif u.get("join_no_space_prev") is True:
            parts.append(tok)
        else:
            parts.append(" " + tok)
    return "".join(parts).strip()


def _contiguous_speaker_spans(orig_units: list[dict[str, Any]], gap_split_s: float = 0.65) -> list[tuple[int, int]]:
    """Partition chunk units into contiguous same-speaker sentence/clause index ranges [start, end)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(orig_units)):
        prev = orig_units[i - 1]
        cur = orig_units[i]
        split = False
        if cur.get("speaker") != prev.get("speaker"):
            split = True
        elif _is_sentence_end_tok(str(prev.get("text") or "")):
            split = True
        elif (
            prev.get("valid_timing")
            and cur.get("valid_timing")
            and prev.get("end") is not None
            and cur.get("start") is not None
            and (float(cur["start"]) - float(prev["end"])) > gap_split_s
        ):
            split = True
        if split:
            spans.append((start, i))
            start = i
    if orig_units:
        spans.append((start, len(orig_units)))
    return spans


def _has_mixed_speaker_overlap(
    diar_list: list[dict[str, Any]],
    raw_text: str,
    min_overlap_s: float = 0.25,
    window: tuple[float, float] | None = None,
) -> bool:
    """Detect whether multiple speakers talk simultaneously in the chunk or localized window."""
    spk_tags = {t.speaker_tag for t in parse_speaker_turns(raw_text) if t.speaker_tag.startswith("<spk:") and t.text.strip()}
    valid_segs: list[tuple[float, float, str]] = []
    for d in diar_list:
        if not (d.get("valid_timing") and d.get("start") is not None and d.get("end") is not None):
            continue
        s, e = float(d["start"]), float(d["end"])
        if window is not None:
            s = max(window[0], s)
            e = min(window[1], e)
            if e <= s:
                continue
        valid_segs.append((s, e, str(d.get("speaker"))))
    valid_segs.sort()
    for i, (s1, e1, sp1) in enumerate(valid_segs):
        for s2, e2, sp2 in valid_segs[i + 1 :]:
            if s2 >= e1:
                break
            if sp1 != sp2 and (min(e1, e2) - max(s1, s2)) >= min_overlap_s:
                return True
    return len(spk_tags) >= 2 and len({sp for _, _, sp in valid_segs}) >= 2 and any(
        min(e1, e2) - max(s1, s2) > 0.15
        for i, (s1, e1, sp1) in enumerate(valid_segs)
        for s2, e2, sp2 in valid_segs[i + 1 :]
        if sp1 != sp2
    )


def _apply_verified_sequence_timings(
    orig_units: list[dict[str, Any]],
    candidate_units: list[dict[str, Any]],
    *,
    recovered_status: str,
    only_untimed: bool = False,
    target_range: tuple[int, int] | None = None,
    min_start_s: float | None = None,
    max_end_s: float | None = None,
    enforce_neighbor_monotonic: bool = False,
) -> dict[str, Any]:
    """Attach timestamps ONLY to verified lexically matching units of ``orig_units``.

    Never mutates ``orig_units[i]['text']``, never inserts extra tokens from candidates,
    and rejects out-of-window or non-monotonic timestamps before mutating any unit.
    """
    lo, hi = target_range if target_range is not None else (0, len(orig_units))
    slice_units = orig_units[lo:hi]
    orig_norms = [_norm_align_tok(u.get("text", "")) for u in slice_units]
    cand_norms = [_norm_align_tok(u.get("text", "")) for u in candidate_units]

    matcher = difflib.SequenceMatcher(None, orig_norms, cand_norms, autojunk=False)
    matched_count = 0
    rejected_bounds_count = 0
    substitutions: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []
    insertions: list[dict[str, Any]] = []

    # Precompute closest valid same-speaker bounds around any index in slice_units when requested.
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for si, cj in zip(range(i1, i2), range(j1, j2)):
                oi = lo + si
                if not orig_norms[si]:
                    continue
                if only_untimed and orig_units[oi].get("valid_timing"):
                    continue
                cu = candidate_units[cj]
                if cu.get("valid_timing") and cu.get("start") is not None and cu.get("end") is not None:
                    st_f = float(cu["start"])
                    en_f = float(cu["end"])
                    if en_f <= st_f:
                        rejected_bounds_count += 1
                        continue
                    if min_start_s is not None and st_f < min_start_s - 0.15:
                        orig_units[oi]["timing_error"] = "external_bounds_violation"
                        rejected_bounds_count += 1
                        continue
                    if max_end_s is not None and en_f > max_end_s + 0.15:
                        orig_units[oi]["timing_error"] = "external_bounds_violation"
                        rejected_bounds_count += 1
                        continue
                    if enforce_neighbor_monotonic:
                        spk = orig_units[oi].get("speaker")
                        left_lb = None
                        for k in range(oi - 1, -1, -1):
                            if orig_units[k].get("speaker") == spk and orig_units[k].get("valid_timing") and orig_units[k].get("start") is not None:
                                left_lb = float(orig_units[k]["start"]) - 0.05
                                break
                        right_ub = None
                        for k in range(oi + 1, len(orig_units)):
                            if orig_units[k].get("speaker") == spk and orig_units[k].get("valid_timing") and orig_units[k].get("end") is not None:
                                right_ub = float(orig_units[k]["end"]) + 0.05
                                break
                        if (left_lb is not None and st_f < left_lb) or (right_ub is not None and en_f > right_ub):
                            orig_units[oi]["timing_error"] = "external_bounds_violation"
                            rejected_bounds_count += 1
                            continue
                    orig_units[oi]["start"] = st_f
                    orig_units[oi]["end"] = en_f
                    orig_units[oi]["raw_start"] = cu.get("raw_start", st_f)
                    orig_units[oi]["raw_end"] = cu.get("raw_end", en_f)
                    orig_units[oi]["valid_timing"] = True
                    orig_units[oi]["timing_error"] = None
                    orig_units[oi]["alignment_status"] = recovered_status
                    matched_count += 1
        elif tag == "replace":
            substitutions.append(
                {
                    "original_tokens": [slice_units[k]["text"] for k in range(i1, i2)],
                    "candidate_tokens": [candidate_units[k]["text"] for k in range(j1, j2)],
                }
            )
        elif tag == "delete":
            omissions.append({"original_tokens": [slice_units[k]["text"] for k in range(i1, i2)]})
        elif tag == "insert":
            insertions.append({"candidate_tokens": [candidate_units[k]["text"] for k in range(j1, j2)]})

    return {
        "matched_units": matched_count,
        "rejected_bounds_count": rejected_bounds_count,
        "substitutions": substitutions,
        "omissions": omissions,
        "insertions": insertions,
    }


def recover_chunk_alignment(
    *,
    raw_text: str,
    ctc_payload: Any,
    source_offset_s: float,
    chunk_index: int,
    model_path: str | None = str(DEFAULT_MODEL_PATH),
    external_aligner: ExternalForcedAligner | None = None,
    chunk_wav_path: Path | None = None,
    source_language: str | None = None,
    subinterval_wav_extractor: Callable[[float, float], Path] | None = None,
) -> dict[str, Any]:
    """Validate CTC timestamps and apply non-destructive recovery on failure.

    Invariants:
    - Original recognized ``raw_text`` is NEVER rewritten, substituted, or stripped of OOV chars.
    - Native observations remain separate from the recognized spine; optional external
      alignment attaches timings only to verified matching original units.
    - Missing/failed timestamps are NEVER fabricated as 0.0 or interpolated silently.
    """
    parsed: ParsedCTCAlignment = parse_ctc_timestamps(
        ctc_payload, source_offset_s=source_offset_s, strict=False
    )
    vocab_chars = load_ctc_vocabulary_chars(model_path) if model_path is not None else None
    vocab_status = ("not_configured" if model_path is None else
                    ("loaded" if vocab_chars else "unavailable"))
    oov_chars = find_ctc_oov_characters(raw_text, vocab_chars)

    # The original recognized text is the lexical spine for every CTC outcome.
    diar_list = [d.to_dict() for d in parsed.diarization]
    orig_units = _build_original_unit_skeleton(
        raw_text, chunk_index=chunk_index, timing_error=parsed.error,
        has_diarization=bool(diar_list),
    )
    native_observations = associate_native_ctc(orig_units, parsed, ctc_payload)
    if parsed.error is None and parsed.units:
        untimed = sum(not u["valid_timing"] for u in orig_units)
        unresolved = sum("source_unit_index" not in o for o in native_observations)
        complete = not (untimed or unresolved or parsed.invalid_timing_count)
        return {
            "units": orig_units,
            "diarization": diar_list,
            "speaker_mapping": dict(parsed.speaker_mapping),
            "alignment_status": "ctc_aligned" if complete else "partial",
            "diagnostic": {
                "chunk_index": chunk_index,
                "status": "ok" if complete else "partial_invalid_units",
                "ctc_error": None,
                "oov_characters": oov_chars,
                "ctc_vocabulary_status": vocab_status,
                "invalid_timing_count": parsed.invalid_timing_count,
                "recovery_attempted": False,
                "recovered_units": 0,
                "untimed_units": untimed,
                "native_ctc_observations": native_observations,
                "unassociated_ctc_observations": unresolved,
            },
        }

    sub_windows = group_diarization_windows(diar_list)
    recovery_attempted = False
    # Optional external aligner hook (runs ONLY per localized untimed span; never replaces native CTC timings).
    ext_diag: dict[str, Any] | None = None
    ext_recovered_count = 0
    if external_aligner is not None and chunk_wav_path is not None and orig_units:
        spans = _contiguous_speaker_spans(orig_units)
        untimed_spans = [
            (lo, hi) for lo, hi in spans if any(not orig_units[k].get("valid_timing") for k in range(lo, hi))
        ]
        if untimed_spans:
            valid_diar = sorted(
                (float(d["start"]), float(d["end"]))
                for d in diar_list
                if d.get("valid_timing") and d.get("start") is not None and d.get("end") is not None
            )
            span_reports: list[dict[str, Any]] = []
            last_backend_diag: dict[str, Any] = {}
            any_skipped_overlap = False

            for lo, hi in untimed_spans:
                span_units = orig_units[lo:hi]
                span_spk = span_units[0].get("speaker")
                span_text = _join_units_for_aligner(span_units)
                if not span_text:
                    continue

                inside_timed = [
                    (float(u["start"]), float(u["end"]))
                    for u in span_units
                    if u.get("valid_timing") and u.get("start") is not None and u.get("end") is not None
                ]
                left_spk_end = next(
                    (
                        float(orig_units[k]["end"])
                        for k in range(lo - 1, -1, -1)
                        if orig_units[k].get("speaker") == span_spk
                        and orig_units[k].get("valid_timing")
                        and orig_units[k].get("end") is not None
                    ),
                    None,
                )
                right_spk_start = next(
                    (
                        float(orig_units[k]["start"])
                        for k in range(hi, len(orig_units))
                        if orig_units[k].get("speaker") == span_spk
                        and orig_units[k].get("valid_timing")
                        and orig_units[k].get("start") is not None
                    ),
                    None,
                )

                win_start: float | None = None
                win_end: float | None = None
                use_full_chunk_wav = False

                if inside_timed:
                    w0 = min(s for s, _ in inside_timed)
                    w1 = max(e for _, e in inside_timed)
                    has_leading_untimed = not span_units[0].get("valid_timing")
                    has_trailing_untimed = not span_units[-1].get("valid_timing")
                    matching_diar = [(ds, de) for ds, de in valid_diar if min(w1, de) - max(w0, ds) >= -0.35]

                    if has_leading_untimed:
                        cand_s = min([w0 - 1.2] + [ds - 0.25 for ds, _ in matching_diar])
                        if left_spk_end is not None:
                            cand_s = max(left_spk_end - 0.05, cand_s)
                    else:
                        cand_s = w0 - 0.35

                    if has_trailing_untimed:
                        cand_e = max([w1 + 1.2] + [de + 0.25 for _, de in matching_diar])
                        if right_spk_start is not None:
                            cand_e = min(right_spk_start + 0.05, cand_e)
                    else:
                        cand_e = w1 + 0.35

                    win_start = round(max(source_offset_s, cand_s), 4)
                    win_end = round(max(win_start + 0.4, cand_e), 4)
                elif len(orig_units) == (hi - lo) and len(sub_windows) <= 1:
                    # Single contiguous span covering the entire clip/chunk (e.g. short clip OOV).
                    use_full_chunk_wav = True
                    win_start = source_offset_s
                    win_end = valid_diar[-1][1] + 1.0 if valid_diar else None
                elif left_spk_end is not None and right_spk_start is not None and right_spk_start > left_spk_end + 0.15:
                    win_start = round(max(source_offset_s, left_spk_end - 0.15), 4)
                    win_end = round(right_spk_start + 0.15, 4)
                elif len(sub_windows) == len(untimed_spans) and len(sub_windows) > 0:
                    s_idx_pos = untimed_spans.index((lo, hi))
                    win_start, win_end = sub_windows[s_idx_pos]
                    win_start = max(source_offset_s, win_start)
                if win_start is None or (win_end is None and not use_full_chunk_wav):
                    if _has_mixed_speaker_overlap(diar_list, raw_text):
                        any_skipped_overlap = True
                        span_reports.append(
                            {"span_range": [lo, hi], "text": span_text, "status": "skipped_mixed_speaker_overlap"}
                        )
                    else:
                        span_reports.append(
                            {"span_range": [lo, hi], "text": span_text, "status": "skipped_unlocalizable_span"}
                        )
                    continue

                check_window = (win_start, win_end) if win_end is not None else None
                if _has_mixed_speaker_overlap(diar_list, raw_text, window=check_window):
                    any_skipped_overlap = True
                    span_reports.append(
                        {
                            "span_range": [lo, hi],
                            "window": [win_start, win_end],
                            "text": span_text,
                            "status": "skipped_mixed_speaker_overlap",
                        }
                    )
                    continue

                target_wav = chunk_wav_path
                target_offset = source_offset_s
                if not use_full_chunk_wav and win_end is not None and subinterval_wav_extractor is not None:
                    target_wav = subinterval_wav_extractor(win_start, win_end)
                    target_offset = win_start

                recovery_attempted = True
                ext_out = external_aligner.align(
                    audio_path=target_wav,
                    text=span_text,
                    language=source_language,
                    source_offset_s=target_offset,
                )
                if isinstance(ext_out, dict):
                    ext_units = list(ext_out.get("units") or [])
                    span_diag = dict(ext_out.get("diagnostic") or {})
                else:
                    ext_units = list(ext_out or [])
                    span_diag = {}
                last_backend_diag = span_diag

                matched_here = 0
                rejected_here = 0
                if ext_units:
                    ext_diff = _apply_verified_sequence_timings(
                        orig_units,
                        ext_units,
                        recovered_status="external_qwen_recovered",
                        only_untimed=True,
                        target_range=(lo, hi),
                        min_start_s=win_start,
                        max_end_s=win_end,
                        enforce_neighbor_monotonic=True,
                    )
                    matched_here = ext_diff["matched_units"]
                    rejected_here = ext_diff.get("rejected_bounds_count", 0)
                    ext_recovered_count += matched_here

                span_reports.append(
                    {
                        "span_range": [lo, hi],
                        "window": [win_start, win_end],
                        "text": span_text,
                        "matched_untimed_units": matched_here,
                        "rejected_bounds_units": rejected_here,
                        "aligner_diagnostic": span_diag,
                    }
                )

            if last_backend_diag:
                ext_diag = {**last_backend_diag, "localized_spans": span_reports}
            elif any_skipped_overlap:
                ext_diag = {
                    "backend": "qwen3_forced_aligner",
                    "status": "skipped_mixed_speaker_overlap",
                    "reason": "linear_forced_aligner_cannot_disentangle_simultaneous_voices",
                    "localized_spans": span_reports,
                }
            elif span_reports:
                ext_diag = {
                    "backend": "qwen3_forced_aligner",
                    "status": span_reports[-1].get("status", "skipped"),
                    "localized_spans": span_reports,
                }

            untimed_after = sum(1 for u in orig_units if not u.get("valid_timing"))
            if ext_recovered_count > 0:
                status = "external_recovered" if untimed_after == 0 else "partial_external_recovery"
                return {
                    "units": orig_units,
                    "diarization": diar_list,
                    "speaker_mapping": dict(parsed.speaker_mapping),
                    "alignment_status": status,
                    "diagnostic": {
                        "chunk_index": chunk_index,
                        "status": status,
                        "native_ctc_observations": native_observations,
                        "ctc_error": parsed.error,
                        "oov_characters": oov_chars,
                        "ctc_vocabulary_status": vocab_status,
                        "failure_cause": "ctc_vocab_oov" if oov_chars else parsed.error,
                        "recovery_attempted": True,
                        "recovered_units": ext_recovered_count,
                        "external_recovered_units": ext_recovered_count,
                        "untimed_units": untimed_after,
                        "external_aligner": ext_diag,
                        "diarization_interval_count": len(diar_list),
                    },
                }

    status = "diarization_bounded_only" if diar_list else "untimed"
    return {
        "units": orig_units,
        "diarization": diar_list,
        "speaker_mapping": dict(parsed.speaker_mapping),
        "alignment_status": status,
        "diagnostic": {
            "chunk_index": chunk_index,
            "status": status,
            "native_ctc_observations": native_observations,
            "ctc_error": parsed.error or "missing_ctc_metadata",
            "oov_characters": oov_chars,
            "ctc_vocabulary_status": vocab_status,
            "failure_cause": "ctc_vocab_oov" if oov_chars else (parsed.error or "missing_ctc_metadata"),
            "recovery_attempted": recovery_attempted,
            "recovered_units": ext_recovered_count,
            "untimed_units": sum(1 for u in orig_units if not u.get("valid_timing")),
            "external_aligner": ext_diag,
            "diarization_interval_count": len(diar_list),
            "external_aligner_hook": "ExternalForcedAligner(audio_path, text, language, source_offset_s)",
        },
    }
