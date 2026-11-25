"""Per-speaker lexical utterance reconstruction preserving simultaneous overlap streams."""

from __future__ import annotations

import unicodedata
from typing import Any

from nvidia_transcribe.parser import EVENT_TAG_RE, STYLE_TAG_RE


def _is_cjk_char(ch: str) -> bool:
    if not ch or ch.isspace():
        return False
    return unicodedata.name(ch, "").startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))


def _is_cjk_punctuation(ch: str) -> bool:
    return ch in "，。！？；：、（）「」『』《》【】“”‘’"


def _is_latin_alnum(ch: str) -> bool:
    if not ch or ch.isspace() or _is_cjk_char(ch) or _is_cjk_punctuation(ch):
        return False
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")


def join_multilingual_units(units: list[dict[str, Any]]) -> str:
    """Reconstruct natural source text using source spacing provenance and script rules."""
    parts: list[str] = []
    prev_is_cjk = False
    prev_text = ""
    for u in units:
        tok = str(u.get("text") or "").strip()
        if not tok:
            continue
        if EVENT_TAG_RE.fullmatch(tok) or STYLE_TAG_RE.fullmatch(tok):
            continue
        first_ch = tok[0]
        last_ch = tok[-1]
        cur_is_cjk = _is_cjk_char(first_ch) or _is_cjk_punctuation(first_ch)
        explicit_no_space = u.get("join_no_space_prev")

        if not parts:
            parts.append(tok)
        elif explicit_no_space is True:
            parts.append(tok)
        elif explicit_no_space is False:
            # Explicit space provenance from recognized source text (preserve e.g. "A B C" or "我用 Python")
            if _is_cjk_punctuation(first_ch) or (len(tok) == 1 and tok in ",.!?;:"):
                parts.append(tok)
            else:
                parts.append(" " + tok)
        elif cur_is_cjk and prev_is_cjk:
            parts.append(tok)
        elif _is_cjk_punctuation(first_ch):
            parts.append(tok)
        elif prev_is_cjk and len(tok) == 1 and tok in ",.!?;:":
            parts.append(tok)
        elif len(tok) == 1 and tok in ",.!?;:":
            parts.append(tok)
        elif prev_text.endswith(("'", "’", "-")):
            parts.append(tok)
        else:
            parts.append(" " + tok)
        prev_is_cjk = _is_cjk_char(last_ch) or _is_cjk_punctuation(last_ch)
        prev_text = tok
    return "".join(parts).strip()


def _ends_sentence(token: str) -> bool:
    t = (token or "").strip()
    return bool(t) and t[-1] in ".?!。？！"


def _speaker_key(unit: dict[str, Any]) -> str:
    gid = unit.get("global_speaker_id")
    if gid:
        return str(gid)
    lid = unit.get("local_speaker_id")
    if lid:
        return str(lid)
    spk = unit.get("speaker")
    if spk is not None and str(spk).strip() != "":
        return str(spk)
    return "unknown"


def _compute_speaker_sort_keys(
    pairs: list[tuple[int, dict[str, Any]]],
    *,
    chunk_windows: dict[int, tuple[float, float]] | None = None,
) -> list[tuple[float, int, int, int, dict[str, Any]]]:
    """Compute order-preserving sort keys for one speaker's timed & untimed units.

    Untimed units are anchored strictly between neighboring timed units or within
    their owning chunk's acoustic bounds instead of an arbitrary ``cidx * 1000.0`` jump.
    """
    if not pairs:
        return []

    by_chunk: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for orig_idx, u in pairs:
        cidx = int(u.get("chunk_index") or 0)
        by_chunk.setdefault(cidx, []).append((orig_idx, u))

    keyed: list[tuple[float, int, int, int, dict[str, Any]]] = []
    for cidx in sorted(by_chunk):
        chunk_pairs = sorted(
            by_chunk[cidx],
            key=lambda p: (
                int(p[1]["source_unit_index"]) if p[1].get("source_unit_index") is not None else p[0],
                p[0],
            ),
        )
        n = len(chunk_pairs)
        valid_t: list[float | None] = [
            float(u["start"]) if (u.get("valid_timing") and u.get("start") is not None) else None
            for _, u in chunk_pairs
        ]
        next_t: list[float | None] = [None] * n
        cur_next: float | None = None
        for i in range(n - 1, -1, -1):
            if valid_t[i] is not None:
                cur_next = valid_t[i]
            next_t[i] = cur_next

        win_s, win_e = (chunk_windows or {}).get(cidx, (cidx * 80.0, cidx * 80.0 + 90.0))
        prev_t: float | None = None
        for i, (orig_idx, u) in enumerate(chunk_pairs):
            src_seq = int(u["source_unit_index"]) if u.get("source_unit_index") is not None else orig_idx
            vt = valid_t[i]
            if vt is not None:
                # Never allow an anomalous earlier timestamp inside the same chunk+speaker
                # stream to scramble recognized lexical emission order.
                if prev_t is not None and vt + 0.05 < prev_t:
                    anchor = prev_t + 1e-6 * (i + 1)
                else:
                    anchor = vt
                    prev_t = vt
            else:
                nxt = next_t[i]
                if prev_t is not None and nxt is not None:
                    span = max(0.0, nxt - prev_t)
                    anchor = prev_t + min(0.5 * span, 1e-4 * (i + 1)) if span > 0 else prev_t + 1e-6 * (i + 1)
                elif prev_t is not None:
                    anchor = prev_t + 1e-4 * (i + 1)
                elif nxt is not None:
                    anchor = max(0.0, nxt - 1e-4 * (n - i))
                else:
                    # Strictly inside owning chunk's window without inventing utterance start/end
                    frac = (i + 1) / float(n + 1)
                    anchor = win_s + frac * max(1.0, win_e - win_s)
            keyed.append((anchor, cidx, src_seq, orig_idx, u))

    keyed.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return keyed


def _extract_timing_gaps(grp: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize contiguous untimed unit runs inside an utterance."""
    gaps: list[dict[str, Any]] = []
    i = 0
    while i < len(grp):
        u = grp[i]
        is_timed = bool(u.get("valid_timing") and u.get("start") is not None and u.get("end") is not None)
        if is_timed:
            i += 1
            continue
        j = i
        while j < len(grp):
            uj = grp[j]
            if bool(uj.get("valid_timing") and uj.get("start") is not None and uj.get("end") is not None):
                break
            j += 1
        prev_end = grp[i - 1].get("end") if i > 0 and grp[i - 1].get("valid_timing") else None
        next_start = grp[j].get("start") if j < len(grp) and grp[j].get("valid_timing") else None
        gaps.append(
            {
                "unit_start_index": i,
                "unit_end_index": j - 1,
                "untimed_count": j - i,
                "text": join_multilingual_units(grp[i:j]),
                "bounded_after_s": prev_end,
                "bounded_before_s": next_start,
                "reason": str(grp[i].get("timing_error") or grp[i].get("alignment_status") or "untimed"),
            }
        )
        i = j
    return gaps


def _infer_chunk_windows(
    units: list[dict[str, Any]],
    explicit: dict[int, tuple[float, float]] | None = None,
) -> dict[int, tuple[float, float]]:
    """Infer chronological [start, end] bounds per chunk_index without inventing timestamps."""
    windows: dict[int, tuple[float, float]] = dict(explicit or {})
    by_chunk_t: dict[int, list[float]] = {}
    all_chunks: set[int] = set()
    for u in units:
        if u.get("chunk_index") is not None:
            cidx = int(u["chunk_index"])
            all_chunks.add(cidx)
            if u.get("valid_timing") and u.get("start") is not None:
                by_chunk_t.setdefault(cidx, []).append(float(u["start"]))
            if u.get("valid_timing") and u.get("end") is not None:
                by_chunk_t.setdefault(cidx, []).append(float(u["end"]))
    for cidx in sorted(all_chunks):
        if cidx in windows:
            continue
        ts = by_chunk_t.get(cidx)
        if ts:
            windows[cidx] = (min(ts), max(ts))
    # Fill remaining untimed chunks smoothly between neighboring chunk windows
    known = sorted(windows)
    for cidx in sorted(all_chunks):
        if cidx in windows:
            continue
        prev_k = max((k for k in known if k < cidx), default=None)
        next_k = min((k for k in known if k > cidx), default=None)
        if prev_k is not None and next_k is not None:
            ps, pe = windows[prev_k]
            ns, ne = windows[next_k]
            frac = (cidx - prev_k) / float(next_k - prev_k)
            ws = pe + frac * max(0.0, ns - pe)
            windows[cidx] = (ws, ws + max(10.0, 0.25 * (ne - pe)))
        elif prev_k is not None:
            _, pe = windows[prev_k]
            windows[cidx] = (pe + 0.5, pe + 60.0)
        elif next_k is not None:
            ns, _ = windows[next_k]
            windows[cidx] = (max(0.0, ns - 60.0), max(1.0, ns - 0.5))
        else:
            windows[cidx] = (cidx * 80.0, cidx * 80.0 + 90.0)
    return windows


def reconstruct_speaker_utterances(
    units: list[dict[str, Any]],
    *,
    source_language: str | None = None,
    pause_threshold_s: float = 0.85,
    sentence_pause_threshold_s: float = 0.35,
    max_utterance_duration_s: float = 16.0,
    max_units_per_utterance: int = 48,
    chunk_windows: dict[int, tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct coherent utterances independently per speaker stream.

    Preserves exact intra-speaker lexical order even when units mix timed and
    untimed states, classifies full vs partial vs failed timing honestly, and
    keeps simultaneous speakers concurrent without mutating missing timestamps.
    """
    if not units:
        return []

    indexed_units = [
        (idx, {**u})
        for idx, u in enumerate(units)
        if not EVENT_TAG_RE.fullmatch(str(u.get("text") or "").strip())
        and not STYLE_TAG_RE.fullmatch(str(u.get("text") or "").strip())
    ]

    inferred_windows = _infer_chunk_windows([u for _, u in indexed_units], chunk_windows)

    # Cross-speaker per-chunk emission ordering: anchor any untimed unit to its nearest
    # timed neighbor (any speaker!) inside the same chunk so missing-timing chunks stay
    # in true chronological order rather than jumping to chunk_index * 1000.0.
    units_by_chunk: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for orig_idx, u in indexed_units:
        cidx = int(u.get("chunk_index") or 0)
        units_by_chunk.setdefault(cidx, []).append((orig_idx, u))
    cross_spk_order_t: dict[int, float] = {}
    for cidx, cpairs in units_by_chunk.items():
        cpairs_sorted = sorted(
            cpairs,
            key=lambda p: (
                int(p[1]["source_unit_index"]) if p[1].get("source_unit_index") is not None else p[0],
                p[0],
            ),
        )
        m = len(cpairs_sorted)
        t_vals = [
            float(u["start"]) if (u.get("valid_timing") and u.get("start") is not None) else None
            for _, u in cpairs_sorted
        ]
        nxt_t = [None] * m
        cur_n: float | None = None
        for i in range(m - 1, -1, -1):
            if t_vals[i] is not None:
                cur_n = t_vals[i]
            nxt_t[i] = cur_n
        ws, we = inferred_windows.get(cidx, (cidx * 80.0, cidx * 80.0 + 90.0))
        cur_p: float | None = None
        for i, (orig_idx, u) in enumerate(cpairs_sorted):
            if t_vals[i] is not None:
                cur_p = t_vals[i]
                cross_spk_order_t[orig_idx] = t_vals[i]  # type: ignore[assignment]
            elif cur_p is not None and nxt_t[i] is not None:
                frac = 0.5
                cross_spk_order_t[orig_idx] = cur_p + frac * max(0.0, nxt_t[i] - cur_p)  # type: ignore[operator]
            elif cur_p is not None:
                cross_spk_order_t[orig_idx] = min(we, cur_p + 0.05 * (i + 1))
            elif nxt_t[i] is not None:
                cross_spk_order_t[orig_idx] = max(ws, nxt_t[i] - 0.05 * (m - i))  # type: ignore[operator]
            else:
                cross_spk_order_t[orig_idx] = ws + ((i + 1) / float(m + 1)) * max(1.0, we - ws)

    orig_index_by_id = {id(u): orig_idx for orig_idx, u in indexed_units}

    by_speaker: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for orig_idx, u in indexed_units:
        by_speaker.setdefault(_speaker_key(u), []).append((orig_idx, u))

    reconstructed: list[dict[str, Any]] = []
    for spk_key, pairs in by_speaker.items():
        ordered = [item[4] for item in _compute_speaker_sort_keys(pairs, chunk_windows=inferred_windows)]

        groups: list[list[dict[str, Any]]] = []
        cur_group: list[dict[str, Any]] = []
        group_start: float | None = None

        for u in ordered:
            u_timed = bool(u.get("valid_timing") and u.get("start") is not None and u.get("end") is not None)
            u_start = float(u["start"]) if u_timed else None
            u_end = float(u["end"]) if u_timed else None
            if not cur_group:
                cur_group = [u]
                group_start = u_start
                continue

            prev = cur_group[-1]
            p_timed = bool(prev.get("valid_timing") and prev.get("start") is not None and prev.get("end") is not None)
            p_end = float(prev["end"]) if p_timed else None

            should_split = False
            if u_start is not None and p_end is not None:
                gap = u_start - p_end
                if len(cur_group) >= max_units_per_utterance:
                    should_split = True
                elif gap > pause_threshold_s:
                    should_split = True
                elif _ends_sentence(str(prev.get("text") or "")) and gap >= sentence_pause_threshold_s:
                    should_split = True
                elif group_start is not None and u_end is not None and (u_end - group_start) > max_utterance_duration_s:
                    should_split = True
            else:
                if _ends_sentence(str(prev.get("text") or "")):
                    should_split = True
                elif u.get("chunk_index") is not None and prev.get("chunk_index") is not None and u["chunk_index"] != prev["chunk_index"]:
                    should_split = True

            # SOT can alternate tags while voices overlap. Preserve distinct raw
            # turn references, but let coherent timed same-voice speech coalesce.
            # Never join unlocalized text across different raw turns by fiat.
            turn_changed = (
                u.get("recognized_turn_index") is not None
                and prev.get("recognized_turn_index") is not None
                and u.get("chunk_index") == prev.get("chunk_index")
                and u["recognized_turn_index"] != prev["recognized_turn_index"]
            )
            if turn_changed and not (u_timed and p_timed):
                should_split = True
            if (u.get("occurrence_span_id") and prev.get("occurrence_span_id")
                    and u["occurrence_span_id"] != prev["occurrence_span_id"]
                    and (u.get("locality_status") == "disputed"
                         or prev.get("locality_status") == "disputed")):
                should_split = True

            if should_split:
                groups.append(cur_group)
                cur_group = [u]
                group_start = u_start
            else:
                cur_group.append(u)
                if group_start is None and u_start is not None:
                    group_start = u_start

        if cur_group:
            groups.append(cur_group)

        for grp in groups:
            valid_starts = [u["start"] for u in grp if u.get("valid_timing") and u.get("start") is not None]
            valid_ends = [u["end"] for u in grp if u.get("valid_timing") and u.get("end") is not None]
            timed_cnt = sum(
                1 for u in grp if u.get("valid_timing") and u.get("start") is not None and u.get("end") is not None
            )
            untimed_cnt = len(grp) - timed_cnt
            u_start = min(valid_starts) if (timed_cnt > 0 and valid_starts) else None
            u_end = max(valid_ends) if (timed_cnt > 0 and valid_ends) else None
            words = [u for u in grp if u.get("unit_type") != "character"]
            chars = [u for u in grp if u.get("unit_type") == "character"]
            first_u = grp[0]
            chunk_indices = sorted({int(u["chunk_index"]) for u in grp if u.get("chunk_index") is not None})
            timed_provenances = sorted(
                {
                    str(u.get("alignment_status") or "ctc_aligned")
                    for u in grp
                    if u.get("valid_timing") and u.get("start") is not None
                }
            )
            gaps = _extract_timing_gaps(grp) if untimed_cnt > 0 else []

            if timed_cnt == len(grp) and timed_cnt > 0:
                utt_align_status = timed_provenances[0] if len(timed_provenances) == 1 else "ctc_aligned"
                timing_status = "full"
            elif timed_cnt > 0 and untimed_cnt > 0:
                utt_align_status = "partial_alignment"
                timing_status = "partial"
            else:
                has_diar_bound = any(u.get("alignment_status") == "diarization_bounded_only" for u in grp)
                has_error = any(u.get("timing_error") not in (None, "no_ctc_timestamps") for u in grp)
                utt_align_status = "diarization_bounded_only" if has_diar_bound else ("alignment_failed" if has_error else "untimed")
                timing_status = "failed" if (has_diar_bound or has_error) else "unavailable"

            backend = "none"
            if timed_cnt > 0:
                if any("external" in p or "qwen" in p for p in timed_provenances):
                    backend = "mixed_ctc_and_external" if any("ctc" in p for p in timed_provenances) else "external_forced_aligner"
                else:
                    backend = "nemotron_ctc"

            min_orig = min(orig_index_by_id.get(id(u), 0) for u in grp)
            min_seq = min(
                (int(u["source_unit_index"]) for u in grp if u.get("source_unit_index") is not None),
                default=min_orig,
            )
            order_t = (
                float(u_start)
                if u_start is not None
                else min(cross_spk_order_t.get(orig_index_by_id.get(id(u), -1), 0.0) for u in grp)
            )
            reconstructed.append(
                {
                    "speaker_id": spk_key,
                    "global_speaker_id": first_u.get("global_speaker_id") or spk_key,
                    "local_speaker_id": first_u.get("local_speaker_id"),
                    "raw_speaker_tag": first_u.get("raw_speaker_tag") or first_u.get("speaker"),
                    "source_language": source_language,
                    "source_text": join_multilingual_units(grp),
                    "source_start": u_start,
                    "source_end": u_end,
                    "source_words": words,
                    "source_characters": chars,
                    "unit_count": len(grp),
                    "timed_unit_count": timed_cnt,
                    "untimed_unit_count": untimed_cnt,
                    "valid_timed_unit_count": timed_cnt,
                    "timing_gaps": gaps,
                    "chunk_indices": chunk_indices,
                    "alignment_status": utt_align_status,
                    "timestamp_provenance": {
                        "backend": backend,
                        "timing_status": timing_status,
                        "unit_count": len(grp),
                        "timed_unit_count": timed_cnt,
                        "untimed_unit_count": untimed_cnt,
                        "invalid_unit_count": untimed_cnt,
                        "is_partial_bounding_interval": timed_cnt > 0 and untimed_cnt > 0,
                        "unit_statuses": timed_provenances,
                    },
                    "_order_key": (
                        order_t,
                        min(chunk_indices) if chunk_indices else 0,
                        min_seq,
                        min_orig,
                        str(spk_key),
                    ),
                }
            )

    reconstructed.sort(key=lambda utt: utt.pop("_order_key"))
    from nvidia_transcribe.source_occurrences import stable_id
    for idx, utt in enumerate(reconstructed):
        units = utt["source_words"] + utt["source_characters"]
        refs = sorted(u["source_unit_id"] for u in units if u.get("source_unit_id"))
        if refs and len(refs) == len(units):
            utt["id"] = stable_id("utt", refs)
            utt["source_unit_ids"] = refs
            utt["recognized_turn_ids"] = sorted({u["recognized_turn_id"] for u in units})
            utt["occurrence_span_ids"] = sorted({u["occurrence_span_id"] for u in units
                                                  if u.get("occurrence_span_id")})
        else:
            utt["id"] = f"utt-{idx:04d}"
    return reconstructed


def _norm_audit_text(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if not ch.isspace() and not _is_cjk_punctuation(ch) and ch not in ".,!?;:'\"-—…()")


def _is_repetition_loop_text(text: str) -> bool:
    norm = _norm_audit_text(text)
    if len(norm) < 16:
        return False
    # High token/2-gram repetition ratio (e.g., Chunk 14 trailing loop)
    grams = [norm[i : i + 3] for i in range(max(0, len(norm) - 2))]
    if len(grams) >= 12 and len(set(grams)) / float(len(grams)) < 0.55:
        return True
    return False


def _longest_common_substring_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    best = 0
    dp = [0] * (len(b) + 1)
    for ca in a:
        new_dp = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                val = dp[j - 1] + 1
                new_dp[j] = val
                if val > best:
                    best = val
        dp = new_dp
    return best


def audit_and_reconstruct_contract(
    contract: dict[str, Any],
    *,
    chunk_windows: dict[int, tuple[float, float]] | None = None,
    chunk_raw_dir: str | Path | None = None,
    manifest_chunks: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Chronologically reorder and classify untimed utterances in an existing contract.

    Does NOT rerun ASR and never invents ``source_start``/``source_end`` values.
    Uses ``chunk_indices``, chunk windows, raw chunk turn sequences, and neighbor
    timed evidence to restore chronological position and classify every untimed
    utterance as ``probable_duplicate``, ``plausible_independent_missing_timing``,
    ``likely_hallucination_or_loop``, or ``uncertain``.
    """
    import json
    from difflib import SequenceMatcher
    from pathlib import Path
    from nvidia_transcribe.parser import parse_speaker_turns

    utts = [{**u} for u in (contract.get("utterances") or [])]
    windows: dict[int, tuple[float, float]] = dict(chunk_windows or {})
    if manifest_chunks:
        for mc in manifest_chunks:
            if mc.get("infer_start_s") is not None and mc.get("infer_end_s") is not None:
                windows[int(mc["chunk_index"])] = (float(mc["infer_start_s"]), float(mc["infer_end_s"]))

    # Derive missing chunk windows from timed utterances per chunk
    by_chunk_t: dict[int, list[float]] = {}
    for u in utts:
        cindices = u.get("chunk_indices") or ([u["chunk_index"]] if u.get("chunk_index") is not None else [])
        if u.get("source_start") is not None and u.get("source_end") is not None:
            for ci in cindices:
                by_chunk_t.setdefault(int(ci), []).extend([float(u["source_start"]), float(u["source_end"])])
    for ci, ts in by_chunk_t.items():
        windows.setdefault(ci, (min(ts), max(ts)))

    raw_dir = Path(chunk_raw_dir) if chunk_raw_dir else None
    raw_turns_by_chunk: dict[int, list[tuple[int, str]]] = {}
    chunk_finish_reason: dict[int, str] = {}
    chunk_raw_len: dict[int, int] = {}
    # Rechunked recordings have effective indices that do not match
    # original planned filenames. Prefer exact manifest observation paths.
    manifest_paths=[(int(mc["chunk_index"]),Path(mc["raw_response_path"]),mc.get("raw_response_sha256"))
                    for mc in (manifest_chunks or []) if mc.get("raw_response_path")]
    paths=manifest_paths
    if not paths and raw_dir and raw_dir.is_dir():
        paths=[(int(p.name.split("_",1)[1].split(".",1)[0]),p,None)
               for p in sorted(raw_dir.glob("chunk_*.raw.json"))
               if p.name.split("_",1)[1].split(".",1)[0].isdigit()]
    if paths:
        import hashlib
        for cidx,p,digest in paths:
            try:
                data=p.read_bytes()
                if digest and hashlib.sha256(data).hexdigest()!=digest:
                    continue
                doc = json.loads(data)
                ch0 = (doc.get("choices") or [{}])[0]
                chunk_finish_reason[cidx] = str(ch0.get("finish_reason") or "")
                content = str((ch0.get("message") or {}).get("content") or "")
                chunk_raw_len[cidx] = len(content)
                turns = parse_speaker_turns(content)
                raw_turns_by_chunk[cidx] = [(ti, _norm_audit_text(t.text)) for ti, t in enumerate(turns) if t.text.strip()]
            except Exception:
                continue

    timed_utts = [u for u in utts if u.get("source_start") is not None and u.get("source_end") is not None]
    untimed_seen_kept: list[tuple[int, dict[str, Any]]] = []
    audit_records: list[dict[str, Any]] = []

    for orig_pos, u in enumerate(utts):
        uid = str(u.get("id") or u.get("utterance_id") or f"utt-{orig_pos:04d}")
        u.setdefault("original_utterance_id", uid)
        cindices = [int(c) for c in (u.get("chunk_indices") or ([u["chunk_index"]] if u.get("chunk_index") is not None else [0]))]
        cidx = cindices[0] if cindices else 0
        ws, we = windows.get(cidx, (cidx * 80.0, cidx * 80.0 + 90.0))

        if u.get("source_start") is not None and u.get("source_end") is not None:
            u["_order_key"] = (float(u["source_start"]), cidx, orig_pos, str(u.get("speaker_id") or ""))
            continue

        # 1. Estimate chronological position within chunk using matched turn index inside chunk_NNN.raw.json
        src_txt = str(u.get("source_text") or "")
        norm_u = _norm_audit_text(src_txt)
        turns_in_chunk = raw_turns_by_chunk.get(cidx) or []
        frac = 0.5
        if turns_in_chunk and norm_u:
            best_ti, best_score = 0, -1.0
            for ti, nt in turns_in_chunk:
                if norm_u in nt or nt in norm_u:
                    score = 1.0
                else:
                    score = SequenceMatcher(None, norm_u[:40], nt[:40]).ratio()
                if score > best_score:
                    best_score = score
                    best_ti = ti
            if best_score >= 0.45:
                frac = (best_ti + 1) / float(len(turns_in_chunk) + 1)
            else:
                frac = (orig_pos % max(1, len(turns_in_chunk))) / float(len(turns_in_chunk) + 1)
        order_t = ws + frac * max(1.0, we - ws)
        u["_order_key"] = (order_t, cidx, orig_pos, str(u.get("speaker_id") or ""))

        # 2. Classify untimed utterance cleanly
        is_loop = (
            _is_repetition_loop_text(src_txt)
            or (chunk_finish_reason.get(cidx) == "length" and chunk_raw_len.get(cidx, 0) > 4000)
        )
        best_dup_sim = 0.0
        best_dup_id: str | None = None

        # Compare against timed utterances within [ws - 45, we + 45] (allows chunk-boundary drift)
        for t in timed_utts:
            ts = float(t["source_start"])
            if ts < ws - 45.0 or ts > we + 45.0:
                continue
            nt = _norm_audit_text(str(t.get("source_text") or ""))
            if not norm_u or not nt:
                continue
            sim = SequenceMatcher(None, norm_u, nt).ratio()
            if len(norm_u) >= 6 and len(nt) >= 6 and (norm_u in nt or nt in norm_u):
                ratio_len = min(len(norm_u), len(nt)) / float(max(len(norm_u), len(nt)))
                if ratio_len >= 0.40:
                    sim = max(sim, 0.90)
            elif len(norm_u) >= 6 and len(nt) >= 6 and _longest_common_substring_len(norm_u, nt) >= 6:
                sim = max(sim, 0.65)
            if sim > best_dup_sim:
                best_dup_sim = sim
                best_dup_id = str(t.get("id") or t.get("utterance_id"))

        # Compare against earlier untimed utterances in adjacent overlapping chunk (e.g., chunk 10 vs 11)
        for prev_cidx, prev_u in untimed_seen_kept:
            if abs(cidx - prev_cidx) > 1:
                continue
            np_u = _norm_audit_text(str(prev_u.get("source_text") or ""))
            if len(norm_u) >= 4 and len(np_u) >= 4:
                sim = SequenceMatcher(None, norm_u, np_u).ratio()
                if len(norm_u) >= 6 and len(np_u) >= 6 and (norm_u in np_u or np_u in norm_u or _longest_common_substring_len(norm_u, np_u) >= 6):
                    sim = max(sim, 0.78)
                if sim > best_dup_sim:
                    best_dup_sim = sim
                    best_dup_id = str(prev_u.get("original_utterance_id") or prev_u.get("id"))

        if is_loop:
            classification = "likely_hallucination_or_loop"
            rationale = "repetition_loop_or_max_tokens_chunk_tail"
        elif best_dup_sim >= 0.58 and len(norm_u) >= 2:
            classification = "probable_duplicate"
            rationale = f"matches_{best_dup_id}_sim={best_dup_sim:.2f}"
        elif len(norm_u) <= 2:
            classification = "uncertain"
            rationale = "short_unlocalized_fragment_or_interjection"
        else:
            classification = "plausible_independent_missing_timing"
            rationale = "coherent_speech_not_present_in_nearby_timed_utterances"
            untimed_seen_kept.append((cidx, u))

        u["untimed_classification"] = classification
        if classification == "probable_duplicate" and best_dup_id:
            u["probable_duplicate_of"] = best_dup_id
        audit_records.append(
            {
                "original_utterance_id": uid,
                "chunk_index": cidx,
                "chunk_window_s": [round(ws, 2), round(we, 2)],
                "estimated_order_s": round(order_t, 2),
                "speaker_id": u.get("speaker_id"),
                "alignment_status": u.get("alignment_status"),
                "source_text": src_txt,
                "classification": classification,
                "duplicate_of": best_dup_id if classification == "probable_duplicate" else None,
                "duplicate_similarity": round(best_dup_sim, 3) if classification == "probable_duplicate" else None,
                "rationale": rationale,
            }
        )

    utts.sort(key=lambda utt: utt.pop("_order_key"))
    for idx, u in enumerate(utts):
        u["id"] = f"utt-{idx:04d}"

    # Map duplicate_of IDs from original_utterance_id -> new sequential ID for audit clarity
    orig_to_new = {str(u.get("original_utterance_id")): u["id"] for u in utts}
    for u in utts:
        if u.get("probable_duplicate_of") in orig_to_new:
            u["probable_duplicate_of"] = orig_to_new[u["probable_duplicate_of"]]
    for rec in audit_records:
        rec["reordered_utterance_id"] = orig_to_new.get(rec["original_utterance_id"])
        if rec.get("duplicate_of") in orig_to_new:
            rec["duplicate_of_reordered"] = orig_to_new[rec["duplicate_of"]]

    new_contract = {**contract, "utterances": utts, "untimed_audit": audit_records}
    return new_contract, audit_records


def apply_validation_overlay(
    contract: dict[str, Any],
    overlay: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply curated coarse-timing recovery, phrasing polish, and duplicate/artifact suppression.

    Preserves ``id``, ``original_utterance_id``, raw ASR timing (``raw_source_start`` /
    ``raw_source_end``), and prior English wording (``pre_overlay_translated_text``) so every
    override remains auditable and reversible without modifying raw ASR responses.
    """
    entries = list(overlay.get("utterances") or overlay.get("entries") or [])
    if not entries:
        return contract, {"status": "empty_overlay", "applied_count": 0}

    utts: list[dict[str, Any]] = [dict(u) for u in (contract.get("utterances") or [])]
    by_id: dict[str, int] = {}
    by_orig_id: dict[str, int] = {}
    for idx, u in enumerate(utts):
        if u.get("id"):
            by_id[str(u["id"])] = idx
        if u.get("utterance_id"):
            by_id[str(u["utterance_id"])] = idx
        if u.get("original_utterance_id"):
            by_orig_id[str(u["original_utterance_id"])] = idx

    recovered_timing = 0
    relocated_timing = 0
    edited_text = 0
    suppressed_count = 0
    skipped: list[str] = []

    for ent in entries:
        tid = str(ent.get("utterance_id") or ent.get("id") or "").strip()
        orig_id = str(ent.get("original_utterance_id") or "").strip()
        pos = by_id.get(tid)
        if pos is None and orig_id:
            pos = by_orig_id.get(orig_id)
        if pos is None:
            skipped.append(tid or orig_id or "unknown")
            continue

        u = utts[pos]
        expected_sub = str(ent.get("expected_source_substring") or "").strip()
        if expected_sub:
            src_hay = f"{u.get('source_text') or ''} {u.get('refined_text') or ''}"
            if expected_sub not in src_hay:
                raise ValueError(
                    f"Validation overlay mismatch for {tid or orig_id}: expected source substring "
                    f"{expected_sub!r} not found in {src_hay!r}"
                )

        u.setdefault("reconstructed_utterance_id", str(u.get("id") or tid))

        # 1. Suppress confirmed duplicate / hallucination artifacts
        if ent.get("suppress") or ent.get("suppressed_duplicate_of"):
            if u.get("translated_text") and "pre_overlay_translated_text" not in u:
                u["pre_overlay_translated_text"] = u["translated_text"]
            u["overlay_suppressed"] = True
            if ent.get("suppressed_duplicate_of"):
                u["suppressed_duplicate_of"] = str(ent["suppressed_duplicate_of"])
            if ent.get("reason"):
                u["overlay_reason"] = str(ent["reason"])
            u["translated_text"] = ""
            suppressed_count += 1
            continue

        # 2. Coarse timing recovery or relocation of misaligned CTC bounds
        new_s = ent.get("proposed_start_s") if "proposed_start_s" in ent else ent.get("source_start")
        new_e = ent.get("proposed_end_s") if "proposed_end_s" in ent else ent.get("source_end")
        if new_s is not None and new_e is not None:
            fs, fe = float(new_s), float(new_e)
            if fe <= fs or fs < 0.0:
                raise ValueError(f"Invalid overlay timing [{fs}, {fe}] for {tid or orig_id}")
            had_timing = u.get("source_start") is not None and u.get("source_end") is not None
            if had_timing and "raw_source_start" not in u:
                u["raw_source_start"] = u["source_start"]
                u["raw_source_end"] = u["source_end"]
            u["source_start"] = round(fs, 3)
            u["source_end"] = round(fe, 3)
            prov = dict(u.get("timestamp_provenance") or {})
            t_status = str(ent.get("timing_status") or ("relocated_reviewed" if had_timing else "coarse_recovered"))
            prov["timing_status"] = t_status
            prov["timing_source"] = "viewing_validation_overlay"
            if ent.get("evidence"):
                prov["evidence"] = list(ent["evidence"])
            u["timestamp_provenance"] = prov
            u["timing_status"] = t_status
            if had_timing:
                relocated_timing += 1
            else:
                recovered_timing += 1

        # 3. Localized English dialogue polish
        if ent.get("translated_text") is not None:
            new_en = " ".join(str(ent["translated_text"]).split()).strip()
            old_en = str(u.get("translated_text") or "").strip()
            if new_en != old_en:
                if old_en and "pre_overlay_translated_text" not in u:
                    u["pre_overlay_translated_text"] = old_en
                u["translated_text"] = new_en
                u["overlay_text_edited"] = True
                edited_text += 1

        if ent.get("reason"):
            u["overlay_reason"] = str(ent["reason"])

    # Maintain chronological ordering for timed/recovered items while preserving IDs cleanly
    timed_pairs = []
    untimed_pairs = []
    for orig_idx, u in enumerate(utts):
        if u.get("source_start") is not None and u.get("source_end") is not None:
            timed_pairs.append(((0, float(u["source_start"]), float(u["source_end"]), orig_idx), u))
        else:
            untimed_pairs.append(((1, orig_idx), u))
    ordered_utts = [u for _, u in sorted(timed_pairs, key=lambda x: x[0])] + [
        u for _, u in sorted(untimed_pairs, key=lambda x: x[0])
    ]

    summary = {
        "status": "applied",
        "overlay_name": str(overlay.get("name") or "custom"),
        "recovered_timing_count": recovered_timing,
        "relocated_timing_count": relocated_timing,
        "edited_text_count": edited_text,
        "suppressed_count": suppressed_count,
        "skipped_ids": skipped,
    }
    return {**contract, "utterances": ordered_utts, "validation_overlay": summary}, summary
