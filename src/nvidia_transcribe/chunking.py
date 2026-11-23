"""Deterministic recording-level chunk planner and boundary deduplication."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from nvidia_transcribe.boundary_evidence import (
    reconcile_displaced_sequences,
    recover_extended_right_context,
    recover_ownerless_straddled_overlap,
)


@dataclass(frozen=True)
class ChunkConfig:
    chunk_seconds: float = 90.0
    context_seconds: float = 10.0
    min_tail_seconds: float = 15.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.chunk_seconds) or self.chunk_seconds <= 0.0:
            raise ValueError(f"chunk_seconds must be positive, got {self.chunk_seconds}")
        if not math.isfinite(self.context_seconds) or self.context_seconds < 0.0:
            raise ValueError(f"context_seconds must be >= 0, got {self.context_seconds}")
        if not math.isfinite(self.min_tail_seconds) or self.min_tail_seconds < 0.0:
            raise ValueError(f"min_tail_seconds must be >= 0, got {self.min_tail_seconds}")

    def to_dict(self) -> dict[str, float]:
        return {
            "chunk_seconds": self.chunk_seconds,
            "context_seconds": self.context_seconds,
            "min_tail_seconds": self.min_tail_seconds,
        }


@dataclass(frozen=True)
class ChunkSpec:
    chunk_index: int
    core_start_s: float
    core_end_s: float
    infer_start_s: float
    infer_end_s: float

    @property
    def core_duration_s(self) -> float:
        return round(self.core_end_s - self.core_start_s, 6)

    @property
    def infer_duration_s(self) -> float:
        return round(self.infer_end_s - self.infer_start_s, 6)

    @property
    def left_context_s(self) -> float:
        return round(self.core_start_s - self.infer_start_s, 6)

    @property
    def right_context_s(self) -> float:
        return round(self.infer_end_s - self.core_end_s, 6)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "core_start_s": self.core_start_s,
            "core_end_s": self.core_end_s,
            "core_duration_s": self.core_duration_s,
            "infer_start_s": self.infer_start_s,
            "infer_end_s": self.infer_end_s,
            "infer_duration_s": self.infer_duration_s,
            "left_context_s": self.left_context_s,
            "right_context_s": self.right_context_s,
            "source_offset_s": self.infer_start_s,
        }


def plan_chunks(total_duration_s: float, config: ChunkConfig | None = None) -> list[ChunkSpec]:
    """Build an exact, gap-free core partition with bounded inference context windows."""
    if not math.isfinite(total_duration_s) or total_duration_s <= 0.0:
        raise ValueError(f"total_duration_s must be positive, got {total_duration_s}")
    cfg = config or ChunkConfig()
    total = round(float(total_duration_s), 6)

    if total <= cfg.chunk_seconds + cfg.min_tail_seconds:
        return [
            ChunkSpec(
                chunk_index=0,
                core_start_s=0.0,
                core_end_s=total,
                infer_start_s=0.0,
                infer_end_s=total,
            )
        ]

    boundaries: list[tuple[float, float]] = []
    cursor = 0.0
    while cursor < total:
        remaining = round(total - cursor, 6)
        if remaining <= cfg.chunk_seconds + cfg.min_tail_seconds:
            boundaries.append((round(cursor, 6), total))
            break
        nxt = round(min(total, cursor + cfg.chunk_seconds), 6)
        boundaries.append((round(cursor, 6), nxt))
        cursor = nxt

    specs: list[ChunkSpec] = []
    for idx, (c_start, c_end) in enumerate(boundaries):
        i_start = round(max(0.0, c_start - cfg.context_seconds), 6)
        i_end = round(min(total, c_end + cfg.context_seconds), 6)
        specs.append(
            ChunkSpec(
                chunk_index=idx,
                core_start_s=c_start,
                core_end_s=c_end,
                infer_start_s=i_start,
                infer_end_s=i_end,
            )
        )
    return specs


def _unit_midpoint(unit: dict[str, Any]) -> float | None:
    s, e = unit.get("start"), unit.get("end")
    if s is None or e is None:
        return None
    return 0.5 * (float(s) + float(e))


def _norm_token(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def _speaker_id_for_boundary(unit: dict[str, Any]) -> str | None:
    gid = unit.get("global_speaker_id")
    if gid is not None and str(gid).strip():
        return str(gid).strip()
    lid = unit.get("local_speaker_id")
    if lid is not None and str(lid).strip():
        return str(lid).strip()
    spk = unit.get("speaker")
    if spk is not None and str(spk).strip():
        return str(spk).strip()
    return None


def _is_unknown_speaker_id(spk: str | None) -> bool:
    if not spk:
        return True
    s = spk.strip()
    return s == "unknown" or s.startswith("SPK_UNK_") or s.endswith(":unknown")


def _speaker_evidence(a: dict[str, Any], b: dict[str, Any]) -> str:
    """Classify speaker relationship between two boundary units.

    Returns:
      - ``"same_known"``: both have resolved, matching non-unknown speaker IDs.
      - ``"distinct_known"``: both have resolved, non-matching non-unknown speaker IDs.
      - ``"unknown_side"``: one or both speakers are unknown/missing (never safe proof of identity).
    """
    sa = _speaker_id_for_boundary(a)
    sb = _speaker_id_for_boundary(b)
    if _is_unknown_speaker_id(sa) or _is_unknown_speaker_id(sb):
        return "unknown_side"
    return "same_known" if sa == sb else "distinct_known"


def _compatible_speaker(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Strictly require known matching speaker identity; unknown is NOT positive evidence."""
    return _speaker_evidence(a, b) == "same_known"


def _assign_monotonic_sort_keys(units: list[dict[str, Any]], chunk_core_starts: dict[int, float]) -> None:
    """Assign internal non-mutating sort anchors so untimed units keep lexical order."""
    # Group units by chunk_index while preserving input order per chunk.
    by_chunk: dict[int, list[dict[str, Any]]] = {}
    for u in units:
        cidx = int(u.get("_chunk_idx") if u.get("_chunk_idx") is not None else u.get("chunk_index", 0))
        by_chunk.setdefault(cidx, []).append(u)

    for cidx, items in by_chunk.items():
        # Preserve local sequence index
        valid_times: list[float | None] = [
            float(x["start"]) if (x.get("valid_timing") and x.get("start") is not None) else None
            for x in items
        ]
        prev_t: float | None = None
        next_valid: list[float | None] = [None] * len(items)
        nv: float | None = None
        for i in range(len(items) - 1, -1, -1):
            if valid_times[i] is not None:
                nv = valid_times[i]
            next_valid[i] = nv

        for i, item in enumerate(items):
            vt = valid_times[i]
            seq = int(item.get("_unit_idx", item.get("source_unit_index", i)))
            if vt is not None:
                item["_sort_anchor"] = vt
                prev_t = vt
            else:
                nxt = next_valid[i]
                if prev_t is not None and nxt is not None:
                    # Strictly interpolate sort key only (never mutating item['start'])
                    item["_sort_anchor"] = prev_t + (nxt - prev_t) * 0.5
                elif prev_t is not None:
                    item["_sort_anchor"] = prev_t + 1e-4 * (i + 1)
                elif nxt is not None:
                    item["_sort_anchor"] = max(0.0, nxt - 1e-4 * (len(items) - i))
                else:
                    item["_sort_anchor"] = chunk_core_starts.get(cidx, 0.0) + 1e-4 * seq


def reconcile_chunk_boundaries(
    chunk_units: list[list[dict[str, Any]]],
    chunks: list[ChunkSpec],
    *,
    boundary_window_s: float = 2.5,
    match_time_tolerance_s: float = 0.85,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge per-chunk timed units across core boundaries without duplicating or losing speech.

    Units with valid timestamps are reconciled across shared boundary zones using
    source-time alignment, normalized lexical equivalence, and compatible global
    speaker identity (preserving genuine simultaneous repetitions from distinct speakers).
    """
    if len(chunk_units) != len(chunks):
        raise ValueError("chunk_units and chunks length mismatch")
    if not chunks:
        return [], {"deduplicated_units": 0, "recovered_boundary_units": 0, "ambiguities": []}

    # Tag each unit with its chunk core ownership status first.
    tagged_by_chunk: list[list[dict[str, Any]]] = []
    for spec, units in zip(chunks, chunk_units):
        tagged: list[dict[str, Any]] = []
        for idx, u in enumerate(units):
            mid = _unit_midpoint(u)
            in_core = False
            if mid is not None:
                if spec.chunk_index == len(chunks) - 1:
                    in_core = spec.core_start_s <= mid <= spec.core_end_s + 1e-4
                else:
                    in_core = spec.core_start_s <= mid < spec.core_end_s
            tagged.append(
                {
                    **u,
                    "_chunk_idx": spec.chunk_index,
                    "_unit_idx": idx,
                    "_mid": mid,
                    "_in_core": in_core,
                    "_drop": False,
                    "_recovered": False,
                }
            )
        tagged_by_chunk.append(tagged)

    dedup_count = 0
    recovered_count = 0
    ambiguities: list[dict[str, Any]] = []
    dedup_examples: list[dict[str, Any]] = []

    for b_idx in range(len(chunks) - 1):
        left_spec = chunks[b_idx]
        right_spec = chunks[b_idx + 1]
        boundary_t = left_spec.core_end_s
        win_lo = boundary_t - boundary_window_s
        win_hi = boundary_t + boundary_window_s

        left_cands = [
            u
            for u in tagged_by_chunk[b_idx]
            if not u["_drop"] and u["_mid"] is not None and win_lo <= u["_mid"] <= win_hi
        ]
        right_cands = [
            u
            for u in tagged_by_chunk[b_idx + 1]
            if not u["_drop"] and u["_mid"] is not None and win_lo <= u["_mid"] <= win_hi
        ]

        # Match same-known-speaker, temporally close, lexically matching boundary units.
        # Unknown-speaker lexical collisions and ambiguous multi-candidate ties are logged
        # and BOTH observations are preserved instead of silently deleting speech.
        used_right: set[int] = set()
        matched_left: set[int] = set()
        for lu in left_cands:
            lnorm = _norm_token(lu.get("text", ""))
            if not lnorm:
                continue
            exact_same_cands: list[tuple[float, dict[str, Any]]] = []
            unknown_cands: list[tuple[float, dict[str, Any]]] = []
            for ru in right_cands:
                if id(ru) in used_right:
                    continue
                if _norm_token(ru.get("text", "")) != lnorm:
                    continue
                dt = abs(float(lu["_mid"]) - float(ru["_mid"]))
                if dt > match_time_tolerance_s:
                    continue
                ev = _speaker_evidence(lu, ru)
                if ev == "same_known":
                    exact_same_cands.append((dt, ru))
                elif ev == "unknown_side":
                    unknown_cands.append((dt, ru))

            if exact_same_cands:
                exact_same_cands.sort(key=lambda item: (item[0], item[1]["_unit_idx"]))
                # If two genuinely indistinguishable candidates exist at virtually identical times, do not guess.
                if len(exact_same_cands) >= 2 and abs(exact_same_cands[1][0] - exact_same_cands[0][0]) < 0.05:
                    ambiguities.append(
                        {
                            "boundary_s": boundary_t,
                            "left_chunk": left_spec.chunk_index,
                            "right_chunk": right_spec.chunk_index,
                            "token": lu.get("text"),
                            "start": lu.get("start"),
                            "left_speaker": _speaker_id_for_boundary(lu),
                            "candidates": [
                                {"start": c[1].get("start"), "dt_s": round(c[0], 4), "speaker": _speaker_id_for_boundary(c[1])}
                                for c in exact_same_cands
                            ],
                            "reason": "ambiguous_multiple_same_speaker_candidates",
                            "resolution": "preserved_both",
                        }
                    )
                    continue

                best_dt, best_ru = exact_same_cands[0]
                used_right.add(id(best_ru))
                matched_left.add(id(lu))
                if lu["_in_core"] and not best_ru["_in_core"]:
                    best_ru["_drop"] = True
                    kept, dropped = lu, best_ru
                elif best_ru["_in_core"] and not lu["_in_core"]:
                    lu["_drop"] = True
                    kept, dropped = best_ru, lu
                elif float(lu["_mid"]) < boundary_t:
                    best_ru["_drop"] = True
                    kept, dropped = lu, best_ru
                else:
                    lu["_drop"] = True
                    kept, dropped = best_ru, lu
                dedup_count += 1
                if len(dedup_examples) < 25:
                    dedup_examples.append(
                        {
                            "boundary_s": boundary_t,
                            "token": kept.get("text"),
                            "speaker_id": _speaker_id_for_boundary(kept),
                            "evidence": "same_known_speaker",
                            "kept_chunk": kept["_chunk_idx"],
                            "kept_start": kept.get("start"),
                            "dropped_chunk": dropped["_chunk_idx"],
                            "dropped_start": dropped.get("start"),
                            "dt_s": round(best_dt, 4),
                        }
                    )
            elif unknown_cands:
                # Unknown speaker identity is NEVER sufficient evidence to delete speech.
                unknown_cands.sort(key=lambda item: item[0])
                best_dt, u_ru = unknown_cands[0]
                ambiguities.append(
                    {
                        "boundary_s": boundary_t,
                        "left_chunk": left_spec.chunk_index,
                        "right_chunk": right_spec.chunk_index,
                        "token": lu.get("text"),
                        "start": lu.get("start"),
                        "left_speaker": _speaker_id_for_boundary(lu),
                        "right_speaker": _speaker_id_for_boundary(u_ru),
                        "dt_s": round(best_dt, 4),
                        "reason": "unknown_speaker_cannot_prove_duplicate",
                        "resolution": "preserved_both",
                    }
                )

        # Check right-context units of left chunk just past boundary_t:
        # if left chunk saw a word just past boundary_t (within 1.5s) and right chunk missed it completely
        # (no matched unit and no nearby core unit), recover it to prevent boundary omission.
        for lu in left_cands:
            if id(lu) in matched_left or lu["_drop"] or lu["_in_core"]:
                continue
            mid = float(lu["_mid"])
            if boundary_t <= mid <= boundary_t + 1.5:
                has_nearby_right = any(
                    not ru["_drop"]
                    and ru["_in_core"]
                    and _compatible_speaker(lu, ru)
                    and abs(float(ru["_mid"]) - mid) <= 0.35
                    for ru in right_cands
                )
                if not has_nearby_right:
                    lu["_in_core"] = True
                    lu["_recovered"] = True
                    recovered_count += 1
                else:
                    ambiguities.append(
                        {
                            "boundary_s": boundary_t,
                            "left_chunk": left_spec.chunk_index,
                            "right_chunk": right_spec.chunk_index,
                            "token": lu.get("text"),
                            "start": lu.get("start"),
                            "reason": "unmatched_boundary_context_overlap",
                        }
                    )

    # Evidence-gated multiword reconciliation handles displaced lexical echoes
    # and more complete neighboring observations that the narrow token matcher
    # cannot see. Recovery outside the historical 1.5s cap requires an entire
    # coherent missing phrase and a neighboring-coverage check.
    sequence_diag = reconcile_displaced_sequences(tagged_by_chunk, chunks)
    extended_diag = recover_extended_right_context(tagged_by_chunk, chunks)
    # Two ASR contexts can agree on a phrase while each timestamps it
    # outside its own nominal core. Preserve an untimed source hypothesis
    # rather than silently dropping the entire phrase or inventing time.
    straddled_diag = recover_ownerless_straddled_overlap(tagged_by_chunk, chunks)
    dedup_count += sequence_diag["sequence_deduplicated_units"]
    recovered_count += extended_diag["extended_recovered_units"]

    chunk_core_starts = {s.chunk_index: s.core_start_s for s in chunks}
    retained_raw: list[dict[str, Any]] = []
    untimed_boundary_count = 0
    for tagged in tagged_by_chunk:
        for u in tagged:
            if u["_drop"]:
                continue
            # Keep core units or untimed units without fabricating boundaries
            if u["_in_core"] or u["_mid"] is None:
                if u["_mid"] is None:
                    untimed_boundary_count += 1
                retained_raw.append(u)

    _assign_monotonic_sort_keys(retained_raw, chunk_core_starts)
    retained_raw.sort(
        key=lambda x: (
            x["_sort_anchor"],
            x["_chunk_idx"],
            x["_unit_idx"],
        )
    )
    merged: list[dict[str, Any]] = []
    for u in retained_raw:
        clean_u = {k: v for k, v in u.items() if not k.startswith("_")}
        if u["_recovered"]:
            clean_u["boundary_provenance"] = u.get("_recovery_kind") or "boundary_recovered"
        elif u["_in_core"]:
            clean_u["boundary_provenance"] = "core"
        else:
            clean_u["boundary_provenance"] = "untimed_preserved"
        merged.append(clean_u)

    return merged, {
        "deduplicated_units": dedup_count,
        "recovered_boundary_units": recovered_count,
        "untimed_preserved_units": untimed_boundary_count,
        "ambiguities": ambiguities,
        "dedup_examples": dedup_examples,
        "sequence_evidence": sequence_diag,
        "extended_recovery": extended_diag,
        "straddled_overlap_source": straddled_diag,
    }
