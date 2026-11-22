"""Deterministic, speech-pause-aware chunk planning for Nemotron inference.

Do not confuse signal-level silence with a verified speaker/word boundary.
The detector only proposes cut candidates. Every core has exact, gap-free
coverage; inference context surrounds every non-edge cut. Fixed chunking
remains available as an explicit compatibility mode.
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import Any, Sequence

from nvidia_transcribe.chunking import ChunkSpec

PLANNER_VERSION = "pause_aware/v1"
_SILENCE_START = re.compile(r"silence_start:\s*([0-9.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9.]+)")


def detect_silence_intervals(
    media_path: str | Path, *,
    audio_stream_index: int,
    duration_s: float,
    noise_db: int = -30,
    min_silence_s: float = 0.45,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Decode the selected audio once with FFmpeg's silencedetect, no writes.

    Detection failure is reported, not silently treated as observed silence.
    The planner can then use a bounded forced-cut fallback with uncertainty.
    """
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if audio_stream_index < 0:
        raise ValueError("audio_stream_index must be nonnegative")
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
        "-i", str(media_path), "-map", f"0:a:{audio_stream_index}",
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence_s}",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
            timeout=max(120, min(900, round(duration_s / 10))),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], {
            "schema": PLANNER_VERSION, "detector": "ffmpeg_silencedetect",
            "status": "failed", "error_type": type(exc).__name__,
            "silence_intervals": 0,
        }
    if proc.returncode:
        return [], {
            "schema": PLANNER_VERSION, "detector": "ffmpeg_silencedetect",
            "status": "failed", "exit_code": proc.returncode,
            "silence_intervals": 0,
        }
    pauses: list[tuple[float, float]] = []
    start: float | None = None
    for line in proc.stderr.splitlines():
        m = _SILENCE_START.search(line)
        if m:
            start = float(m.group(1))
        m = _SILENCE_END.search(line)
        if m and start is not None:
            end = float(m.group(1))
            if math.isfinite(start) and math.isfinite(end):
                lo, hi = max(0.0, start), min(float(duration_s), end)
                if hi - lo >= min_silence_s:
                    pauses.append((round(lo, 6), round(hi, 6)))
            start = None
    if start is not None and math.isfinite(start) and duration_s - start >= min_silence_s:
        pauses.append((round(max(0.0, start), 6), round(float(duration_s), 6)))
    pauses.sort()
    return pauses, {
        "schema": PLANNER_VERSION, "detector": "ffmpeg_silencedetect",
        "status": "complete", "silence_intervals": len(pauses),
        "noise_db": noise_db, "min_silence_s": min_silence_s,
    }


def plan_pause_aware_chunks(
    total_duration_s: float,
    pauses: Sequence[tuple[float, float]],
    *,
    max_requested_core_s: float = 90.0,
    requested_context_s: float = 10.0,
    target_core_s: float = 39.0,
    min_core_s: float = 22.0,
    max_core_s: float = 46.0,
    max_context_s: float = 4.0,
    max_tail_slack_s: float = 8.0,
) -> tuple[list[ChunkSpec], dict[str, Any]]:
    """Choose cuts near actual pauses, never by resizing/acoustically moving words.

    Core partitions remain contiguous and every excerpt includes overlapping
    context. Fallback forced cuts are explicitly reported; they are NOT
    claimed to be safe sentence boundaries. The next qualification measures
    those fallbacks for source omission and speaker loss.
    """
    if not math.isfinite(total_duration_s) or total_duration_s <= 0:
        raise ValueError("total_duration_s must be finite and positive")
    if not (math.isfinite(max_requested_core_s) and max_requested_core_s > 0
            and math.isfinite(requested_context_s) and requested_context_s >= 0):
        raise ValueError("invalid requested core/context")
    max_core = min(max_requested_core_s, max_core_s)
    # Keep small explicit user chunk limits valid, with proportional min cores.
    min_core_s = min(min_core_s, max_core * 0.60)
    target = min(target_core_s, max_core)
    context = min(requested_context_s, max_context_s)
    total = round(float(total_duration_s), 6)
    good_pauses = sorted(
        (max(0.0, float(lo)), min(total, float(hi)))
        for lo, hi in pauses
        if math.isfinite(lo) and math.isfinite(hi)
        and float(hi) > float(lo)
    )
    edges: list[float] = [0.0]
    selections: list[dict[str, Any]] = []
    while total - edges[-1] > max_core + max_tail_slack_s:
        cursor = edges[-1]
        low = cursor + min_core_s
        high = min(cursor + max_core, total - max_tail_slack_s)
        desired = min(cursor + target, high)
        options = []
        for lo, hi in good_pauses:
            mid = (lo + hi) / 2
            if low <= mid <= high:
                # Near-target cuts are preferred, with a small bonus for longer
                # detected pauses, but never beyond the legal core window.
                options.append((abs(mid - desired) - min(hi - lo, 4.0) * 0.5,
                                -min(hi - lo, 4.0), mid))
        if options:
            chosen = min(options)[2]
            why = "detected_pause"
        else:
            chosen = high
            why = "forced_max_core_no_eligible_pause"
        chosen = round(chosen, 6)
        if chosen <= cursor or chosen >= total:
            raise RuntimeError("pause-aware planner failed to advance core boundary")
        edges.append(chosen)
        selections.append({"at_s": chosen, "reason": why})
    edges.append(total)
    chunks = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        chunks.append(ChunkSpec(
            chunk_index=i,
            core_start_s=lo, core_end_s=hi,
            infer_start_s=round(max(0, lo - context), 6),
            infer_end_s=round(min(total, hi + context), 6),
        ))
    assert chunks[0].core_start_s == 0 and chunks[-1].core_end_s == total
    assert all(a.core_end_s == b.core_start_s for a, b in zip(chunks, chunks[1:]))
    return chunks, {
        "schema": PLANNER_VERSION,
        "target_core_s": target, "min_core_s": min_core_s,
        "max_core_s": max_core, "effective_context_s": context,
        "detected_pause_count": len(good_pauses),
        "selected_pause_count": sum(s["reason"] == "detected_pause" for s in selections),
        "forced_cut_count": sum(s["reason"].startswith("forced") for s in selections),
        "boundary_decisions": selections,
        "max_actual_infer_s": round(max(s.infer_duration_s for s in chunks), 6),
        "source_timing_not_inferred": True,
    }
