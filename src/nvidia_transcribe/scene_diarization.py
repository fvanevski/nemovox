"""Experimental scene-bounded Nemotron-3-Diarization evidence adapter (Phase D).

Strictly supplemental and non-authoritative: emits structured acoustic speaker
evidence records linking local scene diarization channels to canonical
``global_speaker_id`` identities without ever overwriting canonical IDs or
forcing ambiguous merges.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

EVIDENCE_SOURCE = "nemotron_3_diarization"
DEFAULT_DIARIZATION_MODEL_PATH = "/workspace/models/Nemotron-3-Diarization"


@dataclass(frozen=True)
class SceneWindow:
    scene_id: str
    start_s: float
    end_s: float
    purpose: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_bounded_scenes(
    total_duration_s: float,
    *,
    scene_seconds: float = 120.0,
    min_tail_seconds: float = 30.0,
) -> list[SceneWindow]:
    """Partition long media into scene-bounded intervals for independent diarization passes."""
    if total_duration_s <= 0:
        return []
    if scene_seconds <= 0:
        raise ValueError("scene_seconds must be positive")
    if total_duration_s <= scene_seconds:
        return [SceneWindow("scene_000", 0.0, round(total_duration_s, 3))]

    windows: list[SceneWindow] = []
    cur = 0.0
    idx = 0
    while cur < total_duration_s:
        nxt = min(total_duration_s, cur + scene_seconds)
        rem = total_duration_s - nxt
        if 0.0 < rem < min_tail_seconds:
            nxt = total_duration_s
        windows.append(SceneWindow(f"scene_{idx:03d}", round(cur, 3), round(nxt, 3)))
        if nxt >= total_duration_s:
            break
        cur = nxt
        idx += 1
    return windows


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for s, e in ordered[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1e-6:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in _merge_intervals(intervals))


def build_scene_speaker_evidence(
    *,
    scene_id: str,
    scene_start_s: float,
    scene_end_s: float,
    diar_segments: list[dict[str, Any]],
    canonical_utterances: list[dict[str, Any]],
    model_revision: str,
    min_overlap_s: float = 0.25,
    ambiguity_margin: float = 0.15,
) -> dict[str, Any]:
    """Reconcile scene-bounded diarization output against canonical utterances non-destructively.

    Never overwrites canonical ``global_speaker_id`` labels. Instead classifies
    each local diarization channel as ``consistent_support``,
    ``ambiguous_one_to_many`` (one diar channel splits across multiple canonical
    speakers), ``ambiguous_many_to_one`` (multiple diar channels compete for one
    canonical speaker — e.g. collapsed child voices), ``ambiguous_split_and_merge``,
    or ``unmatched_acoustic_activity`` (e.g. music/SFX false positives).
    """
    scene_diar: dict[str, list[tuple[float, float]]] = defaultdict(list)
    scene_diar_segments: list[dict[str, Any]] = []
    for seg in diar_segments:
        s_raw = seg.get("start")
        e_raw = seg.get("end")
        if s_raw is None or e_raw is None:
            continue
        s = max(scene_start_s, float(s_raw))
        e = min(scene_end_s, float(e_raw))
        if e <= s:
            continue
        ch_raw = seg.get("speaker")
        ch = f"diar_spk_{ch_raw}" if not str(ch_raw).startswith("diar_spk_") else str(ch_raw)
        scene_diar[ch].append((s, e))
        scene_diar_segments.append(
            {
                "local_speaker_channel": ch,
                "source_time_interval": [round(s, 3), round(e, 3)],
            }
        )

    canon_by_id: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for utt in canonical_utterances:
        us = utt.get("source_start")
        ue = utt.get("source_end")
        if us is None or ue is None:
            continue
        s = max(scene_start_s, float(us))
        e = min(scene_end_s, float(ue))
        if e <= s:
            continue
        gid = str(utt.get("global_speaker_id") or utt.get("speaker_id") or "SPK_UNK_UNKNOWN")
        canon_by_id[gid].append((s, e))

    # First compute pairwise overlap & Jaccard between each local channel and canonical speaker
    ch_candidates: dict[str, list[dict[str, Any]]] = {}
    canon_to_strong_channels: dict[str, list[str]] = defaultdict(list)

    for ch, ch_ints in sorted(scene_diar.items()):
        ch_dur = _union_duration(ch_ints)
        cands: list[dict[str, Any]] = []
        merged_ch = _merge_intervals(ch_ints)
        for gid, g_ints in sorted(canon_by_id.items()):
            merged_g = _merge_intervals(g_ints)
            ov = sum(_interval_overlap(a0, a1, b0, b1) for a0, a1 in merged_ch for b0, b1 in merged_g)
            if ov < min_overlap_s:
                continue
            g_dur = _union_duration(merged_g)
            union = ch_dur + g_dur - ov
            ch_frac = ov / ch_dur if ch_dur > 0 else 0.0
            g_frac = ov / g_dur if g_dur > 0 else 0.0
            jacc = ov / union if union > 0 else 0.0
            cands.append(
                {
                    "global_speaker_id": gid,
                    "overlap_s": round(ov, 3),
                    "channel_overlap_fraction": round(ch_frac, 4),
                    "canonical_overlap_fraction": round(g_frac, 4),
                    "jaccard": round(jacc, 4),
                }
            )
            if ch_frac >= 0.30 and ov >= 0.5:
                canon_to_strong_channels[gid].append(ch)
        cands.sort(key=lambda item: (-item["jaccard"], -item["overlap_s"], item["global_speaker_id"]))
        ch_candidates[ch] = cands

    channel_records: list[dict[str, Any]] = []
    advisory_flags: list[dict[str, Any]] = []

    for ch, ch_ints in sorted(scene_diar.items()):
        cands = ch_candidates[ch]
        ch_dur = round(_union_duration(ch_ints), 3)
        merged_ch = _merge_intervals(ch_ints)
        interval_bounds = [round(merged_ch[0][0], 3), round(merged_ch[-1][1], 3)]
        contradictions: list[dict[str, Any]] = []

        if not cands:
            status = "unmatched_acoustic_activity"
            if ch_dur >= 0.35:
                contradictions.append(
                    {
                        "type": "no_canonical_speech_overlap",
                        "detail": "Acoustic diarization activity has insufficient overlap with canonical ASR speech (possible music/SFX/untranscribed vocalization).",
                        "channel_duration_s": ch_dur,
                    }
                )
        else:
            top = cands[0]
            one_to_many = False
            if len(cands) > 1:
                second = cands[1]
                if second["channel_overlap_fraction"] >= 0.25 or (top["jaccard"] - second["jaccard"]) < ambiguity_margin:
                    one_to_many = True
                    contradictions.append(
                        {
                            "type": "channel_spans_multiple_canonical_speakers",
                            "detail": "Single local diarization channel overlaps substantially with multiple canonical speakers.",
                            "candidates": [top["global_speaker_id"], second["global_speaker_id"]],
                        }
                    )

            competing = canon_to_strong_channels.get(top["global_speaker_id"], [])
            many_to_one = len(competing) > 1
            if many_to_one:
                contradictions.append(
                    {
                        "type": "multiple_channels_for_canonical_speaker",
                        "detail": "Multiple distinct scene diarization channels map onto one canonical global_speaker_id (possible voice collapse/split).",
                        "canonical_speaker_id": top["global_speaker_id"],
                        "competing_channels": list(competing),
                    }
                )

            if one_to_many and many_to_one:
                status = "ambiguous_split_and_merge"
            elif one_to_many:
                status = "ambiguous_one_to_many"
            elif many_to_one:
                status = "ambiguous_many_to_one"
            else:
                status = "consistent_support"

        channel_records.append(
            {
                "evidence_source": EVIDENCE_SOURCE,
                "model_revision": model_revision,
                "scene_id": scene_id,
                "source_time_interval": interval_bounds,
                "active_intervals": [[round(a, 3), round(b, 3)] for a, b in merged_ch],
                "active_duration_s": ch_dur,
                "local_speaker_channel": ch,
                "supported_identity_candidates": cands,
                "contradictions": contradictions,
                "unresolved_status": status,
                "authoritative_override_applied": False,
            }
        )

    for gid, chans in sorted(canon_to_strong_channels.items()):
        if len(chans) > 1:
            advisory_flags.append(
                {
                    "type": "inspect_possible_collapsed_speaker",
                    "scene_id": scene_id,
                    "global_speaker_id": gid,
                    "distinct_diar_channels": chans,
                    "action_taken": "none_experimental_evidence_only",
                }
            )

    return {
        "evidence_source": EVIDENCE_SOURCE,
        "model_revision": model_revision,
        "scene_id": scene_id,
        "source_time_interval": [round(scene_start_s, 3), round(scene_end_s, 3)],
        "channel_count": len(channel_records),
        "canonical_speaker_count": len(canon_by_id),
        "channels": channel_records,
        "segments": scene_diar_segments,
        "advisory_flags": advisory_flags,
    }
