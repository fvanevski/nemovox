"""Generate model-bound AudioRegion *proposals* for bounded source ASR.

This producer never transcribes, assigns source words to a full-window result,
or asserts a canonical speaker/event. Speaker-aware scene tracks take
precedence as separate turn candidates; AED/VAD remains soft overlap evidence.
Known direct-audition events are kept separately. Full coverage-window ASR
remains an unlocalized hypothesis until segmented and acoustically bounded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from nvidia_transcribe.source_recovery import _id, _interval, _overlap

SCHEMA = "nemovox-audio-region-proposals/v1"
SCENE_SCHEMA = "nemovox-scene-turn-evidence/v1"
AUDITED_SCHEMA = "nemovox-audited-source-events/v1"
ACTIVITY_SCHEMA = "nemovox-audio-activity-evidence/v1"
DOSSIER_SCHEMA = "nemovox-source-reconciliation/v1"
MAX_WINDOW_S = 24.0
SPEECH_KINDS = frozenset({"speech", "lexical_shout"})
NEGATIVE_KINDS = frozenset({"singing", "music", "sfx", "silence", "nonlexical_vocalization"})


def _positive_float(x: Any, name: str) -> float:
    if not isinstance(x, (float, int)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive finite")
    return float(x)


def _crop(start: float, end: float, duration: float, margin_s: float) -> tuple[float, float]:
    # Expand short turns to provide ASR context, avoiding unreliable isolated
    # 2-3s inference; do not silently cut away either end of an event.
    left = max(0.0, start - margin_s)
    right = min(duration, end + margin_s)
    if right - left < 4.0 and duration >= 4.0:
        extra = 4.0 - (right-left)
        left = max(0.0, left-extra/2)
        right = min(duration, max(right, left+4.0))
        left = max(0.0, right-4.0)
    if right-left > MAX_WINDOW_S:
        raise ValueError("Audio region requires bounded subsegmentation before transcription")
    return round(left, 6), round(right, 6)


def plan_audio_regions(
    *,
    media_fingerprint_sha256: str,
    source_language: str,
    media_duration_s: float,
    activity: Mapping[str, Any],
    dossier: Mapping[str, Any] | None = None,
    audited_events: Mapping[str, Any] | None = None,
    scene_tracks: Mapping[str, Any] | None = None,
    margin_s: float = 3.0,
) -> dict[str, Any]:
    """Stage scene/acoustic proposals, never a source transcript.

    Scene tracks may represent simultaneous voices and are NOT merged.
    Without scene tracks, adjacent same-detector activity can only propose a
    *review interval*, not a single accepted source event.
    """
    if not isinstance(media_fingerprint_sha256, str) or not media_fingerprint_sha256:
        raise ValueError("Media fingerprint required")
    if source_language not in {"fr", "zh"}:
        raise ValueError("French/Mandarin source language must be explicit")
    duration = _positive_float(media_duration_s, "media duration")
    margin = _positive_float(margin_s, "ASR crop margin")
    for name, doc, schema in (
        ("activity", activity, ACTIVITY_SCHEMA),
        ("dossier", dossier, DOSSIER_SCHEMA),
        ("audited", audited_events, AUDITED_SCHEMA),
        ("scene", scene_tracks, SCENE_SCHEMA),
    ):
        if doc is not None and (doc.get("schema") != schema
                                or doc.get("media_fingerprint_sha256") != media_fingerprint_sha256):
            raise ValueError(f"{name} schema or media fingerprint mismatch")

    evidence = []
    ids = set()
    for e in activity.get("intervals") or []:
        a, b = _interval(e)
        if b > duration + .001 or e.get("kind") not in SPEECH_KINDS | NEGATIVE_KINDS | {"unknown"}:
            raise ValueError("Activity time or kind not supported")
        evid = str(e.get("id") or "")
        if not evid or evid in ids or not e.get("source"):
            raise ValueError("Duplicate or incomplete activity provenance")
        ids.add(evid)
        evidence.append({"id": evid, "start_s": a, "end_s": b,
                         "source": str(e["source"]), "kind": e["kind"]})

    windows = []
    for w in (dossier or {}).get("model_windows") or []:
        a, b = _interval(w)
        if b > duration + .001 or not w.get("id") or w.get("source_content_accepted") is True:
            raise ValueError("Challenger window is invalid or prematurely accepted")
        windows.append({"id": w["id"], "start_s": a, "end_s": b,
                        "has_model_text": bool(w.get("source_text")),
                        "source_status": w.get("status")})

    proposals = []
    def append_region(a: float, b: float, source: str, *, audio_event_id: str | None = None,
                      speaker_channel: str | None = None, input_id: str = "") -> None:
        if b > duration + .001:
            raise ValueError("Source acoustic region extends beyond media duration")
        length = b-a
        # A 25-second activity span is NOT one identifiable speech turn; it
        # must be explicitly split before any model-only source promotion.
        requires_split = length > MAX_WINDOW_S-2*margin
        bounds = (a, b)
        labels = [
            {"id": e["id"], "kind": e["kind"], "source": e["source"]}
            for e in evidence if _overlap(*bounds, e["start_s"], e["end_s"]) > 0
        ]
        context = [w["id"] for w in windows
                   if _overlap(*bounds, w["start_s"], w["end_s"]) > 0]
        crop = None if requires_split else _crop(a, b, duration, margin)
        proposals.append({
            "id": _id("areg", [media_fingerprint_sha256, source, input_id, a, b, speaker_channel]),
            "start_s": a, "end_s": b,
            "region_source": source,
            "source_id": input_id,
            "audio_event_id": audio_event_id,
            "speaker_channel_hypothesis": speaker_channel,
            "speech_activity_ids": [e["id"] for e in labels if e["kind"] in SPEECH_KINDS],
            "non_speech_activity_ids": [e["id"] for e in labels if e["kind"] in NEGATIVE_KINDS],
            "model_context_window_ids": context,
            "bounded_crop_start_s": crop[0] if crop else None,
            "bounded_crop_end_s": crop[1] if crop else None,
            "needs_subsegmentation": requires_split,
            "voice_classification_status": "independent_event_auditioned" if audio_event_id else "model_advisory",
            "model_text_accepted": False,
            "speaker_identity_accepted": False,
        })

    scene = (scene_tracks or {}).get("turns") or []
    if scene:
        seen = set()
        for t in scene:
            tid = str(t.get("id") or "")
            channel = str(t.get("speaker_channel") or "")
            if not tid or tid in seen or not channel:
                raise ValueError("Scene turn needs unique ID and speaker channel")
            seen.add(tid)
            a, b = _interval(t)
            append_region(a, b, "scene_diarization_advisory",
                          speaker_channel=channel, input_id=tid)
    else:
        # Different model detectors may report the same voice. For a
        # reasonable CPU-only crop task list, merge only overlapping/nearby
        # activity from the SAME model; NEVER merge across voice channels.
        grouped: dict[str, list[tuple[float, float, str]]] = {}
        for e in evidence:
            if e["kind"] in SPEECH_KINDS:
                grouped.setdefault(e["source"], []).append((e["start_s"], e["end_s"], e["id"]))
        for source, spans in sorted(grouped.items()):
            accum: list[tuple[float, float, list[str]]] = []
            for a, b, evidence_id in sorted(spans):
                if accum and a <= accum[-1][1] + .15:
                    prev_a, prev_b, prev_ids = accum[-1]
                    accum[-1] = (prev_a, max(prev_b, b), prev_ids + [evidence_id])
                else:
                    accum.append((a, b, [evidence_id]))
            for a, b, evidence_ids in accum:
                append_region(a, b, "activity_region_proposal", input_id=";".join(evidence_ids))

    seen_audited = set()
    for e in (audited_events or {}).get("events") or []:
        if e.get("acoustic_truth") != "independently_auditioned":
            raise ValueError("Cannot import unverified review event")
        eid = str(e.get("id") or "")
        if not eid or eid in seen_audited:
            raise ValueError("Duplicate or empty audited event ID")
        seen_audited.add(eid)
        a, b = _interval(e)
        if e.get("kind") in SPEECH_KINDS:
            append_region(a, b, "direct_audio_audition_event",
                          audio_event_id=eid, input_id=eid)

    proposals.sort(key=lambda x: (x["start_s"], x["end_s"], x["region_source"], x["id"]))
    unlocalized = []
    for window in windows:
        if not window["has_model_text"]:
            continue
        if any(p["model_context_window_ids"] and window["id"] in p["model_context_window_ids"]
               for p in proposals):
            # Window still has transcript parts outside the proposed turn.
            # Always record that the full text remains UNLOCALIZED.
            pass
        unlocalized.append({
            "model_context_window_id": window["id"],
            "start_s": window["start_s"], "end_s": window["end_s"],
            "status": "window_text_unlocalized_requires_bound_asr_or_review",
            "may_contain_multiple_speakers_music_or_repetitions": True,
        })
    return {
        "schema": SCHEMA,
        "media_fingerprint_sha256": media_fingerprint_sha256,
        "source_language": source_language,
        "media_duration_s": duration,
        "regions": proposals,
        "unlocalized_model_windows": unlocalized,
        "policy": {
            "no_negative_vad_audio_exclusion": True,
            "no_scene_turns_merged_across_speaker_channels": True,
            "no_context_window_text_as_source_event": True,
            "auditioned_source_events_distinct_from_model_proposals": True,
            "pending_long_turns_require_subsegmentation": True,
        },
        "diagnostics": {
            "region_proposals": len(proposals),
            "auditioned_source_events": sum(p["audio_event_id"] is not None for p in proposals),
            "scene_turn_proposals": sum(p["region_source"] == "scene_diarization_advisory" for p in proposals),
            "requires_subsegmentation": sum(p["needs_subsegmentation"] for p in proposals),
            "unlocalized_model_windows": len(unlocalized),
            "source_fidelity_certified": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-fingerprint", required=True)
    parser.add_argument("--source-language", choices=["fr", "zh"], required=True)
    parser.add_argument("--media-duration-s", type=float, required=True)
    parser.add_argument("--activity", required=True, type=Path)
    parser.add_argument("--reconciliation-dossier", type=Path)
    parser.add_argument("--audited-events", type=Path)
    parser.add_argument("--scene-turns", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse overwrite of an existing AudioRegion plan")
    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))
    report = plan_audio_regions(
        media_fingerprint_sha256=args.media_fingerprint,
        source_language=args.source_language,
        media_duration_s=args.media_duration_s,
        activity=load(args.activity),
        dossier=load(args.reconciliation_dossier) if args.reconciliation_dossier else None,
        audited_events=load(args.audited_events) if args.audited_events else None,
        scene_tracks=load(args.scene_turns) if args.scene_turns else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["diagnostics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
