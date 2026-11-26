"""Non-destructive assessment of native dialogue coverage and optional acoustic activity.

An uncovered interval is not necessarily speech. A detector observation is a
nomination for investigation, never a missing word, corrected speaker or
accepted source occurrence. All coordinates are on the decoded recording clock.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter

SCHEMA = "nemovox-source-assessment/v1"
ACTIVITY_SCHEMA = "nemovox-speech-activity/v1"


def _finite(v):
    return type(v) in (float, int) and math.isfinite(v)


def _intervals(items, duration):
    """Validate and union positive intervals; preserve overlap semantics."""
    if not isinstance(items, (list, tuple)):
        raise ValueError("Activity intervals must be a list")
    spans = []
    for pair in items:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Activity interval must be a pair")
        s, e = pair
        if not (_finite(s) and _finite(e) and 0 <= s < e <= duration):
            raise ValueError("Activity interval outside decoded recording")
        spans.append([float(s), float(e)])
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _duration(spans):
    return sum(e - s for s, e in spans)


def _gaps(spans, duration):
    result = []
    last = 0.0
    for s, e in spans:
        if s > last:
            result.append([last, s])
        last = e
    if last < duration:
        result.append([last, duration])
    return result


def _intersect(a, b):
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            result.append([s, e])
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return result


def _source_identity(contract):
    native = contract["native_dialogue"]
    rows = [{k: r.get(k) for k in ("id", "source_text", "speaker_id", "source_start", "source_end",
                                    "timestamp_provenance")} for r in contract["utterances"]]
    payload = {"native_dialogue": native, "source": contract["source"],
               "source_language": contract["source_language"], "events": rows}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def assess_native_dialogue(contract, activity=None):
    """Compute a current-source report without modifying, suppressing or retiming events."""
    from nvidia_transcribe.moss_recording import validate_moss_contract

    validate_moss_contract(contract)
    native = contract["native_dialogue"]
    duration = native["decoded_duration_s"]
    rows = contract["utterances"]
    spans = _intervals([(r["source_start"], r["source_end"]) for r in rows
                        if r["source_start"] is not None], duration)
    gaps = _gaps(spans, duration)
    counts = Counter(r["speaker_id"] for r in rows)
    report = {
        "schema": SCHEMA, "status": "assessed", "scope": "geometry_only",
        "source_identity": _source_identity(contract),
        "audio_sha256": native["audio_sha256"], "decoded_duration_s": duration,
        "event_count": len(rows), "observed_event_union_s": _duration(spans),
        "observed_event_intervals": spans,
        "uncovered_intervals": gaps, "uncovered_duration_s": _duration(gaps),
        "untimed_event_ids": native["untimed_event_ids"],
        "clamped_event_ids": native["clamped_event_ids"],
        "speaker_event_counts": dict(sorted(counts.items())),
        "singleton_speaker_labels": sorted(k for k, v in counts.items() if v == 1),
        "speaker_identity_verified": False, "speech_recall_verified": False,
        "source_truth_accepted": False, "applied_corrections": 0,
        "activity": {"status": "not_requested"},
        "investigation_nominations": [],
    }
    if activity is None:
        return report
    if (activity.get("schema") != ACTIVITY_SCHEMA
            or activity.get("audio_sha256") != native["audio_sha256"]
            or activity.get("decoded_duration_s") != duration):
        raise ValueError("Acoustic observations do not bind the native recording audio")
    tracks = activity.get("event_intervals")
    if not isinstance(tracks, dict) or set(tracks) != {"speech", "singing", "music"}:
        raise ValueError("Expected distinct speech, singing and music observations")
    union = {k: _intervals(v, duration) for k, v in tracks.items()}
    speech = union["speech"]
    # Every nomination is an actual detector/geometry intersection. Do not
    # discard speech that overlaps music/singing or treat either as a speaker.
    candidates = _intersect(speech, gaps)
    nominations = []
    for s, e in candidates:
        singing = _duration(_intersect([[s, e]], union["singing"]))
        music = _duration(_intersect([[s, e]], union["music"]))
        nominations.append({"start_s": s, "end_s": e, "speech_activity_s": e-s,
                            "singing_overlap_s": singing, "music_overlap_s": music,
                            "kind": "speech_activity_outside_native_events",
                            "status": "unresolved", "source_truth_accepted": False})
    report.update(scope="geometry_and_acoustic_activity",
                  activity={"status": "observed", "resource_identity": activity["resource_identity"],
                            "speech_union_s": _duration(speech),
                            "speech_outside_event_union_s": _duration(candidates),
                            "speech_inside_event_union_s": _duration(_intersect(speech, spans)),
                            "singing_union_s": _duration(union["singing"]),
                            "music_union_s": _duration(union["music"])},
                  investigation_nominations=nominations)
    return report
