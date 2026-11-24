"""Current-job source quality: bounded acoustic observations before translation.

Source selection consumes fresh, separately extracted audio contexts, not a
historical subtitle or manually constructed revision dossier. All timestamps
and text selected here remain model observations, not human-verified truth.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from nvidia_transcribe.asr_integrity import audit_asr_generation
from nvidia_transcribe.client import InferenceRequestConfig
from nvidia_transcribe.media import extract_wav_segment, probe_media, validate_wav_file
from nvidia_transcribe.parser import split_lexical_units
from nvidia_transcribe.semantic_review import run_tiered_text_reviews


@dataclass(frozen=True)
class SourceQualityPolicy:
    max_windows: int = 6
    max_target_seconds: float = 14.0
    timing_tolerance_s: float = 0.85
    max_text_cases: int = 6
    max_text_tokens: int = 9472

    def __post_init__(self):
        if (not 1 <= self.max_windows <= 24 or not 1 <= self.max_text_cases <= 24
                or not 1792 <= self.max_text_tokens <= 49152
                or not 4 <= self.max_target_seconds <= 14
                or not 0 < self.timing_tolerance_s <= 2):
            raise ValueError("Invalid source quality budget")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


def _bounds(row: Mapping[str, Any], start="source_start", end="source_end"):
    a, b = row.get(start), row.get(end)
    if (isinstance(a, (int, float)) and not isinstance(a, bool)
            and isinstance(b, (int, float)) and not isinstance(b, bool)
            and math.isfinite(a) and math.isfinite(b) and 0 <= a < b):
        return float(a), float(b)
    return None


def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _lex(text):
    # Lexical matching establishes corroboration, not a release WER target.
    result = []
    for item in split_lexical_units(str(text or "")):
        key = "".join(c for c in unicodedata.normalize("NFKC", item["text"]).casefold()
                      if c.isalnum() or unicodedata.category(c).startswith("M"))
        if key:
            result.append(key)
    return tuple(result)


def _contains(outer, inner):
    return bool(inner) and any(outer[i:i + len(inner)] == inner
                               for i in range(len(outer) - len(inner) + 1))


def load_current_ledger(recording, recording_path):
    """Require a completed current recording and its exact observation sidecar."""
    root = Path(recording_path).parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    data = (root / "source-evidence.v1.json").read_bytes()
    ledger = json.loads(data)
    if (manifest.get("asr_run_status") != "completed"
            or (manifest.get("source_evidence") or {}).get("sha256")
            != hashlib.sha256(data).hexdigest()
            or ledger.get("schema_version") != "nemovox-source-evidence/v1"
            or ledger.get("media_fingerprint_sha256")
            != recording["source"]["fingerprint_sha256"]
            or ledger.get("source_language") != recording.get("source_language")):
        raise ValueError("Current source evidence is missing, stale or incompatible")
    expected = {r["chunk_index"]: r["raw_response_sha256"]
                for r in recording["processing"]["chunks"]}
    observed = {r["chunk_index"]: r["raw_response_sha256"]
                for r in ledger["asr_observations"]}
    if (not expected or expected != observed
            or len(expected) != len(recording["processing"]["chunks"])
            or len(observed) != len(ledger["asr_observations"])):
        raise ValueError("Source evidence does not belong to this recording")
    return ledger


def plan_windows(recording, ledger, request, advisory, policy):
    """Prioritize current material boundary conflicts, then unrepresented activity."""
    duration = float(recording["source"]["duration_s"])
    decisions = {r["case_id"]: r.get("assessment", {}).get("decision")
                 for r in (advisory or {}).get("outcomes", [])
                 if r.get("status") == "advisory_only"}
    proposals = []
    for case in request.get("cases", []):
        a, b = case["context_window_s"]
        if not 0 <= a < b <= duration:
            continue
        # Context expansion can discover dialogue immediately preceding a
        # known partial observation, not only recover its exact words.
        target = (max(0., a - 3.), min(duration, b + 1.))
        proposals.append({
            "id": case["case_id"], "window": target,
            "chunks": list(case["source_chunks"]),
            "reason": "boundary_conflict",
            "advisory": decisions.get(case["case_id"], "not_reviewed"),
        })
        displaced = case.get("left_lexical_correspondence", {}).get("left_words") or []
        if displaced:
            starts = [w["start_s"] for w in displaced]
            ends = [w["end_s"] for w in displaced]
            old = (max(0., min(starts) - 1.), min(duration, max(ends) + 1.))
            if old[1] > old[0] and not _overlap(target, old):
                proposals.append({
                    "id": case["case_id"] + ":competing_time",
                    "window": old, "chunks": list(case["source_chunks"]),
                    "reason": "competing_timing_hypothesis",
                    "advisory": decisions.get(case["case_id"], "not_reviewed"),
                })
    parents = {r["id"]: r["chunk_index"] for r in ledger["asr_observations"]}
    for activity in ledger["diarization_activity_hypotheses"]:
        if not activity.get("possible_unrepresented_speech"):
            continue
        bounds = _bounds(activity, "start_s", "end_s")
        if not bounds or bounds[1] - bounds[0] < .65 or bounds[1] > duration:
            continue
        proposals.append({
            "id": activity["id"], "window": (max(0., bounds[0] - 1.),
                                               min(duration, bounds[1] + 1.)),
            "chunks": [parents[activity["asr_observation_id"]]],
            "reason": "activity_without_lexical_timing",
            "advisory": "not_reviewed",
        })
    # A text-only retain decision cannot veto a contradictory acoustic
    # observation. It only affects bounded scheduling priority.
    proposals.sort(key=lambda p: (
        p["reason"] == "activity_without_lexical_timing",
        p["advisory"] == "retain_core",
        p["reason"] == "competing_timing_hypothesis", p["window"][0],
    ))
    windows = []
    for p in proposals:
        a, b = p["window"]
        while a < b:
            part = (a, min(b, a + policy.max_target_seconds))
            existing = next((w for w in windows if
                             _overlap(w["window"], part) >= .8 * (part[1] - part[0])), None)
            if existing is not None:
                existing["evidence_ids"].append(p["id"])
                existing["chunks"] = sorted(set(existing["chunks"] + p["chunks"]))
            else:
                windows.append({"window": part, "evidence_ids": [p["id"]],
                                "chunks": p["chunks"], "reason": p["reason"]})
            a = part[1]
    return windows[:policy.max_windows], windows[policy.max_windows:]


def _crop_windows(target, duration):
    a, b = target
    crops = [(max(0., a - 2.), min(duration, b + 5.)),
             (max(0., a - 5.), min(duration, b + 2.))]
    # Distinct offsets or endings are essential; two replays of identical
    # input are not independently acquired acoustic contexts.
    return crops if crops[0] != crops[1] else []


def _observe(client, media, crop, root, index):
    a, b = crop
    path = root / f"probe-{index:03d}.wav"
    extract_wav_segment(media.path, start_s=a, duration_s=b - a,
                        out_wav=path, audio_stream_index=media.selected_audio_stream)
    actual, wav_sha = validate_wav_file(path)
    if abs(actual - (b - a)) > .1:
        raise ValueError("Acoustic challenge duration disagrees with planned crop")
    result = client.transcribe(path, InferenceRequestConfig(
        mode="speaker", source_language=media.source_language,
        capture_ctc_timestamps=True, source_offset_s=a,
        apply_loop_guardrail=False,
    ))
    raw = result["raw_response"]
    audit = audit_asr_generation(raw)
    raw_text = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content")
    source_lex = _lex(raw_text)
    rows = []
    if audit["status"] == "complete" and audit["finish_reason"] == "stop":
        for u in result["contract"].get("utterances", []):
            bounds = _bounds(u)
            lexical = _lex(u.get("source_text"))
            # CTC-generated text must also occur in actual ASR source text.
            if (bounds and a - .01 <= bounds[0] < bounds[1] <= b + .01
                    and len(lexical) >= 2 and _contains(source_lex, lexical)):
                rows.append({"source_text": u["source_text"], "source_start": bounds[0],
                             "source_end": bounds[1], "local_voice": u.get("speaker_id"),
                             "observation_id": f"probe-{index:03d}:{u['id']}"})
    return {"crop": crop, "audio_sha256": wav_sha, "raw_response_sha256": _digest(raw),
            "model": result["contract"].get("model_provenance", {}).get("served_model"),
            "integrity": audit, "utterances": rows, "raw_response": raw}


def corroborated_events(first, second, target, tolerance):
    """Select unique same-event observations in two different audio contexts.

    Repetition remains distinct when timing separates its occurrences. A
    disagreement or ambiguous pairing remains unresolved, not a deletion.
    """
    if (first["audio_sha256"] == second["audio_sha256"]
            or first["crop"] == second["crop"]):
        return []
    left, right = first["utterances"], second["utterances"]

    def match(a, b):
        x, y = _bounds(a), _bounds(b)
        return (x and y and _lex(a["source_text"]) == _lex(b["source_text"])
                and abs(x[0] - y[0]) <= tolerance and abs(x[1] - y[1]) <= tolerance
                and _overlap(x, y) > 0
                and target[0] <= (x[0] + x[1]) / 2 <= target[1]
                and target[0] <= (y[0] + y[1]) / 2 <= target[1])

    events = []
    for a in left:
        matches = [b for b in right if match(a, b)]
        if len(matches) != 1:
            continue
        b = matches[0]
        if sum(bool(match(other, b)) for other in left) != 1:
            continue
        # Choose an observed interval, never fabricate a midpoint timestamp.
        events.append({**a, "supporting_observation_ids": [a["observation_id"], b["observation_id"]],
                       "audio_sha256": [first["audio_sha256"], second["audio_sha256"]],
                       "raw_response_sha256": [first["raw_response_sha256"], second["raw_response_sha256"]]})
    return events


def select_source_events(recording, events, policy):
    """Use positive corroboration to fill gaps or extend a single coherent turn.

    No deletion of distinct source events based on lexical similarity, lack
    of model coverage, or text review. Displaced repeats and competing voices
    are explicit unresolved alternatives rather than falsely certain repairs.
    """
    rows = [dict(r) for r in recording["utterances"]]
    decisions = []
    media_id = recording["source"]["fingerprint_sha256"]
    def corroborated_at_own_time(row, event):
        original = _bounds(row)
        target = _bounds(event)
        return bool(original and target and not _overlap(original, target)
                    and any(_bounds(other)
                            and _lex(other["source_text"]) == _lex(row["source_text"])
                            and abs(_bounds(other)[0] - original[0]) <= policy.timing_tolerance_s
                            and abs(_bounds(other)[1] - original[1]) <= policy.timing_tolerance_s
                            and _overlap(_bounds(other), original) > 0
                            for other in events))

    for event in sorted(events, key=lambda e: (e["source_start"], e["source_end"])):
        bounds = _bounds(event)
        words = _lex(event["source_text"])
        if not bounds or not words:
            continue
        overlaps = [u for u in rows if _bounds(u) and _overlap(_bounds(u), bounds) > .08]
        nearby_matches = [u for u in rows if
                          (_contains(_lex(u["source_text"]), words)
                           or _contains(words, _lex(u["source_text"])))
                          and (set(u.get("chunk_indices") or []) & set(event["chunks"])
                               or (_bounds(u) and abs(_bounds(u)[0] - bounds[0]) <= 60))]
        status = "unresolved_competing_source"
        replace = None
        if any(_lex(u["source_text"]) == words for u in overlaps):
            status = "already_represented"
        elif (len(overlaps) == 1 and _contains(words, _lex(overlaps[0]["source_text"]))
                and bounds[0] <= _bounds(overlaps[0])[0] + policy.timing_tolerance_s
                and bounds[1] >= _bounds(overlaps[0])[1] - policy.timing_tolerance_s
                and not any(u is not overlaps[0] for u in nearby_matches)):
            replace = overlaps[0]
            status = "extended_corroborated_turn"
        elif not overlaps and len(nearby_matches) == 1 and _bounds(nearby_matches[0]) is None:
            if _lex(nearby_matches[0]["source_text"]) == words:
                replace = nearby_matches[0]
                status = "timed_corroborated_turn"
        elif not overlaps and all(corroborated_at_own_time(u, event) for u in nearby_matches):
            status = "recovered_corroborated_turn"
        elif not overlaps:
            status = "unresolved_possible_displaced_or_repeated_turn"
        if status in {"extended_corroborated_turn", "timed_corroborated_turn",
                      "recovered_corroborated_turn"}:
            uid = replace["id"] if replace else "src-" + _digest(
                [media_id, event["supporting_observation_ids"], bounds])[:24]
            row = {
                "id": uid, "utterance_id": uid, "source_turn_id": uid,
                "source_turn_ids": [uid],
                "source_language": recording.get("source_language"),
                "source_text": event["source_text"], "source_start": bounds[0],
                "source_end": bounds[1],
                "speaker_id": "SPK_UNK", "speaker_identity_accepted": False,
                "source_words": [], "source_characters": [], "unit_count": 0,
                "timed_unit_count": 0, "untimed_unit_count": 0, "timing_gaps": [],
                "chunk_indices": event["chunks"],
                "alignment_status": "corroborated_model_event_bounds",
                "source_quality_status": status,
                "timestamp_provenance": {
                    "backend": "separate_acoustic_contexts",
                    "timing_status": "event_bounds_model_corroborated",
                    "human_verified": False,
                    "audio_sha256": event["audio_sha256"],
                    "raw_response_sha256": event["raw_response_sha256"],
                    "observation_ids": event["supporting_observation_ids"],
                },
            }
            if replace is not None:
                row["prior_source_hypothesis"] = dict(replace)
                rows.remove(replace)
            if any(u["id"] == uid for u in rows):
                raise ValueError("Recovered source event ID collision")
            # Preserve order of unrelated/untimed source rows.
            pos = next((i for i, u in enumerate(rows) if _bounds(u)
                        and _bounds(u)[0] > bounds[0]), len(rows))
            rows.insert(pos, row)
        decisions.append({"status": status, "observations": event["supporting_observation_ids"],
                          "window": bounds,
                          "superseded_source_id": replace["id"] if replace else None})
    return rows, decisions


def review_current_source(recording, *, recording_path, media_path, output_dir,
                          text_scope, asr_scope, policy=None):
    """Review one current job, with explicitly sequential managed model scopes."""
    policy = policy or SourceQualityPolicy()
    ledger = load_current_ledger(recording, recording_path)
    source = recording["source"]
    media = probe_media(media_path, audio_stream_index=source["selected_audio_stream"])
    if (media.fingerprint_sha256 != source["fingerprint_sha256"]
            or abs(media.duration_s - source["duration_s"]) > .01):
        raise ValueError("Acoustic review media does not match current recording")
    # Use immutable media metadata plus language without changing MediaInfo.
    from types import SimpleNamespace
    observed_media = SimpleNamespace(path=media.path, duration_s=media.duration_s,
                                     selected_audio_stream=media.selected_audio_stream,
                                     source_language=recording["source_language"])
    request = recording["processing"]["boundary_reconciliation"]["semantic_review_candidates"]
    advisory = None
    if request["cases"]:
        with text_scope() as text:
            if text is None:
                raise ValueError("Source quality requires an admitted NInfer text client")
            advisory = run_tiered_text_reviews(
                request, client=text, case_limit=policy.max_text_cases,
                token_budget=policy.max_text_tokens, light_tokens=512, reasoned_tokens=1792,
                light_effort="none", reasoned_effort="xhigh",
                light_thinking_budget=0, reasoned_thinking_budget=1024,
            )
    windows, deferred = plan_windows(recording, ledger, request, advisory, policy)
    observations, events, unresolved = [], [], []
    if windows:
        with tempfile.TemporaryDirectory(prefix="source-quality-", dir=output_dir) as tmp:
            with asr_scope() as asr:
                for w in windows:
                    crops = _crop_windows(w["window"], media.duration_s)
                    if not crops:
                        unresolved.append({"window": w, "reason": "insufficient_distinct_context"})
                        continue
                    pair = []
                    for crop in crops:
                        item = _observe(asr, observed_media, crop, Path(tmp), len(observations))
                        observations.append(item)
                        pair.append(item)
                    selected = corroborated_events(*pair, w["window"], policy.timing_tolerance_s)
                    for item in selected:
                        item["chunks"] = w["chunks"]
                        item["trigger_evidence_ids"] = w["evidence_ids"]
                    events.extend(selected)
                    matched = {oid for item in selected for oid in item["supporting_observation_ids"]}
                    unmatched = [u["observation_id"] for observation in pair
                                 for u in observation["utterances"]
                                 if w["window"][0] <= (u["source_start"] + u["source_end"]) / 2 <= w["window"][1]
                                 and u["observation_id"] not in matched]
                    if not selected or unmatched:
                        unresolved.append({"window": w, "reason": "unmatched_acoustic_observations"
                                           if unmatched else "no_unique_corroborated_event",
                                           "unmatched_observation_ids": unmatched})
    rows, decisions = select_source_events(recording, events, policy)
    reviewed_case_ids = {o["case_id"] for o in (advisory or {}).get("outcomes", [])}
    deferred_case_ids = [c["case_id"] for c in request["cases"]
                         if c["case_id"] not in reviewed_case_ids]
    discovery_deferred = request.get("deferred_by_budget", 0)
    uncertain = (bool(unresolved or deferred or deferred_case_ids or discovery_deferred)
                 or any(d["status"].startswith("unresolved") for d in decisions)
                 or bool(advisory and (advisory.get("systemic_failure")
                          or any(o["status"] != "advisory_only" for o in advisory["outcomes"]))))
    report = {
        "schema": "nemovox-current-source-quality/v1",
        "status": "partial" if uncertain else "complete",
        "media_fingerprint_sha256": media.fingerprint_sha256,
        "recording_sha256": _digest(recording),
        "source_language": recording["source_language"],
        "text_advisory": advisory, "deferred_text_case_ids": deferred_case_ids,
        "discovery_deferred_by_budget": discovery_deferred,
        "windows": windows, "deferred_windows": deferred,
        "acoustic_observations": observations, "decisions": decisions,
        "unresolved_windows": unresolved, "audio_requests": len(observations),
        "source_truth_human_verified": False,
        "independent_model_agreement": False,
        "source_word_timing_verified": False,
        "original_recording_modified": False,
    }
    # Store raw challenge observations in the current private job, not a
    # historical correction tree. The canonical recording references them.
    return {**recording, "utterances": rows, "source_quality": {
        k: v for k, v in report.items() if k != "acoustic_observations"
    }}, report
