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
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping

from nvidia_transcribe.asr_integrity import audit_asr_generation
from nvidia_transcribe.client import InferenceRequestConfig
from nvidia_transcribe.media import extract_wav_segment, probe_media, validate_wav_file
from nvidia_transcribe.parser import split_lexical_units
from nvidia_transcribe.semantic_review import run_tiered_text_reviews


@dataclass(frozen=True)
class SourceQualityPolicy:
    max_windows: int = 18
    max_extra_contexts: int = 8
    max_target_seconds: float = 14.0
    timing_tolerance_s: float = 0.85
    max_text_cases: int = 6
    max_text_tokens: int = 9472
    # Relative geometry, not fixed episode-specific crop sizes.
    challenge_scales: tuple[float, ...] = (.65, 1.5, 2.5)
    challenge_max_seconds: float = 45.
    challenge_overlap_fraction: float = .35
    max_extra_per_window: int = 3
    max_acoustic_audio_s: float = 720.
    max_acoustic_wall_s: float = 600.
    max_independent_targets: int = 6
    max_independent_audio_s: float = 240.
    max_independent_wall_s: float = 900.
    scene_context_s: float = 60.
    scene_max_context_s: float = 180.
    scene_context_char_budget: int = 24000
    scene_max_cases: int = 6
    scene_max_frames: int = 6

    def __post_init__(self):
        if (not 1 <= self.max_windows <= 24 or not 0 <= self.max_extra_contexts <= 12
                or not 1 <= self.max_text_cases <= 24
                or not 1792 <= self.max_text_tokens <= 49152
                or not 4 <= self.max_target_seconds <= 14
                or not 0 < self.timing_tolerance_s <= 2
                or not self.challenge_scales
                or any(not math.isfinite(x) or not .3 <= x <= 5 for x in self.challenge_scales)
                or not 14 <= self.challenge_max_seconds <= 90
                or not .1 <= self.challenge_overlap_fraction <= .49
                or not 1 <= self.max_extra_per_window <= 6
                or not 1 <= self.max_independent_targets <= 12
                or any(not math.isfinite(x) or x <= 0 for x in (
                    self.max_acoustic_audio_s,self.max_acoustic_wall_s,
                    self.max_independent_audio_s,self.max_independent_wall_s))
                or not 0 < self.scene_context_s <= self.scene_max_context_s <= 600
                or not 1000 <= self.scene_context_char_budget <= 64000
                or not 1 <= self.scene_max_cases <= 24
                or not 3 <= self.scene_max_frames <= 10):
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


def _same_event_bounds(a, b, tolerance):
    """Temporal correspondence requires agreement at both event boundaries."""
    return bool(a and b and _overlap(a, b) > 0
                and abs(a[0] - b[0]) <= tolerance
                and abs(a[1] - b[1]) <= tolerance)


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
    from nvidia_transcribe.recovery_planner import plan_recovery_windows
    return plan_recovery_windows(recording, ledger, request, advisory, policy)


def _crop_windows(target, duration):
    a, b = target
    crops = [(max(0., a - 2.), min(duration, b + 5.)),
             (max(0., a - 5.), min(duration, b + 2.))]
    # Distinct offsets or endings are essential; two replays of identical
    # input are not independently acquired acoustic contexts.
    return crops if crops[0] != crops[1] else []


def challenge_geometries(target,duration,policy,*,timing_problem=False,seen=()):
    """Order alternative geometries using the observed failure mechanism.

    Resize relative to target length, shift nominal boundary into the
    disputed region with overlap, clip to media, and avoid duplicate crops.
    Each candidate is an independent observation, not a replacement chunk.
    """
    a,b=target
    length=b-a
    midpoint=(a+b)/2
    def window(center,size):
        size=min(policy.challenge_max_seconds,max(.5,size))
        return (max(0.,center-size/2),min(duration,center+size/2))
    compact=[("compact",window(midpoint,length*x)) for x in policy.challenge_scales if x<1]
    expanded=[("expanded",window(midpoint,length*x)) for x in policy.challenge_scales if x>=1]
    overlap=length*policy.challenge_overlap_fraction
    extent=min(policy.challenge_max_seconds/2,max(length*.8,overlap))
    split=[("shift_left",(max(0.,midpoint-extent),min(duration,midpoint+overlap))),
           ("shift_right",(max(0.,midpoint-overlap),min(duration,midpoint+extent)))]
    # Reserve adjacency of the shifted pair before further scales.
    # Prioritize compact audio for alignment failure; expanded audio for
    # lexical ambiguity. Remaining scales stay available within the budget.
    ordered=(compact[:1]+split+compact[1:]+expanded if timing_problem
             else expanded[:1]+split+expanded[1:]+compact)
    used={tuple(x) for x in seen}
    result=[]
    for kind,crop in ordered:
        if crop[1]-crop[0]>=.5 and crop not in used:
            used.add(crop)
            result.append((kind,crop))
    return result


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
    # Apply the same recognized-spine and locality rules as primary ASR,
    # not the unaudited direct transcript projection.
    from nvidia_transcribe.acoustic_observations import reconstruct_acoustic_observation
    reconstructed, inventory = reconstruct_acoustic_observation(
        raw, audio_sha256=wav_sha, crop=crop, language=media.source_language)
    rows = []
    rejected = []
    if audit["status"] == "complete" and audit["finish_reason"] == "stop":
        for u in reconstructed:
            bounds = _bounds(u)
            lexical = _lex(u.get("source_text"))
            # CTC-generated text must also occur in actual ASR source text.
            if (bounds and a - .01 <= bounds[0] < bounds[1] <= b + .01
                    and not u.get("untimed_unit_count")
                    and not any(x.get("locality_status") == "disputed"
                                for x in u["source_words"]+u["source_characters"])
                    and len(lexical) >= 2 and _contains(source_lex, lexical)):
                timed = []
                for unit in list(u.get("source_words") or []) + list(u.get("source_characters") or []):
                    ub = _bounds(unit, "start", "end")
                    lex = _lex(unit.get("text"))
                    if (unit.get("valid_timing") and ub and len(lex) == 1
                            and bounds[0] - .01 <= ub[0] < ub[1] <= bounds[1] + .01):
                        timed.append({"lex": lex[0], "start": ub[0], "end": ub[1]})
                timed.sort(key=lambda x: (x["start"], x["end"]))
                rows.append({"source_text": u["source_text"], "source_start": bounds[0],
                             "source_end": bounds[1], "local_voice": u.get("speaker_id"),
                             "observation_id": u["id"],
                             "source_unit_ids": u["source_unit_ids"],
                             "recognized_turn_ids": u["recognized_turn_ids"],
                             "occurrence_span_ids": u["occurrence_span_ids"],
                             "timed_lexical_units": timed})
            else:
                rejected.append(u)
    return {"crop": crop, "audio_sha256": wav_sha, "raw_response_sha256": _digest(raw),
            "model": result["contract"].get("model_provenance", {}).get("served_model"),
            "integrity": audit, "utterances": rows, "rejected_utterances": rejected,
            "occurrence_inventory": inventory, "raw_response": raw}


def _observation_groups(observation):
    """Adjacent same-local-voice segments are alternative segmentations, not new speech."""
    rows = sorted(observation["utterances"], key=lambda r: (r["source_start"], r["source_end"]))
    groups = []
    for i, first in enumerate(rows):
        group = {**first, "member_ids": [first["observation_id"]]}
        groups.append(group)
        for next_row in rows[i + 1:i + 3]:
            voice = group.get("local_voice")
            if (voice in (None, "", "unknown", "SPK_UNK")
                    or voice != next_row.get("local_voice")
                    or not -.10 <= next_row["source_start"] - group["source_end"] <= 1.5
                    or next_row["source_end"] <= group["source_end"]):
                break
            # No crossing an intervening distinct voice: only consecutive rows.
            group = {**group, "source_text": group["source_text"] + " " + next_row["source_text"],
                     "source_end": next_row["source_end"],
                     "member_ids": group["member_ids"] + [next_row["observation_id"]],
                     "timed_lexical_units": (group.get("timed_lexical_units", [])
                                            + next_row.get("timed_lexical_units", []))}
            groups.append(group)
    return groups


def corroborated_events(first, second, target, tolerance):
    """Compare lexical events independently of ASR segmentation, never merge voices."""
    if first["audio_sha256"] == second["audio_sha256"] or first["crop"] == second["crop"]:
        return []
    left, right = _observation_groups(first), _observation_groups(second)

    def match(a, b):
        x, y = _bounds(a), _bounds(b)
        return (_same_event_bounds(x, y, tolerance)
                and _lex(a["source_text"]) == _lex(b["source_text"])
                and target[0] <= (x[0] + x[1]) / 2 <= target[1]
                and target[0] <= (y[0] + y[1]) / 2 <= target[1])

    # Prefer unambiguous single-row matches, then use remaining observations
    # for split/join cases. No member can support two different events.
    events, used = [], set()
    for compound in (False, True):
        pairs = [(a, b) for a in left for b in right
                 if (len(a["member_ids"]) > 1 or len(b["member_ids"]) > 1) == compound
                 and not used.intersection(a["member_ids"] + b["member_ids"])
                 and match(a, b)]
        for a, b in pairs:
            ids = a["member_ids"] + b["member_ids"]
            if used.intersection(ids):
                continue
            if sum(bool(set(ids).intersection(x["member_ids"] + y["member_ids"]))
                   for x, y in pairs) != 1:
                continue
            # Prefer the actually observed joined text/interval over a
            # constructed concatenation, especially for CJK punctuation.
            chosen = a if len(a["member_ids"]) <= len(b["member_ids"]) else b
            events.append({**chosen, "supporting_observation_ids": ids,
                           "audio_sha256": [first["audio_sha256"], second["audio_sha256"]],
                           "context_crops": [list(first["crop"]), list(second["crop"])],
                           "raw_response_sha256": [first["raw_response_sha256"], second["raw_response_sha256"]]})
            used.update(ids)
    return events


def _third_context(event, observation, tolerance):
    """Additional distinct positive support; absence is not a contradiction."""
    if (observation["audio_sha256"] in event["audio_sha256"]
            or list(observation["crop"]) in event.get("context_crops", [])):
        return False
    matches = [r for r in _observation_groups(observation)
               if _lex(r["source_text"]) == _lex(event["source_text"])
               and _same_event_bounds(_bounds(r), _bounds(event), tolerance)]
    if len(matches) != 1:
        return False
    event["supporting_observation_ids"].extend(matches[0]["member_ids"])
    event["audio_sha256"].append(observation["audio_sha256"])
    event.setdefault("context_crops", []).append(list(observation["crop"]))
    event["raw_response_sha256"].append(observation["raw_response_sha256"])
    return True


def _anchored_intervals(row, observation, bracket):
    """Locate a noisy untimed source hypothesis using observed lexical-unit edges.

    Approximate interior lexical agreement is ONLY timing association, not
    permission to replace speech. Require ordered, unique strong edge anchors.
    """
    words = _lex(row["source_text"])
    cjk = (row.get("source_language") or "").startswith(("zh", "ja"))
    anchor = 4 if cjk else 3
    slack = 2 if cjk else 1
    if len(words) < 2 * anchor + 2:
        return []
    matches = {}
    for group in _observation_groups(observation):
        units = group.get("timed_lexical_units") or []
        if len(units) < 2 * anchor:
            continue
        tokens = tuple(u["lex"] for u in units)
        for first_offset in range(slack + 1):
            prefix = words[first_offset:first_offset + anchor]
            starts = [i for i in range(len(tokens) - anchor + 1)
                      if tokens[i:i + anchor] == prefix]
            for last_offset in range(slack + 1):
                stop = len(words) - last_offset
                suffix = words[stop-anchor:stop]
                ends = [i + anchor for i in range(len(tokens) - anchor + 1)
                        if tokens[i:i + anchor] == suffix]
                for i in starts:
                    for j in ends:
                        if j - i < 2 * anchor:
                            continue
                        if not .65 <= (j-i) / (stop-first_offset) <= 1.45:
                            continue
                        ratio = SequenceMatcher(None, words[first_offset:stop], tokens[i:j],
                                                autojunk=False).ratio()
                        bounds = (units[i]["start"], units[j-1]["end"])
                        if (ratio < .70 or not bracket[0] <= bounds[0] < bounds[1] <= bracket[1]
                                or any(x["end"] > y["start"] + .10
                                       for x, y in zip(units[i:j], units[i+1:j]))):
                            continue
                        # Prefer complete edge matches over an inner fragment.
                        key = (bounds, first_offset, last_offset)
                        matches[key] = (ratio, group["member_ids"])
    if not matches:
        return []
    best_edges = min(k[1] + k[2] for k in matches)
    return [(k[0], matches[k][1], matches[k][0]) for k in matches
            if k[1] + k[2] == best_edges]


def _recover_untimed(recording, observations, policy):
    """Recover only source-event timing, without changing lexical or voice hypotheses."""
    rows = [dict(r) for r in recording["utterances"]]
    decisions = []
    original = list(rows)
    for i, row in enumerate(original):
        if _bounds(row):
            continue
        left = next((_bounds(r) for r in reversed(original[:i]) if _bounds(r)), None)
        right = next((_bounds(r) for r in original[i+1:] if _bounds(r)), None)
        if not left or not right or not left[1] < right[0]:
            continue
        bracket = (left[1], right[0])
        # The row must be attributable to the current challenged chunks.
        candidates = []
        for obs in observations:
            if not set(row.get("chunk_indices") or []).intersection(obs.get("chunks", [])):
                continue
            matches = _anchored_intervals({**row, "source_language": recording["source_language"]},
                                          obs, bracket)
            if len(matches) == 1:
                candidates.append((obs, matches[0]))
        pairs = [(a, b) for k, a in enumerate(candidates) for b in candidates[k+1:]
                 if a[0]["audio_sha256"] != b[0]["audio_sha256"]
                 and a[0]["crop"] != b[0]["crop"]
                 and _same_event_bounds(a[1][0], b[1][0], policy.timing_tolerance_s)]
        if not pairs:
            continue
        a, b = pairs[0]
        bounds = a[1][0]
        if any(not _same_event_bounds(x[1][0], bounds, policy.timing_tolerance_s)
               for x in candidates):
            continue
        # Never assign one observed occurrence to competing untimed rows,
        # or reverse an already recovered source order.
        if any(not _bounds(other) and j != i
               and any(_same_event_bounds(candidate[0], bounds, policy.timing_tolerance_s)
                       for candidate in _anchored_intervals(
                           {**other, "source_language": recording["source_language"]},
                           a[0], bracket))
               for j, other in enumerate(original)):
            continue
        if any(other is not rows[i] and _bounds(other)
               and _overlap(_bounds(other), bounds) > .08 for other in rows):
            continue
        if (any(_bounds(r) and not _bounds(original[j]) and _bounds(r)[1] > bounds[0]
                for j, r in enumerate(rows[:i]))
                or any(_bounds(r) and not _bounds(original[j]) and _bounds(r)[0] < bounds[1]
                       for j, r in enumerate(rows[i+1:], i+1))):
            continue
        obs_ids = a[1][1] + b[1][1]
        rows[i] = {**row, "source_start": bounds[0], "source_end": bounds[1],
                   "alignment_status": "anchored_event_bounds",
                   "source_quality_status": "unresolved_wording_anchored_timing",
                   "timestamp_provenance": {
                       "backend": "separate_acoustic_contexts",
                       "timing_status": "approximate",
                       "human_verified": False, "word_timing_verified": False,
                       "observation_ids": obs_ids,
                       "audio_sha256": [a[0]["audio_sha256"], b[0]["audio_sha256"]],
                       "raw_response_sha256": [a[0]["raw_response_sha256"], b[0]["raw_response_sha256"]],
                       "lexical_association_ratios": [a[1][2], b[1][2]],
                   }}
        decisions.append({"status": "unresolved_wording_anchored_timing",
                          "source_id": row["id"], "window": bounds,
                          "observations": obs_ids})
    return {**recording, "utterances": rows}, decisions


def select_source_events(recording, events, policy):
    """Use positive corroboration to fill gaps or extend a single coherent turn.

    No deletion of distinct source events based on lexical similarity, lack
    of model coverage, or text review. Displaced repeats and competing voices
    are explicit unresolved alternatives rather than falsely certain repairs.
    """
    rows = [dict(r) for r in recording["utterances"]]
    decisions = []
    media_id = recording["source"]["fingerprint_sha256"]
    def crossed_turn_ids(row, target):
        """Do not hide a change in order across an entire other source turn."""
        original = _bounds(row)
        if not original or not target:
            return []
        crossed = []
        for other in rows:
            if other is row:
                continue
            interval = _bounds(other)
            if interval and any(min(old, new) <= interval[0]
                                and interval[1] <= max(old, new)
                                for old, new in zip(original, target)):
                crossed.append(other["id"])
        return crossed

    def corroborated_at_own_time(row, event):
        original = _bounds(row)
        target = _bounds(event)
        return bool(original and target and not _overlap(original, target)
                    and any(_lex(other["source_text"]) == _lex(row["source_text"])
                            and _same_event_bounds(_bounds(other), original,
                                                   policy.timing_tolerance_s)
                            and not crossed_turn_ids(row, _bounds(other))
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
                          and (_lex(u["source_text"]) == words
                               or set(u.get("chunk_indices") or []) & set(event["chunks"])
                               or (_bounds(u) and abs(_bounds(u)[0] - bounds[0]) <= 60))]
        status = "unresolved_competing_source"
        replace = None
        competing_hypotheses = []
        crossed_ids = []
        exact_overlaps = [u for u in overlaps if _lex(u["source_text"]) == words]
        if exact_overlaps:
            # Lexical agreement and minimal overlap do not identify an
            # occurrence. This branch is exclusive: an equal-text timing
            # conflict must not fall through to source-turn extension.
            if len(exact_overlaps) == 1:
                crossed_ids = crossed_turn_ids(exact_overlaps[0], bounds)
            if (len(exact_overlaps) == 1
                    and _same_event_bounds(_bounds(exact_overlaps[0]), bounds,
                                           policy.timing_tolerance_s)
                    and not crossed_ids):
                status = "already_represented"
            else:
                status = "unresolved_possible_displaced_or_repeated_turn"
                competing_hypotheses = [
                    {"source_id": u["id"], "window": _bounds(u)}
                    for u in exact_overlaps
                ]
        elif (len(overlaps) == 1 and _contains(words, _lex(overlaps[0]["source_text"]))
                and bounds[0] <= _bounds(overlaps[0])[0] + policy.timing_tolerance_s
                and bounds[1] >= _bounds(overlaps[0])[1] - policy.timing_tolerance_s
                and not any(u is not overlaps[0] for u in nearby_matches)):
            replace = overlaps[0]
            status = "extended_corroborated_turn"
        elif (len(overlaps) == 1 and not nearby_matches
                and len(set(event["audio_sha256"])) >= 3
                and _same_event_bounds(_bounds(overlaps[0]), bounds, policy.timing_tolerance_s)
                and not crossed_turn_ids(overlaps[0], bounds)
                and len(words) >= 3
                and not any(_bounds(other) and _overlap(_bounds(other), bounds) > 0
                            for other in rows if other is not overlaps[0])):
            # Three distinct contexts positively support the same competing
            # wording at the same occurrence. Keep the prior hypothesis;
            # this is model adjudication, not independent-model truth.
            replace = overlaps[0]
            status = "replaced_corroborated_source_hypothesis"
        elif not overlaps and len(nearby_matches) == 1 and _bounds(nearby_matches[0]) is None:
            if _lex(nearby_matches[0]["source_text"]) == words:
                replace = nearby_matches[0]
                status = "timed_corroborated_turn"
        elif not overlaps and all(corroborated_at_own_time(u, event) for u in nearby_matches):
            status = "recovered_corroborated_turn"
        elif not overlaps:
            status = "unresolved_possible_displaced_or_repeated_turn"
        if (status == "recovered_corroborated_turn"
                and bounds[1] - bounds[0] < .65
                and len(set(event["audio_sha256"])) < 3):
            status = "unresolved_short_insertion_needs_support"
        if status in {"extended_corroborated_turn", "timed_corroborated_turn",
                      "recovered_corroborated_turn", "replaced_corroborated_source_hypothesis"}:
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
            row["source_occurrence_ids"] = [uid]
            row["source_lineage"] = {
                "prior_source_ids": [replace["id"]] if replace else [],
                "acoustic_observation_ids": list(event["supporting_observation_ids"]),
                "trigger_evidence_ids": list(event.get("trigger_evidence_ids", [])),
                "independent_observation_ids": list(event.get("independent_observation_ids", [])),
                "source_word_timing_accepted": False,
            }
            if replace is not None:
                row["prior_source_hypothesis"] = dict(replace)
                # These are origin references, not word-level verification
                # of the revised wording.
                for key in ("source_unit_ids","recognized_turn_ids","occurrence_span_ids"):
                    row[key] = list(replace.get(key, []))
                rows.remove(replace)
            if any(u["id"] == uid for u in rows):
                raise ValueError("Recovered source event ID collision")
            # Preserve order of unrelated/untimed source rows.
            pos = next((i for i, u in enumerate(rows) if _bounds(u)
                        and _bounds(u)[0] > bounds[0]), len(rows))
            rows.insert(pos, row)
        decision = {"status": status, "observations": event["supporting_observation_ids"],
                    "window": bounds,
                    "trigger_evidence_ids": list(event.get("trigger_evidence_ids", [])),
                    "independent_observation_ids": list(event.get("independent_observation_ids", [])),
                    "superseded_source_id": replace["id"] if replace else None}
        if status == "already_represented":
            decision["source_id"] = exact_overlaps[0]["id"]
        elif status in {"extended_corroborated_turn","timed_corroborated_turn",
                        "recovered_corroborated_turn","replaced_corroborated_source_hypothesis"}:
            decision["source_id"] = uid
        if competing_hypotheses:
            decision["competing_source_hypotheses"] = competing_hypotheses
            decision["crossed_source_ids"] = crossed_ids
        decisions.append(decision)
    return rows, decisions


def review_current_source(recording, *, recording_path, media_path, output_dir,
                          text_scope, asr_scope, policy=None, challenger_resource=None):
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
    extra_contexts = 0
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
                        item["chunks"] = w["chunks"]
                        observations.append(item)
                        pair.append(item)
                    selected = corroborated_events(*pair, w["window"], policy.timing_tolerance_s)
                    # Disputed wording, very short proposed insertions, and
                    # failed pairing merit a bounded targeted third context.
                    # Prefer a tight event crop rather than replaying either input.
                    disputed = [e for e in selected if e["source_end"] - e["source_start"] < .65
                                or any(_bounds(r) and _overlap(_bounds(r), _bounds(e)) > .08
                                       and _lex(r["source_text"]) != _lex(e["source_text"])
                                       for r in recording["utterances"])]
                    if extra_contexts < policy.max_extra_contexts and (disputed or not selected):
                        target = (min(e["source_start"] for e in disputed),
                                  max(e["source_end"] for e in disputed)) if disputed else w["window"]
                        timing_problem=(w["reason"]=="untimed_source_dialogue"
                                        or not any(o["utterances"] for o in pair))
                        # Compare variable-length and shifted overlapping
                        # observations; preserve positive support from all
                        # distinct contexts, without treating them as votes.
                        geometries=challenge_geometries(target,media.duration_s,policy,
                            timing_problem=timing_problem,seen=[x["crop"] for x in pair])
                        for kind,crop in geometries[:min(policy.max_extra_per_window,
                                                     policy.max_extra_contexts-extra_contexts)]:
                            third=_observe(asr,observed_media,crop,Path(tmp),len(observations))
                            third["chunks"]=w["chunks"]
                            third["geometry"]=kind
                            observations.append(third)
                            extra_contexts+=1
                            for e in selected:
                                _third_context(e,third,policy.timing_tolerance_s)
                            for previous in pair:
                                for e in corroborated_events(previous,third,w["window"],
                                                            policy.timing_tolerance_s):
                                    if not any(set(e["supporting_observation_ids"]).intersection(
                                            old["supporting_observation_ids"])
                                            or _overlap(_bounds(e),_bounds(old))>.08 for old in selected):
                                        for other in pair:
                                            _third_context(e,other,policy.timing_tolerance_s)
                                        selected.append(e)
                            pair.append(third)
                            # Exact positive support can stop further
                            # escalation for this target, but a failed
                            # geometry never disables that mechanism.
                            if selected and not disputed and all(
                                    len(set(e["audio_sha256"]))>=3 for e in selected):
                                break
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
    timed_recording, timing_decisions = _recover_untimed(recording, observations, policy)
    rows, decisions = select_source_events(timed_recording, events, policy)
    decisions = timing_decisions + decisions
    # Independent source recovery runs only after the strict selector, with
    # Nemotron released. A missing resource is explicit, never a download.
    from nvidia_transcribe.source_recovery import recover_independent_source
    rows, recovery = recover_independent_source(
        {**recording, "utterances": rows}, observations, observed_media,
        challenger_resource=challenger_resource,
        output_dir=output_dir,
    )
    decisions.extend(recovery["decisions"])
    from nvidia_transcribe.scene_evidence import SceneEvidencePolicy, adjudicate_scene
    rows,scene=adjudicate_scene({**recording,"utterances":rows},observations,
        recovery["observations"],observed_media,text_scope,output_dir,
        SceneEvidencePolicy(initial_context_s=policy.scene_context_s,
            maximum_context_s=policy.scene_max_context_s,
            context_char_budget=policy.scene_context_char_budget,
            max_cases=policy.scene_max_cases,max_frames=policy.scene_max_frames))
    reviewed_case_ids = {o["case_id"] for o in (advisory or {}).get("outcomes", [])}
    deferred_case_ids = [c["case_id"] for c in request["cases"]
                         if c["case_id"] not in reviewed_case_ids]
    discovery_deferred = request.get("deferred_by_budget", 0)
    # Occurrence-level alternatives and disputed locality are explicit inputs
    # to the successor recovery planner (#18). The existing review cannot
    # declare them resolved merely because its older window queue is empty.
    pending_spans = sorted({span["id"]
        for inventory in ledger.get("recognized_occurrence_inventories", [])
        for span in inventory.get("locality", {}).get("spans", [])
        if span["locality_status"] == "disputed" or any(
            d["status"] == "context_candidate_unresolved"
            for d in span.get("selection_dispositions", []))})
    uncertain = (bool(unresolved or deferred or deferred_case_ids or discovery_deferred or pending_spans)
                 or any(d["status"].startswith("unresolved") for d in decisions)
                 or bool(recovery["decisions"] or recovery["errors"])
                 or bool(scene["errors"] or scene["deferred_source_ids"])
                 or any(x.get("outcome") not in ("context_supported","supported") for x in scene["decisions"])
                 or recovery["status"] in ("unresolved","unsupported_language")
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
        "pending_occurrence_span_ids": pending_spans,
        "windows": windows, "deferred_windows": deferred,
        "acoustic_observations": observations, "decisions": decisions,
        "unresolved_windows": unresolved, "audio_requests": len(observations),
        "extra_acoustic_contexts": extra_contexts,
        "source_truth_human_verified": False,
        "independent_model_agreement": any(
            d["kind"]=="omission" for d in recovery["decisions"]),
        "independent_recovery": recovery,
        "scene_evidence": scene,
        "source_word_timing_verified": False,
        "original_recording_modified": False,
    }
    # Store raw challenge observations in the current private job, not a
    # historical correction tree. The canonical recording references them.
    return {**recording, "utterances": rows, "source_quality": {
        k: v for k, v in report.items() if k != "acoustic_observations"
    }}, report
