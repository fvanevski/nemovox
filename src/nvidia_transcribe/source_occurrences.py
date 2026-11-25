"""Preserve recognized turn identity separately from provisional native CTC timing."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import unicodedata

from nvidia_transcribe.parser import parse_speaker_turns, split_lexical_units


def stable_id(kind, material):
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"{kind}_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:24]}"


def source_observation_id(media_sha, raw_sha, spec):
    return stable_id("asr", [media_sha, raw_sha, float(spec.infer_start_s),
                            float(spec.infer_end_s), float(spec.core_start_s),
                            float(spec.core_end_s)])


def _key(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return "".join(c for c in text if not c.isspace()
                   and unicodedata.category(c)[0] not in ("P", "S")) or text


def _tag(value):
    if value is None or not str(value).strip():
        return None
    return str(value).removeprefix("<spk:").removesuffix(">")


def associate_native_ctc(units, parsed, payload):
    """Map native observations to the recognized spine, never the reverse.

    Complete equal per-tag sequences permit ordinal association only. If the
    recognized transcript is wholly untagged, use the ordered native lexical
    stream without its speaker labels. Partial streams use unique ordered
    lexical anchors. Crossing anchors and ambiguous repeats stay unresolved.
    No lexical match certifies acoustic locality or supplies a speaker tag.
    """
    native = payload.get("words") if isinstance(payload, dict) else None
    observations = []
    parsed_iter = iter(parsed.units)
    for index, item in enumerate(native if isinstance(native, list) else []):
        if not isinstance(item, dict):
            observations.append({"ctc_word_index": index, "raw_word": item,
                                 "association_status": "malformed_native_word"})
            continue
        observations.append({
            "ctc_word_index": index, "raw_word": item,
            "timing": next(parsed_iter).to_dict(),
            "association_status": "native_error" if parsed.error else "unmatched",
        })
    for unit in units:
        unit["ctc_association_status"] = "native_error" if parsed.error else "unmatched"
    if parsed.error:
        return observations

    left, right = defaultdict(list), defaultdict(list)
    for i, unit in enumerate(units):
        left[_tag(unit.get("speaker"))].append((i, _key(unit["text"])))
    for i, observation in enumerate(observations):
        if "timing" in observation:
            t = observation["timing"]
            right[_tag(t["speaker"])].append((i, _key(t["text"])))
    # Untagged recognition (e.g. verbatim) can still carry native CTC
    # speaker labels. Match the entire native stream in its observed order,
    # including unlabelled observations, so duplicates cannot disappear into
    # another label partition. Never use this fallback for mixed/tagged
    # recognition: native words may already belong to an explicit turn.
    untagged = set(left) == {None} and any(tag is not None for tag in right)
    if untagged:
        right[None] = sorted(item for stream in right.values() for item in stream)

    for tag, source in left.items():
        observed = right[tag]
        a, b = [x[1] for x in source], [x[1] for x in observed]
        ca, cb = Counter(a), Counter(b)
        source_keys = {key: i for i, key in source if ca[key] == 1}
        if a == b:
            pairs = [(x[0], y[0]) for x, y in zip(source, observed)]
            method = ("complete_untagged_sequence_provisional" if untagged
                      else "complete_per_tag_sequence_provisional")
        else:
            pairs = [(source_keys[key], j) for j, key in observed
                     if key and ca[key] == cb[key] == 1]
            if any(x[0] >= y[0] for x, y in zip(pairs, pairs[1:])):
                pairs = []
            method = ("partial_untagged_unique_ordered_anchor_provisional" if untagged
                      else "partial_unique_ordered_anchor_provisional")
        for j, key in observed:
            observations[j]["candidate_source_count"] = ca[key]
            if ca[key]:
                observations[j]["association_status"] = "ambiguous_sequence"
        for i, key in source:
            if cb[key]:
                units[i]["ctc_association_status"] = "ambiguous_sequence"
        for i, j in pairs:
            observation = observations[j]
            timing = observation["timing"]
            unit = units[i]
            observation.update(association_status=method, source_unit_index=unit["source_unit_index"])
            unit.update(ctc_association_status=method,
                        ctc_word_index=observation["ctc_word_index"],
                        ctc_observed_speaker=timing["speaker"])
            for key in ("start", "end", "valid_timing", "timing_error", "raw_start", "raw_end"):
                unit[key] = timing[key]
            unit["alignment_status"] = "ctc_aligned" if timing["valid_timing"] else "invalid_timing"
    return observations


def bind_occurrence_inventory(*, observation_id, raw_text, projected_text, units, native_observations):
    """Bind raw turns and working units; do not pretend a cleaned projection is exact."""
    exact = raw_text == projected_text
    turns = []
    for turn in parse_speaker_turns(raw_text):
        turn_id = stable_id("turn" if exact else "rawturn", [observation_id, turn.turn_index])
        turns.append({
            "id": turn_id, "turn_index": turn.turn_index, "speaker_tag": turn.speaker_tag,
            "raw_text": turn.raw_text, "text": turn.text,
            "events": list(turn.events), "styles": list(turn.styles),
            "tokens": [
                {**token, "position": i, "id": stable_id("token", [turn_id, i])}
                for i, token in enumerate(split_lexical_units(turn.text))
            ],
        })
    by_ordinal = {}
    for unit in units:
        ordinal = unit["source_unit_index"]
        turn_id = stable_id("turn", [observation_id, unit["recognized_turn_index"]])
        unit_id = stable_id("unit", [observation_id, ordinal, unit["text"]])
        if ordinal in by_ordinal:
            raise ValueError("Duplicate recognized source ordinal")
        unit.update(source_unit_id=unit_id, recognized_turn_id=turn_id,
                    asr_observation_id=observation_id,
                    raw_recognized_token_ids=[
                        stable_id("token", [turn_id, i])
                        for i in unit["recognized_token_positions"]
                    ] if exact else [],
                    raw_projection_status="exact" if exact else "loop_cleaned_unmapped")
        by_ordinal[ordinal] = unit_id
    for obs in native_observations:
        obs["id"] = stable_id("ctc", [observation_id, obs["ctc_word_index"]])
        if "source_unit_index" in obs:
            obs["source_unit_id"] = by_ordinal[obs["source_unit_index"]]
    return {
        "asr_observation_id": observation_id, "raw_recognized_text": raw_text,
        "projection_status": "exact" if exact else "loop_cleaned_unmapped",
        "recognized_turns": turns, "native_ctc_observations": native_observations,
        "working_source_unit_ids": list(by_ordinal.values()),
        "acoustic_locality_certified": False,
    }
