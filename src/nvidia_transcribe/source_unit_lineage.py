"""Evidence-only binding of legacy v1 source units to audited audio events.

A raw word's (chunk_index, source_unit_index) identifies a unique immutable
lexical observation in source-evidence/v1. Its model CTC timestamp does not
identify the audio event. Independently auditioned, unit-linked events in a
reconciliation dossier provide a separate positive association. Only when
ALL words of a legacy utterance belong to ONE reviewed acoustic event, and
legacy placement is plausible, may a caller consider same-event replacement.

This module never mutates source, speaker, alignment, or recording contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from nvidia_transcribe.source_recovery import _id, _interval, _overlap

SCHEMA = "nemovox-legacy-source-lineage/v1"
SOURCE_SCHEMA = "nemovox-source-evidence/v1"
DOSSIER_SCHEMA = "nemovox-source-reconciliation/v1"
RECORDING_SCHEMA = "nemotron-recording-contract/v1"
MAX_SUPERSESSION_TIME_SLACK_S = 3.0


def _hash_json(doc: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _id_str(v: Any, name: str) -> str:
    if not isinstance(v, str) or not v:
        raise ValueError(f"{name} must be a nonempty identifier")
    return v


def _source_key(item: Mapping[str, Any], ordinal: str) -> tuple[int, int] | None:
    chunk = item.get("chunk_index")
    index = item.get(ordinal)
    if (not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 0
            or not isinstance(index, int) or isinstance(index, bool) or index < 0):
        return None
    return chunk, index


def bind_legacy_source_lineage(
    recording: Mapping[str, Any],
    ledger: Mapping[str, Any],
    dossier: Mapping[str, Any],
) -> dict[str, Any]:
    if recording.get("schema_version") != RECORDING_SCHEMA:
        raise ValueError("Expected recording contract v1")
    if ledger.get("schema_version") != SOURCE_SCHEMA or dossier.get("schema") != DOSSIER_SCHEMA:
        raise ValueError("Source ledger and reconciliation dossier must have correct schemas")
    media = (recording.get("source") or {}).get("fingerprint_sha256")
    language = recording.get("source_language")
    if not isinstance(media, str) or not media or language not in ("fr", "zh"):
        raise ValueError("Recording lacks verified source fingerprint/language")
    if any(d.get("media_fingerprint_sha256") != media for d in (ledger, dossier)):
        raise ValueError("Recording/ledger/dossier media fingerprint mismatch")
    if ledger.get("source_language") != language or dossier.get("source_language") != language:
        raise ValueError("Recording/ledger/dossier source language mismatch")

    # A pair of index numbers means nothing across DIFFERENT inference
    # layouts. Authenticate the model response and actual input audio window
    # for each source-observation chunk before any index-level join.
    contract_chunks = {}
    for chunk in ((recording.get("processing") or {}).get("chunks") or []):
        idx = chunk.get("chunk_index")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0 or idx in contract_chunks:
            raise ValueError("Ambiguous recording chunk identity")
        contract_chunks[idx] = chunk
    if not contract_chunks:
        raise ValueError("Recording contract lacks source chunk response identities")
    observed_chunks = set()
    for observation in ledger.get("asr_observations") or []:
        idx = observation.get("chunk_index")
        if idx not in contract_chunks or idx in observed_chunks:
            raise ValueError("Source ledger chunk absent or repeated in recording")
        observed_chunks.add(idx)
        recorded = contract_chunks[idx]
        if recorded.get("raw_response_sha256") != observation.get("raw_response_sha256"):
            raise ValueError("Replay ledger does not match recording raw ASR response")
        for edge in ("infer_start_s", "infer_end_s"):
            old, new = recorded.get(edge), observation.get(edge)
            if (not isinstance(old, (int, float)) or not isinstance(new, (int, float))
                    or isinstance(old, bool) or isinstance(new, bool)
                    or not math.isfinite(old) or not math.isfinite(new)
                    or abs(old-new) > 0.001):
                raise ValueError("Replay ledger ASR input window differs from recording")
    lexical_by_index: dict[tuple[int, int], dict[str, Any]] = {}
    lexical_ids = set()
    for lexical in ledger.get("lexical_observations") or []:
        ident = _id_str(lexical.get("id"), "source lexical observation ID")
        key = _source_key(lexical, "source_ordinal")
        if key is None or key in lexical_by_index or ident in lexical_ids:
            raise ValueError("Ambiguous or missing immutable lexical index/identity")
        if lexical.get("unit_type") not in ("word", "character"):
            raise ValueError("Unexpected source unit type")
        if key[0] not in observed_chunks:
            raise ValueError("Lexical unit refers to unverified source chunk")
        lexical_by_index[key] = dict(lexical)
        lexical_ids.add(ident)

    reviewed_turns = {}
    for turn in dossier.get("source_turns_audio_review_attested") or []:
        eid = _id_str(turn.get("audio_event_id"), "reviewed event")
        if eid in reviewed_turns:
            raise ValueError("Duplicate reviewed source event")
        if (turn.get("source_status") != "audio_review_attested_source"
                or turn.get("selection_basis") != "independent_audio_review"
                or turn.get("source_turn_id") != _id("src_turn", [media, eid])):
            raise ValueError("Reviewed audio-event evidence does not match the source-turn identity")
        reviewed_turns[eid] = turn

    reviewed_regions: dict[str, dict[str, Any]] = {}
    unit_to_event: dict[str, str] = {}
    for region in dossier.get("audited_audio_regions") or []:
        eid = _id_str(region.get("audio_event_id"), "audio region event")
        if eid in reviewed_regions:
            raise ValueError("Duplicate reviewed audio-region identity")
        _interval(region)
        reviewed_regions[eid] = dict(region)
        linked = region.get("positive_audio_event_unit_ids") or []
        if not isinstance(linked, list) or len(set(linked)) != len(linked):
            raise ValueError("Event-linked unit references must be a unique ID array")
        if not linked:
            continue
        if (eid not in reviewed_turns
                or region.get("source_status") != "audio_review_attested"
                or region.get("reviewed_source_turn_id") != _id("src_turn", [media, eid])):
            raise ValueError("Only positively reviewed audio events may link lexical units")
        for unit_id in linked:
            if unit_id not in lexical_ids or unit_id in unit_to_event:
                raise ValueError("Unknown or multiply attributed reviewed lexical unit")
            unit_to_event[unit_id] = eid

    entries = []
    seen_legacy_ids = set()
    first_ownership: dict[tuple[int, int], str] = {}
    for utt in recording.get("utterances") or []:
        uid = _id_str(utt.get("id"), "legacy utterance")
        if uid in seen_legacy_ids:
            raise ValueError("Duplicate legacy utterance identity")
        seen_legacy_ids.add(uid)
        words = list(utt.get("source_words") or []) + list(utt.get("source_characters") or [])
        matched = []
        unidentified = 0
        for unit in words:
            key = _source_key(unit, "source_unit_index")
            if key is None or key not in lexical_by_index:
                unidentified += 1
                continue
            lex = lexical_by_index[key]
            if (unit.get("text") != lex.get("text")
                    or unit.get("unit_type") != lex.get("unit_type")):
                unidentified += 1
                continue
            # A single raw model unit cannot silently be owned by multiple
            # different v1 utterances, even if the words look the same.
            if key in first_ownership and first_ownership[key] != uid:
                raise ValueError("Duplicate lexical ownership across legacy utterances")
            first_ownership[key] = uid
            matched.append(lex["id"])

        events = {unit_to_event[u] for u in matched if u in unit_to_event}
        positive_count = sum(u in unit_to_event for u in matched)
        status = "no_positive_same_event_assignment"
        chosen_event = None
        if unidentified:
            status = "incomplete_or_unmatched_legacy_source_units"
        elif not words:
            status = "no_lexical_unit_ownership"
        elif positive_count:
            if positive_count != len(words) or len(events) != 1:
                status = "partial_or_multiple_reviewed_audio_events"
            else:
                chosen_event = next(iter(events))
                region = reviewed_regions[chosen_event]
                s, e = utt.get("source_start"), utt.get("source_end")
                valid_time = (isinstance(s, (int, float)) and not isinstance(s, bool)
                              and isinstance(e, (int, float)) and not isinstance(e, bool)
                              and math.isfinite(s) and math.isfinite(e) and 0 <= s <= e)
                a, b = _interval(region)
                status = (
                    "fully_same_event_time_plausible"
                    if valid_time and _overlap(a-MAX_SUPERSESSION_TIME_SLACK_S,
                                               b+MAX_SUPERSESSION_TIME_SLACK_S, s, e) > 0
                    else "fully_same_event_timing_displaced_or_missing"
                )
                if status != "fully_same_event_time_plausible":
                    chosen_event = None

        entries.append({
            "legacy_utterance_id": uid,
            "unit_count": len(words),
            "source_unit_id_count": len(matched),
            "missing_or_unmatched_unit_count": unidentified,
            "positively_reviewed_unit_count": positive_count,
            "reviewed_event_ids": sorted(events),
            "positive_same_event_id": chosen_event,
            "source_turn_id": (_id("src_turn", [media, chosen_event]) if chosen_event else None),
            "status": status,
            "eligible_for_reviewed_same_event_supersession": (
                status == "fully_same_event_time_plausible"
            ),
            "source_unit_observations_deleted": 0,
        })
    return {
        "schema": SCHEMA,
        "media_fingerprint_sha256": media,
        "source_language": language,
        "recording_content_sha256": _hash_json(recording),
        "source_ledger_content_sha256": _hash_json(ledger),
        "reconciliation_dossier_content_sha256": _hash_json(dossier),
        "source_unit_index_authority": "exact_chunk_index_source_ordinal_and_original_text",
        "legacy_utterances": entries,
        "diagnostics": {
            "legacy_utterances": len(entries),
            "original_legacy_units": sum(x["unit_count"] for x in entries),
            "matched_lexical_units": sum(x["source_unit_id_count"] for x in entries),
            "missing_or_unmatched_units": sum(x["missing_or_unmatched_unit_count"] for x in entries),
            "positive_audio_reviewed_units": len(unit_to_event),
            "eligible_same_event_legacy_utterances": sum(
                x["eligible_for_reviewed_same_event_supersession"] for x in entries
            ),
            "partially_attributed_legacy_utterances": sum(
                x["status"] == "partial_or_multiple_reviewed_audio_events" for x in entries
            ),
            "source_unit_observations_deleted": 0,
            "source_fidelity_certified": False,
        },
        "policy": {
            "exact_index_text_mapping_not_string_similarity": True,
            "raw_response_and_inference_audio_window_must_match": True,
            "all_units_must_belong_to_one_reviewed_audio_event": True,
            "timing_plausibility_is_separate_from_model_ctc_truth": True,
            "partial_or_displaced_utterances_are_preserved": True,
            "immutable_source_ledger_remains_unmodified": True,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording-contract", type=Path, required=True)
    p.add_argument("--source-evidence", type=Path, required=True)
    p.add_argument("--reconciliation-dossier", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    out = bind_legacy_source_lineage(
        read(args.recording_contract), read(args.source_evidence),
        read(args.reconciliation_dossier),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out["diagnostics"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
