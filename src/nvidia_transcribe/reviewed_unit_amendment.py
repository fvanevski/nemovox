"""Direct-audio-reviewed amendments to source-unit evidence, never to raw ASR.

An amendment may (1) attribute an unassigned *owned* lexical unit to a
reviewed acoustic event or (2) retain an unowned raw unit as a positively
reviewed alternate observation of an already-owned SAME acoustic occurrence.
No file, ASR response, lexical unit, speaker ID or subtitle is modified here.
A model or lexical match alone may not submit a positive review.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping

from nvidia_transcribe.evidence_fusion import _digest
from nvidia_transcribe.source_unit_lineage import bind_legacy_source_lineage

SCHEMA = "nemovox-reviewed-unit-amendment/v1"
_DOSSIER_SCHEMA = "nemovox-source-reconciliation/v1"


def _check_window(value: Any, label: str) -> tuple[float, float]:
    if (not isinstance(value, list) or len(value) != 2
            or any(not isinstance(x, (int, float)) or isinstance(x, bool)
                   or not math.isfinite(x) for x in value)
            or not 0 <= value[0] < value[1]):
        raise ValueError(f"{label} must be finite nonnegative [start,end]")
    return float(value[0]), float(value[1])


def _contains(outer: tuple[float, float], inner: tuple[float, float]) -> bool:
    return outer[0] <= inner[0] + .001 and outer[1] >= inner[1] - .001


def _sha(value: Any, label: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise ValueError(f"{label} requires exact SHA256")


def apply_reviewed_unit_amendments(
    recording: Mapping[str, Any], ledger: Mapping[str, Any],
    dossier: Mapping[str, Any], request: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce a NEW evidence dossier and immutable disposition audit.

    This is not acoustic proof by itself: the submitted review explicitly
    attests direct listening to the exact original-media crop. If a competing
    occurrence is not independently excluded, its raw observation stays
    unresolved and this routine refuses to make it an alternate.
    """
    if request.get("schema") != SCHEMA:
        raise ValueError("Unsupported direct-review amendment schema")
    for kind, obj in (("recording", recording), ("ledger", ledger), ("dossier", dossier)):
        if request.get(kind + "_content_sha256") != _digest(obj):
            raise ValueError("Stale amendment source: " + kind)
    media = (recording.get("source") or {}).get("fingerprint_sha256")
    language = recording.get("source_language")
    if (not isinstance(media, str) or not media or
            any(d.get("media_fingerprint_sha256") != media for d in (ledger, dossier, request)) or
            any(d.get("source_language") != language for d in (ledger, dossier, request))):
        raise ValueError("Amendment media fingerprint or language mismatch")
    if dossier.get("schema") != _DOSSIER_SCHEMA:
        raise ValueError("Unsupported source reconciliation schema")
    # Authenticate EVERY original source unit/chunk's raw response SHA, ASR
    # input audio interval and legacy unit ownership before any amendment.
    # The revised dossier will be validated by the same binder at application.
    bind_legacy_source_lineage(recording, ledger, dossier)
    raw = {}
    coords = {}
    for item in ledger.get("lexical_observations") or []:
        unit_id = item.get("id")
        coord = (item.get("chunk_index"), item.get("source_ordinal"))
        if not isinstance(unit_id, str) or not unit_id or unit_id in raw or coord in coords:
            raise ValueError("Duplicate or malformed immutable raw source unit")
        raw[unit_id] = item
        coords[coord] = unit_id
    owners = {}
    for utt in recording.get("utterances") or []:
        for item in list(utt.get("source_words") or []) + list(utt.get("source_characters") or []):
            coord = (item.get("chunk_index"), item.get("source_unit_index"))
            uid = coords.get(coord)
            if uid is None or item.get("text") != raw[uid].get("text"):
                raise ValueError("Original recording has nonmatching lexical source owner")
            if uid in owners:
                raise ValueError("Duplicate original lexical source ownership")
            owners[uid] = utt["id"]
    observed = {}
    for parent in ledger.get("asr_observations") or []:
        idx = parent.get("chunk_index")
        if idx in observed or not isinstance(idx, int):
            raise ValueError("Duplicate ASR inference identity")
        observed[idx] = parent
    regions = {r["audio_event_id"]: r for r in dossier.get("audited_audio_regions") or []}
    turns = {t["audio_event_id"]: t for t in dossier.get("source_turns_audio_review_attested") or []}
    if (len(regions) != len(dossier.get("audited_audio_regions") or [])
            or len(turns) != len(dossier.get("source_turns_audio_review_attested") or [])):
        raise ValueError("Ambiguous reviewed event identity")
    current = {}
    for eid, region in regions.items():
        for uid in region.get("positive_audio_event_unit_ids") or []:
            if uid not in raw or uid in current:
                raise ValueError("Invalid original positive audio unit assignment")
            current[uid] = eid
    original_positive = dict(current)
    actions = request.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Amendment requires nonempty direct-audio-reviewed actions")
    revised = deepcopy(dossier)
    revised_regions = {r["audio_event_id"]: r for r in revised["audited_audio_regions"]}
    amended = set()
    records = []
    for action in actions:
        if not isinstance(action, dict):
            raise ValueError("Invalid review action")
        mode, uid, eid = action.get("kind"), action.get("source_unit_id"), action.get("audio_event_id")
        if mode not in ("assign_owned_unit", "reviewed_alternate_observation"):
            raise ValueError("Unsupported reviewed unit action")
        if uid not in raw or uid in amended or eid not in regions or eid not in turns:
            raise ValueError("Unknown/repeated unit or unreviewed audio event")
        amended.add(uid)
        turn, region = turns[eid], regions[eid]
        if (region.get("source_status") != "audio_review_attested" or
                turn.get("source_status") != "audio_review_attested_source" or
                turn.get("reviewer_has_attested_direct_audio_listening") is not True or
                action.get("review_method") != "direct_audio_listening" or
                action.get("same_audio_event_verified") is not True or
                action.get("source_unit_identity_verified") is not True):
            raise ValueError("Positive independent listening/identity review required")
        _sha(action.get("review_evidence_sha256"), "Review evidence digest")
        _sha(action.get("review_audio_crop_sha256"), "Review audio crop digest")
        crop = _check_window(action.get("review_audio_crop_interval_s"), "Review audio crop")
        event = _check_window([turn["source_start"], turn["source_end"]], "Reviewed speech event")
        if not _contains(crop, event):
            raise ValueError("Amendment review audio does not cover source event")
        parent = observed.get(raw[uid]["chunk_index"])
        if parent is None:
            raise ValueError("Amendment source unit lacks authenticated original ASR parent")
        infer = _check_window([parent["infer_start_s"], parent["infer_end_s"]], "ASR inference window")
        if max(event[0], infer[0]) >= min(event[1], infer[1]):
            raise ValueError("Amendment acoustic event outside original ASR input")
        if mode == "assign_owned_unit":
            if uid not in owners or uid in current:
                raise ValueError("Only previously unassigned owned source units may be linked")
            revised_regions[eid]["positive_audio_event_unit_ids"].append(uid)
            current[uid] = eid
        else:
            preferred = action.get("preferred_owned_source_unit_id")
            if (uid in owners or original_positive.get(uid) != eid or preferred == uid
                    or preferred not in owners or original_positive.get(preferred) != eid):
                raise ValueError("Alternate requires unowned raw and same-event retained owner")
            preferred_parent = observed.get(raw[preferred]["chunk_index"])
            if preferred_parent is None:
                raise ValueError("Alternate owner lacks original ASR parent")
            owner_window = _check_window(
                [preferred_parent["infer_start_s"], preferred_parent["infer_end_s"]], "Owner inference window"
            )
            if max(event[0], owner_window[0]) >= min(event[1], owner_window[1]):
                raise ValueError("Alternate owner input could not cover the reviewed event")
            if action.get("separate_occurrence_excluded_by_listening") is not True:
                raise ValueError("Alternative raw observation still might be a genuine repeated occurrence")
            competing = _check_window(action.get("competing_occurrence_window_s"), "Competing occurrence window")
            if not _contains(crop, competing) or not _contains(competing, event):
                raise ValueError("Alternate review crop must include competing event window")
            # A reviewed alternate preserves raw bytes, the event link, and
            # competing-interval evidence, but cannot become a second subtitle.
            revised_regions[eid]["positive_audio_event_unit_ids"].remove(uid)
            revised_regions[eid].setdefault("reviewed_alternate_source_observations", []).append({
                "source_unit_id": uid, "preferred_owned_source_unit_id": preferred,
                "review_evidence_sha256": action["review_evidence_sha256"],
                "review_audio_crop_sha256": action["review_audio_crop_sha256"],
                "competing_occurrence_window_s": list(competing),
                "status": "same_occurrence_reviewed_alternate_not_an_output_unit",
            })
            current.pop(uid)
        records.append({
            "kind": mode, "source_unit_id": uid, "audio_event_id": eid,
            "parent_chunk_index": raw[uid]["chunk_index"],
            "parent_raw_response_sha256": parent.get("raw_response_sha256"),
            "review_evidence_sha256": action["review_evidence_sha256"],
            "review_audio_crop_sha256": action["review_audio_crop_sha256"],
        })
    if "reviewed_unit_amendment" in revised:
        raise ValueError("Reamendment requires a fresh authoritative dossier, not a chained overlay")
    bind_legacy_source_lineage(recording, ledger, revised)
    revised["reviewed_unit_amendment"] = {
        "schema": SCHEMA, "source_dossier_sha256": _digest(dossier),
        "review_request_sha256": _digest(request),
        "actions": records, "raw_unit_observations_deleted": 0,
        "speaker_or_word_timing_invented": False,
    }
    report = {
        "schema": "nemovox-reviewed-unit-amendment-report/v1",
        "original_dossier_sha256": _digest(dossier),
        "derived_dossier_sha256": _digest(revised),
        "review_request_sha256": _digest(request),
        "action_count": len(records),
        "assigned_owned_units": sum(x["kind"] == "assign_owned_unit" for x in records),
        "reviewed_raw_alternates": sum(x["kind"] == "reviewed_alternate_observation" for x in records),
        "original_ledger_modified": False, "original_recording_modified": False,
        "raw_unit_observations_deleted": 0,
        "actions": records,
    }
    return {"derived_dossier": revised, "report": report}
