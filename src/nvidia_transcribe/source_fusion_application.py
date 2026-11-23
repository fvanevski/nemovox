"""Opt-in, reversible source-event unit partition application (no file writes).

Caller selects complete reviewed connected components. All source observations,
original CTC estimates, prior recordings and audio artifacts remain immutable.
Event-level reviewed timing is *not* word-level timing or global speaker truth.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping

from nvidia_transcribe.evidence_fusion import (
    PARTITION_SCHEMA, _digest, plan_reviewed_source_partitions,
)

SCHEMA = "nemovox-fusion-source-revision-application/v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")


def _canonical_id(kind: str, payload: Any) -> str:
    return kind + "_" + hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:24]


def _event_bounds(turn: Mapping[str, Any], duration: float) -> tuple[float, float]:
    a, b = turn.get("source_start"), turn.get("source_end")
    if (any(not isinstance(v, (float, int)) or isinstance(v, bool)
            or not math.isfinite(v) for v in (a, b))
            or not 0 <= a < b <= duration + 0.001):
        raise ValueError("Invalid directly reviewed event interval")
    return float(a), float(b)


def build_reviewed_source_revision(
    recording: Mapping[str, Any], ledger: Mapping[str, Any], dossier: Mapping[str, Any],
    *, selected_component_ids: list[str], expected_plan_sha256: str, enable: bool = False,
) -> dict[str, Any]:
    """Plan or materialize a *new* v1 recording for explicitly selected components.

    This does not write files, invoke any model or modify its inputs. The caller
    must separately authorize any downstream translation/render and verify all
    source/model/policy decisions. A model-only proposal cannot provide a source
    turn here: every selected event must be direct-audio-reviewed and fully
    linked to immutable raw source units and an original ASR inference window.
    """
    if not isinstance(enable, bool):
        raise ValueError("enable must be explicit boolean")
    if (not isinstance(selected_component_ids, list)
            or any(not isinstance(x, str) or not x for x in selected_component_ids)
            or len(set(selected_component_ids)) != len(selected_component_ids)):
        raise ValueError("Selected components must be unique identifiers")
    if not isinstance(expected_plan_sha256, str) or not _SHA.fullmatch(expected_plan_sha256):
        raise ValueError("Expected immutable partition plan SHA is required")
    plan = plan_reviewed_source_partitions(recording, ledger, dossier)
    if (plan.get("schema") != PARTITION_SCHEMA or _digest(plan) != expected_plan_sha256
            or plan.get("recording_content_sha256") != _digest(recording)
            or plan.get("source_ledger_content_sha256") != _digest(ledger)
            or plan.get("reconciliation_dossier_content_sha256") != _digest(dossier)):
        raise ValueError("Stale or unsupported source-unit partition plan")
    media = plan["media_fingerprint_sha256"]
    language = plan["source_language"]
    duration = (recording.get("source") or {}).get("duration_s")
    if (not isinstance(duration, (float, int)) or isinstance(duration, bool)
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Invalid recording duration")
    components = {c["id"]: c for c in plan["connected_components"]}
    selected = []
    for cid in selected_component_ids:
        c = components.get(cid)
        if c is None:
            raise ValueError("Unknown reviewed event component")
        if c.get("status") != "complete_reviewed_component_requires_application_gate":
            raise ValueError("Cannot apply an incomplete/partially reviewed component")
        if not c.get("reviewed_event_ids"):
            raise ValueError("A replacement component must have reviewed events")
        selected.append(c)
    event_ids = {eid for comp in selected for eid in comp["reviewed_event_ids"]}
    legacy_ids = {uid for comp in selected for uid in comp["legacy_utterance_ids"]}
    if len(legacy_ids) != sum(len(comp["legacy_utterance_ids"]) for comp in selected):
        raise ValueError("Overlapping selected source components")
    if len(event_ids) != sum(len(comp["reviewed_event_ids"]) for comp in selected):
        raise ValueError("Duplicated reviewed source event across components")
    turns = {t["audio_event_id"]: t for t in dossier.get("source_turns_audio_review_attested", [])}
    regions = {r["audio_event_id"]: r for r in dossier.get("audited_audio_regions", [])}
    raw_units = {u["id"]: u for u in ledger.get("lexical_observations", [])}
    asr_by_chunk = {o["chunk_index"]: o for o in ledger.get("asr_observations", [])}
    selected_units: dict[str, list[str]] = {eid: [] for eid in event_ids}
    for row in plan["legacy_utterances"]:
        if row["legacy_utterance_id"] not in legacy_ids:
            continue
        for group in row["reviewed_event_groups"]:
            selected_units[group["audio_event_id"]].extend(group["source_unit_ids"])
    new_turns = []
    for eid in sorted(event_ids):
        turn, region = turns.get(eid), regions.get(eid)
        if not isinstance(turn, dict) or not isinstance(region, dict):
            raise ValueError("Missing reviewed source turn or acoustic event")
        if (turn.get("schema") != "nemovox-source-turn-revision/v1"
                or turn.get("source_status") != "audio_review_attested_source"
                or turn.get("selection_basis") != "independent_audio_review"
                or turn.get("reviewer_has_attested_direct_audio_listening") is not True
                or turn.get("source_language") != language
                or region.get("source_status") != "audio_review_attested"):
            raise ValueError("Source event lacks positive independent audio review")
        a, b = _event_bounds(turn, float(duration))
        crop = region.get("review_audio_crop_provenance")
        if (not isinstance(crop, dict) or crop.get("event_fully_covered_by_review_audio") is not True
                or not isinstance(crop.get("sha256"), str) or not _SHA.fullmatch(crop["sha256"])
                or any(not isinstance(crop.get(k), (int, float)) or isinstance(crop.get(k), bool)
                       or not math.isfinite(crop[k]) for k in ("start_s", "end_s"))
                or crop["start_s"] > a + .001 or crop["end_s"] < b - .001):
            raise ValueError("Positive review audio crop does not cover event")
        review_sha = turn.get("review_evidence_sha256")
        if not isinstance(review_sha, str) or not _SHA.fullmatch(review_sha):
            raise ValueError("Missing direct audio review digest")
        text = turn.get("source_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Reviewed source event lacks verbatim text")
        expected_tid = _canonical_id("src_turn", [media, eid])
        expected_revision = _canonical_id("src_revision", [media, eid, text, review_sha, a, b])
        if turn.get("source_turn_id") != expected_tid or turn.get("source_revision_id") != expected_revision:
            raise ValueError("Reviewed source revision does not match immutable event/review identity")
        actual_units = list(region.get("positive_audio_event_unit_ids") or [])
        group_units = selected_units[eid]
        if (not group_units or len(set(group_units)) != len(group_units)
                or set(actual_units) != set(group_units)):
            raise ValueError("Incomplete or duplicate reviewed event source-unit ownership")
        source_observations = set()
        for unit_id in group_units:
            raw = raw_units.get(unit_id)
            parent = asr_by_chunk.get(raw.get("chunk_index")) if isinstance(raw, dict) else None
            if not isinstance(parent, dict):
                raise ValueError("Reviewed source unit has no original inference parent")
            start, end = parent.get("infer_start_s"), parent.get("infer_end_s")
            if (any(not isinstance(v, (int, float)) or isinstance(v, bool)
                    or not math.isfinite(v) for v in (start, end))
                    or max(a, float(start)) >= min(b, float(end))):
                raise ValueError("Reviewed audio event lies outside original raw inference window")
            source_observations.add(parent["id"])
        new_turns.append({
            "id": expected_tid, "utterance_id": expected_tid,
            "source_turn_id": expected_tid, "source_turn_ids": [expected_tid],
            "source_revision_id": expected_revision, "audio_event_id": eid,
            "source_text": text, "source_language": language,
            "source_start": a, "source_end": b,
            "source_words": [], "source_characters": [], "unit_count": 0,
            "timed_unit_count": 0, "untimed_unit_count": 0, "timing_gaps": [],
            "speaker_id": "SPK_UNK",
            "speaker_identity_accepted": False,
            "alignment_status": "audio_event_bounds_only_word_timing_unverified",
            "timestamp_provenance": {"status": "audio_reviewed_event_bounds_only",
                                    "original_ctc_may_be_displaced": True,
                                    "source_revision_id": expected_revision,
                                    "review_audio_sha256": crop["sha256"]},
            "source_acceptance_status": "audio_review_attested_source",
            "source_text_review_attested": True,
            "source_text_human_reference_verified": False,
            "source_observation_ids": sorted(source_observations),
            "source_unit_lineage_ids": sorted(group_units),
            "source_unit_lineage_count": len(group_units),
            "source_revision_provenance": {"review_evidence_sha256": review_sha,
                                            "original_recording_sha256": plan["recording_content_sha256"]},
        })
    original = recording.get("utterances")
    if not isinstance(original, list):
        raise ValueError("Expected immutable original utterances")
    original_ids = {u["id"] for u in original}
    if not legacy_ids <= original_ids:
        raise ValueError("Selected legacy source turns absent in recording")
    if any(turn["id"] in original_ids for turn in new_turns):
        raise ValueError("Reviewed event id collides with original recording")
    report = {
        "schema": SCHEMA, "media_fingerprint_sha256": media,
        "recording_content_sha256": plan["recording_content_sha256"],
        "source_ledger_content_sha256": plan["source_ledger_content_sha256"],
        "reconciliation_dossier_content_sha256": plan["reconciliation_dossier_content_sha256"],
        "partition_plan_sha256": expected_plan_sha256,
        "selected_component_ids": sorted(selected_component_ids),
        "superseded_legacy_ids": sorted(legacy_ids),
        "reviewed_event_ids": sorted(event_ids),
        "reviewed_event_to_original_source_unit_ids": {
            x["audio_event_id"]: x["source_unit_lineage_ids"] for x in new_turns},
        "source_unit_observations_deleted": 0,
        "source_word_level_timing_certified": False,
        "speaker_identity_certified": False,
        "original_recording_mutated": False,
        "activation_enabled": enable,
    }
    if not enable:
        return {"report": report, "new_recording": None}
    new_recording = deepcopy(recording)
    retained = [u for u in new_recording["utterances"] if u["id"] not in legacy_ids]
    # Respect actual event bounds, not the legacy model's possibly displaced CTC.
    decorated = [(row, idx) for idx, row in enumerate(retained + new_turns)]
    decorated.sort(key=lambda x: (x[0].get("source_start") if isinstance(
        x[0].get("source_start"), (int, float)) and math.isfinite(
        x[0]["source_start"]) else math.inf, x[1]))
    new_recording["utterances"] = [row for row, _ in decorated]
    new_recording["source_fusion_revision"] = dict(report)
    return {"report": report, "new_recording": new_recording}
