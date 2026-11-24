"""Explicit, audio-attested recovery of omitted source events.

The QC queue is triage, never authority to insert or discard speech. This
separate opt-in contract preserves the existing owned-unit partition gate.
Audio attestations are supplied by a reviewer; a digest proves identity,
not the correctness of the listening judgment. No models or files are written.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import math
import wave
from typing import Any, Mapping

from nvidia_transcribe.evidence_fusion import _digest
from nvidia_transcribe.semantic_review_followups import build_meaning_first_review_queue
from nvidia_transcribe.source_fusion_application import _canonical_id
from nvidia_transcribe.source_unit_lineage import bind_legacy_source_lineage

REVIEW_SCHEMA = "nemovox-audio-source-recovery-review/v1"
REPORT_SCHEMA = "nemovox-audio-source-recovery-application/v1"


def _sha(x: Any) -> bool:
    return isinstance(x, str) and len(x) == 64 and all(c in "0123456789abcdef" for c in x)


def _bounds(x: Any, duration: float) -> tuple[float, float]:
    if (not isinstance(x, list) or len(x) != 2
            or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                   or not math.isfinite(v) for v in x)
            or not 0 <= x[0] < x[1] <= duration + .001):
        raise ValueError("Invalid audio review interval")
    return float(x[0]), float(x[1])


def _contains(outer, inner):
    return outer[0] <= inner[0] + .001 and outer[1] >= inner[1] - .001


def _overlaps(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def _ids(value, label, *, nonempty=False):
    if (not isinstance(value, list) or (nonempty and not value)
            or any(not isinstance(v, str) or not v for v in value)
            or len(set(value)) != len(value)):
        raise ValueError(f"Invalid or duplicate {label}")
    return set(value)


def apply_audio_source_recovery(
    recording: Mapping[str, Any], ledger: Mapping[str, Any],
    request: Mapping[str, Any], batch: Mapping[str, Any], policy: Mapping[str, Any],
    queue: Mapping[str, Any], review: Mapping[str, Any],
    audio_crops: Mapping[str, bytes], *,
    expected_review_sha256: str, enable: bool = False,
) -> dict[str, Any]:
    """Validate a private listening dossier and preview/materialize new source.

    The caller authenticates original media and crop extraction outside this
    pure function. Each crop's bytes and PCM duration are checked here. The
    reviewer must explicitly account for every raw unit of any superseded
    legacy row, including displaced units. Unowned units may be recovered;
    an audio-only event has no synthetic lexical observations.
    """
    if not isinstance(enable, bool):
        raise ValueError("Explicit enable boolean required")
    if not _sha(expected_review_sha256) or _digest(review) != expected_review_sha256:
        raise ValueError("Stale audio review identity")
    if queue != build_meaning_first_review_queue(request, batch, policy, recording, ledger):
        raise ValueError("Stale or altered semantic review queue")
    media = recording["source"]["fingerprint_sha256"]
    language = recording["source_language"]
    duration = float(recording["source"]["duration_s"])
    if (review.get("schema") != REVIEW_SCHEMA
            or review.get("queue_sha256") != queue["queue_sha256"]
            or review.get("recording_content_sha256") != _digest(recording)
            or review.get("source_ledger_content_sha256") != _digest(ledger)
            or review.get("media_fingerprint_sha256") != media
            or review.get("source_language") != language
            or review.get("review_method") != "direct_audio_listening"
            or review.get("reviewer_attests_original_media_and_crop_extraction") is not True):
        raise ValueError("Recovery requires independent original-media audio review")
    # Authenticate the same legacy index, text, type, ASR response and
    # inference-window identities used by the existing strict partition path.
    empty_dossier = {"schema": "nemovox-source-reconciliation/v1",
                     "media_fingerprint_sha256": media, "source_language": language}
    binding = bind_legacy_source_lineage(recording, ledger, empty_dossier)
    raw = {u["id"]: u for u in ledger["lexical_observations"]}
    coordinates = {(u["chunk_index"], u["source_ordinal"]): u["id"] for u in raw.values()}
    parents = {p["id"]: p for p in ledger["asr_observations"]}
    if len(parents) != len(ledger["asr_observations"]):
        raise ValueError("Duplicate parent ASR identity")
    originals = {u["id"]: u for u in recording["utterances"]}
    if len(originals) != len(recording["utterances"]):
        raise ValueError("Duplicate original utterance identity")
    ownership = {}
    for row in binding["legacy_utterances"]:
        uid = row["legacy_utterance_id"]
        if row["missing_or_unmatched_unit_count"]:
            raise ValueError("Incomplete original raw source binding")
        utt = originals[uid]
        units = list(utt.get("source_words") or []) + list(utt.get("source_characters") or [])
        ids = [coordinates[(w["chunk_index"], w["source_unit_index"])] for w in units]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated original raw source unit")
        ownership[uid] = set(ids)
    items = {r["case_id"]: r for r in queue["items"]}
    windows = review.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("Nonempty independent audio review windows required")
    used_units = {}
    events = []
    rejected = []
    seen_events = set()
    event_cases = {}
    seen_crops = set()
    superseded = _ids(review.get("superseded_legacy_ids"), "superseded legacy IDs")
    if not superseded <= originals.keys() or any(not ownership[u] for u in superseded):
        raise ValueError("Cannot supersede unknown or unbound legacy source")
    def touches_case(crop, case):
        if _overlaps(crop, case["candidate_source_window_s"]):
            return True
        # Displaced CTC hypotheses may be many seconds from the actual
        # dialogue. A separately reviewed competing interval is legitimate,
        # but only if it touches an original source-linked candidate unit.
        for uid in case["source_unit_ids"]:
            unit = raw[uid]
            a, b = unit.get("ctc_start_s"), unit.get("ctc_end_s")
            if (isinstance(a, (int, float)) and isinstance(b, (int, float))
                    and math.isfinite(a) and math.isfinite(b) and a < b
                    and _overlaps(crop, (a, b))):
                return True
        return False

    for window in windows:
        crop = _bounds(window.get("window_s"), duration)
        cases = _ids(window.get("case_ids"), "review case IDs", nonempty=True)
        if (not cases <= items.keys() or crop[1] - crop[0] > 60
                or any(not touches_case(crop, items[c]) for c in cases)):
            raise ValueError("Audio review must be bounded and linked to QC cases")
        chunks = {idx for c in cases for idx in items[c]["source_chunks"]}
        crop_sha = window.get("crop_sha256")
        data = audio_crops.get(crop_sha) if _sha(crop_sha) else None
        if not isinstance(data, bytes) or hashlib.sha256(data).hexdigest() != crop_sha:
            raise ValueError("Missing or changed reviewed audio crop bytes")
        if crop_sha in seen_crops:
            raise ValueError("Duplicate audio review crop")
        seen_crops.add(crop_sha)
        try:
            with wave.open(io.BytesIO(data), "rb") as wav:
                seconds = wav.getnframes() / wav.getframerate()
                if len(wav.readframes(wav.getnframes())) != (
                        wav.getnframes() * wav.getnchannels() * wav.getsampwidth()):
                    raise ValueError("Truncated reviewed PCM crop")
        except (wave.Error, EOFError, ZeroDivisionError) as exc:
            raise ValueError("Review crop must be PCM WAV") from exc
        if abs(seconds - (crop[1] - crop[0])) > .05:
            raise ValueError("Review crop duration disagrees with original-media interval")

        def claim(unit_id, status):
            if unit_id in used_units or unit_id not in raw:
                raise ValueError("Unknown or multiply disposed raw unit")
            unit = raw[unit_id]
            parent = parents.get(unit.get("asr_observation_id"))
            if (not parent or parent["chunk_index"] != unit["chunk_index"]
                    or unit["chunk_index"] not in chunks):
                raise ValueError("Raw unit outside reviewed source parents")
            used_units[unit_id] = status
            return parent

        for event in window.get("events", []):
            eid = event.get("audio_event_id")
            if not isinstance(eid, str) or not eid or eid in seen_events:
                raise ValueError("Duplicate or invalid reviewed event ID")
            seen_events.add(eid)
            event_cases[eid] = set(cases)
            bounds = _bounds(event.get("event_s"), duration)
            text = event.get("source_text")
            if (not _contains(crop, bounds) or not isinstance(text, str) or not text.strip()
                    or event.get("spoken_source_text_attested") is not True
                    or event.get("event_bounds_attested") is not True):
                raise ValueError("Recovery event requires positive reviewed speech and timing")
            ids = _ids(event.get("source_unit_ids"), "event source units")
            kind = event.get("recovery_kind")
            if kind == "raw_unit_recovery":
                if not ids or event.get("same_occurrence_unit_identity_attested") is not True:
                    raise ValueError("Raw recovery requires positively identified lexical observations")
            elif kind == "asr_absent_speech":
                if ids or event.get("absence_from_original_raw_asr_attested") is not True:
                    raise ValueError("Audio-only recovery must explicitly attest raw ASR absence")
            else:
                raise ValueError("Unknown recovery evidence kind")
            source_parents = _ids(event.get("parent_asr_observation_ids"),
                                  "event ASR parents", nonempty=True)
            if not source_parents <= parents.keys():
                raise ValueError("Unknown event ASR parent")
            for pid in source_parents:
                parent = parents[pid]
                if (parent["chunk_index"] not in chunks
                        or not _overlaps(bounds, [parent["infer_start_s"], parent["infer_end_s"]])):
                    raise ValueError("Reviewed event outside original inference window")
            if not any(_contains([parents[pid]["infer_start_s"], parents[pid]["infer_end_s"]],
                                 bounds) for pid in source_parents):
                raise ValueError("No original ASR parent fully covers reviewed event")
            for uid in ids:
                parent = claim(uid, eid)
                if parent["id"] not in source_parents:
                    raise ValueError("Event unit missing its original inference parent")
            tid = _canonical_id("src_turn", [media, eid])
            revision = _canonical_id("src_revision", [media, eid, text, expected_review_sha256,
                                                      bounds[0], bounds[1]])
            events.append({
                "id": tid, "utterance_id": tid, "source_turn_id": tid,
                "source_turn_ids": [tid], "source_revision_id": revision, "audio_event_id": eid,
                "source_text": text, "source_language": language,
                "source_start": bounds[0], "source_end": bounds[1],
                "source_words": [], "source_characters": [], "unit_count": 0,
                "timed_unit_count": 0, "untimed_unit_count": 0, "timing_gaps": [],
                "speaker_id": "SPK_UNK", "speaker_identity_accepted": False,
                "alignment_status": "audio_event_bounds_only_word_timing_unverified",
                "source_acceptance_status": "audio_review_attested_source",
                "source_text_review_attested": True,
                "source_text_human_reference_verified": False,
                "source_observation_ids": sorted(source_parents),
                "source_unit_lineage_ids": sorted(ids), "source_unit_lineage_count": len(ids),
                "source_recovery_kind": kind,
                "timestamp_provenance": {"status": "audio_reviewed_event_bounds_only",
                                         "review_audio_sha256": crop_sha,
                                         "source_revision_id": revision},
                "source_revision_provenance": {"review_evidence_sha256": expected_review_sha256,
                                               "original_recording_sha256": _digest(recording),
                                               "queue_sha256": queue["queue_sha256"]},
            })
        for row in window.get("rejected_observations", []):
            uid = row.get("source_unit_id")
            parent = claim(uid, "rejected")
            kind = row.get("reason")
            raw_unit = raw[uid]
            ctc = _bounds([raw_unit.get("ctc_start_s"), raw_unit.get("ctc_end_s")], duration)
            if (kind not in ("not_spoken_asr_artifact", "same_occurrence_alternate")
                    or row.get("direct_audio_disposition_attested") is not True
                    or not _contains(crop, ctc)
                    or not _overlaps(crop, [parent["infer_start_s"], parent["infer_end_s"]])):
                raise ValueError("Rejected raw unit lacks direct review of reported occurrence")
            rejected.append({"source_unit_id": uid, "reason": kind,
                             "review_audio_sha256": crop_sha,
                             "replacement_audio_event_id": row.get("replacement_audio_event_id"),
                             "case_ids": sorted(cases),
                             "separate_occurrence_excluded_by_listening":
                                 row.get("separate_occurrence_excluded_by_listening") is True})
    if not events or len({e["id"] for e in events}) != len(events):
        raise ValueError("Recovery requires distinct positive reviewed events")
    if set(e["id"] for e in events) & originals.keys():
        raise ValueError("Recovery event collides with original source")
    for row in rejected:
        eid = row["replacement_audio_event_id"]
        if row["reason"] == "same_occurrence_alternate":
            if (eid not in seen_events or not event_cases[eid].intersection(row["case_ids"])
                    or not row["separate_occurrence_excluded_by_listening"]):
                raise ValueError("Alternate requires independent exclusion of distinct speech and a linked event")
        elif eid is not None:
            raise ValueError("ASR artifact cannot claim a replacement event")
    accounted = set(used_units)
    for uid, units in ownership.items():
        if uid in superseded and not units <= accounted:
            raise ValueError("Partial legacy supersession would lose original source units")
        if uid not in superseded and units & accounted:
            raise ValueError("Recovered or rejected unit still owned by retained legacy source")
    report = {
        "schema": REPORT_SCHEMA, "activation_enabled": enable,
        "media_fingerprint_sha256": media, "recording_content_sha256": _digest(recording),
        "source_ledger_content_sha256": _digest(ledger),
        "queue_sha256": queue["queue_sha256"], "review_evidence_sha256": expected_review_sha256,
        "superseded_legacy_ids": sorted(superseded),
        "reviewed_event_ids": sorted(seen_events),
        "reviewed_event_to_original_source_unit_ids": {
            e["audio_event_id"]: e["source_unit_lineage_ids"] for e in events},
        "audio_only_event_ids": sorted(e["audio_event_id"] for e in events
                                       if e["source_recovery_kind"] == "asr_absent_speech"),
        "rejected_raw_observations": rejected,
        "source_unit_observations_deleted": 0,
        "source_word_level_timing_certified": False, "speaker_identity_certified": False,
        "original_recording_mutated": False,
    }
    if not enable:
        return {"report": report, "new_recording": None}
    output = deepcopy(recording)
    output["utterances"] = [u for u in output["utterances"] if u["id"] not in superseded] + events
    output["utterances"].sort(key=lambda u: u["source_start"] if isinstance(
        u.get("source_start"), (int, float)) and math.isfinite(u["source_start"]) else math.inf)
    output["source_fusion_revision"] = report
    return {"report": report, "new_recording": output}


def postedited_audio_recovery_translation(
    assembled: Mapping[str, Any], *, config: Any, text_client: Any,
    enable: bool = False,
) -> dict[str, Any]:
    """Review only newly translated recovery events with retained context.

    Explicitly supplied managed text client; no implicit resource startup.
    Primary MT is preserved on errors and on the existing polarity veto.
    """
    from nvidia_transcribe.translation_review import review_translated_utterances
    if not isinstance(enable, bool):
        raise ValueError("Explicit postedit enable boolean required")
    report = assembled.get("source_fusion_revision")
    bridge = assembled.get("source_revision_translation_bridge")
    if (not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA
            or report.get("activation_enabled") is not True
            or not isinstance(bridge, dict) or bridge.get("enabled") is not True
            or bridge.get("source_fusion_report_sha256") != _digest(report)):
        raise ValueError("Expected activated audio recovery translation")
    targets = _ids(bridge.get("new_source_turn_ids"), "new translation targets", nonempty=True)
    rows = assembled["utterances"]
    if targets != {u["id"] for u in rows if u.get("audio_event_id") in report["reviewed_event_ids"]}:
        raise ValueError("Translation targets differ from reviewed recovery events")
    if not enable:
        return {"translated_contract": None, "target_ids": sorted(targets)}
    if text_client is None or config is None or config.profile == "off":
        raise ValueError("Explicit managed review client and enabled config required")
    if config.target_language != assembled.get("target_language"):
        raise ValueError("Postedit target language differs from primary translation")
    if any(u["id"] in targets and u.get("translation_review_status")
           != "not_reviewed_for_source_revision" for u in rows):
        raise ValueError("Only fresh reviewed-source translations may be postedited")
    updated, review_report = review_translated_utterances(
        deepcopy(rows), source_language=assembled["source_language"], config=config,
        text_client=text_client, resume=False, target_utterance_ids=targets)
    output = deepcopy(assembled)
    output["utterances"] = updated
    output["translation_review"] = {**review_report,
        "reviewed_source_revisions": review_report["reviewed"],
        "target_ids": sorted(targets),
        "historical_status_retained_per_utterance": True,
        "pre_postedit_contract_sha256": _digest(assembled)}
    return {"translated_contract": output, "report": output["translation_review"]}
