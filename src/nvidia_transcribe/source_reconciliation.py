"""Audio-region evidence reconciliation for long-form source language ASR.

The reconciler does NOT make two model answers into acoustic ground truth.
Long model windows can contain multiple voices and phrases. A matching window
provides context, not a localized utterance or deduplication authorization.
Only an independently auditioned, verbatim-reviewed source event can produce
an audio-reviewed source turn. All model-only events remain provisional.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from nvidia_transcribe.source_recovery import (
    ACTIVITY_SCHEMA, LANGUAGES, SCHEMA as PLAN_SCHEMA, _id, _interval, _overlap,
)

SCHEMA = "nemovox-source-reconciliation/v1"
CHALLENGER_SCHEMA = "nemovox-challenger-observations/v1"
AUDIT_SCHEMA = "nemovox-audited-source-events/v1"
TURN_SCHEMA = "nemovox-source-turn-revision/v1"
ACOUSTIC_KINDS = frozenset({
    "speech", "lexical_shout", "singing", "music", "sfx",
    "silence", "nonlexical_vocalization", "unknown",
})
COMPLETE = frozenset({"complete", "completed"})
INCOMPLETE = frozenset({"incomplete", "failed", "not_run"})
HEX = frozenset("0123456789abcdef")


def _digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value.lower()) <= HEX:
        raise ValueError(f"{field} must be an exact SHA-256 hex digest")
    return value.lower()


def _overlap_rows(bounds: tuple[float, float], rows: list[dict[str, Any]]) -> list[str]:
    return [r["id"] for r in rows if _overlap(*bounds, r["start_s"], r["end_s"]) > 0]


def reconcile_source_observations(
    ledger: Mapping[str, Any],
    plan: Mapping[str, Any],
    challenger: Mapping[str, Any],
    *,
    activity: Mapping[str, Any] | None = None,
    audited_events: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a deterministic, lossless evidence dossier; not a transcript.

    Strictly separate a crop-wide challenger response from an independently
    localized source event. Model agreement creates review hints, not a
    source-turn selection. Human-auditioned verbatim reference is mandatory
    before a source turn may be recorded as audio-reviewed.
    """
    if ledger.get("schema_version") != "nemovox-source-evidence/v1":
        raise ValueError("Unsupported source-evidence ledger")
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("Unsupported recovery plan")
    if challenger.get("schema") != CHALLENGER_SCHEMA:
        raise ValueError("Unsupported challenger evidence")
    media = ledger.get("media_fingerprint_sha256")
    lang = ledger.get("source_language")
    if not isinstance(media, str) or not media or lang not in LANGUAGES:
        raise ValueError("Expected source media identity and supported source language")
    for name, doc in (("plan", plan), ("challenger", challenger),
                      ("activity", activity), ("audited events", audited_events)):
        if doc is not None and doc.get("media_fingerprint_sha256") != media:
            raise ValueError(f"{name} media fingerprint mismatch")
    if plan.get("source_language") != lang or challenger.get("source_language") != lang:
        raise ValueError("Source language mismatch")
    model = challenger.get("model")
    if not isinstance(model, dict) or not isinstance(model.get("repo_id"), str) or not model["repo_id"]:
        raise ValueError("Missing challenger model identity")
    if not isinstance(model.get("revision"), str) or not model["revision"]:
        raise ValueError("Missing challenger checkpoint revision")
    if model.get("pinned_language") != LANGUAGES[lang]:
        raise ValueError("Challenger source language must be pinned, not auto-detected")
    if activity is not None and activity.get("schema") != ACTIVITY_SCHEMA:
        raise ValueError("Unsupported activity schema")
    if audited_events is not None and audited_events.get("schema") != AUDIT_SCHEMA:
        raise ValueError("Unsupported independently auditioned event schema")

    pending = plan.get("unplanned_due_to_budget") or []
    admitted = plan.get("requests") or []
    if not isinstance(admitted, list) or not isinstance(pending, list):
        raise ValueError("Plan requests must be arrays")
    requests = {}
    for request in admitted:
        rid = request.get("id")
        if not rid or rid in requests:
            raise ValueError("Missing or duplicate request ID")
        _interval(request)
        if request.get("language") != lang or request.get("source_content_accepted") is not False:
            raise ValueError("Untrusted request language or premature source acceptance")
        requests[rid] = request
    pending_ids = set()
    for r in pending:
        if not r.get("id") or r["id"] in requests or r["id"] in pending_ids:
            raise ValueError("Invalid or duplicate pending request ID")
        pending_ids.add(r["id"])

    windows = []
    seen = set()
    for row in challenger.get("windows") or []:
        rid = row.get("request_id")
        if rid in pending_ids:
            raise ValueError("Response for request that planner explicitly deferred")
        if rid not in requests or rid in seen:
            raise ValueError("Challenger response must match unique admitted request")
        seen.add(rid)
        req = requests[rid]
        a, b = _interval(row)
        ra, rb = _interval(req)
        if abs(a-ra) > .001 or abs(b-rb) > .001:
            raise ValueError("Challenger crop does not match authenticated plan interval")
        crop_sha = _digest(row.get("audio_crop_sha256"), "audio_crop_sha256")
        raw_sha = _digest(row.get("raw_response_sha256"), "raw_response_sha256")
        state = row.get("finish_status")
        if state not in COMPLETE | INCOMPLETE:
            raise ValueError("Unknown model completion state")
        output = row.get("source_text")
        if not isinstance(output, str):
            raise ValueError("Source transcript must be a string, including if empty")
        if state in INCOMPLETE and output:
            # A partial response may be retained but cannot be treated as
            # completed. It is stored only as a rejected/partial hypothesis.
            status = "incomplete_generation_do_not_accept"
        else:
            status = "window_only_unlocalized_hypothesis" if output and state in COMPLETE else "no_source_words_accepted"
        windows.append({
            "id": _id("challenger", [media, model["repo_id"], model["revision"], rid, crop_sha, raw_sha]),
            "request_id": rid,
            "start_s": a, "end_s": b,
            "model": dict(model),
            "audio_crop_sha256": crop_sha,
            "raw_response_sha256": raw_sha,
            "source_text": output,
            "text_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "finish_status": state,
            "status": status,
            "source_turn_id": None,
            "timestamp_acoustically_verified": False,
            "speaker_identity_accepted": False,
            "source_content_accepted": False,
        })
    windows.sort(key=lambda r: (r["start_s"], r["end_s"], r["id"]))
    observed_ids = {row.get("id") for row in ledger.get("asr_observations") or []}
    if len(observed_ids) != len(ledger.get("asr_observations") or []) or None in observed_ids:
        raise ValueError("Invalid or repeated ASR ledger observation IDs")
    lexical_ids = [r.get("id") for r in ledger.get("lexical_observations") or []]
    if None in lexical_ids or len(set(lexical_ids)) != len(lexical_ids):
        raise ValueError("Lexical unit identities must be unique and nonempty")
    valid_unit_ids = set(lexical_ids)
    ledger_windows = []
    window_by_id: dict[str, tuple[float, float]] = {}
    for row in ledger.get("asr_observations") or []:
        a, b = _interval(row, start="infer_start_s", end="infer_end_s")
        window_by_id[row["id"]] = (a, b)
        ledger_windows.append({
            "id": row["id"], "start_s": a, "end_s": b,
            "raw_response_sha256": _digest(row.get("raw_response_sha256"), "ASR response"),
        })
    unit_source_window: dict[str, tuple[float, float]] = {}
    for row in ledger.get("lexical_observations") or []:
        owner = row.get("asr_observation_id")
        if owner not in window_by_id:
            raise ValueError("Source lexical unit lacks a valid parent ASR observation")
        unit_source_window[row["id"]] = window_by_id[owner]

    acoustic_activity = []
    for row in (activity or {}).get("intervals") or []:
        a, b = _interval(row)
        if not row.get("id") or not row.get("source") or row.get("kind") not in ACOUSTIC_KINDS:
            raise ValueError("Activity evidence lacks source/identity/classification")
        acoustic_activity.append({
            "id": str(row["id"]), "start_s": a, "end_s": b,
            "kind": row["kind"], "source": row["source"],
        })

    turns = []
    regions = []
    seen_events = set()
    # Only a direct acoustic review may bind *individual unit IDs* to a
    # single event. Never infer this assignment from word similarity,
    # CTC displacement or a 24-second challenger window.
    unit_to_reviewed_event: dict[str, str] = {}
    for ev in (audited_events or {}).get("events") or []:
        eid = ev.get("id")
        if not isinstance(eid, str) or not eid or eid in seen_events:
            raise ValueError("Missing or duplicate audio event ID")
        seen_events.add(eid)
        a, b = _interval(ev)
        kind = ev.get("kind")
        if kind not in ACOUSTIC_KINDS:
            raise ValueError("Unknown audited acoustic event kind")
        if ev.get("acoustic_truth") != "independently_auditioned":
            raise ValueError("Event cannot be treated as audited without independent listening")
        linked = ev.get("linked_source_unit_ids") or []
        if not isinstance(linked, list) or any(not isinstance(x, str) for x in linked):
            raise ValueError("Linked source unit identities must be a list")
        if len(set(linked)) != len(linked) or any(x not in valid_unit_ids for x in linked):
            raise ValueError("Linked source units must exist uniquely in the ledger")
        if linked:
            if (ev.get("same_audio_event_verified") is not True
                    or ev.get("review_method") != "direct_audio_listening"):
                raise ValueError("Linking units needs positive independent audio-event review")
            _digest(ev.get("review_evidence_sha256"), "review_evidence_sha256")
            for unit_id in linked:
                # No lexical observation from an inference request covering
                # [25s, 45s] could have heard an event near 1132s. CTC can
                # drift, but the request's *audio input bounds* cannot.
                if _overlap(a, b, *unit_source_window[unit_id]) <= 0:
                    raise ValueError("Reviewed unit has no parent audio-window overlap with the event")
                if unit_id in unit_to_reviewed_event:
                    raise ValueError("One lexical source unit cannot be bound to two independent events")
                unit_to_reviewed_event[unit_id] = eid
        reviewed = ev.get("source_text_verified") is True
        text = ev.get("source_text")
        if reviewed:
            if kind not in {"speech", "lexical_shout", "singing"}:
                raise ValueError("Verbatim source text is inconsistent with non-lexical event")
            if ev.get("review_method") != "direct_audio_listening":
                raise ValueError("Verbatim source reference requires direct audio review")
            review_digest = _digest(ev.get("review_evidence_sha256"), "review_evidence_sha256")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Human-verbatim text must be nonempty")
            if ev.get("source_language") != lang:
                raise ValueError("Reviewed source text must use recording source language")
            turn_id = _id("src_turn", [media, eid])
            revision_id = _id("src_revision", [
                media, eid, text, review_digest, a, b,
            ])
            turns.append({
                "schema": TURN_SCHEMA,
                "source_turn_id": turn_id,
                "source_revision_id": revision_id,
                "audio_event_id": eid,
                "source_language": lang,
                "source_text": text,
                "source_start": a,
                "source_end": b,
                "source_observation_ids": _overlap_rows((a, b), ledger_windows),
                "selection_basis": "independent_audio_review",
                "source_status": "audio_review_attested_source",
                "word_alignment_status": "unverified",
                "speaker_identity_status": "unverified",
                "review_evidence_sha256": review_digest,
                "reviewer_has_attested_direct_audio_listening": True,
                "acoustic_bounds_are_event_level_not_word_level": True,
            })
        elif text:
            raise ValueError("Unverified reference cannot declare trusted verbatim text")

        supporting_activity = [
            {k: signal[k] for k in ("id", "kind", "source")}
            for signal in acoustic_activity
            if _overlap(a, b, signal["start_s"], signal["end_s"]) > 0
        ]
        qwen_windows = _overlap_rows((a, b), windows)
        nemotron_windows = _overlap_rows((a, b), ledger_windows)
        regions.append({
            "id": _id("audio_region", [media, eid, a, b]),
            "audio_event_id": eid,
            "start_s": a, "end_s": b,
            "kind": kind,
            "reviewed_source_turn_id": _id("src_turn", [media, eid]) if reviewed else None,
            "source_status": "audio_review_attested" if reviewed else "audio_event_source_text_unverified",
            "model_activity_evidence": supporting_activity,
            "challenger_window_context_ids": qwen_windows,
            "legacy_asr_window_context_ids": nemotron_windows,
            "model_windows_do_not_localize_or_verify_words": True,
            "positive_audio_event_unit_ids": list(linked),
            "unit_link_review_evidence_sha256": (
                ev.get("review_evidence_sha256") if linked else None
            ),
            "shared_non_speech_risk": any(s["kind"] in {"singing", "music", "sfx"} for s in supporting_activity),
            "speaker_identity_accepted": False,
            "cross_chunk_echo_decision": "unresolved_without_positive_same_event_identity",
        })
    seen_corr = set()
    echo_hypotheses = []
    for row in ledger.get("cross_chunk_correspondence_candidates") or []:
        rid = row.get("id")
        if not rid or rid in seen_corr:
            raise ValueError("Invalid or repeated cross-chunk candidate ID")
        seen_corr.add(rid)
        left = list(row.get("left_observation_unit_ids") or [])
        right = list(row.get("right_observation_unit_ids") or [])
        resolved_left = {unit_to_reviewed_event.get(u) for u in left}
        resolved_right = {unit_to_reviewed_event.get(u) for u in right}
        decision = "preserve_both_until_positive_audio_event_adjudication"
        matched_event = None
        if left and right and None not in resolved_left | resolved_right:
            if len(resolved_left | resolved_right) == 1:
                matched_event = next(iter(resolved_left))
                decision = "same_audio_event_reviewed_keep_observation_lineage"
            elif len(resolved_left) == len(resolved_right) == 1:
                decision = "distinct_audio_events_reviewed_preserve_both"
        echo_hypotheses.append({
            "id": rid,
            "left_observation_unit_ids": left,
            "right_observation_unit_ids": right,
            "ctc_relative_offset_hypothesis_s": row.get("ctc_relative_offset_hypothesis_s"),
            "repeated_phrase_ambiguity": row.get("repeated_phrase_ambiguity"),
            "positively_verified_event_id": matched_event,
            "decision": decision,
            "source_unit_observations_deleted": 0,
        })
    return {
        "schema": SCHEMA,
        "media_fingerprint_sha256": media,
        "source_language": lang,
        "challenger_model": dict(model),
        "model_windows": windows,
        "audited_audio_regions": regions,
        "source_turns_audio_review_attested": turns,
        "echo_correspondence_hypotheses": echo_hypotheses,
        "unexecuted_request_ids": sorted(set(requests) - seen),
        "deferred_request_ids": sorted(pending_ids),
        "policy": {
            "window_text_does_not_imply_event_text": True,
            "model_agreement_is_not_acoustic_truth": True,
            "alignment_is_not_transcript_verification": True,
            "no_auto_drop_from_music_or_sfx": True,
            "nonreviewed_source_turns_remain_unresolved": True,
            "legacy_asr_observations_preserved_unchanged": True,
            "source_fidelity_certified": False,
        },
        "diagnostics": {
            "challenger_windows_recorded": len(windows),
            "challenger_windows_incomplete": sum(w["finish_status"] in INCOMPLETE for w in windows),
            "pending_challenger_requests": len(set(requests) - seen),
            "model_only_source_turns_automatically_accepted": 0,
            "reviewed_audio_source_turns": len(turns),
            "audited_events_without_verbatim_source_text": len(regions) - len(turns),
            "unresolved_echo_candidates": sum(
                r["decision"] == "preserve_both_until_positive_audio_event_adjudication"
                for r in echo_hypotheses
            ),
            "audio_reviewed_same_event_correspondences": sum(
                r["decision"] == "same_audio_event_reviewed_keep_observation_lineage"
                for r in echo_hypotheses
            ),
            "source_unit_observations_deleted": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-evidence", type=Path, required=True)
    parser.add_argument("--recovery-plan", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--activity", type=Path)
    parser.add_argument("--audited-events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refuse to overwrite a prior reconciliation dossier")
    def read(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))
    dossier = reconcile_source_observations(
        read(args.source_evidence), read(args.recovery_plan), read(args.challenger),
        activity=read(args.activity) if args.activity else None,
        audited_events=read(args.audited_events) if args.audited_events else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(dossier, fh, ensure_ascii=False, indent=2)
    print(json.dumps({"schema": SCHEMA, **dossier["diagnostics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
