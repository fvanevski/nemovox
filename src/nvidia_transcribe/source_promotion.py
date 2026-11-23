"""Opt-in source-turn promotion into a v1-compatible recording contract.

Only independently audio-reviewed turns, or model observations explicitly
localized by separate bounded ASR with corroborating speech-activity evidence,
can be rendered. Model-only output is PROVISIONAL, not acoustic truth.
Existing utterances remain intact unless a positively audited same-event
supersession explicitly names the legacy IDs; raw evidence is never deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from nvidia_transcribe.source_reconciliation import SCHEMA as DOSSIER_SCHEMA
from nvidia_transcribe.source_recovery import LANGUAGES, _id, _interval, _overlap
from nvidia_transcribe.source_unit_lineage import bind_legacy_source_lineage

SCHEMA = "nemovox-bounded-source-segments/v1"
PROMOTION_SCHEMA = "nemovox-source-promotion/v1"
RECORDING_SCHEMA = "nemotron-recording-contract/v1"
MAX_EVENT_S = 18.0
MAX_BOUNDED_AUDIO_S = 24.0
SHA_HEX = frozenset("0123456789abcdef")
NON_SPEECH_CLASSES = frozenset({"music", "singing", "sfx", "silence", "nonlexical_vocalization"})
# Conservative review screens, NOT evidence of a word's acoustic origin.
# A 2.1-second scene cannot safely inherit 19 words from an 8.1-second crop.
# The small absolute allowances avoid blocking short real exclamations.
MAX_PROVISIONAL_FR_WORDS_PER_SECOND = 5.5
PROVISIONAL_FR_WORD_SLACK = 2
MAX_PROVISIONAL_ZH_HAN_PER_SECOND = 12.0
PROVISIONAL_ZH_HAN_SLACK = 6


def _lexical_density_evidence(text: str, language: str, duration_s: float) -> dict[str, Any]:
    """Flag physically implausible source-text scope, not language correctness.

    This only gates UNREVIEWED model output. Real rapid speech can exceed
    a heuristic; quarantining retains the raw model hypothesis for separate
    audition, never deletes an actual source observation.
    """
    if language == "zh":
        count = sum(
            0x3400 <= ord(ch) <= 0x9FFF or 0x20000 <= ord(ch) <= 0x2FA1F
            for ch in text
        )
        limit = MAX_PROVISIONAL_ZH_HAN_PER_SECOND * duration_s + PROVISIONAL_ZH_HAN_SLACK
        unit = "han_characters"
    elif language == "fr":
        count = len(text.split())
        limit = MAX_PROVISIONAL_FR_WORDS_PER_SECOND * duration_s + PROVISIONAL_FR_WORD_SLACK
        unit = "whitespace_words"
    else:
        raise ValueError("Unsupported source language for lexical-density screen")
    return {
        "unit": unit,
        "count": count,
        "event_duration_s": round(duration_s, 4),
        "units_per_second": round(count / duration_s, 4),
        "conservative_limit": round(limit, 4),
        "requires_review": count > limit,
        "not_an_acoustic_transcript_validation": True,
    }



def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or not set(value.lower()) <= SHA_HEX:
        raise ValueError(f"{name} requires a 64-character SHA-256")
    return value.lower()


def _str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value.strip()


def _same_bounds(left: tuple[float, float], right: tuple[float, float], eps: float = 0.001) -> bool:
    return abs(left[0] - right[0]) <= eps and abs(left[1] - right[1]) <= eps


def promote_bounded_source_turns(
    recording: Mapping[str, Any],
    dossier: Mapping[str, Any],
    segments: Mapping[str, Any],
    *,
    activity: Mapping[str, Any] | None = None,
    source_evidence: Mapping[str, Any] | None = None,
    include_model_provisional: bool = True,
) -> dict[str, Any]:
    """Create a v1-compatible *new* recording contract; never mutate inputs.

    Invariant: a 24-second language-model window is not a short speech turn.
    For model-only text, the candidate must be separately re-transcribed on
    a bounded input with its own raw-response and crop hashes, and match a
    positive speech activity window. It remains explicitly unverified.
    """
    if recording.get("schema_version") != RECORDING_SCHEMA:
        raise ValueError("Unsupported v1 recording contract")
    if dossier.get("schema") != DOSSIER_SCHEMA or segments.get("schema") != SCHEMA:
        raise ValueError("Unexpected source dossier or bounded segment schema")
    media = (recording.get("source") or {}).get("fingerprint_sha256")
    language = recording.get("source_language")
    if language not in LANGUAGES or not isinstance(media, str) or not media:
        raise ValueError("Recording requires valid media fingerprint and source language")
    if any(doc.get("media_fingerprint_sha256") != media for doc in (dossier, segments)):
        raise ValueError("Source/dossier/segment media fingerprint mismatch")
    if any(doc.get("source_language") != language for doc in (dossier, segments)):
        raise ValueError("Source/dossier/segment language mismatch")
    if activity is not None and (activity.get("schema") != "nemovox-audio-activity-evidence/v1"
                                 or activity.get("media_fingerprint_sha256") != media):
        raise ValueError("Activity schema/fingerprint mismatch")
    if not isinstance(include_model_provisional, bool):
        raise ValueError("Model provisional enable flag must be boolean")

    duration = (recording.get("source") or {}).get("duration_s")
    if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Recording requires positive finite duration")
    baseline = deepcopy(recording)
    original = baseline.get("utterances") or []
    if not isinstance(original, list):
        raise ValueError("v1 recording utterances must be an array")
    original_ids = [str(u.get("id") or "") for u in original]
    if not all(original_ids) or len(set(original_ids)) != len(original_ids):
        raise ValueError("Missing or duplicate legacy utterance identities")

    # Recompute provenance from the exact original raw-unit ledger and
    # independently reviewed source events. Do NOT trust a free-form
    # user-supplied event/utterance ID correspondence as deletion authority.
    lineage = (
        bind_legacy_source_lineage(recording, source_evidence, dossier)
        if source_evidence is not None else None
    )
    eligible_by_event: dict[str, set[str]] = {}
    partially_linked_by_event: dict[str, set[str]] = {}
    if lineage is not None:
        for row in lineage["legacy_utterances"]:
            events = row["reviewed_event_ids"]
            uid = row["legacy_utterance_id"]
            if row["eligible_for_reviewed_same_event_supersession"]:
                eligible_by_event.setdefault(row["positive_same_event_id"], set()).add(uid)
            elif events:
                for eid in events:
                    partially_linked_by_event.setdefault(eid, set()).add(uid)

    verified_turns = {}
    for turn in dossier.get("source_turns_audio_review_attested") or []:
        eid = _str(turn.get("audio_event_id"), "audio_event_id")
        if eid in verified_turns:
            raise ValueError("Duplicate reviewed audio event")
        if turn.get("schema") != "nemovox-source-turn-revision/v1":
            raise ValueError("Unsupported reviewed source-turn contract")
        if (turn.get("source_status") != "audio_review_attested_source"
                or turn.get("selection_basis") != "independent_audio_review"
                or turn.get("reviewer_has_attested_direct_audio_listening") is not True
                or turn.get("source_language") != language):
            raise ValueError("Source event lacks positive direct audio-review attestation")
        _sha(turn.get("review_evidence_sha256"), "direct audio-review evidence")
        _str(turn.get("source_text"), "reviewed source text")
        _interval(turn, start="source_start", end="source_end")
        verified_turns[eid] = turn
    regions = {}
    for region in dossier.get("audited_audio_regions") or []:
        eid = _str(region.get("audio_event_id"), "audio_event_id")
        if eid in regions:
            raise ValueError("Duplicate audio event region")
        _interval(region)
        regions[eid] = region
    selected_model = dossier.get("challenger_model") or {}
    window_contexts = {}
    for window in dossier.get("model_windows") or []:
        wid = _str(window.get("id"), "model_window_id")
        if wid in window_contexts:
            raise ValueError("Duplicate model window identity")
        _interval(window)
        window_contexts[wid] = window

    activity_rows = {}
    for row in (activity or {}).get("intervals") or []:
        aid = _str(row.get("id"), "activity_id")
        if aid in activity_rows:
            raise ValueError("Duplicate acoustic activity evidence ID")
        _str(row.get("source"), "activity_source")
        activity_rows[aid] = {**row, "_bounds": _interval(row)}

    raw_candidates: dict[str, list[dict[str, Any]]] = {}
    seen_candidates = set()
    for proposal in segments.get("segments") or []:
        cid = _str(proposal.get("id"), "candidate_id")
        eid = _str(proposal.get("audio_event_id"), "audio_event_id")
        if cid in seen_candidates:
            raise ValueError("Duplicate bounded candidate ID")
        seen_candidates.add(cid)
        a, b = _interval(proposal)
        if b > duration + .001 or b - a > MAX_EVENT_S:
            raise ValueError("Source event bounds exceed media or bounded turn limit")
        _str(proposal.get("source_text"), "candidate source text")
        kind = proposal.get("kind")
        if kind not in {"speech", "lexical_shout"}:
            raise ValueError("Only lexical spoken events are eligible for source promotion")
        if eid in regions:
            region = regions[eid]
            if not _same_bounds((a, b), _interval(region), eps=.1):
                raise ValueError("Candidate source bounds mismatch audited acoustic event")
            if region.get("kind") not in {"speech", "lexical_shout"}:
                raise ValueError("Candidate speech conflicts with reviewed non-speech event")
        raw_candidates.setdefault(eid, []).append(dict(proposal))

    accepted: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    used_supersessions: dict[str, str] = {}
    for eid in sorted(raw_candidates):
        proposals = sorted(raw_candidates[eid], key=lambda r: r["id"])
        reasons: list[str] = []
        if len(proposals) > 1:
            # Exact repeated candidate copies are not a new acoustic event.
            signatures = {(x["start_s"], x["end_s"], x["source_text"]) for x in proposals}
            if len(signatures) > 1:
                reasons.append("conflicting_candidates_same_audio_event")
        candidate = proposals[0]
        a, b = _interval(candidate)
        refs = candidate.get("context_window_ids") or []
        if not isinstance(refs, list) or any(not isinstance(r, str) for r in refs):
            raise ValueError("Model context references must be an array of IDs")
        if len(set(refs)) != len(refs) or any(r not in window_contexts for r in refs):
            raise ValueError("Unknown or repeated model context window ID")
        for r in refs:
            if _overlap(a, b, *_interval(window_contexts[r])) <= 0:
                raise ValueError("Model context window cannot have heard proposed event")
        reviewed = verified_turns.get(eid)
        if reviewed:
            if (candidate["source_text"] != reviewed.get("source_text")
                    or not _same_bounds((a, b), (
                        float(reviewed["source_start"]), float(reviewed["source_end"])
                    ), eps=.1)):
                reasons.append("candidate_not_identical_to_directly_reviewed_text")
            basis = "reviewed"
        else:
            basis = "model_provisional"
            # Window-wide Qwen text can contain several neighboring voices.
            # Matching a digest only proves the text came from the crop, not
            # that every word fits into this one acoustic event. A gross
            # source-text/duration mismatch must not become an English cue.
            density = _lexical_density_evidence(
                candidate["source_text"], language, b - a
            )
            if density["requires_review"]:
                reasons.append("model_text_scope_exceeds_audio_event_duration")
            # A direct independent audition of ONLY music/SFX/singing is
            # stronger contrary evidence than any number of model activity
            # flags. However, an overlapping independently heard SPEECH
            # region means real simultaneous voice remains possible.
            audited_overlaps = [
                region for region in regions.values()
                if _overlap(a, b, *_interval(region)) > 0
            ]
            negative_dominant = any(
                region.get("kind") in {"singing", "sfx", "silence", "nonlexical_vocalization"}
                and _overlap(a, b, *_interval(region)) >= 0.5*(b-a)
                for region in audited_overlaps
            )
            independently_heard_speech = any(
                region.get("kind") in {"speech", "lexical_shout"}
                for region in audited_overlaps
            )
            if negative_dominant and not independently_heard_speech:
                reasons.append("contradicts_independently_auditioned_non_speech")
            bound = candidate.get("bounded_asr") or {}
            if not isinstance(bound, dict):
                raise ValueError("bounded_asr must be an object")
            if not bound:
                reasons.append("missing_bounded_asr_for_model_only_text")
            else:
                cropped = _interval(bound, start="crop_start_s", end="crop_end_s")
                crop_length = cropped[1] - cropped[0]
                # Existing 24s sweep text cannot be relabeled as a 2s voice.
                # Demand a separate materially narrower model input crop.
                bounded_limit = min(MAX_BOUNDED_AUDIO_S, 16.0, max(6.0, (b-a)+8.0))
                if (crop_length > bounded_limit
                        or cropped[0] > a or cropped[1] < b or cropped[1] > duration + .001):
                    raise ValueError("Bounded ASR crop must tightly contain the acoustic event")
                repo = _str(bound.get("model_repo_id"), "bounded_asr model")
                revision = _str(bound.get("model_revision"), "bounded_asr revision")
                if repo != selected_model.get("repo_id") or revision != selected_model.get("revision"):
                    raise ValueError("Bounded ASR model identity differs from authenticated challenger")
                _sha(bound.get("audio_crop_sha256"), "bounded audio crop")
                _sha(bound.get("raw_response_sha256"), "bounded ASR response")
                if (_sha(bound.get("source_text_sha256"), "bounded transcript") !=
                        hashlib.sha256(candidate["source_text"].encode("utf-8")).hexdigest()):
                    raise ValueError("Bounded ASR transcript text does not match authenticated digest")
                if bound.get("finish_status") not in {"complete", "completed"}:
                    reasons.append("bounded_asr_not_complete")
                if bound.get("language") != LANGUAGES[language]:
                    raise ValueError("Bounded model language must be pinned to source")
            if not include_model_provisional:
                reasons.append("model_provisional_not_enabled")
            support_ids = candidate.get("speech_activity_ids") or []
            if not isinstance(support_ids, list) or len(set(support_ids)) != len(support_ids):
                raise ValueError("Speech support IDs must be a unique array")
            if not support_ids:
                reasons.append("no_positive_speech_activity_for_model_only_text")
            positive_families = set()
            for aid in support_ids:
                support = activity_rows.get(aid)
                if support is None or support.get("kind") not in {"speech", "lexical_shout"}:
                    raise ValueError("Claimed positive activity ID is not known speech")
                if _overlap(a, b, *support["_bounds"]) <= 0:
                    raise ValueError("Positive speech support must intersect source event")
                positive_families.add(support["source"])
            # Singing and effects can trigger false speech detectors. Do not
            # call this an absolute audio-negative test: if multiple
            # independent families detect speech, retain provisional content.
            # A single model's speech flag on predominantly song/SFX audio
            # is insufficient to promote new, unreviewed model words.
            nonlexical_dominance = any(
                r.get("kind") in {"singing", "sfx", "nonlexical_vocalization"}
                and _overlap(a, b, *r["_bounds"]) >= 0.5*(b-a)
                for r in activity_rows.values()
            )
            if nonlexical_dominance and len(positive_families) < 2:
                reasons.append("high_nonlexical_risk_without_independent_speech_corroboration")

        replacement_ids = candidate.get("supersedes_legacy_utterance_ids") or []
        if (not isinstance(replacement_ids, list) or len(set(replacement_ids)) != len(replacement_ids)
                or any(not isinstance(x, str) for x in replacement_ids)):
            raise ValueError("Supersessions must be distinct legacy IDs")
        if reviewed and lineage is not None:
            eligible = eligible_by_event.get(eid, set())
            partial = partially_linked_by_event.get(eid, set())
            if eligible and set(replacement_ids) != eligible:
                reasons.append("reviewed_same_event_requires_complete_explicit_supersession")
            # Partially linked legacy text is NOT a reason to discard a
            # newly heard real phrase. Keep both source observations, expose
            # the unresolved overlap, and prohibit replacing partial rows.
            if replacement_ids and not set(replacement_ids) <= eligible:
                reasons.append("requested_supersession_not_proven_by_immutable_source_units")
        if replacement_ids and not reviewed:
            reasons.append("model_only_candidate_may_not_replace_legacy_speech")
        if any(uid not in original_ids for uid in replacement_ids):
            raise ValueError("Cannot supersede unknown legacy utterance")
        for uid in replacement_ids:
            if uid in used_supersessions:
                raise ValueError("Legacy utterance claimed by multiple source events")
            # Never delete an unrelated legacy turn merely because its time
            # overlaps this reviewed speech. Supersession requires positive
            # event-ID lineage already on that v1-compatible utterance.
            original_turn = original[original_ids.index(uid)]
            original_lineage = original_turn.get("source_turn_ids") or (
                [original_turn["source_turn_id"]] if original_turn.get("source_turn_id") else []
            )
            if (_id("src_turn", [media, eid]) not in original_lineage
                    and (source_evidence is None
                         or uid not in eligible_by_event.get(eid, set()))):
                reasons.append("legacy_turn_lacks_positive_same_event_lineage")
            old_s, old_e = original_turn.get("source_start"), original_turn.get("source_end")
            # Matching IDs are necessary but not alone sufficient to erase
            # legacy content if its placement is wildly inconsistent with
            # the purported same acoustic event. Preserve for audit.
            if (not isinstance(old_s, (int, float))
                    or not isinstance(old_e, (int, float))
                    or isinstance(old_s, bool) or isinstance(old_e, bool)
                    or not math.isfinite(old_s) or not math.isfinite(old_e)
                    or old_s > old_e or _overlap(a-3, b+3, old_s, old_e) <= 0):
                reasons.append("legacy_supersession_audio_bounds_unverified")

        overlapping_legacy = []
        for utterance, uid in zip(original, original_ids):
            us, ue = utterance.get("source_start"), utterance.get("source_end")
            if (isinstance(us, (int, float)) and isinstance(ue, (int, float))
                    and not isinstance(us, bool) and not isinstance(ue, bool)
                    and math.isfinite(us) and math.isfinite(ue)
                    and us <= ue and _overlap(a, b, us, ue) > 0):
                overlapping_legacy.append(uid)
        if not reviewed and any(uid not in replacement_ids for uid in overlapping_legacy):
            # A speech detector and bounded ASR could be rediscovering a
            # phrase already present in v1; this cannot be inferred away.
            reasons.append("model_candidate_collides_with_existing_legacy_speech")
        if reviewed:
            event_id = _id("src_turn", [media, eid])
            if event_id != reviewed.get("source_turn_id"):
                raise ValueError("Reviewed source event ID does not match canonical lineage")
            rev_id = reviewed.get("source_revision_id")
            _str(rev_id, "source_revision_id")
            status = "audio_review_attested_source"
        else:
            event_id = _id("src_turn", [media, eid])
            rev_id = _id("src_revision", [
                media, eid, candidate["source_text"], a, b,
                (candidate.get("bounded_asr") or {}).get("raw_response_sha256"),
            ])
            status = "model_source_candidate_provisional"

        if reasons:
            quarantined.append({
                "audio_event_id": eid,
                "candidate_ids": [x["id"] for x in proposals],
                "reasons": sorted(set(reasons)),
                "candidate_source_sha256": hashlib.sha256(
                    candidate["source_text"].encode("utf-8")
                ).hexdigest(),
                "overlapping_legacy_utterance_ids": overlapping_legacy,
                "cannot_be_presented_as_accepted_speech": True,
                "model_text_scope_screen": (
                    density if not reviewed else None
                ),
            })
            continue

        for uid in replacement_ids:
            used_supersessions[uid] = eid
        selected = {
            "id": event_id,
            "utterance_id": event_id,
            "source_turn_id": event_id,
            "source_turn_ids": [event_id],
            "source_revision_id": rev_id,
            "audio_event_id": eid,
            "source_text": candidate["source_text"],
            "source_language": language,
            "source_start": a, "source_end": b,
            "source_words": [], "source_characters": [],
            "unit_count": 0,
            "timed_unit_count": 0, "untimed_unit_count": 0,
            "timing_gaps": [],
            "speaker_id": "SPK_UNK",
            "alignment_status": "turn_bounded_word_timing_unverified",
            "timestamp_provenance": {
                "status": "audio_event_bounds_only_not_word_timing",
                "source_revision_id": rev_id,
            },
            "source_acceptance_status": status,
            "source_observation_ids": (
                list(reviewed.get("source_observation_ids") or []) if reviewed else []
            ),
            "bounded_candidate_ids": [x["id"] for x in proposals],
            "source_window_context_ids": list(refs),
            "speaker_identity_accepted": False,
            "source_text_review_attested": bool(reviewed),
            "source_text_human_reference_verified": False,
            "source_text_scope_screen": (
                density if not reviewed else None
            ),
        }
        accepted.append({
            "utterance": selected,
            "supersedes": replacement_ids,
            "overlapping_legacy_utterance_ids": overlapping_legacy,
            "partially_attributed_legacy_utterance_ids": sorted(
                partially_linked_by_event.get(eid, set())
            ),
        })

    retained = [u for u, uid in zip(original, original_ids) if uid not in used_supersessions]
    new_utts = [x["utterance"] for x in accepted]
    decorated = [(u, i) for i, u in enumerate(retained + new_utts)]
    # Stable ordering, but never invent ordering for unlocated legacy text.
    decorated.sort(key=lambda item: (
        float(item[0].get("source_start")) if isinstance(item[0].get("source_start"), (int, float))
        and math.isfinite(item[0]["source_start"]) else float("inf"),
        item[1],
    ))
    baseline["utterances"] = [row for row, _ in decorated]
    if len({u["id"] for u in baseline["utterances"]}) != len(baseline["utterances"]):
        raise ValueError("New turn identity collides with legacy utterance identity")
    report = {
        "schema": PROMOTION_SCHEMA,
        "media_fingerprint_sha256": media,
        "source_language": language,
        "policy": {
            "model_provisional_enabled": include_model_provisional,
            "model_source_is_not_human_verified": True,
            "no_window_wide_text_as_localized_source": True,
            "bounded_text_requires_separate_crop_and_transcript_digest": True,
            "model_only_may_not_replace_legacy": True,
            "no_auto_music_sfx_source": True,
            "dominant_singing_sfx_requires_two_independent_speech_sources_or_direct_review": True,
            "independent_non_speech_only_acoustic_reference_blocks_model_promotion": True,
            "no_source_unit_deletion": True,
            "legacy_supersession_requires_existing_same_event_id": True,
            "legacy_supersession_requires_plausible_event_time": True,
            "optional_positive_all_unit_lineage_enables_legacy_supersession": True,
            "no_automatic_legacy_suppression_from_similarity_or_ctc": True,
            "word_timing_and_speaker_identity_not_certified": True,
            "unreviewed_text_scope_rate_screen_is_not_acoustic_truth": True,
        },
        "accepted": [
            {
                "audio_event_id": x["utterance"]["audio_event_id"],
                "source_turn_id": x["utterance"]["source_turn_id"],
                "source_revision_id": x["utterance"]["source_revision_id"],
                "status": x["utterance"]["source_acceptance_status"],
                "supersedes_legacy_utterance_ids": x["supersedes"],
                "original_legacy_utterances_preserved_in_input": True,
                "overlapping_legacy_utterance_ids": x["overlapping_legacy_utterance_ids"],
                "partially_attributed_legacy_utterance_ids": x["partially_attributed_legacy_utterance_ids"],
            }
            for x in accepted
        ],
        "quarantined": quarantined,
        "legacy_lineage_diagnostics": (lineage["diagnostics"] if lineage is not None else None),
        "preserved_legacy_utterance_count": len(retained),
        "superseded_legacy_utterance_count": len(used_supersessions),
        "source_utterances_input": len(original),
        "source_utterances_output": len(baseline["utterances"]),
        "new_source_turns": len(accepted),
        "new_model_provisional_turns": sum(
            x["utterance"]["source_acceptance_status"] == "model_source_candidate_provisional"
            for x in accepted
        ),
        "new_audio_review_attested_turns": sum(
            x["utterance"]["source_acceptance_status"] == "audio_review_attested_source"
            for x in accepted
        ),
        "incomplete_source_recovery_not_claimed_complete": True,
        "source_fidelity_certified": False,
    }
    baseline["source_promotion"] = report
    return baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-contract", required=True, type=Path)
    parser.add_argument("--reconciliation-dossier", required=True, type=Path)
    parser.add_argument("--bounded-segments", required=True, type=Path)
    parser.add_argument("--activity-evidence", type=Path)
    parser.add_argument("--source-evidence", type=Path,
                        help="Optional authenticated raw-unit replay for positive same-event legacy supersession")
    parser.add_argument("--disable-model-provisional", action="store_true")
    parser.add_argument("--output-contract", required=True, type=Path)
    args = parser.parse_args()
    if args.output_contract.exists():
        parser.error("Refuse overwrite of source recording contract")
    def read(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))
    output = promote_bounded_source_turns(
        read(args.recording_contract),
        read(args.reconciliation_dossier),
        read(args.bounded_segments),
        activity=read(args.activity_evidence) if args.activity_evidence else None,
        source_evidence=read(args.source_evidence) if args.source_evidence else None,
        include_model_provisional=not args.disable_model_provisional,
    )
    args.output_contract.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output_contract, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    report = output["source_promotion"]
    print(json.dumps({
        "output_contract": str(args.output_contract),
        "new_source_turns": report["new_source_turns"],
        "model_provisional": report["new_model_provisional_turns"],
        "reviewed": report["new_audio_review_attested_turns"],
        "quarantined": len(report["quarantined"]),
        "fidelity_certified": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
