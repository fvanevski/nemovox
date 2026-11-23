"""Private, opt-in bridge from reviewed source revisions to downstream artifacts.

Never assigns a global speaker identity, fabricates word timing, invokes a
model or changes existing jobs. Translation receipt is tied to an exact new
source revision. Historical translations are reused only for unchanged IDs.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping

_SHA = re.compile(r"^[0-9a-f]{64}$")
_VOICE = re.compile(r"^voice_[0-9]{1,4}$")
_SCHEMA = "nemovox-revision-translation-bridge/v1"


def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                      separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _require_sha(v: Any, label: str) -> str:
    if not isinstance(v, str) or not _SHA.fullmatch(v):
        raise ValueError(f"{label} must be SHA256")
    return v


def _bounds(obj: Mapping[str, Any]) -> tuple[float, float]:
    a, b = obj.get("source_start"), obj.get("source_end")
    if (any(not isinstance(v, (int, float)) or isinstance(v, bool)
            or not math.isfinite(v) for v in (a, b)) or a < 0 or a >= b):
        raise ValueError("Untrusted reviewed event bounds")
    return float(a), float(b)


def attach_local_acoustic_voices(
    revised: Mapping[str, Any],
    dossier: Mapping[str, Any],
    voice_review: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep attested *local* voice equivalence as evidence, never speaker_id.

    The caller must use an independently auditioned review record. Every
    accepted label is explicitly scoped to one media/review context and
    cannot become an episode-wide character or cross-language identity.
    """
    report = revised.get("source_fusion_revision")
    if not isinstance(report, dict) or not report.get("activation_enabled"):
        raise ValueError("Source contract is not an activated private revision")
    media = (revised.get("source") or {}).get("fingerprint_sha256")
    if (voice_review.get("schema") != "nemovox-local-acoustic-voice-review/v1"
            or voice_review.get("media_fingerprint_sha256") != media
            or dossier.get("media_fingerprint_sha256") != media):
        raise ValueError("Local voice/media review schema mismatch")
    _require_sha(voice_review.get("review_scope_sha256"), "local voice review scope")
    _require_sha(voice_review.get("review_evidence_sha256"), "voice review evidence")
    rows = voice_review.get("events")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Local voice evidence requires event rows")
    accepted_events = set(report.get("reviewed_event_ids") or [])
    turns = {t.get("audio_event_id"): t for t in dossier.get("source_turns_audio_review_attested") or []}
    copied = deepcopy(revised)
    by_event = {u.get("audio_event_id"): u for u in copied.get("utterances") or [] if u.get("audio_event_id")}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid local voice evidence")
        eid = row.get("audio_event_id")
        if eid in seen or eid not in accepted_events or eid not in by_event or eid not in turns:
            raise ValueError("Unknown, repeated or unselected local voice event")
        seen.add(eid)
        turn = turns[eid]
        out = by_event[eid]
        if (row.get("speaker_evidence_status") != "direct_audio_reviewed"
                or row.get("speaker_identity_scope") != "local_acoustic_only"
                or row.get("review_method") != "direct_audio_listening"
                or row.get("review_evidence_sha256") != voice_review["review_evidence_sha256"]
                or not turn.get("reviewer_has_attested_direct_audio_listening")
                or out.get("source_revision_id") != turn.get("source_revision_id")
                or _bounds(out) != _bounds(turn)):
            raise ValueError("Local voice identity lacks matching acoustic event review")
        label = row.get("voice_label")
        if not isinstance(label, str) or not _VOICE.fullmatch(label):
            raise ValueError("Local voice label must be an opaque voice_number")
        if out.get("speaker_id") != "SPK_UNK" or out.get("speaker_identity_accepted") is not False:
            raise ValueError("Local voice must not overwrite global speaker identity")
        out["reviewed_local_acoustic_voice"] = {
            "label": label,
            "scope_sha256": voice_review["review_scope_sha256"],
            "review_evidence_sha256": row["review_evidence_sha256"],
            "status": "local_audio_review_only_not_global_speaker",
        }
    copied["source_fusion_local_voice_review_sha256"] = _hash(voice_review)
    return copied


def assemble_revised_translation(
    original_recording: Mapping[str, Any],
    previous_translated: Mapping[str, Any],
    revised_recording: Mapping[str, Any],
    new_translations: list[dict[str, Any]],
    *,
    enable: bool = False,
) -> dict[str, Any]:
    """Build isolated translated.contract from prior + new reviewed events.

    enable=False only reports missing translation IDs. enable=True requires
    an authenticated translation result for every new source-revision ID.
    Existing translated text is never copied to a changed source event.
    """
    if not isinstance(enable, bool) or not isinstance(new_translations, list):
        raise ValueError("Explicit enable boolean and translation list required")
    report = revised_recording.get("source_fusion_revision")
    if not isinstance(report, dict) or not report.get("activation_enabled"):
        raise ValueError("Input requires an opt-in source fusion revision")
    orig_hash = _hash(original_recording)
    if report.get("recording_content_sha256") != orig_hash:
        raise ValueError("Revised source does not identify exact original recording")
    source = (revised_recording.get("source") or {}).get("fingerprint_sha256")
    lang = revised_recording.get("source_language")
    if (not source or original_recording.get("source", {}).get("fingerprint_sha256") != source
            or previous_translated.get("source", {}).get("fingerprint_sha256") != source
            or previous_translated.get("source_language") != lang
            or original_recording.get("source_language") != lang):
        raise ValueError("Source media/language mismatch")
    originals = {u["id"]: u for u in original_recording.get("utterances") or []}
    revised = {u["id"]: u for u in revised_recording.get("utterances") or []}
    # The historical untimed-chronology audit may reorder and renumber the
    # translated contract. Its original_utterance_id is the source identity;
    # the *current* translated row ID is only a presentation/order identity.
    # Verify a complete bijection before reusing a single English word.
    previous_rows = previous_translated.get("utterances") or []
    prior: dict[str, dict[str, Any]] = {}
    prior_display_to_source: dict[str, str] = {}
    for translated_row in previous_rows:
        display_id = translated_row.get("id")
        source_id = translated_row.get("original_utterance_id", display_id)
        if (not isinstance(display_id, str) or not display_id
                or not isinstance(source_id, str) or not source_id
                or display_id in prior_display_to_source or source_id in prior):
            raise ValueError("Duplicate or invalid historical translation source identity")
        prior_display_to_source[display_id] = source_id
        prior[source_id] = translated_row
    if (len(originals) != len(original_recording.get("utterances") or [])
            or len(prior) != len(previous_rows)
            or len(revised) != len(revised_recording.get("utterances") or [])
            or set(originals) != set(prior)):
        raise ValueError("Duplicate utterance identity or baseline translation mismatch")
    replaced = set(report.get("superseded_legacy_ids") or [])
    new_ids = set(revised) - set(originals)
    if (not replaced or not new_ids or set(originals) - set(revised) != replaced
            or new_ids & set(originals)):
        raise ValueError("Revised recording IDs do not match superseded source")
    events = set(report.get("reviewed_event_ids") or [])
    if {revised[x].get("audio_event_id") for x in new_ids} != events:
        raise ValueError("New reviewed turns differ from source revision report")
    received: dict[str, dict[str, Any]] = {}
    for row in new_translations:
        if not isinstance(row, dict) or row.get("utterance_id") not in new_ids:
            raise ValueError("Unexpected fresh translation ID")
        uid = row["utterance_id"]
        if uid in received:
            raise ValueError("Duplicated fresh translation ID")
        u = revised[uid]
        text = row.get("translated_text")
        if (row.get("source_revision_id") != u.get("source_revision_id")
                or row.get("source_text_sha256") != hashlib.sha256(
                    str(u.get("source_text", "")).encode("utf-8")).hexdigest()
                or row.get("source_language") != lang
                or row.get("target_language") != previous_translated.get("target_language")
                or row.get("status") != "translated"
                or not isinstance(text, str) or not text.strip()):
            raise ValueError("Fresh translation is unbound or incomplete")
        _require_sha(row.get("translation_evidence_sha256"), "translation result provenance")
        received[uid] = dict(row)
    missing = sorted(new_ids - set(received))
    out_report = {
        "schema": _SCHEMA,
        "original_recording_sha256": orig_hash,
        "original_translated_sha256": _hash(previous_translated),
        "revised_recording_sha256": _hash(revised_recording),
        "source_fusion_report_sha256": _hash(report),
        "new_source_turn_ids": sorted(new_ids),
        "superseded_source_turn_ids": sorted(replaced),
        "fresh_translations_supplied": len(received),
        "missing_fresh_translation_ids": missing,
        "old_translation_reused_only_when_source_unchanged": True,
        "original_recording_mutated": False,
        "enabled": enable,
    }
    if not enable:
        return {"report": out_report, "translated_contract": None}
    if missing:
        raise ValueError("Fresh translation missing for reviewed source revisions")
    output = deepcopy(revised_recording)
    translated_rows = []
    for u in revised_recording["utterances"]:
        uid = u["id"]
        if uid in new_ids:
            row = deepcopy(u)
            row["raw_source_text"] = row["source_text"]
            row["refined_text"] = row["source_text"]
            row["translated_text"] = received[uid]["translated_text"]
            row["translation_status"] = "translated_from_reviewed_source_revision"
            row["translation_review_status"] = "not_reviewed_for_source_revision"
            row["translation_evidence_sha256"] = received[uid]["translation_evidence_sha256"]
            row["grouped_utterance_ids"] = [uid]
        else:
            old = originals.get(uid)
            translated = prior.get(uid)
            if old is None or translated is None:
                raise ValueError("Missing retained original source translation")
            fields = ("source_text", "source_start", "source_end", "speaker_id")
            if any(old.get(k) != u.get(k) or translated.get(k) != old.get(k) for k in fields):
                raise ValueError("Cannot reuse translation for changed retained source")
            if not str(translated.get("translated_text") or "").strip():
                raise ValueError("Retained translation lacks valid English text")
            def to_source_id(value: Any) -> str:
                if not isinstance(value, str) or value not in prior_display_to_source:
                    raise ValueError("Historical translation references unknown row ID")
                return prior_display_to_source[value]

            original_grouping = translated.get("grouped_utterance_ids") or [translated["id"]]
            if not isinstance(original_grouping, list):
                raise ValueError("Invalid historical translation grouping")
            grouping = [to_source_id(x) for x in original_grouping]
            if len(grouping) != len(set(grouping)):
                raise ValueError("Historical translation grouping is ambiguous")
            merged = translated.get("merged_into")
            stable_merged = to_source_id(merged) if merged is not None else None
            if any(x in replaced for x in grouping) or stable_merged in replaced:
                raise ValueError("Retained translated turn depends on removed legacy ID")
            row = deepcopy(translated)
            if translated["id"] != uid:
                row["historical_translated_row_id"] = translated["id"]
            row["id"] = uid
            if "utterance_id" in row:
                if to_source_id(row["utterance_id"]) != uid:
                    raise ValueError("Contradictory retained translation utterance identity")
                row["utterance_id"] = uid
            row["original_utterance_id"] = uid
            row["grouped_utterance_ids"] = grouping
            if merged is not None:
                row["merged_into"] = stable_merged
            # Duplicate similarity is only a legacy *hypothesis*. Keep the
            # historical reference for traceability, but don't advertise an
            # obsolete dependency as an active source ID after supersession.
            duplicate_of = translated.get("probable_duplicate_of")
            if duplicate_of is not None:
                stable_dup = to_source_id(duplicate_of)
                if stable_dup in replaced:
                    row["historical_probable_duplicate_of_superseded"] = stable_dup
                    row.pop("probable_duplicate_of", None)
                else:
                    row["probable_duplicate_of"] = stable_dup
            # Preserve additional verified revision-local metadata without
            # treating local acoustic voice as global speaker identity.
            if "reviewed_local_acoustic_voice" in u:
                row["reviewed_local_acoustic_voice"] = deepcopy(u["reviewed_local_acoustic_voice"])
        translated_rows.append(row)
    output["utterances"] = translated_rows
    output["target_language"] = previous_translated["target_language"]
    output["translation"] = {
        "backend": "reviewed_source_revision_mixed",
        "historical_unmodified_rows": len(translated_rows) - len(new_ids),
        "newly_translated_reviewed_rows": len(new_ids),
        "revision_report_sha256": _hash(report),
        "original_translation_contract_sha256": _hash(previous_translated),
    }
    output["translation_review"] = {
        "status": "new_revisions_not_postedited",
        "reviewed_source_revisions": 0,
        "historical_status_retained_per_utterance": True,
    }
    output["source_revision_translation_bridge"] = out_report
    return {"report": out_report, "translated_contract": output}
