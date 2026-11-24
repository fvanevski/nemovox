"""Offline meaning-first QC review queue bound to immutable source observations.

A valid LLM advisory is a triage result, not proof of audio, voice, timing or
source equivalence. This module only links its observations to existing raw
source units and plans narrowly scoped listening; it cannot revise subtitles.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping

from nvidia_transcribe.evidence_fusion import _digest
from nvidia_transcribe.semantic_review import (
    POLICY_SCHEMA, RESULT_SCHEMA, REVIEW_SCHEMA, validate_semantic_review,
)
from nvidia_transcribe.source_events import SCHEMA_VERSION as LEDGER_SCHEMA

QUEUE_SCHEMA = "nemovox-meaning-first-review-queue/v1"
BATCH_SCHEMA = "nemovox-tiered-semantic-review-batch/v1"
_SHA_CHARS = frozenset("0123456789abcdef")
_OBSERVATIONS = ("right_context_words", "left_core_overlap_words",
                 "left_lexical_correspondence")
_NEXT_STEP = {
    "possible_duplicate": "audio_review_same_occurrence_vs_distinct_repetition",
    "unresolved": "audio_review_missing_speech_and_competing_hypotheses",
    "retain_core": "confirm_no_change_against_audio_if_material",
    "adopt_neighbor_context": "review_positive_source_event_before_opt_in_revision",
}


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _SHA_CHARS


def _window(value: Any, duration: float) -> tuple[float, float]:
    if (not isinstance(value, list) or len(value) != 2 or any(
        not isinstance(n, (float, int)) or isinstance(n, bool) or not math.isfinite(n)
        for n in value)):
        raise ValueError("Review event window is malformed")
    a, b = float(value[0]), float(value[1])
    if not 0 <= a < b <= duration + .001:
        raise ValueError("Review event exceeds original source duration")
    return a, b


def build_meaning_first_review_queue(
    request: Mapping[str, Any],
    batch: Mapping[str, Any],
    policy: Mapping[str, Any],
    original_recording: Mapping[str, Any],
    raw_ledger: Mapping[str, Any],
    *, audio_padding_s: float = 4.0,
) -> dict[str, Any]:
    """Bind an advisory-only batch to original media and raw lexical units.

    Accepted model decisions remain advisory. Every proposed direct-audio
    review MUST be completed outside this function and separately submitted
    to the existing reviewed_unit_amendment/source revision contracts.
    """
    if (request.get("schema") != RESULT_SCHEMA
            or policy.get("schema") != POLICY_SCHEMA
            or request.get("policy_sha256") != _digest(policy)):
        raise ValueError("Incompatible or stale QC policy/request")
    copied = dict(request)
    digest = copied.pop("request_sha256", None)
    if not _sha(digest) or digest != _digest(copied):
        raise ValueError("QC request identity mismatch")
    if (batch.get("schema") != BATCH_SCHEMA or batch.get("request_sha256") != digest
            or batch.get("model_review_execution_sha256") != _digest(policy.get("model_review") or {})
            or batch.get("source_or_subtitles_modified") is not False):
        raise ValueError("Review batch does not match immutable nonapplying QC request")
    if (raw_ledger.get("schema_version") != LEDGER_SCHEMA
            or not isinstance(original_recording.get("source"), dict)):
        raise ValueError("Unsupported source recording/evidence schema")
    source = original_recording["source"]
    media = source.get("fingerprint_sha256")
    duration = source.get("duration_s")
    if (not isinstance(media, str) or not media
            or raw_ledger.get("media_fingerprint_sha256") != media
            or raw_ledger.get("source_language") != original_recording.get("source_language")
            or not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Original media/source evidence mismatch")
    if (not isinstance(audio_padding_s, (int, float)) or isinstance(audio_padding_s, bool)
            or not math.isfinite(audio_padding_s) or not 0 <= audio_padding_s <= 15):
        raise ValueError("Invalid bounded listening context")
    cases = request.get("cases")
    outcomes = batch.get("outcomes")
    if (not isinstance(cases, list) or not isinstance(outcomes, list)
            or len({c.get("case_id") for c in cases}) != len(cases)):
        raise ValueError("Review candidates or outcomes malformed")
    by_case = {c["case_id"]: c for c in cases}
    seen = set()
    asr_by_chunk: dict[int, dict[str, Any]] = {}
    for row in raw_ledger.get("asr_observations") or []:
        idx = row.get("chunk_index")
        if not isinstance(idx, int) or isinstance(idx, bool) or idx in asr_by_chunk:
            raise ValueError("Nonunique ASR observation/chunk identity")
        if not _sha(row.get("raw_response_sha256")) or not isinstance(row.get("id"), str):
            raise ValueError("Unverifiable raw response identity")
        asr_by_chunk[idx] = row
    coordinates: dict[tuple[int, int], dict[str, Any]] = {}
    raw_ids: set[str] = set()
    for row in raw_ledger.get("lexical_observations") or []:
        coordinate = (row.get("chunk_index"), row.get("source_ordinal"))
        if (not all(isinstance(x, int) and not isinstance(x, bool) for x in coordinate)
                or coordinate in coordinates or not isinstance(row.get("id"), str)
                or row["id"] in raw_ids):
            raise ValueError("Duplicate/invalid raw source-unit evidence")
        if coordinate[0] not in asr_by_chunk or row.get("asr_observation_id") != asr_by_chunk[coordinate[0]]["id"]:
            raise ValueError("Source lexical unit does not bind to raw parent ASR")
        coordinates[coordinate] = row
        raw_ids.add(row["id"])

    def bind_words(words: Any) -> list[dict[str, Any]]:
        if not isinstance(words, list):
            raise ValueError("Candidate source observations are not a list")
        bound: list[dict[str, Any]] = []
        for word in words:
            if not isinstance(word, dict):
                raise ValueError("Invalid candidate lexical observation")
            coordinate = (word.get("chunk_index"), word.get("source_unit_index"))
            if (not all(isinstance(x, int) and not isinstance(x, bool) for x in coordinate)
                    or coordinate not in coordinates):
                raise ValueError("QC candidate has no matching original lexical unit")
            raw = coordinates[coordinate]
            if word.get("text") != raw.get("text"):
                raise ValueError("QC candidate text diverges from immutable raw lexical source")
            parent = asr_by_chunk[coordinate[0]]
            bound.append({
                "source_unit_id": raw["id"], "chunk_index": coordinate[0],
                "source_ordinal": coordinate[1], "parent_asr_observation_id": parent["id"],
                "parent_raw_response_sha256": parent["raw_response_sha256"],
                "local_speaker_hypothesis": raw.get("local_speaker_id"),
                "ctc_timing_is_acoustic_truth": False,
            })
        if len({w["source_unit_id"] for w in bound}) != len(bound):
            raise ValueError("Repeated lexical unit within candidate group")
        return bound

    queued = []
    for outcome in outcomes:
        cid = outcome.get("case_id")
        if cid not in by_case or cid in seen:
            raise ValueError("Duplicate or unknown QC outcome")
        seen.add(cid)
        c = by_case[cid]
        status = outcome.get("status")
        if status not in ("advisory_only", "unaccepted_model_response",
                          "deferred_token_budget", "deferred_systemic_model_failure"):
            raise ValueError("Unknown QC outcome status")
        if status == "advisory_only":
            advisory = outcome.get("assessment")
            if (not isinstance(advisory, dict)
                    or advisory.get("schema") != "nemovox-semantic-review-advisory/v1"
                    or advisory.get("request_sha256") != digest
                    or advisory.get("case_id") != cid
                    or advisory.get("source_application_authorized") is not False
                    or advisory.get("quality_review_only") is not True):
                raise ValueError("Advisory is not a source-bound, nonapplying assessment")
            # Reapply the original semantic acceptance rules to the assessment,
            # without trusting an arbitrary 'advisory_only' status label.
            projected = {"schema": REVIEW_SCHEMA, "request_sha256": digest,
                         "case_id": cid, "decision": advisory.get("decision"),
                         "rationale": advisory.get("rationale"),
                         "supporting_evidence": advisory.get("supporting_evidence"),
                         "contrary_evidence": advisory.get("contrary_evidence"),
                         "unresolved": advisory.get("unresolved", [])}
            checked = validate_semantic_review(request, projected)
            for k in ("decision", "rationale", "supporting_evidence",
                      "contrary_evidence", "unresolved"):
                if checked[k] != advisory.get(k):
                    raise ValueError("Advisory assessment changed after validation")
            # Source JSON cases contain NO independently attested audio/visual
            # review objects. A model cannot supply such proof on its own.
            if {"independent_audio_review", "independent_visual_review"} & set(
                checked["supporting_evidence"] + checked["contrary_evidence"]
            ):
                raise ValueError("Unbacked audio/visual citation in QC advisory")
            decision = checked["decision"]
        else:
            advisory = None
            decision = None
        a, b = _window(c.get("context_window_s"), float(duration))
        parent_chunks = c.get("source_chunks")
        if (not isinstance(parent_chunks, list) or len(parent_chunks) != 2
                or any(idx not in asr_by_chunk for idx in parent_chunks)):
            raise ValueError("QC boundary has unknown original inference parents")
        links = {key: bind_words(c.get(key, [])) for key in _OBSERVATIONS[:2]}
        lexical = c.get("left_lexical_correspondence") or {}
        links["left_lexical_correspondence"] = bind_words(lexical.get("left_words", []))
        all_units = {row["source_unit_id"] for group in links.values() for row in group}
        crop = [round(max(0.0, a - audio_padding_s), 4),
                round(min(float(duration), b + audio_padding_s), 4)]
        queued.append({
            "case_id": cid, "status": status, "model_decision": decision,
            "advisory": deepcopy(advisory), "candidate_source_window_s": [a, b],
            "original_media_listen_window_s": crop, "source_chunks": parent_chunks,
            "source_unit_evidence": links,
            "source_unit_ids": sorted(all_units),
            "material_semantic_review": True,
            "next_action": (_NEXT_STEP[decision] if decision is not None
                            else "inspect_failed_or_deferred_model_review_before_acoustic_decision"),
            "direct_audio_review_required_for_source_change": True,
            "speaker_identity_verified": False, "word_timing_verified": False,
            "source_application_authorized": False,
            "english_postedit_authorized": False,
        })
    if len(seen) != len(outcomes) or batch.get("model_calls") != sum(
        x.get("status") in ("advisory_only", "unaccepted_model_response") for x in outcomes
    ):
        raise ValueError("Reported model-call accounting disagrees with batch evidence")
    output = {
        "schema": QUEUE_SCHEMA, "request_sha256": digest,
        "policy_sha256": request["policy_sha256"],
        "execution_sha256": batch["model_review_execution_sha256"],
        "recording_content_sha256": _digest(original_recording),
        "source_ledger_content_sha256": _digest(raw_ledger),
        "media_fingerprint_sha256": media,
        "source_language": raw_ledger["source_language"],
        "items": queued, "model_calls": batch["model_calls"],
        "advisories": sum(x["status"] == "advisory_only" for x in queued),
        "source_or_translation_or_subtitles_mutated": False,
        "automatic_source_revision": False,
        "review_advisory_is_not_independent_audio_evidence": True,
        "downstream_source_gate": "reviewed_unit_amendment_then_complete_reviewed_source_revision",
        "downstream_english_gate": "existing_source_bound_translation_review_or_verified_new_receipts",
    }
    output["queue_sha256"] = _digest(output)
    return output



def write_meaning_first_review_queue(
    *, request_path: str, batch_path: str, policy_path: str,
    recording_path: str, source_evidence_path: str, output_path: str,
    audio_padding_s: float = 4.0,
) -> dict[str, Any]:
    """Materialize one NEW private review file, without models or source edits."""
    import os
    from pathlib import Path

    def read(path: str) -> Any:
        with Path(path).open("r", encoding="utf-8") as stream:
            return json.load(stream)

    result = build_meaning_first_review_queue(
        read(request_path), read(batch_path), read(policy_path),
        read(recording_path), read(source_evidence_path),
        audio_padding_s=audio_padding_s,
    )
    dest = Path(output_path)
    if not dest.parent.is_dir():
        raise FileNotFoundError("Existing private output parent directory required")
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # O_EXCL prevents overwriting old evidence; restrictive permissions avoid
    # exposing bounded copyrighted source words on shared development hosts.
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise
    return {
        "status": "private_review_queue_created",
        "path": str(dest),
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "queue_sha256": result["queue_sha256"],
        "items": len(result["items"]),
        "advisories": result["advisories"],
        "source_or_subtitles_modified": False,
    }
