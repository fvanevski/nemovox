"""Configurable, source-preserving evidence assessment for subtitle correction.

This is a CPU-only opt-in review scaffold, NOT a production correction engine.
It never modifies the primary recording, source ledger, or any job output.
Numerical salience is a tunable review preference, not calibrated probability.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re
from typing import Any, Mapping

CASE_SCHEMA = "nemovox-evidence-case/v1"
POLICY_SCHEMA = "nemovox-evidence-policy/v1"
REVIEW_SCHEMA = "nemovox-evidence-model-review/v1"
ASSESSMENT_SCHEMA = "nemovox-evidence-assessment/v1"
PARTITION_SCHEMA = "nemovox-reviewed-unit-partitions/v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_STATUS = {"consider", "limited", "exclude"}
_DECISIONS = {"support", "reject", "abstain"}


def _required_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"{label} must be an exact SHA256 digest")
    return value


def _interval(value: Any, duration: float, label: str) -> tuple[float, float]:
    if (not isinstance(value, list) or len(value) != 2
            or any(not isinstance(x, (int, float)) or isinstance(x, bool)
                   or not math.isfinite(x) for x in value)):
        raise ValueError(f"{label} needs two finite seconds")
    start, end = map(float, value)
    if not 0 <= start < end <= duration + 0.001:
        raise ValueError(f"{label} outside source media")
    return start, end


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _policy_rule(rule: Mapping[str, Any], observation: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    match = rule.get("match")
    if not isinstance(match, dict):
        raise ValueError("Policy rule match must be a mapping")
    allowed = {"modality", "claim_kind", "source_family", "scene_kind", "av_sync_status"}
    if not set(match) <= allowed:
        raise ValueError("Unknown policy selector")
    actual = {"modality": observation["modality"], "claim_kind": observation["claim_kind"],
              "source_family": observation["source_family"],
              "scene_kind": context.get("scene_kind", "unknown"),
              "av_sync_status": context.get("av_sync_status", "unverified")}
    return all(actual[k] == v for k, v in match.items())


def prepare_evidence_review(case: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate identities/dependencies and annotate context-specific applicability.

    Rules provide advisory review salience only. The configured model/reviewer
    weighs supporting and conflicting claims; a salience cannot certify speech.
    """
    if case.get("schema") != CASE_SCHEMA or policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("Unknown case/policy schema")
    _required_string(case.get("case_id"), "case_id")
    media = _sha(case.get("media_fingerprint_sha256"), "media fingerprint")
    duration = case.get("duration_s")
    if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(duration) or duration <= 0):
        raise ValueError("Source duration must be positive finite")
    context = case.get("context")
    if not isinstance(context, dict):
        raise ValueError("Missing source context")
    for field in ("scene_kind", "av_sync_status"):
        _required_string(context.get(field), field)
    rules = policy.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("Policy rules must be a list")
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("status") not in _STATUS:
            raise ValueError("Invalid policy status")
        salience = rule.get("salience", 1.0)
        if (not isinstance(salience, (int, float)) or isinstance(salience, bool)
                or not math.isfinite(salience) or not 0 <= salience <= 10):
            raise ValueError("Invalid advisory salience")
        _required_string(rule.get("rationale"), "rule rationale")
        if not isinstance(rule.get("priority", 0), int) or isinstance(rule.get("priority", 0), bool):
            raise ValueError("Policy priority must be an integer")
    observations = case.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Case requires observations")
    by_id: dict[str, dict[str, Any]] = {}
    for obs in observations:
        if not isinstance(obs, dict):
            raise ValueError("Observation must be an object")
        oid = _required_string(obs.get("id"), "observation id")
        if oid in by_id:
            raise ValueError("Duplicate observation ID")
        for field in ("modality", "claim_kind", "subject_id", "source_family"):
            _required_string(obs.get(field), field)
        _sha(obs.get("artifact_sha256"), "observation artifact")
        if obs.get("media_fingerprint_sha256") != media:
            raise ValueError("Cross-media evidence is not admissible")
        _interval(obs.get("interval_s"), float(duration), "observation interval")
        if "assertion" not in obs:
            raise ValueError("Evidence lacks a testable assertion")
        if not isinstance(obs.get("derived_from", []), list):
            raise ValueError("derived_from must be an array")
        if not isinstance(obs.get("method"), str) or not obs["method"]:
            raise ValueError("Evidence method is required")
        by_id[oid] = deepcopy(obs)
    stack: set[str] = set()

    def roots(oid: str) -> set[str]:
        if oid in stack:
            raise ValueError("Cyclic derived evidence")
        obs = by_id[oid]
        parents = obs.get("derived_from", [])
        if len(parents) != len(set(parents)):
            raise ValueError("Duplicate evidence dependency")
        stack.add(oid)
        try:
            for parent in parents:
                if parent not in by_id or parent == oid:
                    raise ValueError("Unknown/self evidence dependency")
            if not parents:
                return {obs["source_family"]}
            result: set[str] = set()
            for parent in parents:
                result.update(roots(parent))
            return result
        finally:
            stack.remove(oid)

    evaluated = []
    for oid, obs in by_id.items():
        source_families = sorted(roots(oid))
        matching = [r for r in rules if _policy_rule(r, obs, context)]
        # Longest selector wins; conflicting equally specific rules fail closed.
        if matching:
            rank = max((r.get("priority", 0), len(r["match"])) for r in matching)
            selected = [r for r in matching if (r.get("priority", 0), len(r["match"])) == rank]
            if len(selected) != 1:
                raise ValueError("Ambiguous equally specific evidence policy rules")
            rule = selected[0]
            status, salience, reason = rule["status"], float(rule.get("salience", 1)), rule["rationale"]
        else:
            status, salience, reason = "consider", 1.0, "Unweighted observation; independently assess relevance"
        evaluated.append({"id": oid, "observation": obs,
                          "root_source_families": source_families,
                          "applicability": status, "review_salience": salience,
                          "applicability_rationale": reason})
    candidates = case.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Case requires proposed alternatives, including abstention")
    candidate_ids = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Malformed candidate")
        cid = _required_string(candidate.get("id"), "candidate id")
        _required_string(candidate.get("action"), "candidate action")
        if cid in candidate_ids:
            raise ValueError("Duplicate candidate")
        candidate_ids.add(cid)
    return {"schema": "nemovox-evidence-review-request/v1", "case_id": case["case_id"],
            "media_fingerprint_sha256": media, "context": deepcopy(context),
            "policy_id": _required_string(policy.get("policy_id"), "policy_id"),
            "policy_sha256": _digest(policy), "case_sha256": _digest(case),
            "reviewer_guidance": str(policy.get("reviewer_guidance", "")),
            "observations": evaluated, "candidates": deepcopy(candidates),
            "review_rules": ["Distinguish observable events from inferred identity and source text",
                             "A/V agreement is not independent when one observation derived from the other",
                             "Explicitly compare counterevidence and abstain if unresolved",
                             "Do not treat advisory salience as calibrated probability",
                             "Do not authorize source mutation or speaker promotion"]}


def adjudicate_review(review_request: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a reasoner's evidence citations, returning review-only proposals."""
    if response.get("schema") != REVIEW_SCHEMA or response.get("case_id") != review_request.get("case_id"):
        raise ValueError("Review response identity/schema mismatch")
    if response.get("case_sha256") != review_request.get("case_sha256") or response.get("policy_sha256") != review_request.get("policy_sha256"):
        raise ValueError("Stale case/policy review")
    all_obs = {e["id"]: e for e in review_request["observations"]}
    all_candidates = {c["id"] for c in review_request["candidates"]}
    choices = response.get("decisions")
    if (not isinstance(choices, list) or any(not isinstance(c, dict) for c in choices)
            or len({c.get("candidate_id") for c in choices}) != len(choices)):
        raise ValueError("Repeated or missing reviewed decisions")
    if {c.get("candidate_id") for c in choices} != all_candidates:
        raise ValueError("Not all candidates were assessed")
    out = []
    for choice in choices:
        decision = choice.get("decision")
        if decision not in _DECISIONS:
            raise ValueError("Invalid reviewer decision")
        supporting = choice.get("supporting_evidence_ids")
        contrary = choice.get("contrary_evidence_ids")
        if (not isinstance(supporting, list) or not isinstance(contrary, list)
                or len(set(supporting)) != len(supporting)
                or len(set(contrary)) != len(contrary)
                or set(supporting) & set(contrary)
                or any(v not in all_obs for v in supporting + contrary)):
            raise ValueError("Invalid or conflicted evidence citations")
        if any(all_obs[v]["applicability"] == "exclude" for v in supporting):
            raise ValueError("Reviewer used policy-excluded evidence as support")
        summary = _required_string(choice.get("rationale"), "review rationale")
        unresolved = choice.get("unresolved", [])
        if not isinstance(unresolved, list) or any(not isinstance(x, str) for x in unresolved):
            raise ValueError("Unresolved concerns must be an array")
        families = sorted({fam for v in supporting for fam in all_obs[v]["root_source_families"]})
        # An output labelled support remains a *reviewed hypothesis* until a
        # separate source-unit reconciliation and explicit application gate.
        out.append({"candidate_id": choice["candidate_id"], "model_decision": decision,
                    "supporting_evidence_ids": supporting, "contrary_evidence_ids": contrary,
                    "independent_root_families": families,
                    "rationale": summary, "unresolved": unresolved,
                    "correction_application": "not_authorized_evidence_review_only"})
    return {"schema": ASSESSMENT_SCHEMA, "case_id": review_request["case_id"],
            "case_sha256": review_request["case_sha256"],
            "policy_sha256": review_request["policy_sha256"],
            "decisions": out, "original_source_mutated": False}


def plan_reviewed_source_partitions(
    recording: Mapping[str, Any], ledger: Mapping[str, Any], dossier: Mapping[str, Any],
) -> dict[str, Any]:
    """Dry-run general reviewed event partition across old utterance boundaries.

    A complete one-to-many unit mapping may be proposed without trusting CTC
    temporal proximity. This never bypasses the current production promotion
    contract and never creates a replacement recording or deletes source units.
    """
    from nvidia_transcribe.source_unit_lineage import bind_legacy_source_lineage

    lineage = bind_legacy_source_lineage(recording, ledger, dossier)
    obs = {(w["chunk_index"], w["source_ordinal"]): w
           for w in ledger["lexical_observations"]}
    by_unit = {}
    for region in dossier.get("audited_audio_regions", []):
        for uid in region.get("positive_audio_event_unit_ids", []):
            if uid in by_unit:
                raise ValueError("One raw source unit assigned to multiple reviewed events")
            by_unit[uid] = region["audio_event_id"]
    reviewed_ids = {t["audio_event_id"] for t in dossier.get("source_turns_audio_review_attested", [])}
    results = []
    by_utt = {r["legacy_utterance_id"]: r for r in lineage["legacy_utterances"]}
    all_identified_units: set[str] = set()
    for utt in recording["utterances"]:
        units = list(utt.get("source_words") or []) + list(utt.get("source_characters") or [])
        ids: list[str] = []
        missing = False
        for item in units:
            original = obs.get((item.get("chunk_index"), item.get("source_unit_index")))
            if original is None or item.get("text") != original.get("text"):
                missing = True
                continue
            ids.append(original["id"])
        if set(ids) & all_identified_units:
            raise ValueError("Duplicated immutable lexical unit ownership")
        all_identified_units.update(ids)
        groups = defaultdict(list)
        for uid in ids:
            if uid in by_unit:
                groups[by_unit[uid]].append(uid)
        covered = (not missing and bool(ids) and len(ids) == len(units)
                   and len(set(ids)) == len(ids)
                   and sum(map(len, groups.values())) == len(ids)
                   and set(groups) <= reviewed_ids)
        status = ("complete_reviewed_unit_partition_requires_application_gate" if covered
                  else "incomplete_or_unreviewed_unit_ownership")
        results.append({"legacy_utterance_id": utt["id"],
                        "existing_lineage_status": by_utt[utt["id"]]["status"],
                        "partition_status": status,
                        "original_source_unit_count": len(units),
                        "accounted_for_unit_count": sum(map(len, groups.values())),
                        "reviewed_event_groups": [{"audio_event_id": eid, "source_unit_ids": members}
                                                  for eid, members in sorted(groups.items())],
                        "replacement_recording_created": False})
    return {"schema": PARTITION_SCHEMA, "media_fingerprint_sha256": lineage["media_fingerprint_sha256"],
            "source_language": lineage["source_language"],
            "recording_content_sha256": lineage["recording_content_sha256"],
            "source_ledger_content_sha256": lineage["source_ledger_content_sha256"],
            "reconciliation_dossier_content_sha256": lineage["reconciliation_dossier_content_sha256"],
            "legacy_utterances": results,
            "diagnostics": {"fully_partitioned": sum(x["partition_status"].startswith("complete") for x in results),
                            "missing_or_unreviewed": sum(x["partition_status"].startswith("incomplete") for x in results)},
            "policy": {"allows_multiple_reviewed_events_per_legacy_utterance_in_dry_run": True,
                       "CTC_proximity_not_application_authority": True,
                       "application_requires_separate_reversible_source_contract": True,
                       "original_source_immutable": True}}
