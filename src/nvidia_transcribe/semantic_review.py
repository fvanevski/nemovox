"""Conservative, tunable triage for missing/displaced overlapping ASR speech.

Produces evidence-only review cases, never rewrites ASR, CTC, speaker labels,
translations, or source ownership. Text/AV reasoning must return a separately
validated assessment before any reviewed-revision application is possible.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from typing import Any, Mapping, Sequence
from difflib import SequenceMatcher

POLICY_SCHEMA = "nemovox-semantic-review-policy/v1"
REVIEW_SCHEMA = "nemovox-semantic-review-decision/v1"
RESULT_SCHEMA = "nemovox-boundary-semantic-review-candidates/v1"
_SAFE_DECISIONS = {"adopt_neighbor_context", "retain_core", "possible_duplicate", "unresolved"}

# A conservative, metadata-only default. Users may supply a versioned policy
# through reconcile_chunk_boundaries(semantic_review_policy=...). The default
# never loads a reasoning model, vision service or new checkpoint.
DEFAULT_SEMANTIC_REVIEW_POLICY = {
    "schema": POLICY_SCHEMA,
    "policy_id": "meaning-first-boundary-qc-default-v1",
    "min_phrase_words": 3,
    "max_interword_gap_s": 1.2,
    "max_nominal_core_coverage_ratio": 0.45,
    "max_cases_per_run": 24,
    "reviewer_guidance": (
        "Prioritize intended meaning, polarity, readability and narrative continuity. "
        "Compare competing ASR/CTC/diarization evidence, do not equate spelling or "
        "word-for-word agreement with acoustic truth, and abstain on uncertain speech. "
        "Use light text QC first; escalate to audio-based reasoning when material, "
        "and to vision only when it can resolve relevant remaining ambiguity."
    ),
}


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _norm(s: Any) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s or "").casefold()) if c.isalnum() and not unicodedata.combining(c))


def _finite(v: Any) -> bool:
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v)


def _valid(u: Mapping[str, Any]) -> bool:
    return (not u.get("_drop") and u.get("valid_timing") is True and u.get("unit_type") != "character"
            and _finite(u.get("start")) and _finite(u.get("end"))
            and 0 <= u["start"] < u["end"] and _norm(u.get("text")))


def _local_voice(u: Mapping[str, Any]) -> str:
    return str(u.get("local_speaker_id") or u.get("speaker") or "unknown")


def _words(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{"text": str(u.get("text", "")), "start_s": float(u["start"]),
             "end_s": float(u["end"]), "source_unit_index": u.get("_unit_idx"),
             "chunk_index": u.get("_chunk_idx"), "local_voice": _local_voice(u)} for u in units]


def plan_boundary_semantic_reviews(
    tagged_by_chunk: Sequence[Sequence[dict[str, Any]]], chunks: Sequence[Any],
    *, policy: Mapping[str, Any], activity_by_chunk: Mapping[int, Sequence[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Find unowned *candidate* left-context speech, including displaced core echoes.

    Case text comes from actual returned source observations, not synthesized
    audio. Cases are review requests, NEVER automatically accepted corrections.
    """
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError("Invalid semantic review policy schema")
    if len(chunks) != len(tagged_by_chunk):
        raise ValueError("Chunk plans and lexical observations disagree")
    min_words = policy.get("min_phrase_words", 3)
    max_gap = policy.get("max_interword_gap_s", 1.2)
    coverage = policy.get("max_nominal_core_coverage_ratio", .45)
    if (not isinstance(min_words, int) or isinstance(min_words, bool) or min_words < 2
            or not _finite(max_gap) or not 0 < max_gap <= 5
            or not _finite(coverage) or not 0 <= coverage <= 1):
        raise ValueError("Invalid semantic triage thresholds")
    if not isinstance(policy.get("policy_id"), str) or not policy["policy_id"].strip():
        raise ValueError("Policy ID is required")
    if not isinstance(policy.get("reviewer_guidance"), str) or not policy["reviewer_guidance"].strip():
        raise ValueError("Reviewer guidance is required")
    max_cases = policy.get("max_cases_per_run", 24)
    if not isinstance(max_cases, int) or isinstance(max_cases, bool) or max_cases < 1:
        raise ValueError("Invalid review case budget")
    activities = activity_by_chunk or {}
    cases: list[dict[str, Any]] = []
    for b in range(len(chunks) - 1):
        left, right = chunks[b:b + 2]
        overlap_start = max(float(left.infer_start_s), float(right.infer_start_s))
        overlap_end = min(float(left.core_end_s), float(right.core_start_s))
        if overlap_end <= overlap_start:
            continue
        right_context = [u for u in tagged_by_chunk[b + 1] if _valid(u)
                         and overlap_start - .001 <= float(u["start"])
                         and float(u["end"]) <= overlap_end + .001]
        # Consecutive within voice and lexical order; never aggregate
        # another speaker's overlapping interjection as the same phrase.
        by_voice: dict[str, list[dict[str, Any]]] = {}
        for u in right_context:
            by_voice.setdefault(_local_voice(u), []).append(u)
        core = [u for u in tagged_by_chunk[b] if _valid(u) and u.get("_in_core")]
        left_by_voice: dict[str, list[dict[str, Any]]] = {}
        for unit in tagged_by_chunk[b]:
            if _valid(unit):
                left_by_voice.setdefault(_local_voice(unit), []).append(unit)
        for stream in left_by_voice.values():
            stream.sort(key=lambda u: (int(u.get("_unit_idx", 0)), float(u["start"])))
        for voice, stream in by_voice.items():
            stream.sort(key=lambda u: (int(u.get("_unit_idx", 0)), float(u["start"])))
            runs: list[list[dict[str, Any]]] = []
            for u in stream:
                if (runs and int(u.get("_unit_idx", 0)) <= int(runs[-1][-1].get("_unit_idx", 0)) + 2
                        and float(u["start"]) >= float(runs[-1][-1]["start"]) - .15
                        and float(u["start"]) - float(runs[-1][-1]["end"]) <= max_gap):
                    runs[-1].append(u)
                else:
                    runs.append([u])
            for run in runs:
                if len(run) < min_words:
                    continue
                start, end = min(float(x["start"]) for x in run), max(float(x["end"]) for x in run)
                same_time = [u for u in core if float(u["start"]) < end and float(u["end"]) > start]
                ratio = min(1.0, len(same_time) / len(run))
                if ratio > coverage:
                    continue  # already represented or ambiguous; don't duplicate
                norm_right = [_norm(u["text"]) for u in run]
                # Source order must be evaluated per left-local-voice stream.
                # Interleaved other voices can otherwise fragment a matching
                # multiword turn into several misleading short correspondences.
                # Cross-chunk voice labels are NOT assumed to be identical.
                voice_matches: list[tuple[int, str, list[dict[str, Any]]]] = []
                for left_voice, left_stream in left_by_voice.items():
                    match = SequenceMatcher(
                        None, [_norm(u["text"]) for u in left_stream],
                        norm_right, autojunk=False,
                    ).find_longest_match(0, len(left_stream), 0, len(norm_right))
                    if match.size:
                        voice_matches.append((
                            match.size, left_voice,
                            left_stream[match.a:match.a + match.size],
                        ))
                voice_matches.sort(key=lambda x: (-x[0], x[1]))
                best = voice_matches[0] if voice_matches else (0, None, [])
                displaced = best[2] if best[0] >= 2 else []
                offsets = [float(x["start"]) for x in displaced]
                activity = [a for a in activities.get(int(left.chunk_index), ())
                            if a.get("possible_unrepresented_speech") is True
                            and _finite(a.get("start_s")) and _finite(a.get("end_s"))
                            and a["start_s"] < end and a["end_s"] > start]
                case_id = "boundary_" + _digest([left.chunk_index, right.chunk_index, voice,
                                                   [(x.get("_unit_idx"), x.get("text")) for x in run]])[:20]
                cases.append({
                    "case_id": case_id, "status": "review_required_not_auto_adopted",
                    "boundary_s": float(left.core_end_s),
                    "source_chunks": [int(left.chunk_index), int(right.chunk_index)],
                    "context_window_s": [round(start, 4), round(end, 4)],
                    "right_context_words": _words(run),
                    "left_core_overlap_words": _words(same_time),
                    "nominal_core_word_coverage_ratio": round(ratio, 4),
                    "left_lexical_correspondence": {"matched_words": best[0],
                        "left_local_voice_hypothesis": best[1],
                        "left_words": _words(displaced),
                        "left_ctc_start_s": min(offsets) if offsets else None,
                        "competing_left_voice_matches": [
                            {"left_local_voice_hypothesis": v, "matched_words": n}
                            for n, v, _ in voice_matches if n == best[0]
                        ],
                        "alignment_is_acoustic_truth": False},
                    "unrepresented_activity_intervals": [{"start_s": a["start_s"],
                        "end_s": a["end_s"]} for a in activity],
                    "distinct_voice_identity_verified": False,
                    "evidence_limitations": [
                        "Model ASR words and CTC timestamps are hypotheses",
                        "Same words in two chunks may be different spoken occurrences",
                        "The candidate may be a hallucination over non-speech audio",
                        "No acoustic truth or character identity is certified",
                    ],
                    "suggested_review_tier": (
                        "reasoned_text" if activity or displaced or same_time else "light_text"
                    ),
                    "optional_multimodal_escalation": "only_if_visual_evidence_can_resolve_remaining_speaker_or_scene_ambiguity",
                    "proposed_source_action": "none_requires_evidence_bound_adjudication",
                })
    # Bounded priority order: strong independent activity/lexical disagreement
    # first; do not silently pretend other candidates were fully evaluated.
    total_candidates = len(cases)
    cases.sort(key=lambda c: (
        -int(bool(c["unrepresented_activity_intervals"])),
        -c["left_lexical_correspondence"]["matched_words"],
        c["boundary_s"], c["case_id"],
    ))
    cases = cases[:max_cases]
    request = {"schema": RESULT_SCHEMA, "policy_id": policy.get("policy_id"),
               "total_candidates_before_budget": total_candidates,
               "deferred_by_budget": max(0, total_candidates - len(cases)),
               "policy_sha256": _digest(policy), "reviewer_guidance": policy["reviewer_guidance"],
               "cases": cases, "raw_source_modified": False,
               "note": "Light text review first; audiovisual reasoning only for residual relevant ambiguity"}
    request["request_sha256"] = _digest(request)
    return request


def validate_semantic_review(request: Mapping[str, Any], response: Mapping[str, Any]) -> dict[str, Any]:
    """Validate LLM/reviewer citations; return an advisory, non-applying decision."""
    if request.get("schema") != RESULT_SCHEMA or response.get("schema") != REVIEW_SCHEMA:
        raise ValueError("Wrong semantic review schema")
    bound = dict(request)
    digest = bound.pop("request_sha256", None)
    if digest != _digest(bound) or response.get("request_sha256") != digest:
        raise ValueError("Stale quality-review request")
    ids = {c["case_id"] for c in request["cases"]}
    if response.get("case_id") not in ids or response.get("decision") not in _SAFE_DECISIONS:
        raise ValueError("Unknown candidate or decision")
    if not isinstance(response.get("rationale"), str) or not response["rationale"].strip():
        raise ValueError("Evidence-weighting rationale missing")
    if response["decision"] != "unresolved" and not response.get("supporting_evidence"):
        raise ValueError("Non-abstaining review must cite source observations")
    cited = response.get("supporting_evidence", [])
    contrary = response.get("contrary_evidence")
    if not isinstance(contrary, list):
        raise ValueError("Contrary evidence list required, even if empty")
    allowed = {"right_context_words", "left_core_overlap_words", "left_lexical_correspondence",
               "unrepresented_activity_intervals", "independent_audio_review", "independent_visual_review"}
    if (len(contrary) != len(set(contrary)) or any(x not in allowed for x in contrary)):
        raise ValueError("Invalid contrary evidence citations")
    if (not isinstance(cited, list) or len(cited) != len(set(cited))
            or any(s not in allowed for s in cited)):
        raise ValueError("Invalid reviewed evidence citations")
    if response["decision"] == "adopt_neighbor_context":
        audio_sha = response.get("independent_audio_review_sha256")
        if (not response.get("independent_acoustic_support")
                or "independent_audio_review" not in cited
                or not isinstance(audio_sha, str) or len(audio_sha) != 64
                or any(c not in "0123456789abcdef" for c in audio_sha)):
            raise ValueError("Cannot certify neighbor speech from source-text agreement alone")
    return {"schema": "nemovox-semantic-review-advisory/v1", "request_sha256": request["request_sha256"],
            "case_id": response["case_id"], "decision": response["decision"],
            "rationale": response["rationale"], "supporting_evidence": cited,
            "contrary_evidence": contrary, "unresolved": response.get("unresolved", []),
            "source_application_authorized": False, "quality_review_only": True}


def build_semantic_review_messages(request: Mapping[str, Any], case: Mapping[str, Any]) -> list[dict[str, str]]:
    """Bounded source/meaning review prompt; no image or acoustic truth fabricated."""
    if case.get("case_id") not in {x["case_id"] for x in request.get("cases", [])}:
        raise ValueError("Unknown semantic review candidate")
    system = (request["reviewer_guidance"] + "\n" +
              "Return a single JSON object with schema nemovox-semantic-review-decision/v1, "
              "request_sha256, case_id, decision, rationale, supporting_evidence, "
              "contrary_evidence and unresolved. Decision must be one of: adopt_neighbor_context, "
              "retain_core, possible_duplicate, unresolved. Describe both supporting "
              "and contrary evidence. Do not claim independent audio/video review "
              "unless actual authenticated observations were supplied. A plausible "
              "nonliteral English paraphrase is acceptable but never acoustic proof. "
              "Never invent dialogue, speaker names, timing or verified identities.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps({
                "request_sha256": request["request_sha256"], "case": case,
            }, ensure_ascii=False, sort_keys=True)}]


def run_tiered_text_reviews(
    request: Mapping[str, Any], *, client: Any,
    case_limit: int = 6, token_budget: int = 5000,
    light_tokens: int = 512, reasoned_tokens: int = 1536,
    light_effort: str = "low", reasoned_effort: str = "xhigh",
    light_thinking_budget: int | None = None,
    reasoned_thinking_budget: int | None = None,
) -> dict[str, Any]:
    """Explicit injected text-model QC, capped by case and total token budget.

    The caller owns service admission and client lifecycle. Results are advisory
    and MUST pass source-unit revision gates before affecting output.
    """
    if request.get("schema") != RESULT_SCHEMA:
        raise ValueError("Invalid semantic review request")
    if (not isinstance(case_limit, int) or isinstance(case_limit, bool) or not 1 <= case_limit <= 50
            or not isinstance(token_budget, int) or isinstance(token_budget, bool) or token_budget < 1
            or not all(isinstance(n, int) and not isinstance(n, bool) and 256 <= n <= token_budget
                       for n in (light_tokens, reasoned_tokens))
            or not all(isinstance(e, str) and e for e in (light_effort, reasoned_effort))):
        raise ValueError("Invalid model review budgets")
    # Defaults adapt to legacy callers that override just max_tokens. An
    # explicitly supplied impossible cap NEVER gets silently shrunk.
    if light_thinking_budget is None:
        light_thinking_budget = min(128, light_tokens - 128)
    if reasoned_thinking_budget is None:
        reasoned_thinking_budget = min(512, reasoned_tokens - 128)
    if (any(not isinstance(n, int) or isinstance(n, bool) or n < 0
            for n in (light_thinking_budget, reasoned_thinking_budget))
            or light_thinking_budget > light_tokens - 128
            or reasoned_thinking_budget > reasoned_tokens - 128
            or (light_thinking_budget == 0 and light_effort != "none")
            or (reasoned_thinking_budget == 0 and reasoned_effort != "none")):
        raise ValueError("Invalid model review budgets: reserve at least 128 output tokens")
    outcomes: list[dict[str, Any]] = []
    remaining = token_budget
    for case in request.get("cases", [])[:case_limit]:
        reasoned = case.get("suggested_review_tier") == "reasoned_text"
        per_case = reasoned_tokens if reasoned else light_tokens
        effort = reasoned_effort if reasoned else light_effort
        thought_cap = reasoned_thinking_budget if reasoned else light_thinking_budget
        if remaining < per_case:
            outcomes.append({"case_id": case["case_id"], "status": "deferred_token_budget"})
            continue
        remaining -= per_case
        response = client.complete(
            messages=build_semantic_review_messages(request, case),
            max_tokens=per_case, temperature=0.0,
            response_format={"type": "json_object"},
            reasoning_effort=effort,
            thinking_budget=thought_cap,
            strict_reasoning_effort=True,
        )
        choice = (response.get("choices") or [{}])[0]
        raw = str((choice.get("message") or {}).get("content") or "")
        usage = response.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        telemetry = {
            "requested_reasoning_effort": effort,
            "thinking_budget": thought_cap,
            "max_completion_tokens": per_case,
            "reserved_answer_headroom_tokens": per_case - thought_cap,
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "request_diagnostics": response.get("_client_request_diagnostics"),
        }
        try:
            if choice.get("finish_reason") == "length" and raw.strip():
                raise ValueError("Review response reached output limit; completion not certified")
            if not raw.strip():
                if (choice.get("finish_reason") == "length" and
                        isinstance(details.get("reasoning_tokens"), int) and
                        details["reasoning_tokens"] >= int(usage.get("completion_tokens") or 0)):
                    raise ValueError("Completion budget exhausted by reasoning before final JSON")
                raise ValueError("Model returned no final JSON content")

            decoded = json.loads(raw)
            assessed = validate_semantic_review(request, decoded)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            outcomes.append({"case_id": case["case_id"], "status": "unaccepted_model_response",
                             "error": str(exc), "telemetry": telemetry,
                             "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest()})
        else:
            outcomes.append({"case_id": case["case_id"], "status": "advisory_only",
                             "assessment": assessed, "telemetry": telemetry,
                             "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest()})
    return {"schema": "nemovox-tiered-semantic-review-batch/v1",
            "request_sha256": request["request_sha256"], "model_calls": sum(
                o["status"] != "deferred_token_budget" for o in outcomes),
            "tokens_reserved": token_budget - remaining,
            "outcomes": outcomes, "source_or_subtitles_modified": False}



def run_configured_tiered_text_reviews(
    request: Mapping[str, Any], *, client: Any, policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the exact versioned review profile that produced the cases."""
    if request.get("policy_sha256") != _digest(policy):
        raise ValueError("Tiered review policy differs from the pinned candidate plan")
    config = policy.get("model_review") or {}
    allowed = {
        "case_limit", "token_budget", "light_tokens", "reasoned_tokens",
        "light_effort", "reasoned_effort", "light_thinking_budget",
        "reasoned_thinking_budget",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Invalid model_review execution configuration")
    batch = run_tiered_text_reviews(request, client=client, **config)
    batch["model_review_execution_sha256"] = _digest(config)
    return batch
