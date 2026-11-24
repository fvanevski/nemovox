"""Source-aware post-editing of dedicated machine translations.

Only the target text may change. Original MT, source text, speaker and timing
remain available for inspection and rerendering.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nvidia_transcribe.resource_lifecycle import ResourceLifecycleError
from nvidia_transcribe.refinement import GlossaryConfig, OpenAIChatTextClient

VALID_REVIEW_PROFILES = ("off", "conservative", "balanced", "expressive")
VALID_REVIEW_STYLES = ("natural", "literal", "expressive")
_PROMPT_VERSION = "mt_postedit_v2_polarity_guard"
_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "machine_translation_postedit", "strict": True,
        "schema": {"type": "object", "properties": {
            "reviews": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "utterance_id": {"type": "string"},
                    "changed": {"type": "boolean"},
                    "corrected_text": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["utterance_id", "changed", "corrected_text", "reason"],
                "additionalProperties": False,
            }},
        }, "required": ["reviews"], "additionalProperties": False},
    },
}
_PROFILE = {
    "conservative": "Fix only unambiguous mistranslations, misheard proper names, or clear errors. Keep the original otherwise.",
    "balanced": "Repair well-supported semantic/contextual errors and awkward idioms while preserving facts, intent and register.",
    "expressive": "Repair errors and improve dialogue naturalness without adding events, exaggeration or changing the original intent.",
}
_STYLE = {
    "natural": "Use concise, idiomatic subtitle dialogue.",
    "literal": "Stay close to the source meaning and phrasing.",
    "expressive": "Preserve the speaker's tone and emotion without inventing information.",
}


@dataclass(frozen=True)
class TranslationReviewConfig:
    profile: str = "off"
    style: str = "natural"
    target_language: str = "en"
    context_utterances_before: int = 5
    context_utterances_after: int = 4
    batch_size: int = 6
    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "chat"
    resource_fingerprint: str = ""
    api_key: str | None = None
    timeout_s: float = 180.0
    max_tokens: int = 4096
    use_json_schema: bool = True
    temperature: float = 0.0
    reasoning_effort: str | None = "low"


def _uid(u: dict[str, Any], i: int) -> str:
    return str(u.get("utterance_id") or u.get("id") or f"utt-{i:04d}")


def _entry(u: dict[str, Any], i: int) -> dict[str, Any]:
    return {
        "utterance_id": _uid(u, i),
        "speaker": u.get("speaker_id") or u.get("global_speaker_id"),
        "start": u.get("source_start"), "end": u.get("source_end"),
        # Give the reviewer the original ASR text, never a rewritten source
        # language hypothesis. The English machine translation is the only
        # text that this stage is permitted to change.
        "source_text": str(u.get("source_text") or ""),
        "machine_translation": str(u.get("machine_translated_text") or ""),
    }


_FRENCH_NEGATIVE = re.compile(
    r"(?i)(?:\b(?:pas|jamais|rien|personne|aucun|aucune|sans)\b|(?:^|\W)n['’]|\bne\b(?:\W+\w+){0,5}\W+plus\b)"
)
_ENGLISH_NEGATIVE = re.compile(
    r"(?i)(?:\b(?:not|never|no|none|nobody|nothing|neither|nor|without|cannot)\b|n['’]t\b)"
)


def _unsupported_polarity_inversion(
    source: str, original: str, candidate: str, language: str | None,
) -> bool:
    """Only veto loss of lexical negation corroborated by BOTH French and Riva.

    This is deliberately conservative: changes retaining lexical negatives
    (e.g. 'not ... or' -> 'neither ... nor') remain allowed, as do
    corrections with no jointly corroborated negative evidence.
    """
    return bool(
        language and language.lower().startswith("fr")
        and _FRENCH_NEGATIVE.search(source)
        and _ENGLISH_NEGATIVE.search(original)
        and not _ENGLISH_NEGATIVE.search(candidate)
    )


def review_translated_utterances(
    utterances: list[dict[str, Any]], *,
    source_language: str | None,
    config: TranslationReviewConfig | None = None,
    glossary: GlossaryConfig | None = None,
    text_client: Any = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    target_utterance_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Review primary translations with contextual evidence, never retranslate source."""
    cfg = config or TranslationReviewConfig()
    if cfg.profile not in VALID_REVIEW_PROFILES:
        raise ValueError(f"Unsupported translation-review profile {cfg.profile!r}")
    if cfg.style not in VALID_REVIEW_STYLES:
        raise ValueError(f"Unsupported translation-review style {cfg.style!r}")
    out = [{**u} for u in utterances]
    all_ids = [_uid(u, i) for i, u in enumerate(out)]
    if target_utterance_ids is not None and (
            not isinstance(target_utterance_ids, set) or not target_utterance_ids
            or len(set(all_ids)) != len(all_ids) or not target_utterance_ids <= set(all_ids)):
        raise ValueError("Selective translation review requires unique existing target IDs")
    selected = set(all_ids) if target_utterance_ids is None else target_utterance_ids
    for i, u in enumerate(out):
        if all_ids[i] in selected or "machine_translated_text" not in u:
            u["machine_translated_text"] = str(u.get("translated_text") or "")
    if cfg.profile == "off":
        for i, u in enumerate(out):
            if all_ids[i] in selected:
                u["translation_review_status"] = "off"
        return [u if all_ids[i] in selected else {**utterances[i]} for i, u in enumerate(out)], {
            "status": "disabled", "profile": "off", "reviewed": 0, "changed": 0}
    gl = glossary or GlossaryConfig()
    signature = {
        "version": _PROMPT_VERSION, "profile": cfg.profile, "style": cfg.style,
        "source_language": source_language, "target_language": cfg.target_language,
        "model": cfg.model, "endpoint": cfg.base_url, "resource": cfg.resource_fingerprint,
        "glossary": gl.digest(), "before": cfg.context_utterances_before,
        "after": cfg.context_utterances_after, "batch": cfg.batch_size,
        "temperature": cfg.temperature, "schema": cfg.use_json_schema,
        "reasoning": cfg.reasoning_effort,
    }
    ckpt = Path(checkpoint_path) if checkpoint_path else None
    cached: dict[str, dict[str, Any]] = {}
    if resume and ckpt and ckpt.is_file():
        try:
            document = json.loads(ckpt.read_text(encoding="utf-8"))
            if document.get("schema") == "translation_review/v1" and isinstance(document.get("items"), dict):
                cached = document["items"]
        except (OSError, ValueError, TypeError):
            cached = {}
    client = text_client or OpenAIChatTextClient(
        base_url=cfg.base_url, model=cfg.model, api_key=cfg.api_key,
        timeout_s=cfg.timeout_s, reasoning_effort=cfg.reasoning_effort,
    )
    errors: list[str] = []
    blocked_reason: str | None = None
    calls = hits = changed_count = reviewed_count = rejected_polarity = 0
    started = time.perf_counter()
    n = len(out)
    batch = max(1, int(cfg.batch_size))
    # Only consecutive selected rows become targets. Other rows remain
    # read-only context and are restored exactly in the returned result.
    ranges = []
    pos = 0
    while pos < n:
        if all_ids[pos] not in selected:
            pos += 1
            continue
        end = pos + 1
        while end < n and end - pos < batch and all_ids[end] in selected:
            end += 1
        ranges.append((pos, end))
        pos = end
    for start, end in ranges:
        if blocked_reason is not None:
            for i in range(start, end):
                out[i]["translation_review_status"] = "review_error_preserved_machine_translation"
            continue
        before = [_entry(out[i], i) for i in range(max(0, start - cfg.context_utterances_before), start)]
        targets = [_entry(out[i], i) for i in range(start, end)]
        after = [_entry(out[i], i) for i in range(end, min(n, end + cfg.context_utterances_after))]
        key = hashlib.sha256(json.dumps(
            {"config": signature, "before": before, "targets": targets, "after": after},
            ensure_ascii=False, sort_keys=True,
        ).encode("utf-8")).hexdigest()
        ids = [_uid(out[i], i) for i in range(start, end)]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate utterance IDs in translation post-edit batch")
        entries: dict[str, dict[str, Any]] | None = None
        if key in cached and isinstance(cached[key], dict):
            hit = cached[key]
            if set(hit) == set(ids) and all(isinstance(x, dict) for x in hit.values()):
                entries = hit
                hits += len(ids)
        if entries is None:
            payload = {
                "glossary_characters": list(gl.characters), "glossary_terms": list(gl.terms),
                "series_notes": gl.notes, "context_before": before,
                "target_utterances": targets, "context_after": after,
            }
            messages = [
                {"role": "system", "content": (
                    f"Review existing machine translations from {source_language or 'source'} "
                    f"into {cfg.target_language}. " + _PROFILE[cfg.profile] + " " + _STYLE[cfg.style]
                    + " Only edit TARGET translated English cues; neighboring entries and the original "
                    + "ASR source-language transcript are read-only evidence, never material to rewrite. "
                    + "Evaluate each individual target cue against the untranslated source and its adjacent "
                    + "original machine translations. This is NOT a fresh translation request or a blanket "
                    + "localization rewrite. Look for concrete mistranslations, missing meaning, incorrect names, "
                    + "misresolved references and context-dependent idioms. Preserve original meaning, names, "
                    + "numbers, speaker intent and narrative facts. Stylistic preferences alone do not justify edits. "
                    + "Return exactly one review for every target utterance_id. "
                    + "When the existing translation is adequate or evidence is insufficient, set changed=false "
                    + "and return the machine_translation exactly. For a change, give a short reason tied to "
                    + "the source or surrounding raw translations; never invent dialogue or source words."
                )},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            try:
                calls += 1
                response = client.complete(
                    messages=messages, max_tokens=cfg.max_tokens, temperature=cfg.temperature,
                    response_format=_SCHEMA if cfg.use_json_schema else None,
                )
                raw = str(((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
                first = raw.find("{")
                if first < 0:
                    raise ValueError("Missing JSON response")
                parsed, _ = json.JSONDecoder().raw_decode(raw[first:])
                reviews = parsed.get("reviews") if isinstance(parsed, dict) else None
                if not isinstance(reviews, list):
                    raise ValueError("Missing reviews array")
                entries = {}
                for item in reviews:
                    if not isinstance(item, dict) or not isinstance(item.get("utterance_id"), str):
                        raise ValueError("Invalid review item")
                    uid = item["utterance_id"]
                    if uid not in ids or uid in entries:
                        raise ValueError("Unexpected or duplicate utterance ID")
                    entries[uid] = item
                if set(entries) != set(ids):
                    raise ValueError("Incomplete review IDs")
            except Exception as exc:
                if isinstance(exc, ResourceLifecycleError):
                    # Do not repeatedly attempt a blocked model acquisition on
                    # subsequent batches. Still preserve all original Riva MT.
                    blocked_reason = exc.status
                errors.append(f"batch[{start}:{end}]:{type(exc).__name__}:{exc}")
                for i in range(start, end):
                    out[i]["translation_review_status"] = "review_error_preserved_machine_translation"
                continue
        valid_for_cache = True
        for i in range(start, end):
            uid = _uid(out[i], i)
            item = entries[uid]
            original = out[i]["machine_translated_text"]
            candidate = item.get("corrected_text")
            changed = item.get("changed")
            if (not isinstance(candidate, str) or not isinstance(changed, bool)
                    or not isinstance(item.get("reason"), str)
                    or (not candidate.strip() and bool(original.strip()))
                    or changed != (candidate.strip() != original.strip())):
                errors.append(f"invalid_review:{uid}")
                out[i]["translation_review_status"] = "review_invalid_preserved_machine_translation"
                valid_for_cache = False
                continue
            out[i]["translation_review_reason"] = item["reason"]
            out[i]["translation_reviewed_text"] = candidate.strip()
            if changed and _unsupported_polarity_inversion(
                str(out[i].get("source_text") or ""),
                original, candidate, source_language,
            ):
                out[i]["translation_review_status"] = "review_rejected_polarity_preserved_machine_translation"
                out[i]["translation_review_rejection_reason"] = "source_and_primary_mt_negative_but_review_affirmative"
                rejected_polarity += 1
                reviewed_count += 1
                continue
            out[i]["translation_review_status"] = "corrected" if changed else "unchanged"
            if changed:
                out[i]["translated_text"] = candidate.strip()
                changed_count += 1
            reviewed_count += 1
        if ckpt and valid_for_cache and key not in cached:
            cached[key] = entries
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            tmp = ckpt.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                {"schema": "translation_review/v1", "items": cached},
                ensure_ascii=False, indent=2,
            ), encoding="utf-8")
            tmp.replace(ckpt)
    if target_utterance_ids is not None:
        out = [u if all_ids[i] in selected else {**utterances[i]} for i, u in enumerate(out)]
    return out, {
        "status": "completed_with_errors" if errors else "completed",
        "profile": cfg.profile, "style": cfg.style,
        "resource_fingerprint": cfg.resource_fingerprint,
        "reviewed": reviewed_count, "changed": changed_count,
        "rejected_polarity": rejected_polarity,
        "cache_hits": hits, "api_calls": calls, "errors": errors,
        "resource_blocked_reason": blocked_reason,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
