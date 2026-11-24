"""Experimental localized transcript refinement and structured patch proposals (Phase C).

Strictly non-authoritative: emits structured patch proposals referencing
immutable ``source_unit_ids`` with ``status="proposed_only"``. Never mutates
canonical recording transcripts automatically.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from nvidia_transcribe.metrics import compute_cer, compute_wer, normalize_latin_words, normalize_zh_chars


@dataclass(frozen=True)
class PatchProposal:
    proposal_id: str
    case_id: str
    language: str
    method: str  # "C1_contextual_audio" | "C2_text_reasoning_only" | "C2_audio_verified"
    source_unit_ids: list[str]
    utterance_id: str | None
    source_start_s: float | None
    source_end_s: float | None
    before_text: str
    proposed_text: str
    changed: bool
    audio_verified: bool
    reasoning_content: str | None
    rationale: str
    status: str = "proposed_only"  # Experimental safety boundary: NEVER auto-applied.

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _levenshtein(a: list[str], b: list[str]) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def select_suspicious_utterances(
    utterances: list[dict[str, Any]],
    *,
    glossary_terms: list[str] | None = None,
    max_candidates: int = 20,
) -> list[dict[str, Any]]:
    """Bounded localization: flag only genuinely suspicious or uncertain utterances."""
    glossary_low = [t.strip().lower() for t in (glossary_terms or []) if t.strip()]
    flagged: list[dict[str, Any]] = []

    for utt in utterances:
        text = str(utt.get("source_text") or "").strip()
        if not text:
            continue
        reasons: list[str] = []
        prov = utt.get("timestamp_provenance") or {}
        timing_status = prov.get("timing_status") or "unknown"
        if timing_status in {"partial", "failed", "unavailable"}:
            reasons.append(f"timing_{timing_status}")

        has_cjk = any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text)
        has_latin = any("a" <= ch.lower() <= "z" for ch in text)
        if has_cjk and has_latin:
            reasons.append("mixed_cjk_latin")

        words = [w for w in re.split(r"\s+", text) if w]
        if len(text) <= 4 or (not has_cjk and len(words) <= 2):
            reasons.append("very_short_fragment")

        if glossary_low:
            tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']{3,}", text.lower())
            for tok in tokens:
                for g in glossary_low:
                    if tok != g and abs(len(tok) - len(g)) <= 2 and _levenshtein(list(tok), list(g)) <= 2:
                        reasons.append(f"glossary_near_miss:{tok}->{g}")
                        break

        if reasons:
            flagged.append(
                {
                    "utterance_id": utt.get("utterance_id") or utt.get("id"),
                    "source_start": utt.get("source_start"),
                    "source_end": utt.get("source_end"),
                    "source_text": text,
                    "reasons": reasons,
                }
            )
            if len(flagged) >= max_candidates:
                break
    return flagged


def build_c2_text_probe_messages(
    *,
    target_text: str,
    language: str,
    prev_text: str = "",
    next_text: str = "",
    glossary_terms: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build deterministic text-only reasoning prompt for suspect-span detection (C2a)."""
    lang_label = "Mandarin Chinese" if language == "zh" else ("French" if language == "fr" else language)
    terms_str = ", ".join(glossary_terms) if glossary_terms else "(none)"
    sys_msg = (
        "You are an ASR transcript auditor. Inspect ONLY the target segment using neighboring context "
        "and domain terms. If the target looks completely accurate, respond with changed=false and keep "
        "proposed_text identical to target_text. Never translate into another language. "
        "Return strict JSON with keys: changed (bool), proposed_text (string), confidence (0..1 float), rationale (string)."
    )
    user_msg = (
        f"Language: {lang_label}\n"
        f"Known glossary/proper nouns: {terms_str}\n"
        f"Previous segment: {prev_text or '(none)'}\n"
        f"Target segment: {target_text}\n"
        f"Next segment: {next_text or '(none)'}\n"
        "Respond with JSON only."
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def parse_c2_probe_response(raw_response: dict[str, Any], *, fallback_text: str) -> dict[str, Any]:
    """Parse OpenAI-compatible vLLM chat completion including Nemotron reasoning_content safely."""
    choices = raw_response.get("choices") or []
    if not choices:
        return {
            "status": "empty_choices",
            "reasoning_content": None,
            "changed": False,
            "proposed_text": fallback_text,
            "rationale": "empty_choices",
        }
    msg = choices[0].get("message") or {}
    content = str(msg.get("content") or "").strip()
    reasoning = msg.get("reasoning_content")
    if reasoning is None:
        reasoning = msg.get("reasoning")
    reasoning_str = str(reasoning).strip() if reasoning else None

    # Handle fallback <think>...</think> inline tags if reasoning-parser didn't split them
    if reasoning_str is None and "<think>" in content and "</think>" in content:
        m = re.search(r"<think>(.*?)</think>(.*)", content, flags=re.DOTALL)
        if m:
            reasoning_str = m.group(1).strip()
            content = m.group(2).strip()

    # Extract first valid JSON object using raw_decode so repeated trailing blocks don't fail parsing
    first_brace = content.find("{")
    if first_brace < 0:
        return {
            "status": "parse_error",
            "reasoning_content": reasoning_str,
            "raw_content": content,
            "degenerate_repetition": False,
            "changed": False,
            "proposed_text": fallback_text,
            "rationale": "no_json_object_in_response",
        }
    try:
        decoder = json.JSONDecoder()
        parsed, end_idx = decoder.raw_decode(content, first_brace)
    except json.JSONDecodeError as exc:
        return {
            "status": "parse_error",
            "reasoning_content": reasoning_str,
            "raw_content": content,
            "degenerate_repetition": False,
            "changed": False,
            "proposed_text": fallback_text,
            "rationale": f"json_decode_error: {exc}",
        }

    trailing = content[end_idx:].strip()
    degenerate_repetition = bool(trailing) and trailing.startswith("{")
    prop_text = str(parsed.get("proposed_text") or fallback_text).strip()
    text_differs = prop_text != fallback_text.strip()
    changed = bool(parsed.get("changed")) and text_differs
    flag_contradiction = (not bool(parsed.get("changed"))) and text_differs
    return {
        "status": "ok",
        "reasoning_content": reasoning_str,
        "degenerate_repetition": degenerate_repetition,
        "flag_contradiction": flag_contradiction,
        "changed": changed,
        "proposed_text": prop_text if changed else fallback_text,
        "confidence": parsed.get("confidence"),
        "rationale": str(parsed.get("rationale") or ""),
    }


def evaluate_refinement_against_reference(
    *,
    before_text: str,
    proposed_text: str,
    reference_text: str | None,
    language: str,
) -> dict[str, Any]:
    """Compute exact WER/CER deltas, fixes, regressions, and false-correction classification."""
    changed = proposed_text.strip() != before_text.strip()
    if reference_text is None:
        return {
            "reference_verified": False,
            "changed": changed,
            "before_metric": None,
            "after_metric": None,
            "fixed_to_exact_match": None,
            "false_correction": None,
            "outcome": "unverified_changed" if changed else "unverified_unchanged",
        }

    if language == "zh":
        m_before = compute_cer(reference_text, before_text)
        m_after = compute_cer(reference_text, proposed_text)
        b_exact = normalize_zh_chars(before_text) == normalize_zh_chars(reference_text)
        a_exact = normalize_zh_chars(proposed_text) == normalize_zh_chars(reference_text)
        metric_name = "cer"
        val_before = m_before["cer"]
        val_after = m_after["cer"]
        edits_before = m_before["edits"]
        edits_after = m_after["edits"]
    else:
        m_before = compute_wer(reference_text, before_text)
        m_after = compute_wer(reference_text, proposed_text)
        b_exact = normalize_latin_words(before_text) == normalize_latin_words(reference_text)
        a_exact = normalize_latin_words(proposed_text) == normalize_latin_words(reference_text)
        metric_name = "wer"
        val_before = m_before["wer"]
        val_after = m_after["wer"]
        edits_before = m_before["edits"]
        edits_after = m_after["edits"]

    false_correction = bool(b_exact and changed and not a_exact)
    if not changed:
        outcome = "unchanged_correct" if b_exact else "missed_error"
    elif false_correction:
        outcome = "false_correction"
    elif edits_after < edits_before:
        outcome = "improved"
    elif edits_after > edits_before:
        outcome = "degraded"
    else:
        outcome = "neutral_edit"

    return {
        "reference_verified": True,
        "metric_name": metric_name,
        "changed": changed,
        "before_metric": val_before,
        "after_metric": val_after,
        "edits_before": edits_before,
        "edits_after": edits_after,
        "net_edits_reduced": edits_before - edits_after,
        "initially_exact": b_exact,
        "after_exact": a_exact,
        "false_correction": false_correction,
        "outcome": outcome,
    }


# --------------------------------------------------------------------------- #
# Production Batched SwiftU Contextual Refinement (Work Package B)
# --------------------------------------------------------------------------- #

VALID_REFINEMENT_PROFILES = ("off", "conservative", "balanced", "expressive")

_PROFILE_INSTRUCTIONS = {
    "conservative": (
        "Only fix clear spelling mistakes, punctuation, obvious ASR recognition artifacts, "
        "and recurring glossary names when strongly supported by surrounding context. Leave wording unchanged when unsure."
    ),
    "balanced": (
        "Fix contextually clear mistranscriptions, homophones (including CJK homophone errors), character names, "
        "glossary terms, repeated ASR stutters, and broken idioms while preserving the original meaning, register, and source language."
    ),
    "expressive": (
        "Repair corrupted, garbled, or awkward ASR dialogue into natural, coherent phrasing in the SAME source language "
        "using full scene context and glossary, preserving speaker meaning and narrative events without inventing new plot turns."
    ),
}


@dataclass(frozen=True)
class GlossaryConfig:
    characters: tuple[str, ...] = ()
    terms: tuple[str, ...] = ()
    notes: str | None = None

    def all_terms(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in (*self.characters, *self.terms):
            t = str(item).strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        return out

    def digest(self) -> str:
        import hashlib

        payload = json.dumps(
            {"characters": list(self.characters), "terms": list(self.terms), "notes": self.notes or ""},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def load_glossary(path_or_terms: str | Path | Sequence[str] | None) -> GlossaryConfig:
    """Load character/term glossary from YAML, JSON, or a comma-separated/list of terms."""
    if path_or_terms is None:
        return GlossaryConfig()
    if isinstance(path_or_terms, (list, tuple)):
        return GlossaryConfig(terms=tuple(str(x).strip() for x in path_or_terms if str(x).strip()))
    p = Path(str(path_or_terms))
    if p.is_file():
        raw_text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(raw_text) or {}
        else:
            data = json.loads(raw_text)
        if isinstance(data, list):
            return GlossaryConfig(terms=tuple(str(x).strip() for x in data if str(x).strip()))
        if not isinstance(data, dict):
            raise ValueError(f"Glossary file {p.name!r} must contain a mapping or list")
        chars_raw = data.get("characters") or []
        terms_raw = data.get("terms") or []
        if not isinstance(chars_raw, list) or not isinstance(terms_raw, list):
            raise ValueError(f"Glossary file {p.name!r} character and term fields must be lists")
        chars = tuple(str(x).strip() for x in chars_raw if str(x).strip())
        terms = tuple(str(x).strip() for x in terms_raw if str(x).strip())
        notes = str(data.get("notes")).strip() if data.get("notes") else None
        return GlossaryConfig(characters=chars, terms=terms, notes=notes)
    terms = tuple(x.strip() for x in str(path_or_terms).split(",") if x.strip())
    return GlossaryConfig(terms=terms)


@dataclass(frozen=True)
class RefinementConfig:
    backend: str = "openai_chat"  # "off" | "openai_chat" | legacy "swiftu"
    profile: str = "balanced"  # "off" | "conservative" | "balanced" | "expressive"
    context_utterances_before: int = 6
    context_utterances_after: int = 4
    batch_size: int = 6
    audio_verification: bool = False
    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "chat"
    resource_fingerprint: str = ""  # effective identity, no credentials
    api_key: str | None = None
    timeout_s: float = 180.0
    max_tokens: int = 8192
    temperature: float = 0.0
    use_json_schema: bool = True
    reasoning_effort: str | None = "low"


_REFINEMENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "utterance_refinement_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "corrections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "utterance_id": {"type": "string"},
                            "original_text": {"type": "string"},
                            "refined_text": {"type": "string"},
                            "changed": {"type": "boolean"},
                            "edit_type": {"type": "string"},
                            "rationale": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["utterance_id", "original_text", "refined_text", "changed", "edit_type", "rationale", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["corrections"],
            "additionalProperties": False,
        },
    },
}


def _language_display_name(code: str | None) -> str:
    low = (code or "").strip().lower()
    mapping = {"fr": "French", "zh": "Mandarin Chinese", "en": "English", "es": "Spanish", "de": "German", "ja": "Japanese", "ko": "Korean"}
    return mapping.get(low, low or "source language")


def build_batched_refinement_messages(
    *,
    target_utterances: list[dict[str, Any]],
    context_before: list[dict[str, Any]],
    context_after: list[dict[str, Any]],
    source_language: str | None,
    profile: str,
    glossary: GlossaryConfig,
) -> list[dict[str, str]]:
    """Build multi-utterance contextual prompt returning utterance-ID-anchored JSON."""
    lang_name = _language_display_name(source_language)
    prof_rule = _PROFILE_INSTRUCTIONS.get(profile, _PROFILE_INSTRUCTIONS["balanced"])
    chars_s = ", ".join(glossary.characters) if glossary.characters else "(none)"
    terms_s = ", ".join(glossary.terms) if glossary.terms else "(none)"

    def _fmt(utts: list[dict[str, Any]], use_refined: bool = False) -> str:
        if not utts:
            return "(none)"
        lines = []
        for u in utts:
            uid = u.get("utterance_id") or u.get("id")
            spk = u.get("speaker_id") or u.get("global_speaker_id") or "SPK"
            txt = (u.get("refined_text") if (use_refined and u.get("refined_text")) else None) or u.get("source_text") or ""
            lines.append(f"- [{uid}] ({spk}): {txt}")
        return "\n".join(lines)

    sys_prompt = (
        f"You are refining an automatic speech recognition (ASR) transcript in {lang_name}. "
        f"Profile '{profile}': {prof_rule} "
        "NEVER translate into another language. Preserve speaker intent and keep every target utterance_id separate. "
        "Do NOT modify or include READ-ONLY context utterances in output. "
        "Return strict JSON matching {\"corrections\": [{\"utterance_id\": str, \"original_text\": str, \"refined_text\": str, \"changed\": bool, \"edit_type\": str, \"rationale\": str, \"confidence\": float}]}."
    )
    user_prompt = (
        f"Source language: {lang_name} ({source_language or 'unknown'})\n"
        f"Glossary characters: {chars_s}\n"
        f"Glossary terms: {terms_s}\n"
        + (f"Series notes: {glossary.notes}\n" if glossary.notes else "")
        + f"\nREAD-ONLY CONTEXT BEFORE:\n{_fmt(context_before, use_refined=True)}\n"
        + f"\nTARGET UTTERANCES (RETURN EXACTLY THESE {len(target_utterances)} IDs):\n{_fmt(target_utterances, use_refined=False)}\n"
        + f"\nREAD-ONLY CONTEXT AFTER:\n{_fmt(context_after, use_refined=False)}\n"
        + "\nReturn JSON only."
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]


def parse_batched_refinement_response(raw_response: dict[str, Any]) -> dict[str, Any]:
    """Parse batched OpenAI-compatible chat completion JSON cleanly."""
    choices = raw_response.get("choices") or []
    if not choices:
        return {"status": "empty_choices", "corrections": []}
    msg = choices[0].get("message") or {}
    content = str(msg.get("content") or "").strip()
    if "<think>" in content and "</think>" in content:
        m = re.search(r"<think>.*?</think>(.*)", content, flags=re.DOTALL)
        if m:
            content = m.group(1).strip()

    first_brace = content.find("{")
    first_bracket = content.find("[")
    start_idx = first_brace if first_brace >= 0 and (first_bracket < 0 or first_brace < first_bracket) else first_bracket
    if start_idx < 0:
        return {"status": "parse_error", "corrections": [], "raw_content": content[:400]}
    try:
        parsed, _ = json.JSONDecoder().raw_decode(content, start_idx)
    except json.JSONDecodeError as exc:
        return {"status": "parse_error", "corrections": [], "error": str(exc), "raw_content": content[:400]}

    items = parsed if isinstance(parsed, list) else (parsed.get("corrections") or []) if isinstance(parsed, dict) else []
    if not isinstance(items, list):
        return {"status": "parse_error", "corrections": []}
    return {"status": "ok", "corrections": [it for it in items if isinstance(it, dict)]}


_ENGLISH_LEAK_RE = re.compile(r"\b(?:the|this|that|with|from|you|are|they|were|have|not|what|there)\b", re.IGNORECASE)


def screen_refinement_edit(
    *,
    original_text: str,
    refined_text: str,
    profile: str,
    source_language: str | None = None,
) -> tuple[bool, str]:
    """Lightweight sanity screening for batched contextual edits without blocking expressive fixes."""
    orig = (original_text or "").strip()
    ref = (refined_text or "").strip()
    if not orig:
        return True, "empty_original"
    if not ref:
        return False, "empty_replacement"
    if orig == ref:
        return True, "unchanged"

    # Reject pathological repetition loops (e.g., same multi-word phrase 3+ times).
    words = ref.lower().split()
    if len(words) >= 12:
        for span_len in (3, 4):
            grams = [" ".join(words[i : i + span_len]) for i in range(len(words) - span_len + 1)]
            counts: dict[str, int] = {}
            for g in grams:
                counts[g] = counts.get(g, 0) + 1
                if counts[g] >= 3 and words.count(words[0]) >= 3:
                    return False, "repetition_loop"

    ratio = len(ref) / max(1, len(orig))
    max_ratio = 3.5 if profile == "conservative" else (5.0 if profile == "balanced" else 7.5)
    min_ratio = 0.25 if profile == "conservative" else 0.12
    if len(orig) >= 8 and ratio > max_ratio:
        return False, "excessive_expansion"
    if len(orig) >= 16 and ratio < min_ratio:
        return False, "excessive_contraction"

    # Guard against accidental translation instead of same-language refinement
    low_lang = (source_language or "").strip().lower()
    if low_lang == "zh":
        orig_cjk = sum(1 for c in orig if 0x4E00 <= ord(c) <= 0x9FFF)
        ref_cjk = sum(1 for c in ref if 0x4E00 <= ord(c) <= 0x9FFF)
        if orig_cjk >= 4 and ref_cjk == 0:
            return False, "wrong_language_lost_cjk"
    elif low_lang == "fr":
        orig_en_hits = len(_ENGLISH_LEAK_RE.findall(orig))
        ref_en_hits = len(_ENGLISH_LEAK_RE.findall(ref))
        if orig_en_hits == 0 and ref_en_hits >= 3 and len(words) >= 4:
            return False, "wrong_language_english_leak"

    return True, "accepted"


class OpenAIChatTextClient:
    """Lightweight OpenAI-compatible text client with schema fallback."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        self._schema_rejected = False
        # A rejected effort value does not imply other levels are unsupported.
        # Keep request-level diagnostics for resource-owned QA probes.
        self._rejected_reasoning_efforts: set[str] = set()
        self._reasoning_rejected = False

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
        strict_reasoning_effort: bool = False,
    ) -> dict[str, Any]:
        import httpx

        # Standard Chat Completions does NOT define thinking_budget, and the
        # installed NInfer parser silently ignored it. Never claim a bounded
        # reasoning request using this route; use the Messages adapter.
        if thinking_budget is not None:
            raise ValueError("Chat Completions cannot enforce thinking_budget; use NInferMessagesTextClient")
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        self._reasoning_rejected = False
        if effort is not None and (strict_reasoning_effort or effort not in self._rejected_reasoning_efforts):
            payload["reasoning_effort"] = effort
        if response_format is not None and not self._schema_rejected:
            payload["response_format"] = response_format

        with httpx.Client(headers=headers, timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload)
            if resp.status_code in (400, 422) and "reasoning_effort" in payload:
                message = resp.text.lower()
                if ("reasoning" in message or "invalid prompt" in message) and not strict_reasoning_effort:
                    self._reasoning_rejected = True
                    self._rejected_reasoning_efforts.add(str(effort))
                    payload.pop("reasoning_effort", None)
                    resp = client.post(f"{self.base_url}/chat/completions", json=payload)
            if (resp.status_code in (400, 422) and "response_format" in payload
                    and not strict_reasoning_effort):
                # Legacy relaxed schema fallback; strict tiered QC never
                # retries with a relaxed schema after a rejected effort.
                self._schema_rejected = True
                payload.pop("response_format", None)
                resp = client.post(f"{self.base_url}/chat/completions", json=payload)
            resp.raise_for_status()
            result = resp.json()
            if not isinstance(result, dict):
                raise ValueError("Expected an OpenAI-compatible completion object")
            result["_client_request_diagnostics"] = {
                "reasoning_effort_requested": effort,
                "reasoning_effort_sent": payload.get("reasoning_effort"),
                "reasoning_effort_fallback": self._reasoning_rejected,
                "thinking_budget_sent": payload.get("thinking_budget"),
                "response_format_sent": "response_format" in payload,
            }
            return result


def _copy_utt_unrefined(u: dict[str, Any], profile: str) -> dict[str, Any]:
    raw_txt = str(u.get("source_text") or "")
    return {
        **u,
        "raw_source_text": u.get("raw_source_text") or raw_txt,
        "refined_text": raw_txt,
        "refinement_profile": profile,
        "refinement_applied": False,
        "refinement_edit_type": "none",
        "refinement_rationale": "",
        "refinement_confidence": 1.0,
        "refinement_screen_status": "skipped_profile_off" if profile == "off" else "unchanged",
    }


def _batch_cache_key(
    *,
    config_sig: dict[str, Any],
    targets: list[dict[str, Any]],
    ctx_before: list[dict[str, Any]],
    ctx_after: list[dict[str, Any]],
) -> str:
    import hashlib

    payload = {
        "cfg": config_sig,
        "targets": [
            (str(u.get("utterance_id") or u.get("id") or ""), str(u.get("speaker_id") or u.get("global_speaker_id") or ""), str(u.get("raw_source_text") or u.get("source_text") or ""))
            for u in targets
        ],
        "ctx_before": [
            (str(u.get("utterance_id") or u.get("id") or ""), str(u.get("speaker_id") or u.get("global_speaker_id") or ""), str(u.get("raw_source_text") or u.get("source_text") or ""))
            for u in ctx_before
        ],
        "ctx_after": [
            (str(u.get("utterance_id") or u.get("id") or ""), str(u.get("speaker_id") or u.get("global_speaker_id") or ""), str(u.get("raw_source_text") or u.get("source_text") or ""))
            for u in ctx_after
        ],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def refine_utterances_with_text_model(
    utterances: list[dict[str, Any]],
    *,
    source_language: str | None,
    config: RefinementConfig | None = None,
    glossary: GlossaryConfig | None = None,
    text_client: Any = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refine recording utterances using a compatible text-inference model.

    Preserves ``raw_source_text`` / ``source_text`` untouched, attaches ``refined_text``,
    keeps source acoustic timing intact, and uses exact input+context cache keys so
    edited source lines or context changes never reuse stale cached refinements.
    """
    import time

    cfg = config or RefinementConfig()
    if cfg.profile not in VALID_REFINEMENT_PROFILES:
        raise ValueError(f"Unsupported refinement profile {cfg.profile!r}; expected one of {VALID_REFINEMENT_PROFILES}")
    if cfg.profile == "off" or cfg.backend == "off":
        out = [_copy_utt_unrefined(u, "off") for u in utterances]
        return out, {"backend": cfg.backend, "profile": "off", "total_utterances": len(out), "changed_count": 0}

    gl = glossary or GlossaryConfig()
    gl_digest = gl.digest()
    config_sig = {
        "prompt_version": "ref_v2",
        "backend": cfg.backend,
        "model": cfg.model,
        "endpoint": cfg.base_url,
        "resource_fingerprint": cfg.resource_fingerprint,
        "profile": cfg.profile,
        "source_language": source_language or "",
        "glossary_digest": gl_digest,
        "ctx_before": cfg.context_utterances_before,
        "ctx_after": cfg.context_utterances_after,
        "batch_size": cfg.batch_size,
        "temperature": cfg.temperature,
        "use_json_schema": cfg.use_json_schema,
        "reasoning_effort": cfg.reasoning_effort,
    }
    client = text_client or OpenAIChatTextClient(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        reasoning_effort=cfg.reasoning_effort,
    )

    ckpt_file = Path(checkpoint_path) if checkpoint_path else None
    cached_by_key: dict[str, dict[str, Any]] = {}
    if resume and ckpt_file and ckpt_file.is_file():
        try:
            ckpt = json.loads(ckpt_file.read_text(encoding="utf-8"))
            if isinstance(ckpt.get("items"), dict):
                cached_by_key = dict(ckpt.get("items") or {})
        except Exception:
            cached_by_key = {}

    out_utts = [_copy_utt_unrefined(u, cfg.profile) for u in utterances]
    n = len(out_utts)
    bsize = max(1, int(cfg.batch_size))
    t0 = time.perf_counter()
    batch_errors: list[str] = []
    rejected_edits: list[dict[str, Any]] = []
    api_calls = 0
    cache_hits = 0

    def _apply_batch(batch_indices: list[int], retries_left: int = 1, cache_key: str | None = None) -> None:
        nonlocal api_calls
        if not batch_indices:
            return
        targets = [out_utts[i] for i in batch_indices]
        lo = batch_indices[0]
        hi = batch_indices[-1] + 1
        ctx_before = out_utts[max(0, lo - cfg.context_utterances_before) : lo]
        ctx_after = out_utts[hi : min(n, hi + cfg.context_utterances_after)]

        msgs = build_batched_refinement_messages(
            target_utterances=targets,
            context_before=ctx_before,
            context_after=ctx_after,
            source_language=source_language,
            profile=cfg.profile,
            glossary=gl,
        )
        try:
            api_calls += 1
            raw = client.complete(
                messages=msgs,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                response_format=_REFINEMENT_JSON_SCHEMA if cfg.use_json_schema else None,
            )
            parsed = parse_batched_refinement_response(raw)
        except Exception as exc:
            parsed = {"status": "request_error", "error": f"{type(exc).__name__}: {exc}", "corrections": []}

        if parsed.get("status") != "ok":
            if retries_left > 0 and len(batch_indices) > 1:
                mid = len(batch_indices) // 2
                _apply_batch(batch_indices[:mid], retries_left=0)
                _apply_batch(batch_indices[mid:], retries_left=0)
                return
            batch_errors.append(f"batch[{lo}:{hi}] -> {parsed.get('status')}: {parsed.get('error') or ''}")
            return

        expected_ids = {str(out_utts[i].get("utterance_id") or out_utts[i].get("id")): i for i in batch_indices}
        seen_ids: set[str] = set()
        for item in parsed["corrections"]:
            uid = str(item.get("utterance_id") or "").strip()
            if uid not in expected_ids or uid in seen_ids:
                continue
            seen_ids.add(uid)
            idx = expected_ids[uid]
            orig = str(out_utts[idx]["raw_source_text"])
            cand = str(item.get("refined_text") or orig).strip()
            changed = bool(item.get("changed")) and cand != orig.strip()
            if changed:
                ok_screen, reason = screen_refinement_edit(
                    original_text=orig, refined_text=cand, profile=cfg.profile, source_language=source_language
                )
                if not ok_screen:
                    rejected_edits.append({"utterance_id": uid, "original": orig, "candidate": cand, "reason": reason})
                    out_utts[idx]["refinement_screen_status"] = f"rejected:{reason}"
                    continue
                out_utts[idx]["refined_text"] = cand
                out_utts[idx]["refinement_applied"] = True
                out_utts[idx]["refinement_edit_type"] = str(item.get("edit_type") or "contextual")
                out_utts[idx]["refinement_rationale"] = str(item.get("rationale") or "")
                out_utts[idx]["refinement_confidence"] = float(item.get("confidence") or 0.8)
                out_utts[idx]["refinement_screen_status"] = "accepted"
            else:
                out_utts[idx]["refinement_screen_status"] = "unchanged"

            ck_entry_key = f"{uid}:{cache_key}" if cache_key else None
            if ck_entry_key:
                cached_by_key[ck_entry_key] = {
                    "cache_key": cache_key,
                    "utterance_id": uid,
                    "refined_text": out_utts[idx]["refined_text"],
                    "refinement_applied": out_utts[idx]["refinement_applied"],
                    "refinement_edit_type": out_utts[idx]["refinement_edit_type"],
                    "refinement_rationale": out_utts[idx]["refinement_rationale"],
                    "refinement_confidence": out_utts[idx]["refinement_confidence"],
                    "refinement_screen_status": out_utts[idx]["refinement_screen_status"],
                }

    for lo in range(0, n, bsize):
        hi = min(n, lo + bsize)
        batch_indices = list(range(lo, hi))
        targets = [out_utts[i] for i in batch_indices]
        ctx_before = out_utts[max(0, lo - cfg.context_utterances_before) : lo]
        ctx_after = out_utts[hi : min(n, hi + cfg.context_utterances_after)]
        bkey = _batch_cache_key(config_sig=config_sig, targets=targets, ctx_before=ctx_before, ctx_after=ctx_after)

        all_hit = True
        for i in batch_indices:
            uid = str(out_utts[i].get("utterance_id") or out_utts[i].get("id"))
            if f"{uid}:{bkey}" not in cached_by_key:
                all_hit = False
                break

        if all_hit:
            for i in batch_indices:
                uid = str(out_utts[i].get("utterance_id") or out_utts[i].get("id"))
                citem = cached_by_key[f"{uid}:{bkey}"]
                for k in (
                    "refined_text",
                    "refinement_applied",
                    "refinement_edit_type",
                    "refinement_rationale",
                    "refinement_confidence",
                    "refinement_screen_status",
                ):
                    if k in citem:
                        out_utts[i][k] = citem[k]
                cache_hits += 1
        else:
            _apply_batch(batch_indices, retries_left=1, cache_key=bkey)
            if ckpt_file is not None:
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                ckpt_file.write_text(
                    json.dumps(
                        {"schema": "refinement_cache/v2", "profile": cfg.profile, "glossary_digest": gl_digest, "items": cached_by_key},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    elapsed_s = round(time.perf_counter() - t0, 3)
    changed_count = sum(1 for u in out_utts if u["refinement_applied"])
    summary = {
        "backend": cfg.backend,
        "resource_fingerprint": cfg.resource_fingerprint,
        "profile": cfg.profile,
        "glossary_digest": gl_digest,
        "audio_verification": cfg.audio_verification,
        "total_utterances": n,
        "changed_count": changed_count,
        "rejected_count": len(rejected_edits),
        "rejected_edits": rejected_edits,
        "batch_errors": batch_errors,
        "api_calls": api_calls,
        "cache_hits": cache_hits,
        "elapsed_s": elapsed_s,
    }
    return out_utts, summary

# Legacy API name retained for existing callers.
refine_utterances_with_swiftu = refine_utterances_with_text_model
