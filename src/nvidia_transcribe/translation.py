"""Local multilingual subtitle translation backends and utterance-ID-anchored contract."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from contextlib import nullcontext
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nvidia_transcribe.refinement import GlossaryConfig, OpenAIChatTextClient
from nvidia_transcribe.resource_lifecycle import ModelLifecycle

VALID_TRANSLATION_BACKENDS = ("openai_chat", "swiftu", "riva", "none")
VALID_TRANSLATION_STYLES = ("natural", "literal", "expressive")

DEFAULT_RIVA_MODEL_PATH = Path("/workspace/models/Riva-Translate-4B-Instruct-v2")
DEFAULT_RIVA_PYTHON = Path("/workspace/nvidia-transcribe/outputs/modular-eval/.venv/bin/python")
RIVA_WORKER_SCRIPT = Path(__file__).with_name("_riva_translate_worker.py")

_STYLE_GUIDANCE = {
    "natural": (
        "Produce natural, concise English dialogue suitable for subtitles. Preserve meaning, character intent, "
        "character names, and key narrative facts. Prefer idiomatic conversational English over stiff literal "
        "word-for-word translation. Never add outside explanations."
    ),
    "literal": (
        "Translate closely and faithfully into clear, grammatical English, staying tight to the source wording and "
        "structure without inventing or omitting clauses."
    ),
    "expressive": (
        "Localize into vivid, emotionally authentic English subtitle dialogue that captures tone, banter, and "
        "colloquialisms while strictly keeping all factual events and names intact."
    ),
}

_RIVA_PAIR_MAP = {
    "fr": "fr-en",
    "zh": "zh-en",
    "zh-cn": "zh-cn-en",
    "zh-tw": "zh-tw-en",
    "es": "es-en",
    "de": "de-en",
    "it": "it-en",
    "pt": "pt-en",
    "ja": "ja-en",
    "ko": "ko-en",
    "ru": "ru-en",
    "ar": "ar-en",
    "vi": "vi-en",
    "th": "th-en",
    "id": "id-en",
    "hi": "hi-en",
    "tr": "tr-en",
    "nl": "nl-en",
    "pl": "pl-en",
}


def is_english_source_to_english(source_language: str | None, target_language: str | None) -> bool:
    """Recognize already-English speech destined for English subtitle rendering."""
    def base(value: str | None) -> str:
        return (value or "").strip().lower().replace("_", "-").split("-", 1)[0]

    return base(source_language) in {"en", "eng", "english"} and base(target_language) in {"en", "eng", "english"}


def map_language_for_riva(source_language: str | None) -> str | None:
    if not source_language:
        return None
    low = source_language.strip().lower().replace("_", "-")
    if low in _RIVA_PAIR_MAP:
        return _RIVA_PAIR_MAP[low]
    base = low.split("-", 1)[0]
    return _RIVA_PAIR_MAP.get(base)


@dataclass(frozen=True)
class TranslationConfig:
    backend: str = "openai_chat"  # "openai_chat" | legacy "swiftu" | "riva" | "none"
    style: str = "natural"  # "natural" | "literal" | "expressive"
    target_language: str = "en"
    context_utterances_before: int = 6
    context_utterances_after: int = 4
    batch_size: int = 6
    allow_adjacent_regrouping: bool = True
    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "chat"
    resource_fingerprint: str = ""  # selected resource identity, no credentials
    api_key: str | None = None
    riva_model_path: str = str(DEFAULT_RIVA_MODEL_PATH)
    riva_python: str = str(DEFAULT_RIVA_PYTHON)
    riva_device: str = "cuda:0"
    timeout_s: float = 180.0
    max_tokens: int = 8192
    temperature: float = 0.0
    use_json_schema: bool = True
    reasoning_effort: str | None = "low"


_TRANSLATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "utterance_translation_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "utterance_id": {"type": "string"},
                            "translated_text": {"type": "string"},
                            "grouped_with_prev": {"type": "boolean"},
                            "notes": {"type": "string"},
                        },
                        "required": ["utterance_id", "translated_text", "grouped_with_prev", "notes"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["translations"],
            "additionalProperties": False,
        },
    },
}


def _lang_name(code: str | None) -> str:
    low = (code or "").strip().lower()
    return {"fr": "French", "zh": "Mandarin Chinese", "en": "English", "es": "Spanish", "de": "German", "ja": "Japanese", "ko": "Korean"}.get(low, low or "source language")


def _src_text(utt: dict[str, Any]) -> str:
    txt = utt.get("refined_text")
    if txt is not None and str(txt).strip():
        return str(txt).strip()
    return str(utt.get("source_text") or "").strip()


def build_batched_translation_messages(
    *,
    target_utterances: list[dict[str, Any]],
    context_before: list[dict[str, Any]],
    context_after: list[dict[str, Any]],
    source_language: str | None,
    target_language: str,
    style: str,
    glossary: GlossaryConfig,
    allow_adjacent_regrouping: bool,
) -> list[dict[str, str]]:
    """Build context-aware batched translation prompt anchored to utterance IDs."""
    src_name = _lang_name(source_language)
    tgt_name = _lang_name(target_language)
    style_desc = _STYLE_GUIDANCE.get(style, _STYLE_GUIDANCE["natural"])
    chars_s = ", ".join(glossary.characters) if glossary.characters else "(none)"
    terms_s = ", ".join(glossary.terms) if glossary.terms else "(none)"

    def _fmt(utts: list[dict[str, Any]]) -> str:
        if not utts:
            return "(none)"
        lines = []
        for u in utts:
            uid = u.get("utterance_id") or u.get("id")
            spk = u.get("speaker_id") or u.get("global_speaker_id") or "SPK"
            s = u.get("source_start")
            e = u.get("source_end")
            ts = f"[{s:.2f}-{e:.2f}]" if s is not None and e is not None else "[untimed]"
            lines.append(f"- [{uid}] ({spk} {ts}): {_src_text(u)}")
        return "\n".join(lines)

    regroup_rule = (
        "If two consecutive target utterances by the SAME speaker form one grammatical sentence, you may set grouped_with_prev=true on the second only if doing so makes subtitles clearer (keep both IDs present; put merged English on the first and empty string or continuation on the second). NEVER merge two different speakers or simultaneous speakers."
        if allow_adjacent_regrouping
        else "Keep grouped_with_prev=false on every item."
    )

    sys_prompt = (
        f"You are an expert subtitle localizer translating {src_name} into {tgt_name}. "
        f"Style '{style}': {style_desc} "
        "Keep every target utterance_id in output cleanly without translating READ-ONLY context. "
        f"{regroup_rule} "
        "Return strict JSON matching {\"translations\": [{\"utterance_id\": str, \"translated_text\": str, \"grouped_with_prev\": bool, \"notes\": str}]}."
    )
    user_prompt = (
        f"Glossary characters (keep exact spellings): {chars_s}\n"
        f"Glossary terms: {terms_s}\n"
        + (f"Series context: {glossary.notes}\n" if glossary.notes else "")
        + f"\nREAD-ONLY CONTEXT BEFORE:\n{_fmt(context_before)}\n"
        + f"\nTARGET UTTERANCES (TRANSLATE ALL {len(target_utterances)} IDs):\n{_fmt(target_utterances)}\n"
        + f"\nREAD-ONLY CONTEXT AFTER:\n{_fmt(context_after)}\n"
        + "\nReturn JSON only."
    )
    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]


def parse_batched_translation_response(raw_response: dict[str, Any]) -> dict[str, Any]:
    choices = raw_response.get("choices") or []
    if not choices:
        return {"status": "empty_choices", "translations": []}
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
        return {"status": "parse_error", "translations": [], "raw_content": content[:400]}
    try:
        parsed, _ = json.JSONDecoder().raw_decode(content, start_idx)
    except json.JSONDecodeError as exc:
        return {"status": "parse_error", "translations": [], "error": str(exc)}

    items = parsed if isinstance(parsed, list) else (parsed.get("translations") or []) if isinstance(parsed, dict) else []
    if not isinstance(items, list):
        return {"status": "parse_error", "translations": []}
    return {"status": "ok", "translations": [it for it in items if isinstance(it, dict)]}


_TRANSLATION_PROMPT_VERSION = "trans_v2"
_CONDENSE_PROMPT_VERSION = "condense_v1"

_MANDARIN_PARTICLES = frozenset("的了着过得地之吗呢吧呀啦呗嘛")
_MANDARIN_INTERJECTIONS = frozenset("嗯啊哦哼喂哎哇滚好对走不谁是看听快停等对")
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def _norm_en_clause(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", (text or "").lower()))


def _dedup_merged_translation(anc_t: str, cur_t: str) -> str:
    """Combine anchor + continuation English without repeating already-included clauses."""
    a = " ".join((anc_t or "").split()).strip()
    c = " ".join((cur_t or "").split()).strip()
    if not c:
        return _dedup_internal_trailing_clause(a)
    if not a:
        return _dedup_internal_trailing_clause(c)

    na = _norm_en_clause(a)
    nc = _norm_en_clause(c)
    if not nc or nc == na or (len(nc.split()) >= 2 and na.endswith(nc)):
        return _dedup_internal_trailing_clause(a)
    if len(na.split()) >= 2 and nc in na and na.find(nc) >= max(0, len(na) - len(nc) - 8):
        return _dedup_internal_trailing_clause(a)
    if len(na.split()) >= 2 and nc.startswith(na):
        return _dedup_internal_trailing_clause(c)
    return _dedup_internal_trailing_clause(f"{a} {c}".strip())


def _dedup_internal_trailing_clause(text: str) -> str:
    """Collapse an accidental full-clause echo inside one merged cue (keep short spoken repeats)."""
    # Match repeated >=3-word tail after punctuation or em-dash, e.g. "... foo—bar. foo—bar." or "... A. B. B."
    parts = [p.strip() for p in re.split(r"(?<=[.?!—])\s+", text.strip()) if p.strip()]
    if len(parts) >= 2:
        cleaned = [parts[0]]
        for p in parts[1:]:
            np = _norm_en_clause(p)
            nprev = _norm_en_clause(cleaned[-1])
            if np and len(np.split()) >= 3 and (np == nprev or (nprev.endswith(np) and len(np) >= 12)):
                continue
            cleaned.append(p)
        out = " ".join(cleaned)
    else:
        out = text.strip()
    # Also check duplicated suffix separated cleanly by dash/comma of >=3 words
    m = re.match(r"^(.*?)\s+([—\-–]?[^.?!]{12,}?)\s+\2\s*$", out)
    if m and len(_norm_en_clause(m.group(2)).split()) >= 3:
        return f"{m.group(1).strip()} {m.group(2).strip()}".strip()
    return out


def cleanup_mandarin_fragments(
    utterances: list[dict[str, Any]],
    *,
    source_language: str | None,
    target_language: str = "en",
) -> int:
    """Suppress isolated Mandarin overlap fragments & untranslated particles safely.

    Preserves legitimate standalone one-character interjections (e.g. ``嗯``, ``啊``)
    while suppressing stray single-character particles (``的``, ``了``, ...) or obvious
    duplicate overlap slips (e.g. ``峰`` overlapping ``巅峰``) and blocking CJK leakage
    from entering English subtitles.
    """
    suppressed = 0
    is_zh = (source_language or "").strip().lower().startswith("zh")
    is_en_target = (target_language or "en").strip().lower().startswith("en")
    n = len(utterances)
    for i, u in enumerate(utterances):
        if u.get("merged_into"):
            continue
        src = str(u.get("refined_text") or u.get("source_text") or "").strip()
        src_cjk = _CJK_CHAR_RE.findall(src)
        tr = str(u.get("translated_text") or "").strip()

        # Suppress isolated 1-char untranslated CJK particles/fragments (e.g. '的') while never wiping multi-char error fallbacks
        tr_cjk = _CJK_CHAR_RE.findall(tr)
        if is_en_target and tr and len(tr_cjk) == 1 and not re.search(r"[A-Za-z]", tr):
            ch_tr = tr_cjk[0]
            if ch_tr in _MANDARIN_PARTICLES or u.get("translation_status") == "fallback_empty_translation":
                u["translated_text_unfiltered"] = tr
                u["translated_text"] = ""
                u["fragment_suppressed"] = True
                u["translation_status"] = "suppressed_cjk_fallback_leak"
                suppressed += 1
                continue

        if not is_zh or len(src_cjk) != 1:
            continue
        ch = src_cjk[0]
        if ch in _MANDARIN_INTERJECTIONS and ch not in _MANDARIN_PARTICLES:
            continue

        # Grammatical particle standalone OR overlaps/immediately touches a neighbor containing that char
        s_i = u.get("source_start")
        e_i = u.get("source_end")
        dur_i = (float(e_i) - float(s_i)) if (s_i is not None and e_i is not None) else None
        shares_neighbor_overlap = False
        if s_i is not None and e_i is not None:
            for j in range(max(0, i - 3), min(n, i + 4)):
                if j == i:
                    continue
                nb = utterances[j]
                ns, ne = nb.get("source_start"), nb.get("source_end")
                if ns is None or ne is None:
                    continue
                if float(ns) <= float(e_i) + 0.8 and float(ne) >= float(s_i) - 0.8:
                    nb_src = str(nb.get("refined_text") or nb.get("source_text") or "")
                    if ch in nb_src and len(_CJK_CHAR_RE.findall(nb_src)) >= 2:
                        shares_neighbor_overlap = True
                        break

        if ch in _MANDARIN_PARTICLES or (shares_neighbor_overlap and (dur_i is None or dur_i <= 0.45)):
            u["translated_text_unfiltered"] = tr
            u["translated_text"] = ""
            u["fragment_suppressed"] = True
            u["translation_status"] = "suppressed_mandarin_fragment"
            suppressed += 1
    return suppressed


def _apply_safe_adjacent_regrouping(out_utts: list[dict[str, Any]]) -> int:
    """Apply safe same-speaker adjacent regrouping only when speakers and acoustics do not conflict."""
    merged_count = 0
    anchor_idx: int | None = None
    for i, u in enumerate(out_utts):
        uid = str(u.get("utterance_id") or u.get("id") or f"utt-{i:04d}")
        u["grouped_utterance_ids"] = [uid]
        u.pop("merged_into", None)

        if anchor_idx is None or not u.get("_grouped_with_prev"):
            anchor_idx = i
            continue

        anchor = out_utts[anchor_idx]
        # Accepted/provisional v2 source-event identities are never merged
        # merely because English text is similar. A translator is not an
        # authority to erase a separately observed acoustic source event.
        anc_source_ids = anchor.get("source_turn_ids") or ([anchor["source_turn_id"]] if anchor.get("source_turn_id") else [])
        cur_source_ids = u.get("source_turn_ids") or ([u["source_turn_id"]] if u.get("source_turn_id") else [])
        if (anc_source_ids or cur_source_ids) and set(anc_source_ids) != set(cur_source_ids):
            anchor_idx = i
            continue
        spk_cur = u.get("speaker_id") or u.get("global_speaker_id")
        spk_anc = anchor.get("speaker_id") or anchor.get("global_speaker_id")
        # Never merge different speakers, unknown speakers, or huge gaps (>1.5s) / >12s spans.
        if not spk_cur or spk_cur != spk_anc or str(spk_cur).startswith("SPK_UNK"):
            anchor_idx = i
            continue
        a_s = anchor.get("source_start")
        a_e = anchor.get("source_end")
        c_s = u.get("source_start")
        c_e = u.get("source_end")
        if a_s is None or a_e is None or c_s is None or c_e is None:
            anchor_idx = i
            continue
        if (float(c_s) - float(a_e)) > 1.5 or (float(c_e) - float(a_s)) > 12.0:
            anchor_idx = i
            continue
        # Never merge across an intervening utterance from another speaker
        if i != anchor_idx + len(anchor["grouped_utterance_ids"]):
            anchor_idx = i
            continue

        # Dedup trailing/contained continuation clauses cleanly (fixes EP166 utt-0016/0017 & utt-0185/0186)
        anc_t = str(anchor.get("translated_text") or "").strip()
        cur_t = str(u.get("translated_text") or "").strip()
        anchor["translated_text"] = _dedup_merged_translation(anc_t, cur_t)
        anchor["source_end"] = round(max(float(a_e), float(c_e)), 3)
        anchor["grouped_utterance_ids"].append(uid)
        u["merged_into"] = str(anchor.get("utterance_id") or anchor.get("id"))
        merged_count += 1
    return merged_count


def _translation_batch_cache_key(
    *,
    config_sig: dict[str, Any],
    targets: list[dict[str, Any]],
    ctx_before: list[dict[str, Any]],
    ctx_after: list[dict[str, Any]],
) -> str:
    payload = {
        "cfg": config_sig,
        "targets": [
            (str(u.get("utterance_id") or u.get("id") or ""), str(u.get("speaker_id") or u.get("global_speaker_id") or ""), _src_text(u))
            for u in targets
        ],
        "ctx_before": [
            (str(u.get("utterance_id") or u.get("id") or ""), str(u.get("speaker_id") or u.get("global_speaker_id") or ""), _src_text(u))
            for u in ctx_before
        ],
        "ctx_after": [
            (str(u.get("utterance_id") or u.get("id") or ""), str(u.get("speaker_id") or u.get("global_speaker_id") or ""), _src_text(u))
            for u in ctx_after
        ],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def translate_utterances(
    utterances: list[dict[str, Any]],
    *,
    source_language: str | None,
    config: TranslationConfig | None = None,
    glossary: GlossaryConfig | None = None,
    text_client: Any = None,
    lifecycle: ModelLifecycle | None = None,
    riva_resource_name: str = "riva_local",
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Translate utterances into target English subtitles while preserving source IDs and timing."""
    cfg = config or TranslationConfig()
    if cfg.backend not in VALID_TRANSLATION_BACKENDS:
        raise ValueError(f"Unsupported translator backend {cfg.backend!r}")
    if cfg.style not in VALID_TRANSLATION_STYLES:
        raise ValueError(f"Unsupported translation style {cfg.style!r}")

    out_utts = [{**u} for u in utterances]
    for i, u in enumerate(out_utts):
        uid = str(u.get("utterance_id") or u.get("id") or f"utt-{i:04d}")
        u["grouped_utterance_ids"] = [uid]

    if cfg.backend == "none":
        for u in out_utts:
            u["translated_text"] = _src_text(u)
            u["translation_status"] = "bypassed_source_kept"
        return out_utts, {"backend": "none", "style": cfg.style, "total_utterances": len(out_utts)}

    gl = glossary or GlossaryConfig()
    trans_cfg_sig = {
        "prompt_version": _TRANSLATION_PROMPT_VERSION,
        "backend": cfg.backend,
        "model": cfg.model,
        "endpoint": cfg.base_url,
        "resource_fingerprint": cfg.resource_fingerprint,
        "riva_model_path": cfg.riva_model_path if cfg.backend == "riva" else None,
        "riva_python": cfg.riva_python if cfg.backend == "riva" else None,
        "riva_device": cfg.riva_device if cfg.backend == "riva" else None,
        "style": cfg.style,
        "source_language": source_language or "",
        "target_language": cfg.target_language,
        "glossary_digest": gl.digest(),
        "ctx_before": cfg.context_utterances_before,
        "ctx_after": cfg.context_utterances_after,
        "batch_size": cfg.batch_size,
        "allow_adjacent_regrouping": cfg.allow_adjacent_regrouping,
        "temperature": cfg.temperature,
        "use_json_schema": cfg.use_json_schema,
        "reasoning_effort": cfg.reasoning_effort,
    }
    gl_sig = hashlib.sha256(json.dumps(trans_cfg_sig, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    ckpt_file = Path(checkpoint_path) if checkpoint_path else None
    cached: dict[str, dict[str, Any]] = {}
    if resume and ckpt_file and ckpt_file.is_file():
        try:
            doc = json.loads(ckpt_file.read_text(encoding="utf-8"))
            if isinstance(doc.get("items"), dict):
                cached = dict(doc.get("items") or {})
        except Exception:
            cached = {}

    t0 = time.perf_counter()
    errors: list[str] = []
    api_calls = 0
    cache_hits = 0

    if cfg.backend == "riva":
        if cfg.target_language.strip().lower() not in ("en", "english"):
            raise ValueError("Riva primary translation currently supports an English target only")
        if is_english_source_to_english(source_language, cfg.target_language):
            # English subtitles need no MT or post-edit of the source ASR.
            # This bypass works even if a Riva worker has not been installed.
            for u in out_utts:
                u["translated_text"] = str(u.get("source_text") or "").strip()
                u["translation_status"] = "bypassed_same_language"
            return out_utts, {
                "backend": "riva", "status": "bypassed_same_language",
                "total_utterances": len(out_utts), "resource_fingerprint": cfg.resource_fingerprint,
            }
        lang_pair = map_language_for_riva(source_language)
        if lang_pair is None:
            raise ValueError(f"Riva-Translate-4B-Instruct-v2 does not support source_language={source_language!r}")
        pending_items = []
        riva_keys: dict[str, str] = {}
        for i, u in enumerate(out_utts):
            uid = str(u.get("utterance_id") or u.get("id") or f"utt-{i:04d}")
            src_txt = _src_text(u)
            rkey = hashlib.sha256(
                json.dumps({"backend": "riva", "resource": cfg.resource_fingerprint, "checkpoint": cfg.riva_model_path, "worker": cfg.riva_python, "device": cfg.riva_device, "lang_pair": lang_pair, "uid": uid, "text": src_txt}, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:24]
            ekey = f"{uid}:{rkey}"
            riva_keys[uid] = ekey
            if ekey in cached:
                u["translated_text"] = cached[ekey]["translated_text"]
                u["translation_status"] = "translated_cached"
                cache_hits += 1
            else:
                pending_items.append({"utterance_id": uid, "text": src_txt, "lang_pair": lang_pair})

        riva_meta: dict[str, Any] = {}
        if pending_items:
            # "auto" may choose CUDA inside the subprocess. Admit it before
            # device selection so the worker never bypasses the shared GPU lease.
            with (lifecycle.worker(resource_name=riva_resource_name)
                  if lifecycle is not None and (cfg.riva_device == "auto" or cfg.riva_device.startswith("cuda"))
                  else nullcontext()):
                proc = subprocess.run(
                    [cfg.riva_python, str(RIVA_WORKER_SCRIPT)],
                    input=json.dumps(
                        {"model_path": cfg.riva_model_path, "items": pending_items, "device": cfg.riva_device},
                        ensure_ascii=False,
                    ),
                    capture_output=True,
                    text=True,
                    timeout=max(cfg.timeout_s, len(pending_items) * 15.0),
                )
            if proc.returncode != 0:
                raise RuntimeError(f"Riva worker failed ({proc.returncode}): {proc.stderr[-500:]}")
            resp = json.loads(proc.stdout.strip().splitlines()[-1])
            if resp.get("status") != "ok":
                raise RuntimeError("Riva worker did not report a successful translation")
            if cfg.riva_device != "auto" and resp.get("device_used") != cfg.riva_device:
                raise RuntimeError("Riva worker reported a different device than explicitly configured")
            riva_meta = {k: resp.get(k) for k in ("device_used", "dtype", "load_s", "infer_s", "peak_vram_mib")}
            expected_uids = {item["utterance_id"] for item in pending_items}
            rows = resp.get("results")
            if not isinstance(rows, list):
                raise RuntimeError("Riva worker response is missing translation results")
            by_uid: dict[str, str] = {}
            for row in rows:
                if (not isinstance(row, dict) or row.get("utterance_id") not in expected_uids
                        or row["utterance_id"] in by_uid
                        or not isinstance(row.get("translated_text"), str)):
                    raise RuntimeError("Riva worker returned invalid or duplicate utterance IDs")
                by_uid[row["utterance_id"]] = row["translated_text"]
            if set(by_uid) != expected_uids or any(
                not by_uid[item["utterance_id"]].strip() and item["text"].strip()
                for item in pending_items
            ):
                raise RuntimeError("Riva worker returned missing or empty translations")
            for i, u in enumerate(out_utts):
                uid = str(u.get("utterance_id") or u.get("id") or f"utt-{i:04d}")
                if uid in by_uid:
                    u["translated_text"] = by_uid[uid]
                    u["translation_status"] = "translated"
                    cached[riva_keys[uid]] = {"translated_text": by_uid[uid], "grouped_with_prev": False}
            if ckpt_file:
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                ckpt_file.write_text(json.dumps({"schema": "translation_cache/v2", "signature": gl_sig, "items": cached}, ensure_ascii=False, indent=2), encoding="utf-8")

        return out_utts, {
            "backend": "riva",
            "resource_fingerprint": cfg.resource_fingerprint,
            "lang_pair": lang_pair,
            "total_utterances": len(out_utts),
            "cache_hits": cache_hits,
            "elapsed_s": round(time.perf_counter() - t0, 3),
            **riva_meta,
        }

    # OpenAI-compatible contextual batch translator
    client = text_client or OpenAIChatTextClient(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        reasoning_effort=cfg.reasoning_effort,
    )
    n = len(out_utts)
    bsize = max(1, int(cfg.batch_size))

    def _translate_batch(
        indices: list[int],
        retries_left: int = 1,
        allow_regrouping_override: bool | None = None,
        cache_key: str | None = None,
    ) -> None:
        nonlocal api_calls
        if not indices:
            return
        allow_grp = cfg.allow_adjacent_regrouping if allow_regrouping_override is None else allow_regrouping_override
        targets = [out_utts[i] for i in indices]
        lo, hi = indices[0], indices[-1] + 1
        ctx_before = out_utts[max(0, lo - cfg.context_utterances_before) : lo]
        ctx_after = out_utts[hi : min(n, hi + cfg.context_utterances_after)]
        msgs = build_batched_translation_messages(
            target_utterances=targets,
            context_before=ctx_before,
            context_after=ctx_after,
            source_language=source_language,
            target_language=cfg.target_language,
            style=cfg.style,
            glossary=gl,
            allow_adjacent_regrouping=allow_grp,
        )
        try:
            api_calls += 1
            raw = client.complete(
                messages=msgs,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                response_format=_TRANSLATION_JSON_SCHEMA if cfg.use_json_schema else None,
            )
            parsed = parse_batched_translation_response(raw)
        except Exception as exc:
            parsed = {"status": "request_error", "error": f"{type(exc).__name__}: {exc}", "translations": []}

        if parsed.get("status") != "ok":
            if retries_left > 0 and len(indices) > 1:
                mid = len(indices) // 2
                _translate_batch(indices[:mid], retries_left=0, allow_regrouping_override=allow_regrouping_override)
                _translate_batch(indices[mid:], retries_left=0, allow_regrouping_override=allow_regrouping_override)
                return
            errors.append(f"batch[{lo}:{hi}] -> {parsed.get('status')}: {parsed.get('error') or ''}")
            for i in indices:
                out_utts[i]["translated_text"] = _src_text(out_utts[i])
                out_utts[i]["translation_status"] = "fallback_source_after_error"
            return

        expected = {str(out_utts[i].get("utterance_id") or out_utts[i].get("id") or f"utt-{i:04d}"): i for i in indices}
        seen: set[str] = set()
        for item in parsed["translations"]:
            uid = str(item.get("utterance_id") or "").strip()
            if uid not in expected or uid in seen:
                continue
            seen.add(uid)
            idx = expected[uid]
            trans = " ".join(str(item.get("translated_text") or "").split()).strip()
            grp = bool(item.get("grouped_with_prev")) and (allow_regrouping_override is not False)
            if not trans and not grp and _src_text(out_utts[idx]):
                trans = _src_text(out_utts[idx])
                status_str = "fallback_empty_translation"
            else:
                status_str = "translated"
            out_utts[idx]["translated_text"] = trans
            out_utts[idx]["translation_status"] = status_str
            out_utts[idx]["translation_notes"] = str(item.get("notes") or "")
            out_utts[idx]["_grouped_with_prev"] = grp
            if cache_key and status_str == "translated":
                cached[f"{uid}:{cache_key}"] = {
                    "cache_key": cache_key,
                    "translated_text": trans,
                    "translation_status": status_str,
                    "grouped_with_prev": grp,
                    "notes": out_utts[idx]["translation_notes"],
                }

        for uid, idx in expected.items():
            if uid not in seen:
                errors.append(f"missing_translation_id:{uid}")
                out_utts[idx]["translated_text"] = _src_text(out_utts[idx])
                out_utts[idx]["translation_status"] = "fallback_missing_id"

    for lo in range(0, n, bsize):
        hi = min(n, lo + bsize)
        batch_indices = list(range(lo, hi))
        targets = [out_utts[i] for i in batch_indices]
        ctx_before = out_utts[max(0, lo - cfg.context_utterances_before) : lo]
        ctx_after = out_utts[hi : min(n, hi + cfg.context_utterances_after)]
        bkey = _translation_batch_cache_key(config_sig=trans_cfg_sig, targets=targets, ctx_before=ctx_before, ctx_after=ctx_after)

        all_hit = True
        for i in batch_indices:
            uid = str(out_utts[i].get("utterance_id") or out_utts[i].get("id") or f"utt-{i:04d}")
            if f"{uid}:{bkey}" not in cached:
                all_hit = False
                break

        if all_hit:
            for i in batch_indices:
                uid = str(out_utts[i].get("utterance_id") or out_utts[i].get("id") or f"utt-{i:04d}")
                ci = cached[f"{uid}:{bkey}"]
                out_utts[i]["translated_text"] = ci["translated_text"]
                out_utts[i]["translation_status"] = ci.get("translation_status", "translated")
                out_utts[i]["translation_notes"] = ci.get("notes", "")
                out_utts[i]["_grouped_with_prev"] = bool(ci.get("grouped_with_prev"))
                cache_hits += 1
        else:
            _translate_batch(batch_indices, retries_left=1, cache_key=bkey)
            if ckpt_file:
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                ckpt_file.write_text(json.dumps({"schema": "translation_cache/v2", "signature": gl_sig, "items": cached}, ensure_ascii=False, indent=2), encoding="utf-8")

    regrouped_count = _apply_safe_adjacent_regrouping(out_utts) if cfg.allow_adjacent_regrouping else 0

    # Repair any orphan empty translation caused when the translator merged into a previous
    # utterance whose merge was safely vetoed by _apply_safe_adjacent_regrouping (gap/span/speaker).
    orphan_indices = [
        i
        for i in range(n)
        if not str(out_utts[i].get("translated_text") or "").strip()
        and _src_text(out_utts[i])
        and not out_utts[i].get("merged_into")
    ]
    if orphan_indices:
        repair_set: set[int] = set()
        for i in orphan_indices:
            repair_set.add(i)
            if i > 0 and not out_utts[i - 1].get("merged_into"):
                repair_set.add(i - 1)
        repair_sorted = sorted(repair_set)
        for off in range(0, len(repair_sorted), bsize):
            _translate_batch(repair_sorted[off : off + bsize], retries_left=1, allow_regrouping_override=False)
        if ckpt_file:
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            ckpt_file.write_text(json.dumps({"schema": "translation_cache/v2", "signature": gl_sig, "items": cached}, ensure_ascii=False, indent=2), encoding="utf-8")
        regrouped_count = _apply_safe_adjacent_regrouping(out_utts) if cfg.allow_adjacent_regrouping else 0

    for u in out_utts:
        u.pop("_grouped_with_prev", None)

    fragment_suppressed_count = cleanup_mandarin_fragments(
        out_utts,
        source_language=source_language,
        target_language=cfg.target_language,
    )

    return out_utts, {
        "backend": cfg.backend,
        "resource_fingerprint": cfg.resource_fingerprint,
        "style": cfg.style,
        "target_language": cfg.target_language,
        "signature": gl_sig,
        "total_utterances": n,
        "regrouped_adjacent_utterances": regrouped_count,
        "fragment_suppressed_count": fragment_suppressed_count,
        "api_calls": api_calls,
        "cache_hits": cache_hits,
        "errors": errors,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


_CONDENSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "subtitle_condensation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "utterance_id": {"type": "string"},
                "condensed_text": {"type": "string"},
                "changed": {"type": "boolean"},
                "rationale": {"type": "string"},
            },
            "required": ["utterance_id", "condensed_text", "changed", "rationale"],
            "additionalProperties": False,
        },
    },
}


def condense_high_cps_cues_with_text_model(
    utterances: list[dict[str, Any]],
    *,
    source_language: str | None,
    config: TranslationConfig | None = None,
    cps_threshold: float = 25.0,
    target_cps: float = 20.0,
    text_client: Any = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Selectively condense only overly fast cues (> cps_threshold) while keeping original English."""
    cfg = config or TranslationConfig()
    out_utts = [{**u} for u in utterances]
    if cfg.backend == "none":
        return out_utts, {"status": "skipped_backend_none", "candidate_count": 0, "condensed_count": 0}
    if cfg.backend not in ("openai_chat", "swiftu"):
        raise ValueError(
            "Subtitle condensation requires an OpenAI-compatible text resource, "
            f"not translation backend {cfg.backend!r}"
        )

    client = text_client or OpenAIChatTextClient(
        base_url=cfg.base_url,
        model=cfg.model,
        api_key=cfg.api_key,
        timeout_s=cfg.timeout_s,
        reasoning_effort=cfg.reasoning_effort,
    )
    ckpt_file = Path(checkpoint_path) if checkpoint_path else None
    cached: dict[str, dict[str, Any]] = {}
    if resume and ckpt_file and ckpt_file.is_file():
        try:
            doc = json.loads(ckpt_file.read_text(encoding="utf-8"))
            if isinstance(doc.get("items"), dict):
                cached = dict(doc.get("items") or {})
        except Exception:
            cached = {}

    t0 = time.perf_counter()
    n = len(out_utts)
    candidates: list[tuple[int, float, int]] = []
    for i, u in enumerate(out_utts):
        if u.get("merged_into") or u.get("fragment_suppressed"):
            continue
        s, e = u.get("source_start"), u.get("source_end")
        en = str(u.get("translated_text") or "").strip()
        if s is None or e is None or not en:
            continue
        dur = max(0.35, float(e) - float(s))
        est_cps = len(en) / dur
        budget = max(24, int(round(dur * target_cps)))
        if est_cps > cps_threshold and len(en) > budget + 8 and len(en) >= 32:
            candidates.append((i, dur, budget))

    condensed_count = 0
    api_calls = 0
    cache_hits = 0
    errors: list[str] = []

    for i, dur, budget in candidates:
        u = out_utts[i]
        uid = str(u.get("utterance_id") or u.get("id") or f"utt-{i:04d}")
        en = str(u.get("translated_text") or "").strip()
        src = _src_text(u)
        prev_ctx = [str(out_utts[j].get("translated_text") or "").strip() for j in range(max(0, i - 2), i) if out_utts[j].get("translated_text")]
        next_ctx = [str(out_utts[j].get("translated_text") or "").strip() for j in range(i + 1, min(n, i + 3)) if out_utts[j].get("translated_text")]
        ckey_payload = {
            "prompt_version": _CONDENSE_PROMPT_VERSION,
            "uid": uid,
            "en": en,
            "src": src,
            "prev": prev_ctx,
            "next": next_ctx,
            "budget": budget,
            "style": cfg.style,
            "model": cfg.model,
            "endpoint": cfg.base_url,
            "resource_fingerprint": cfg.resource_fingerprint,
            "temperature": cfg.temperature,
            "use_json_schema": cfg.use_json_schema,
            "reasoning_effort": cfg.reasoning_effort,
        }
        ckey = f"{uid}:{hashlib.sha256(json.dumps(ckey_payload, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:24]}"

        u.setdefault("uncondensed_translated_text", en)
        if ckey in cached:
            entry = cached[ckey]
            cache_hits += 1
        else:
            sys_p = (
                f"You are tightening English subtitle wording (style '{cfg.style}') without losing plot facts, "
                "character names, emotional intent, or meaningful dialogue. Remove redundant phrasing, literal filler, "
                "or translation repetition to fit comfortably within the display duration. Return strict JSON."
            )
            usr_p = (
                f"utterance_id: {uid}\n"
                f"Source ({source_language or 'src'}): {src}\n"
                f"Current English ({len(en)} chars): {en}\n"
                f"Available display duration: {dur:.2f}s (target <= ~{budget} characters if natural; never invent facts)\n"
                f"Neighboring English before: {' | '.join(prev_ctx) or '(none)'}\n"
                f"Neighboring English after: {' | '.join(next_ctx) or '(none)'}\n"
                "Return JSON only."
            )
            try:
                api_calls += 1
                raw = client.complete(
                    messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
                    max_tokens=4096,
                    temperature=cfg.temperature,
                    response_format=_CONDENSE_JSON_SCHEMA if cfg.use_json_schema else None,
                )
                msg = ((raw.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
                if "</think>" in msg:
                    msg = msg.split("</think>", 1)[-1].strip()
                first_b = msg.find("{")
                parsed, _ = json.JSONDecoder().raw_decode(msg, first_b) if first_b >= 0 else ({}, 0)
                cand_txt = " ".join(str(parsed.get("condensed_text") or en).split()).strip()
                ok = bool(parsed.get("changed")) and cand_txt and 0.30 * len(en) <= len(cand_txt) < len(en)
                entry = {
                    "condensed_text": cand_txt if ok else en,
                    "changed": ok,
                    "rationale": str(parsed.get("rationale") or ""),
                }
                cached[ckey] = entry
                if ckpt_file:
                    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                    ckpt_file.write_text(json.dumps({"items": cached}, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                errors.append(f"{uid}:{type(exc).__name__}")
                entry = {"condensed_text": en, "changed": False, "rationale": "error"}

        if entry.get("changed"):
            u["translated_text"] = entry["condensed_text"]
            u["condensation_applied"] = True
            u["condensation_rationale"] = entry.get("rationale", "")
            condensed_count += 1
        else:
            u["condensation_applied"] = False

    return out_utts, {
        "status": "ok",
        "resource_fingerprint": cfg.resource_fingerprint,
        "cps_threshold": cps_threshold,
        "candidate_count": len(candidates),
        "condensed_count": condensed_count,
        "api_calls": api_calls,
        "cache_hits": cache_hits,
        "errors": errors,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }

# Legacy API name retained for existing callers.
condense_high_cps_cues_with_swiftu = condense_high_cps_cues_with_text_model
