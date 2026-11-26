"""Current-row evidence projection; hypotheses never establish entity identity."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from copy import deepcopy

SCHEMA = "nemovox-row-semantic-context/v1"


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def independent_support(candidate):
    support = candidate.get("support", [])
    return any(a.get("backend") == "nemotron" and b.get("backend") == "independent_asr"
               and a.get("id") and b.get("id") and a["id"] != b["id"]
               and a.get("audio_sha256") and b.get("audio_sha256")
               for a in support for b in support)


def literal_match(text, literal):
    if not isinstance(text,str) or not isinstance(literal,str) or not literal:
        return False
    return bool(text==literal or (len(literal)>=2 and (
        (re.search(r"[\u3400-\u9fff\u3040-\u30ff]",literal) and literal in text)
        or re.search(r"(?<!\w)"+re.escape(literal)+r"(?!\w)",text))))


def row_nominations(rows, context):
    index = {r["id"]: r for r in rows}
    by_id, unavailable = {}, []
    for prop in (context or {}).get("proposals", []):
        entity = prop.get("kind") in ("entity_variant", "relationship")
        citations = prop.get("citations", [])
        targets = ({c["source_id"] for c in citations} if entity
                   else {prop.get("target_source_id")})
        valid=[]
        for citation in citations:
            uid=citation.get("source_id")
            row=index.get(uid)
            span=citation.get("character_span")
            text=(row or {}).get("source_text","")
            if (row is not None and isinstance(span,list) and len(span)==2
                    and all(type(x) is int for x in span)
                    and 0<=span[0]<span[1]<=len(text)
                    and text[span[0]:span[1]]==citation.get("quote")
                    and citation.get("source_row_unit_ids")==row.get("source_unit_ids")):
                valid.append({"source_id":uid,"quote":citation["quote"],"character_span":span})
        for uid in sorted(targets,key=str):
            matches=[c for c in valid if c["source_id"]==uid]
            if not matches:
                unavailable.append({"proposal_id":prop["proposal_id"],"source_id":uid,
                                    "reason":"source_reference_changed_or_retired"})
                continue
            by_id.setdefault(uid,[]).append({"proposal_id":prop["proposal_id"],
                "kind":prop["kind"],"citations":matches,
                "related_citations":[c for c in valid if c["source_id"]!=uid],
                "hypothesis":prop.get("hypothesis",""),
                "status":"advisory_only"})
    return by_id, unavailable


def accepted_row_context(row, alternatives, visual):
    text = row["source_text"]
    acoustic = []
    supported = [alt for alt in alternatives if independent_support(alt)]
    if len(supported) == 1 and supported[0].get("text") == text:
        acoustic.extend(deepcopy(supported[0]["support"]))
    labels = []
    for frame in visual:
        literal = frame.get("visible_text")
        if (frame.get("inside_event") is True and frame.get("text_kind") in ("entity_label", "title")
                and frame.get("literal_observation_status")=="model_observed"
                and not frame.get("uncertainty") and isinstance(literal, str)
                and 2 <= len(literal.strip()) <= 120 and literal_match(text,literal.strip())
                and isinstance(frame.get("sha256"), str)
                and re.fullmatch(r"[a-f0-9]{64}", frame["sha256"])):
            labels.append({"literal": literal.strip(), "frame_id": frame["id"],
                "frame_sha256": frame["sha256"], "requested_seek_s": frame["requested_seek_s"],
                "kind": frame["text_kind"], "status": "model_observed_visual_text"})
    if not acoustic and not labels:
        return None
    return {"schema": SCHEMA, "source_id": row["id"], "source_text_sha256": text_digest(text),
            "acoustic_support": acoustic, "visual_literals": labels,
            "entity_identity_accepted": False, "human_verified": False}


def localized_wording_evidence(row):
    """Bound timing-recovery wording alternatives to the current source units."""
    value=row.get("source_lineage",{}).get("localized_wording_evidence")
    text=row.get("source_text")
    ids=row.get("source_unit_ids",[])
    if (not isinstance(value,dict) or not isinstance(text,str)
            or value.get("source_id")!=row.get("id") or value.get("source_text_sha256")!=text_digest(text)
            or value.get("source_unit_ids")!=ids or value.get("entity_identity_accepted") is not False
            or value.get("status")!="advisory_model_hypotheses"):
        return None
    alternatives=value.get("alternatives")
    if not isinstance(alternatives,list) or not alternatives:
        return None
    for a in alternatives:
        if not isinstance(a,dict):
            return None
        span=a.get("source_unit_range")
        if (not isinstance(span,list) or len(span)!=2 or any(type(x) is not int for x in span)
                or not 0<=span[0]<span[1]<=len(ids)
                or not isinstance(a.get("tokens"),list) or not a["tokens"]
                or not all(isinstance(t,str) and t for t in a["tokens"])
                or not isinstance(a.get("audio_sha256"),str) or not re.fullmatch(r"[a-f0-9]{64}",a["audio_sha256"])
                or not isinstance(a.get("observation_ids"),list) or not a["observation_ids"]
                or not all(isinstance(x,str) and x for x in a["observation_ids"])
                or a.get("backend") not in ("nemotron","independent_asr")):
            return None
    if {a["backend"] for a in alternatives}!={"nemotron","independent_asr"}:
        return None
    return deepcopy(value)


def current_row_context(row):
    value = row.get("semantic_context")
    text = row.get("source_text")
    if not isinstance(text,str):
        return None
    if (isinstance(value,dict) and value.get("schema")==SCHEMA
            and value.get("source_id")==row.get("id")
            and value.get("source_text_sha256")==text_digest(text)
            and value.get("entity_identity_accepted") is False
            and value.get("human_verified") is False):
        result={k:deepcopy(value[k]) for k in (
            "schema","source_id","source_text_sha256","acoustic_support",
            "visual_literals","entity_identity_accepted","human_verified") if k in value}
    else:
        result=None
    # Recovered-event alternative wording is context, not canonical text,
    # accepted spelling, speaker identity or permission to invent a name.
    wording=row.get("source_lineage",{}).get("contextual_wording")
    if (isinstance(wording,dict) and wording.get("source_id")==row.get("id")
            and wording.get("source_text_sha256")==text_digest(text)
            and wording.get("status")=="advisory_model_hypotheses"
            and wording.get("entity_identity_accepted") is False):
        alternatives=wording.get("alternatives",[])
        if (isinstance(alternatives,list) and alternatives
                and all(isinstance(a,dict) and isinstance(a.get("text"),str)
                        and a.get("observation_ids")
                        and isinstance(a.get("audio_sha256"),str)
                        and re.fullmatch(r"[a-f0-9]{64}",a["audio_sha256"])
                        and a.get("backend") in ("nemotron","independent_asr")
                        for a in alternatives)
                and {a["backend"] for a in alternatives}=={"nemotron","independent_asr"}):
            if result is None:
                result={"schema":SCHEMA,"source_id":row["id"],
                        "source_text_sha256":text_digest(text),"acoustic_support":[],
                        "visual_literals":[],"entity_identity_accepted":False,"human_verified":False}
            result["advisory_wording_alternatives"]=deepcopy(alternatives)
    local=localized_wording_evidence(row)
    if local is not None:
        if result is None:
            result={"schema":SCHEMA,"source_id":row["id"],"source_text_sha256":text_digest(text),
                    "acoustic_support":[],"visual_literals":[],"entity_identity_accepted":False,"human_verified":False}
        result["localized_wording_alternatives"]=local["alternatives"]
    return result



def literal_examples(row, glossary, context_rows=()):
    """Select explicit operator bilingual pairs for a literal target or a
    neighboring independently supported source row. Never synthesize the
    target spelling from a source-only observation.
    """
    import json
    text = row.get("source_text", "")
    related = [r["source_text"] for r in context_rows
               if (current_row_context(r) or {}).get("acoustic_support")]
    result = []
    for source, target in glossary.translation_examples:
        if not source or not target:
            continue
        if any(literal_match(passage,source) for passage in [text,*related]):
            item = {"source":source,"target":target}
            item["id"] = text_digest(json.dumps(item,ensure_ascii=False,sort_keys=True))
            result.append(item)
    return result[:4]


# Retained only for literal, previously recorded evidence interpretation.
SPECIAL_TAG_RE = re.compile(r"<[^<>\\s]+>")

def _is_cjk_char(ch: str) -> bool:
    if not ch or ch.isspace():
        return False
    name = unicodedata.name(ch, "")
    return name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH"))


def split_lexical_units(text: str) -> list[dict[str, str]]:
    """Split text into lexical units, splitting CJK ideographs individually without whitespace assumption."""
    cleaned = SPECIAL_TAG_RE.sub(" ", text or "")
    units: list[dict[str, str]] = []
    for chunk in cleaned.split():
        buf: list[str] = []
        for ch in chunk:
            if _is_cjk_char(ch):
                if buf:
                    units.append({"text": "".join(buf), "unit_type": "word"})
                    buf.clear()
                units.append({"text": ch, "unit_type": "character"})
            else:
                buf.append(ch)
        if buf:
            units.append({"text": "".join(buf), "unit_type": "word"})
    return units


def _lex(text):
    # Lexical matching establishes corroboration, not a release WER target.
    result = []
    for item in split_lexical_units(str(text or "")):
        key = "".join(c for c in unicodedata.normalize("NFKC", item["text"]).casefold()
                      if c.isalnum() or unicodedata.category(c).startswith("M"))
        if key:
            result.append(key)
    return tuple(result)


def _contains(outer, inner):
    return bool(inner) and any(outer[i:i + len(inner)] == inner
                               for i in range(len(outer) - len(inner) + 1))


