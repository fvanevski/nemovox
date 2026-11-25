"""Current-row evidence projection; hypotheses never establish entity identity."""
from __future__ import annotations

import hashlib
import re
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


def current_row_context(row):
    value = row.get("semantic_context")
    text = row.get("source_text")
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or value.get("source_id") != row.get("id") or not isinstance(text, str)
            or value.get("source_text_sha256") != text_digest(text)
            or value.get("entity_identity_accepted") is not False
            or value.get("human_verified") is not False):
        return None
    return {k: deepcopy(value[k]) for k in (
        "schema", "source_id", "source_text_sha256", "acoustic_support",
        "visual_literals", "entity_identity_accepted", "human_verified") if k in value}



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
