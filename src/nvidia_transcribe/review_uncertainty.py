"""Downstream constraints on unresolved current-source evidence.

Projection is after dedicated MT: no new source interpretation or MT cache
identity, no automatic name mapping. Reports remain model hypotheses.
"""
from __future__ import annotations

from copy import deepcopy
import re

from nvidia_transcribe.semantic_context import row_nominations, text_digest

SCHEMA="nemovox-review-source-uncertainty/v1"
_MATERIAL={"hallucination","missing_dialogue","wording","polarity"}
_ENTITY={"entity_variant","relationship"}
_WORD=re.compile(r"[^\W_]+(?:['’][^\W_]+)*",re.UNICODE)
_ARTICLES={"a","an","the"}
# Exclude common sentence-leading/function words, not a general NER model.
_COMMON={"i","he","she","it","we","they","you","his","her","our","their","my","your",
         "a","an","the","this","that","these","those","and","but","or","so","if",
         "no","yes","well","oh","now","then","here","there","what","who","why","how",
         "is","are","was","were","will","would","can","could","do","did","does"}


def bind_review_uncertainty(rows, quality):
    """Project only current-row citations and exact existing scene IDs.

    A changed or retired citation does not transfer to another row. Scene
    outcomes are dispositions, never accepted lexical evidence.
    """
    result=[{k:v for k,v in row.items() if k!="review_source_uncertainty"} for row in rows]
    if not isinstance(quality,dict) or quality.get("schema")!="nemovox-current-source-quality/v1":
        return result
    context=quality.get("recording_context") or {}
    if context and (not quality.get("recording_sha256")
                    or context.get("recording_sha256")!=quality["recording_sha256"]):
        raise ValueError("Unresolved context recording identity mismatch")
    dispositions={x["proposal_id"]:x for x in quality.get("context_proposal_dispositions",[])
                  if x.get("status")=="hypothesis_unresolved"}
    props=[p for p in context.get("proposals",[])
           if p.get("status")=="advisory_only" and p.get("source_truth_accepted") is False
           and p.get("proposal_id") in dispositions
           and dispositions[p["proposal_id"]].get("kind")==p.get("kind")]
    nominations,_=row_nominations(result,{"proposals":props})
    scene=quality.get("scene_evidence") or {}
    outcomes={x["source_id"]:x.get("outcome") for x in scene.get("decisions",[])
              if x.get("outcome") in ("unresolved","partial","error")}
    for uid in scene.get("deferred_source_ids",[]):
        outcomes.setdefault(uid,"deferred")
    for row in result:
        uid=row["id"]
        cited=[{"proposal_id":n["proposal_id"],"kind":n["kind"],
                "citations":deepcopy(n["citations"])}
               for n in nominations.get(uid,[])
               if uid in dispositions[n["proposal_id"]].get("source_ids",[])]
        outcome=outcomes.get(uid)
        status=str(row.get("source_quality_status") or "")
        if not cited and not outcome and not status.startswith("unresolved_wording"):
            continue
        row["review_source_uncertainty"]={"schema":SCHEMA,"source_id":uid,
            "source_text_sha256":text_digest(row["source_text"]),
            "source_unit_ids":deepcopy(row.get("source_unit_ids",[])),
            "nominations":cited,"scene_outcome":outcome,
            "source_quality_status":status,"source_truth_accepted":False}
    return result


def current_uncertainty(row):
    value=row.get("review_source_uncertainty")
    text=row.get("source_text")
    if (not isinstance(value,dict) or value.get("schema")!=SCHEMA
            or value.get("source_id")!=row.get("id") or not isinstance(text,str)
            or value.get("source_text_sha256")!=text_digest(text)
            or value.get("source_unit_ids")!=row.get("source_unit_ids",[])
            or value.get("source_truth_accepted") is not False):
        return None
    for n in value.get("nominations",[]):
        for cite in n.get("citations",[]):
            span=cite.get("character_span")
            if (cite.get("source_id")!=row["id"] or not isinstance(span,list) or len(span)!=2
                    or any(type(x) is not int for x in span)
                    or not 0<=span[0]<span[1]<=len(text)
                    or text[span[0]:span[1]]!=cite.get("quote")):
                return None
    return deepcopy(value)


def _tokens(text):
    return [w.casefold().replace("’","'") for w in _WORD.findall(text)]


def _anchors(text):
    """Conservative English name/number heuristic, not entity recognition."""
    return [w.casefold().replace("’","'") for w in _WORD.findall(text)
            if any(c.isdigit() for c in w)
            or (w[0].isupper() and w.casefold() not in _COMMON)]


def uncertainty_veto(row, original, candidate, glossary):
    uncertainty=current_uncertainty(row)
    if uncertainty is None:
        return None
    kinds={n["kind"] for n in uncertainty["nominations"]}
    if any(source==row["source_text"] and _tokens(target)==_tokens(candidate)
           for source,target in glossary.translation_examples):
        # Explicit operator authority for this complete source/target pair.
        return None
    if kinds&_MATERIAL or uncertainty["source_quality_status"].startswith("unresolved_wording"):
        # In materially disputed source, retain content words, order,
        # polarity, and numbers. Permit punctuation, casing and articles.
        # No claim that primary MT is correct; no invented resolution.
        if [w for w in _tokens(original) if w not in _ARTICLES]!=[
                w for w in _tokens(candidate) if w not in _ARTICLES]:
            return "unresolved_source_material_lexical_change"
    elif kinds&_ENTITY or uncertainty["scene_outcome"] in ("unresolved","partial","error"):
        # A name/number stability guard where the uncertainty is less
        # specific. Preserve grammatical/idiomatic edits around anchors.
        if _anchors(original)!=_anchors(candidate):
            return "unresolved_source_name_or_number_change"
    return None
