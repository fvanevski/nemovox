"""Downstream constraints on unresolved current-source evidence.

Projection is after dedicated MT: no new source interpretation or MT cache
identity, no automatic name mapping. Reports remain model hypotheses.
"""
from __future__ import annotations

from copy import deepcopy
import re

from nvidia_transcribe.semantic_context import row_nominations, text_digest, localized_wording_evidence

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
        timing_only=status=="event_timing_recovered_original_wording"
        if not cited and not outcome and not status.startswith("unresolved_wording") and not timing_only:
            continue
        row["review_source_uncertainty"]={"schema":SCHEMA,"source_id":uid,
            "source_text_sha256":text_digest(row["source_text"]),
            "source_unit_ids":deepcopy(row.get("source_unit_ids",[])),
            "nominations":cited,"scene_outcome":outcome,
            "source_quality_status":status,"source_truth_accepted":False,
            **({"wording_status":row.get("wording_status"),"timing_only_recovery":localized_wording_evidence(row) is not None}
               if timing_only else {})}
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


# Closed grammatical classes, not an episode-specific entity dictionary.
# Retain possessives on actual names (e.g. Nadia's) and all numeric tokens.
_FUNCTION_CONTRACTIONS = re.compile(
    r"(?:i|you|he|she|it|we|they|that|this|there|here|what|who|where|when|why|how)"
    r"'(?:m|re|s|ve|d|ll)\Z"
)
_PREPOSITIONS = {"in", "on", "at", "to", "from", "for", "of", "by", "with", "as"}


def _anchor_candidate(word):
    folded = word.casefold().replace("’", "'")
    if any(c.isdigit() for c in word):
        return True
    return (word[0].isupper() and folded not in _COMMON
            and folded not in _PREPOSITIONS
            and _FUNCTION_CONTRACTIONS.fullmatch(folded) is None)


def _anchors_preserved(original, candidate):
    """Compare lexical occurrences of potential names/numbers across both texts.

    Capitalization is a nomination, not an identity: once a token is nominated
    on either side, compare its case-folded occurrences in both texts. This
    preserves count and order, including sentence-initial names and numbers,
    without mistaking a case-only change at a cue boundary for substitution.
    This deliberately conservative heuristic is not general entity recognition.
    """
    before = _WORD.findall(original)
    after = _WORD.findall(candidate)
    nominated = {w.casefold().replace("’", "'")
                 for w in before + after if _anchor_candidate(w)}
    def sequence(words):
        return [w for w in _tokens(" ".join(words)) if w in nominated]
    return sequence(before) == sequence(after)


_MONTHS=("January","February","March","April","May","June","July","August","September","October","November","December")
_MONTH_WORDS=("one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve")
_ZH_MONTHS={"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10,"十一":11,"十二":12}


def _relative_month_anchor(row, original, candidate):
    """Do not treat an erroneous English calendar month as a protected name.

    Only the explicit future relative-duration construction, independently
    supported in the current source span, permits this narrow normalization.
    The comparator still judges the actual unmodified translation strings.
    """
    from nvidia_transcribe.source_quality import _lex, _contains
    evidence=localized_wording_evidence(row)
    if evidence is None:
        return original,candidate
    matches=list(re.finditer(r"(?:将于|将在|会在)([一二两三四五六七八九十]{1,2})(?:个)?月后",row["source_text"]))
    if len(matches)!=1:
        return original,candidate
    n=_ZH_MONTHS.get(matches[0].group(1))
    if n is None:
        return original,candidate
    phrase=_lex(matches[0].group())
    support={a["backend"] for a in evidence["alternatives"] if _contains(tuple(a["tokens"]),phrase)}
    if support!={"nemotron","independent_asr"}:
        return original,candidate
    numeral=rf"(?:{n}|{_MONTH_WORDS[n-1]}"+(r"|a" if n==1 else "")+")"
    duration=numeral+(r"\s+month" if n==1 else r"\s+months")
    before=re.compile(r"\bin\s+"+_MONTHS[n-1]+r"\b",re.I)
    after=re.compile(r"\b(?:in\s+"+duration+r"|"+duration+r"\s+later|after\s+"+duration+r")\b",re.I)
    if len(before.findall(original))!=1 or len(after.findall(candidate))!=1:
        return original,candidate
    return before.sub(" relative_duration ",original),after.sub(" relative_duration ",candidate)


def uncertainty_veto(row, original, candidate, glossary):
    uncertainty=current_uncertainty(row)
    if uncertainty is None:
        return None
    kinds={n["kind"] for n in uncertainty["nominations"]}
    if any(source==row["source_text"] and _tokens(target)==_tokens(candidate)
           for source,target in glossary.translation_examples):
        # Explicit operator authority for this complete source/target pair.
        return None
    timing_only=uncertainty["source_quality_status"]=="event_timing_recovered_original_wording"
    valid_timing_only=(timing_only and uncertainty.get("timing_only_recovery") is True
                       and localized_wording_evidence(row) is not None)
    if (kinds&_MATERIAL or uncertainty["source_quality_status"].startswith("unresolved_wording")
            or (timing_only and not valid_timing_only)):
        # In materially disputed source, retain content words, order,
        # polarity, and numbers. Permit punctuation, casing and articles.
        # No claim that primary MT is correct; no invented resolution.
        if [w for w in _tokens(original) if w not in _ARTICLES]!=[
                w for w in _tokens(candidate) if w not in _ARTICLES]:
            return "unresolved_source_material_lexical_change"
    elif (kinds&_ENTITY or uncertainty["scene_outcome"] in ("unresolved","partial","error","deferred")
          or (valid_timing_only and uncertainty.get("wording_status")=="model_variants_unresolved")):
        # A name/number stability guard where the uncertainty is less
        # specific. Preserve grammatical/idiomatic edits around anchors.
        # Recovery of timing without choosing new wording is not a
        # material source rewrite. Preserve genuinely uncertain anchors,
        # but don't freeze an evidenced relative duration as a month name.
        before,after=_relative_month_anchor(row,original,candidate) if valid_timing_only else (original,candidate)
        if not _anchors_preserved(before, after):
            return "unresolved_source_name_or_number_change"
    return None
