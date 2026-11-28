"""Compact immutable episode context and source-cited discourse hints."""
from __future__ import annotations

import hashlib
import json
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class RecordingContext:
    """Complete dialogue where admitted; explicitly bounded subsets otherwise.

    UTF-8 bytes are a conservative input-size proxy, not tokenizer receipts.
    The caller reserves separate headroom for prompts, schemas and completion.
    """
    def __init__(self, rows, mode, max_bytes):
        if mode not in ("english", "bilingual"):
            raise ValueError("Unsupported review context representation")
        if type(max_bytes) is not int or not 1024 <= max_bytes <= 180000:
            raise ValueError("Review context budget must be 1024..180000 UTF-8 bytes")
        self.mode = mode
        self.max_bytes = max_bytes
        self.rows = []
        for i, row in enumerate(rows):
            self.rows.append({
                "id": str(row.get("utterance_id") or row.get("id") or f"utt-{i:04d}"),
                "speaker": row.get("speaker_id") or row.get("global_speaker_id"),
                "interval": [row.get("source_start"), row.get("source_end")],
                "source": str(row.get("source_text") or ""),
                "english": str(row.get("machine_translated_text") or ""),
            })
        self.identity = digest(self.rows)

    def for_targets(self, start, end):
        bilingual = self.mode == "bilingual"
        def render(i):
            row = self.rows[i]
            return [row["id"], row["speaker"], *row["interval"],
                    *([row["source"]] if bilingual else []), row["english"]]
        records = [render(i) for i in range(len(self.rows))]
        # Reserve envelope overhead, not only the encoded row data.
        available = self.max_bytes - 512
        sizes = [len(encode(row).encode()) + 1 for row in records]
        if sum(sizes) <= available:
            selected = list(range(len(records)))
        else:
            # Keep whole observations, both preceding and following. Detailed
            # targets/local evidence are supplied separately even if omitted here.
            selected = []
            for i in sorted(range(len(records)),
                            key=lambda i: (max(start-i, i-end+1, 0), i)):
                if sizes[i] <= available:
                    selected.append(i)
                    available -= sizes[i]
            selected.sort()
        return {"representation": self.mode,
                "coverage": "complete" if len(selected) == len(records) else "bounded",
                "total_occurrences": len(records), "included_occurrences": len(selected),
                "columns": ["id", "speaker", "start", "end",
                            *(["source"] if bilingual else []), "provisional_english"],
                "dialogue": [records[i] for i in selected]}


# Optional for a simple literal term; required by the prompt for entity, alias
# and referent hypotheses. These are advisory interpretations, not identity IDs.
INTERPRETATION_SCHEMA = {"type":"object","properties":{
    "kind":{"type":"string","enum":["entity","title","referent","term"]},
    "sense":{"type":"string"},"referent":{"type":"string"},
    "confidence":{"type":"string","enum":["tentative","supported"]},
    "basis":{"type":"string"},"source_alternative":{"type":"string"},
    "mentions":{"type":"array","maxItems":4,"items":{"type":"object","properties":{
        "source":{"type":"string"},"reference_id":{"type":"string"},
        "relation":{"type":"string","enum":["alias","context"]}},
        "required":["source","reference_id","relation"],"additionalProperties":False}}},
    "required":["kind","sense","referent","confidence","basis","source_alternative","mentions"],
    "additionalProperties":False}


def _bounded_text(value, maximum, *, empty=False):
    return isinstance(value,str) and len(value)<=maximum and (empty or bool(value.strip()))


def _interpretation(term, item, target, all_rows):
    info=term.get("interpretation")
    if (not isinstance(info,dict) or set(info)!=set(INTERPRETATION_SCHEMA["required"])
            or info["kind"] not in ("entity","title","referent","term")
            or info["confidence"] not in ("tentative","supported")
            or not _bounded_text(info["sense"],120)
            or not _bounded_text(info["referent"],120,empty=info["kind"]=="term")
            or not _bounded_text(info["basis"],300)
            or not _bounded_text(info["source_alternative"],80,empty=True)
            or not isinstance(info["mentions"],list) or len(info["mentions"])>4):
        return None
    anchored=set()
    for mention in info["mentions"]:
        if (not isinstance(mention,dict) or set(mention)!={"source","reference_id","relation"}
                or not _bounded_text(mention["source"],80)
                or not isinstance(mention["reference_id"],str)
                or mention["reference_id"] not in term["reference_ids"]
                or mention["relation"] not in ("alias","context")
                or mention["source"] not in all_rows[mention["reference_id"]]["source_text"]):
            return None
        anchored.add(mention["reference_id"])
    if any(term["source"] not in all_rows[ref]["source_text"] and ref not in anchored
           for ref in term["reference_ids"]):
        return None
    alternative=info["source_alternative"]
    cited=[]
    if alternative:
        # Alternative literal must be actually observed and explicitly used
        # by this occurrence's admitted supplementary proposal.
        cited=[v["caption_id"] for v in item.get("caption_observations_used",[])
               if v.get("source_language_match") is True and v.get("text_kind")=="dialogue_caption"
               and target["utterance_id"] in v.get("related_occurrence_ids",[])
               and alternative in v.get("literal_text","")]
        if not cited:
            return None
    return {"interpretation":info,"caption_evidence_ids":sorted(set(cited)),
            "evidence_status":"advisory_model_interpretation_not_identity_confirmation"}


def supported_terms(item, target, all_rows):
    """Validate literal citations; a model's accepted term is still advisory."""
    result = []
    for term in item.get("discourse_terms", []):
        if (not isinstance(term,dict) or not {"source","english","reference_ids"}<=set(term)
                or set(term)-{"source","english","reference_ids","interpretation"}):
            continue
        source, english, refs = term["source"], term["english"], term["reference_ids"]
        if (not isinstance(source, str) or not 1 <= len(source.strip()) <= 80
                or source not in target["source_text"]
                or not isinstance(english, str) or not 1 <= len(english.strip()) <= 100
                or not re.search(r"(?<!\w)"+re.escape(english.casefold())+r"(?!\w)",
                                 item["corrected_text"].casefold())
                or not isinstance(refs, list) or not 1 <= len(refs) <= 4
                or not any(ref != target["utterance_id"] for ref in refs)
                or any(not isinstance(ref,str) or ref not in all_rows for ref in refs)):
            continue
        structured={}
        if "interpretation" in term:
            structured=_interpretation(term,item,target,all_rows)
            if structured is None:
                continue
        elif any(source not in all_rows[ref]["source_text"] for ref in refs):
            continue
        result.append({"source": source, "english": english,
                       "reference_ids": sorted(set(refs)), "accepted_target_id": target["utterance_id"],
                       "status": "advisory_translation_interpretation",**structured})
    return result[:4]


def relevant_terms(ledger, targets, limit=32):
    # Never apply a global string replacement. Supply conflicting interpretations
    # as alternatives, restricted to literally present mentions.
    def applies(term,t):
        info=term.get("interpretation") or {}
        # A pronoun/referent hypothesis belongs to cited contextual occurrences;
        # do not propagate it just because the same pronoun appears elsewhere.
        if info.get("kind")=="referent":
            return t["utterance_id"] in term["reference_ids"]
        aliases=[m["source"] for m in info.get("mentions",[]) if m["relation"]=="alias"]
        return (t["utterance_id"] in term["reference_ids"] or
                any(literal in t["source_text"] for literal in [term["source"],*aliases]))
    relevant = [term for term in ledger if any(applies(term,t) for t in targets)]
    # Collapse repeated equal interpretations while preserving competing senses
    # and their exact accepted occurrences. Do not let frequent repetitions evict
    # every distinct term in the current target batch.
    grouped = {}
    for term in relevant:
        info=term.get("interpretation")
        # Collapse repeated identical semantic descriptions, not competing
        # senses. Preserve each bounded supporting record, not a guessed merge.
        key = (term["source"],term["english"].casefold(),
               tuple(info[k].strip().casefold() for k in
                     ("kind","sense","referent","source_alternative")) if info else None)
        prior = grouped.get(key)
        ids = [*(prior["accepted_target_ids"] if prior else []), term["accepted_target_id"]]
        record={**term,"accepted_target_ids":list(dict.fromkeys(ids))[-8:]}
        if info:
            context={"accepted_target_id":term["accepted_target_id"],"reference_ids":term["reference_ids"],
                     "interpretation":info,"caption_evidence_ids":term["caption_evidence_ids"]}
            contexts=[*(prior.get("supporting_contexts",[]) if prior else [])]
            if context not in contexts:
                contexts.append(context)
            record["supporting_contexts"]=contexts[-4:]
        grouped[key]=record
    return list(grouped.values())[-limit:]
