"""Current-version native review progress, keyed decision transport and scene context.

The immutable batch input selects a progress record; its actual advisory
snapshot is stored and hashed, never recomputed when continuing that record.
Each successful stage is durably committed before the next stage starts.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import httpx

from nvidia_transcribe.recording_context import digest


SCHEMA = "native-review-progress/v1"


def retryable_review_transport(exc):
    """Shared policy for text review and supplementary acquisition transport."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in (408, 429) or 500 <= status <= 599
    return isinstance(exc, httpx.TransportError)


class ReviewProgress:
    def __init__(self, path, resume):
        self.path = Path(path) if path else None
        self.records = {}
        if resume and self.path and self.path.is_file():
            document = json.loads(self.path.read_text(encoding="utf-8"))
            # Clean-forward: do not silently promote old batch-cache records.
            if document.get("schema") == SCHEMA:
                records = document.get("records")
                if not isinstance(records,dict):
                    raise ValueError("Invalid native review progress envelope")
                for key, wrapped in records.items():
                    if (not isinstance(wrapped,dict) or set(wrapped)!={"record","sha256"}
                            or digest(wrapped["record"])!=wrapped["sha256"]):
                        raise ValueError("Native review progress digest mismatch")
                    self.records[key]=wrapped["record"]

    def admit(self, base, terms, scenes):
        key=digest(base)
        record=self.records.get(key)
        if record is None:
            snapshot={"discourse_terms":deepcopy(terms),"scene_observations":deepcopy(scenes)}
            record={"base_sha256":key,"snapshot":snapshot,
                    "context_sha256":digest({"base":base,"snapshot":snapshot}),
                    "entries":None,"requests":{},"split":False}
            self.records[key]=record
        elif (record.get("base_sha256")!=key
                or record.get("context_sha256")!=digest({"base":base,"snapshot":record["snapshot"]})):
            raise ValueError("Native review snapshot identity mismatch")
        return record

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True,exist_ok=True)
        document={"schema":SCHEMA,"records":{k:{"record":v,"sha256":digest(v)}
                                               for k,v in self.records.items()}}
        tmp=self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(document,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
        tmp.replace(self.path)

    def entries(self, record, entries):
        record["entries"]=deepcopy(entries)
        self.save()

    def request(self, record, stage, messages, schema, response):
        # Store the exact inference context, including local aliases, not
        # just the latest mutable advisory ledger. A response digest is an
        # integrity receipt, not an independent semantic judgment.
        body={"stage":stage,"messages":messages,"schema":schema,
              "response":response.get("choices",[]),
              "diagnostics":response.get("_client_request_diagnostics",{}),
              "transport_error":response.get("transport_error")}
        key=digest(body)
        record["requests"][key]=deepcopy(body)
        return key


def scene_projection(ledger, targets, limit=8):
    """Nearby observed scene facts are advisory, not proof of scene continuity."""
    selected=[]
    for obs in ledger:
        interval=obs.get("requested_seek_interval")
        if not isinstance(interval,list) or len(interval)!=2:
            continue
        if any(t["utterance_id"] in obs.get("related_occurrence_ids",[]) or
               (isinstance(t.get("start"),(int,float)) and isinstance(t.get("end"),(int,float))
                and t["start"]<=interval[1]+8 and t["end"]>=interval[0]-8) for t in targets):
            if obs not in selected:
                selected.append(obs)
    return selected[-limit:]


def observed_scenes(receipt):
    if receipt.get("status")=="observed" and receipt.get("kind")=="visual":
        return [x for x in receipt.get("observations",[]) if x.get("observation_id")]
    return []


def visual_digest(receipt):
    return digest({k:receipt.get(k) for k in ("frames","observations","visible_text","focus")})


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key:"+key)
        result[key]=value
    return result


def keyed_schema(ids):
    aliases={f"d{i+1:02d}":uid for i,uid in enumerate(ids)}
    value={"type":"object","properties":{"choice":{"type":"string","enum":["original","proposal"]},
                                         "reason":{"type":"string"}},
           "required":["choice","reason"],"additionalProperties":False}
    schema={"type":"object","properties":{"decisions":{"type":"object",
                "properties":{k:value for k in aliases},"required":list(aliases),
                "additionalProperties":False}},"required":["decisions"],"additionalProperties":False}
    return aliases,schema


def keyed_decisions(reply, aliases):
    choice=(reply.get("choices") or [{}])[0]
    if choice.get("finish_reason")!="stop":
        raise ValueError("incomplete_keyed_comparison")
    parsed=json.loads(choice["message"]["content"],object_pairs_hook=_unique)
    if not isinstance(parsed,dict) or set(parsed)!={"decisions"}:
        raise ValueError("invalid_keyed_comparison_envelope")
    values=parsed["decisions"]
    if not isinstance(values,dict) or set(values)!=set(aliases):
        raise ValueError("invalid_keyed_comparison_ids")
    result={}
    for alias,item in values.items():
        if (not isinstance(item,dict) or set(item)!={"choice","reason"}
                or item["choice"] not in ("original","proposal")
                or not isinstance(item["reason"],str) or not item["reason"].strip()):
            raise ValueError("invalid_keyed_comparison_value")
        uid=aliases[alias]
        result[uid]={"utterance_id":uid,**item}
    return result


def comparison_messages(messages, ids, *, recovery=False):
    """Keep detailed source observations once; short aliases own each decision."""
    alternate=[dict(x) for x in messages]
    data=json.loads(alternate[-1]["content"])
    aliases,schema=keyed_schema(ids)
    by_id={v:k for k,v in aliases.items()}
    # Exact detailed rows once, including connected alternatives and source
    # uncertainty. Other rows are read-only; no redundant joined strings.
    decisions=[]
    owned=set()
    for item in data["comparisons"]:
        uid=item["utterance_id"]
        if uid in by_id:
            decisions.append({"key":by_id[uid],"target_ids":[uid],"atomic":False})
            owned.add(uid)
    for passage in data["passage_checks"]:
        uid=passage["utterance_id"]
        if uid in by_id:
            decisions.append({"key":by_id[uid],"target_ids":passage["target_ids"],"atomic":True})
            owned.update(passage["target_ids"])
    if {x["key"] for x in decisions}!=set(aliases):
        raise ValueError("Missing comparison ownership")
    data["comparisons"]=decisions
    del data["passage_checks"]
    data["caption_interpretations"]=[x for x in data["caption_interpretations"] if x["id"] in owned]
    cited={v for x in data["caption_interpretations"] for v in x["caption_ids"]}
    data["supplementary_observations"]=[x for x in data["supplementary_observations"]
        if x.get("target_id") in owned or any(v["caption_id"] in cited for v in x.get("visible_text",[]))]
    # Strip transport/usage metadata, retain exact observation content and
    # identities. Full receipts remain in progress and output.
    data["supplementary_observations"]=[{k:v for k,v in x.items() if k not in
        ("model_receipt","provider")} for x in data["supplementary_observations"]]
    alternate[-1]["content"]=json.dumps(data,ensure_ascii=False,separators=(",",":"))
    alternate[0]["content"]=alternate[0]["content"].replace(
        "Return one decision per supplied row AND passage ID.","").replace(
        "For each passage_checks ID","For each atomic comparison key").replace(
        "Inspect source_occurrences separately","Inspect constituent complete_target_passage rows separately").replace(
        "Return only supplied verdict IDs.","").replace(
        "For an atomic_connected_allocation passage","For a comparison with atomic=true")
    alternate[0]["content"]+=(
        " RESPONSE CONTRACT: comparisons contains the only requested short keys. "
        "Use complete_target_passage rows in target_ids order to read exact source, primary and proposal; "
        "atomic=true means evaluate the entire allocation as one decision. Other rows are read-only. "
        "Return decisions as an OBJECT with exactly the supplied short keys, not an array, no utterance_id "
        "fields or canonical passage hashes. Each value contains choice and a concise specific reason. "
        "Avoid repeated whitespace and long reasoning in the final answer."
        + (" This is a bounded recovery subset; judge only its keys." if recovery else ""))
    return alternate,aliases,schema
