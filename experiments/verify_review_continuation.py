"""Inspect current-version native continuation, not only process exit or cache count."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nvidia_transcribe import connected_units
from nvidia_transcribe.recording_context import digest
from nvidia_transcribe.review_execution import SCHEMA


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def records(path):
    doc=read(path)
    if doc.get("schema")!=SCHEMA:
        raise ValueError("Unexpected current progress schema")
    result={}
    for key,wrapper in doc["records"].items():
        record=wrapper["record"]
        if digest(record)!=wrapper["sha256"] or record["base_sha256"]!=key:
            raise ValueError("Progress record identity mismatch")
        for ref,receipt in record["requests"].items():
            if digest(receipt)!=ref:
                raise ValueError("Stage request identity mismatch")
        result[key]=record
    return result


def pending_ids(contract):
    review=contract["translation_review"]
    pending=set(review.get("deferred_comparison_ids",[]))
    pending.update(x["target_id"] for x in review.get("supplementary_proposal_errors",[]))
    rows=contract["utterances"]
    for group in connected_units.groups(rows):
        if any(rows[i]["id"] in pending for i in group):
            pending.update(rows[i]["id"] for i in group)
    return pending


def audit(fresh,resumed,first,second):
    errors=[]
    before,after=fresh["utterances"],resumed["utterances"]
    pending=pending_ids(fresh)
    new_pending=pending_ids(resumed)
    if [r["id"] for r in before]!=[r["id"] for r in after]:
        raise ValueError("Occurrence order/coverage changed")
    connected_units.validate_selected_units(before,require_complete=True)
    connected_units.validate_selected_units(after,require_complete=True)
    for a,b in zip(before,after):
        for key in ("id","source_text","speaker_id","source_start","source_end",
                    "machine_translated_text","timestamp_provenance"):
            if a.get(key)!=b.get(key):
                errors.append("canonical_or_primary_changed:"+a["id"]+":"+key)
    changed=[a["id"] for a,b in zip(before,after) if a!=b]
    unexpected=[uid for uid in changed if uid not in pending]
    errors.extend("completed_row_changed:"+uid for uid in unexpected)
    errors.extend("new_pending:"+uid for uid in sorted(new_pending-pending))
    added=[]
    for key,old in first.items():
        new=second.get(key)
        if new is None:
            errors.append("lost_progress_record:"+key)
            continue
        if new["snapshot"]!=old["snapshot"] or new["context_sha256"]!=old["context_sha256"]:
            errors.append("changed_snapshot:"+key)
        for ref,req in old["requests"].items():
            if new["requests"].get(ref)!=req:
                errors.append("lost_or_changed_request:"+ref)
        for uid,item in (old["entries"] or {}).items():
            obs=item.get("supplementary_observation")
            if isinstance(obs,dict) and obs.get("status")=="observed":
                if (new["entries"] or {}).get(uid,{}).get("supplementary_observation")!=obs:
                    errors.append("lost_or_replaced_acquisition:"+uid)
            if uid not in pending and (new["entries"] or {}).get(uid)!=item:
                errors.append("completed_stage_changed:"+uid)
        for ref,req in new["requests"].items():
            if ref in old["requests"]:
                continue
            data=json.loads(req["messages"][-1]["content"])
            stage=req["stage"]
            owned=([uid for d in data["comparisons"] for uid in d["target_ids"]]
                   if stage in ("comparison","comparison_recovery") else
                   [t["utterance_id"] for t in data.get("target_utterances",[])])
            added.append({"stage":stage,"sha256":ref,"owned_ids":owned})
            if stage not in ("comparison","comparison_recovery","evidence_proposal") or not set(owned)<=pending:
                errors.append("unexpected_inference:"+ref)
    if set(second)-set(first):
        errors.append("new_batch_progress_records")
    review=resumed["translation_review"]
    if review.get("cache_hits",0)<len(before)-len(pending):
        errors.append("completed_cache_coverage_shortfall")
    if not pending and (review.get("api_calls") or review.get("review_execution",{}).get("provider_requests")):
        errors.append("inference_on_complete_recording")
    return {"status":"PASS" if not errors else "FAIL","errors":errors,
            "occurrences":len(before),"original_pending_unit_ids":sorted(pending),
            "remaining_pending_unit_ids":sorted(new_pending),"changed_row_ids":changed,
            "unexpected_changed_row_ids":unexpected,"new_requests":added,
            "review_execution":review.get("review_execution"),"cache_hits":review.get("cache_hits"),
            "api_calls":review.get("api_calls")}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fresh-contract",required=True)
    p.add_argument("--resumed-contract",required=True)
    p.add_argument("--fresh-progress",required=True)
    p.add_argument("--resumed-progress",required=True)
    args=p.parse_args()
    result=audit(read(args.fresh_contract),read(args.resumed_contract),
                 records(args.fresh_progress),records(args.resumed_progress))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result["status"]=="PASS" else 1)


if __name__=="__main__":
    main()
