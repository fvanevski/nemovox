"""Private offline replay of the actual qualified native-review responses.

Run from each exact checkout with PYTHONPATH=src:. . No model or media access.
No original contract/checkpoint is modified. Never publish private payloads.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tarfile

import nvidia_transcribe
from nvidia_transcribe.recording_context import digest
from nvidia_transcribe.translation_review import (
    TranslationReviewConfig, review_translated_utterances,
)
from nvidia_transcribe.subtitle_pipeline import validate_deferred_review

ARCHIVE_SHA = "cacf2df53ad58c724a7d286ab2d165e734421b3fc85430c345480a499e4e4390"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_archive(path):
    require(path.stat().st_size == 7302520, "archive size mismatch")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == ARCHIVE_SHA, "archive digest mismatch")
    with tarfile.open(path, "r:gz") as archive:
        manifest = json.load(archive.extractfile("./MANIFEST.json"))
        inventory = {x["path"].removeprefix("./"):x for x in manifest["files"]}
        data = {}
        for phase in ("zh-fresh","zh-resumed"):
            for name in ("translated.contract.json","translation-review.checkpoint.json"):
                key = phase + "/" + name
                raw = archive.extractfile("./"+key).read()
                entry = inventory[key]
                require(len(raw)==entry["bytes"] and hashlib.sha256(raw).hexdigest()==entry["sha256"],
                        "archive member mismatch: "+key)
                data[phase,name] = json.loads(raw)
    return data


def requests(progress):
    result = {}
    require(progress["schema"]=="native-review-progress/v1","wrong progress schema")
    for key,wrapper in progress["records"].items():
        record = wrapper["record"]
        require(digest(record)==wrapper["sha256"] and record["base_sha256"]==key,
                "progress record digest mismatch")
        for ref,receipt in record["requests"].items():
            require(digest(receipt)==ref,"request digest mismatch")
            require(ref not in result,"duplicate request receipt")
            require(receipt["transport_error"] is None,"unexpected transport failure in fixture")
            result[ref]=receipt
    return result


def acquisitions(progress):
    result = {}
    for wrapper in progress["records"].values():
        for uid,item in (wrapper["record"]["entries"] or {}).items():
            obs=item.get("supplementary_observation")
            if obs and obs.get("status")=="observed":
                require(uid not in result and obs["target_id"]==uid,"duplicate acquisition")
                result[uid]=obs
    return result


class RecordedClient:
    def __init__(self, receipts):
        self.pending = {}
        self.consumed = []
        for ref,record in receipts.items():
            key=digest([record["messages"],record["schema"]])
            require(key not in self.pending,"ambiguous same-phase request")
            self.pending[key]=(ref,record)

    def complete(self, *, messages, **kwargs):
        # The exact request must match the actual archived messages and schema.
        # No fuzzy or ID-only lookup, generated answer, alternate output or API.
        key=digest([messages,kwargs["response_format"]["schema"]])
        require(key in self.pending,"unrecorded request identity: "+key)
        ref,record=self.pending.pop(key)
        self.consumed.append(ref)
        return {"choices":deepcopy(record["response"]),
                "_client_request_diagnostics":deepcopy(record["diagnostics"])}


class RecordedProvider:
    def __init__(self, identity, all_receipts, expected):
        self.identity=identity
        self.all_receipts=all_receipts
        self.pending=dict(expected)
        self.consumed=[]
        self.restored=set()

    def collect(self, uid, request):
        require(uid in self.pending,"unexpected acquisition for "+uid)
        obs=self.pending.pop(uid)
        require(obs["request"]==request and obs["provider"]==self.identity,
                "acquisition request or provider changed")
        self.consumed.append(uid)
        return deepcopy(obs)

    def restore(self, receipt):
        uid=receipt["target_id"]
        require(self.all_receipts.get(uid)==receipt,"restored acquisition differs")
        self.restored.add(uid)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--scratch",type=Path,required=True)
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args()
    require(Path(nvidia_transcribe.__file__).resolve().is_relative_to(
            (args.root/"src").resolve()),"wrong package import identity")
    args.scratch.mkdir(parents=True,exist_ok=False)
    data=load_archive(args.archive)
    fresh=data["zh-fresh","translated.contract.json"]
    resumed=data["zh-resumed","translated.contract.json"]
    old=data["zh-fresh","translation-review.checkpoint.json"]
    final=data["zh-resumed","translation-review.checkpoint.json"]
    first_requests,last_requests=requests(old),requests(final)
    first_acq,last_acq=acquisitions(old),acquisitions(final)
    require(len(first_requests)==70 and len(last_requests)==74,"recorded request count changed")
    require(len(first_acq)==15 and len(last_acq)==18,"recorded acquisition count changed")
    require(all(last_requests.get(k)==v for k,v in first_requests.items()),"original request changed")
    require(all(last_acq.get(k)==v for k,v in first_acq.items()),"original acquisition changed")
    identity=next(iter(first_acq.values()))["provider"]
    require(all(x["provider"]==identity for x in last_acq.values()),"provider mismatch")
    # Authenticated original runtime config; the complete request and base
    # identities below independently prove its fidelity. No host config read.
    policy=fresh["translation_review"]["review_policy"]
    cfg=TranslationReviewConfig(profile=fresh["translation_review"]["profile"],
        style=fresh["translation_review"]["style"],target_language=policy["target_language"],
        context_utterances_before=policy["context_before"],context_utterances_after=policy["context_after"],
        batch_size=fresh["translation_review"]["batch_size"],base_url="http://127.0.0.1:8001/v1",
        model="chat",resource_fingerprint=fresh["translation_review"]["resource_fingerprint"],
        max_tokens=policy["max_tokens"],temperature=policy["temperature"],
        reasoning_effort=policy["reasoning_effort"],wire_protocol=policy["wire_protocol"],
        thinking_budget=policy["thinking_budget"],context_mode="bilingual",
        context_max_bytes=160000,max_evidence_requests_per_batch=3)
    cp=args.scratch/"progress.json"
    pending_requests={k:v for k,v in last_requests.items() if k not in first_requests}
    pending_acq={k:v for k,v in last_acq.items() if k not in first_acq}
    results=[]
    # Same archived source+primary inputs (the reviewer itself resets prior
    # review fields). Riva status metadata is preserved, not normalized away.
    for phase,contract,expected_progress,recorded,expected_acq,resume in (
        ("fresh",fresh,old,first_requests,first_acq,False),
        ("resume",resumed,final,pending_requests,pending_acq,True),
        ("complete_replay",resumed,final,{}, {},True),
    ):
        model=RecordedClient(recorded)
        provider=RecordedProvider(identity,last_acq,expected_acq)
        rows,summary=review_translated_utterances(deepcopy(contract["utterances"]),
            source_language=contract["source_language"],config=cfg,text_client=model,
            checkpoint_path=cp,resume=resume,evidence_provider=provider)
        require(not model.pending and not provider.pending,"unused recorded inputs: "+phase)
        require(not summary["errors"],"review errors during "+phase)
        require(len(rows)==329,"row count mismatch")
        changed=[a["id"] for a,b in zip(contract["utterances"],rows) if a!=b]
        require(not changed,"unexpected row differences in "+phase+": "+",".join(changed))
        actual=json.loads(cp.read_text())
        require(actual==expected_progress,"checkpoint differs from recorded "+phase)
        require(summary["status"]==contract["translation_review"]["status"],"status differs")
        validate_deferred_review(rows,summary,contract["translation"])
        require(summary["api_calls"]==len(recorded),"text call count differs")
        require(summary["review_execution"]["provider_requests"]==len(expected_acq),
                "acquisition count differs")
        expected_hits={"fresh":0,"resume":325,"complete_replay":329}[phase]
        require(summary["cache_hits"]==expected_hits,"cache hit count differs")
        for key in ("reviewed","changed","deferred_comparison_ids","supplementary_proposal_errors",
                    "discourse_terms","evidence_assessments","scene_observations",
                    "recording_context_receipts","supplementary_evidence"):
            require(summary[key]==contract["translation_review"][key],"review receipt differs: "+key)
        results.append({"phase":phase,"status":summary["status"],"row_count":len(rows),
            "row_sha256":digest(rows),"checkpoint_sha256":digest(actual),
            "request_count":len(model.consumed),"acquisition_count":len(provider.consumed),
            "restored_count":len(provider.restored),"cache_hits":summary["cache_hits"]})
    print(json.dumps({"status":"PASS","import":str(nvidia_transcribe.__file__),
        "archive_sha256":ARCHIVE_SHA,"phases":results},indent=2))


if __name__=="__main__":
    main()
