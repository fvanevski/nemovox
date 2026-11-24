"""QC advisory -> original raw-unit review queue: no source/output promotion."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from nvidia_transcribe.cli import main
from nvidia_transcribe.evidence_fusion import _digest
from nvidia_transcribe.semantic_review import (
    plan_boundary_semantic_reviews, validate_semantic_review,
)
from nvidia_transcribe.semantic_review_followups import (
    build_meaning_first_review_queue, write_meaning_first_review_queue,
)


class Chunk:
    def __init__(self,index,core_start,core_end,infer_start,infer_end):
        self.chunk_index=index
        self.core_start_s=core_start
        self.core_end_s=core_end
        self.infer_start_s=infer_start
        self.infer_end_s=infer_end


def fixture():
    policy={
        "schema":"nemovox-semantic-review-policy/v1",
        "policy_id":"qc-queue-test-v1",
        "reviewer_guidance":"Meaning and contextual correctness before literal words.",
        "model_review":{"case_limit":1,"token_budget":512,
                        "light_tokens":512,"reasoned_tokens":512,
                        "light_effort":"none","reasoned_effort":"none",
                        "light_thinking_budget":0,"reasoned_thinking_budget":0},
    }
    chunks=[Chunk(0,0,90,0,100),Chunk(1,90,180,80,190)]
    right=[{"text":t,"start":80.0+i*.30,"end":80.2+i*.30,
            "valid_timing":True,"unit_type":"word","speaker":"local_0",
            "local_speaker_id":"local_0"}
           for i,t in enumerate(("The","door","was","not","open"))]
    tagged=[[ ],[dict(unit, _drop=False, _in_core=False, _chunk_idx=1,
                       _unit_idx=i) for i,unit in enumerate(right)]]
    request=plan_boundary_semantic_reviews(tagged,chunks,policy=policy)
    assert len(request["cases"])==1
    case=request["cases"][0]
    model={"schema":"nemovox-semantic-review-decision/v1",
           "request_sha256":request["request_sha256"],"case_id":case["case_id"],
           "decision":"unresolved","rationale":"Cannot attest speech without listening.",
           "supporting_evidence":["right_context_words"],"contrary_evidence":[],
           "unresolved":["audio_source"]}
    advisory=validate_semantic_review(request,model)
    batch={"schema":"nemovox-tiered-semantic-review-batch/v1",
           "request_sha256":request["request_sha256"],
           "model_review_execution_sha256":_digest(policy["model_review"]),
           "model_calls":1,"source_or_subtitles_modified":False,
           "outcomes":[{"case_id":case["case_id"],"status":"advisory_only",
                        "assessment":advisory}]}
    media="a"*64
    record={"source_language":"fr","source":{"duration_s":1200.0,
             "fingerprint_sha256":media},"utterances":[]}
    parent={"id":"asr_1","chunk_index":1,
            "raw_response_sha256":"b"*64,"infer_start_s":80.0,"infer_end_s":190.0}
    other={"id":"asr_0","chunk_index":0,
           "raw_response_sha256":"c"*64,"infer_start_s":0.0,"infer_end_s":100.0}
    ledger={"schema_version":"nemovox-source-evidence/v1",
            "media_fingerprint_sha256":media,"source_language":"fr",
            "asr_observations":[other,parent],
            "lexical_observations":[{
                "id":"unit_"+str(i),"chunk_index":1,"source_ordinal":i,
                "asr_observation_id":"asr_1","text":word["text"],
                "local_speaker_id":"local_0"} for i,word in enumerate(right)]}
    return request,batch,policy,record,ledger


def test_review_queue_binds_original_source_units_and_audio_crops_without_promotion():
    inputs=fixture()
    pristine=deepcopy(inputs)
    out=build_meaning_first_review_queue(*inputs)
    assert out["schema"]=="nemovox-meaning-first-review-queue/v1"
    assert out["advisories"]==1
    assert out["queue_sha256"]==_digest({k:v for k,v in out.items() if k!="queue_sha256"})
    entry=out["items"][0]
    assert entry["model_decision"]=="unresolved"
    assert entry["next_action"]=="audio_review_missing_speech_and_competing_hypotheses"
    assert entry["original_media_listen_window_s"]==[76.0,85.4]
    assert entry["source_unit_ids"]==["unit_0","unit_1","unit_2","unit_3","unit_4"]
    assert all(x["parent_raw_response_sha256"]=="b"*64
               for x in entry["source_unit_evidence"]["right_context_words"])
    assert entry["source_application_authorized"] is False
    assert entry["english_postedit_authorized"] is False
    assert out["source_or_translation_or_subtitles_mutated"] is False
    assert inputs==pristine


@pytest.mark.parametrize("mutation,match",[
    ("policy_changed","stale QC policy"),
    ("request_changed","QC request identity mismatch"),
    ("batch_changed","Review batch does not match"),
    ("batch_applying","Review batch does not match"),
    ("media_changed","media/source evidence mismatch"),
    ("wrong_word","candidate text diverges"),
    ("wrong_ordinal","no matching original lexical"),
    ("tamper_advisory","not a source-bound"),
    ("duplicate_outcome","Duplicate or unknown QC outcome"),
    ("fake_audio","Unbacked audio/visual citation"),
])
def test_untrusted_tampering_cannot_become_actionable_evidence(mutation,match):
    items=list(deepcopy(fixture()))
    request,batch,policy,record,ledger=items
    if mutation=="policy_changed":
        policy["policy_id"]="new_version"
    elif mutation=="request_changed":
        request["cases"][0]["context_window_s"]=[80,89]
    elif mutation=="batch_changed":
        batch["request_sha256"]="f"*64
    elif mutation=="batch_applying":
        batch["source_or_subtitles_modified"]=True
    elif mutation=="media_changed":
        ledger["media_fingerprint_sha256"]="0"*64
    elif mutation=="wrong_word":
        ledger["lexical_observations"][0]["text"]="UNTRUSTED"
    elif mutation=="wrong_ordinal":
        ledger["lexical_observations"][0]["source_ordinal"]=100
    elif mutation=="tamper_advisory":
        batch["outcomes"][0]["assessment"]["source_application_authorized"]=True
    elif mutation=="duplicate_outcome":
        batch["outcomes"].append(deepcopy(batch["outcomes"][0]))
    elif mutation=="fake_audio":
        batch["outcomes"][0]["assessment"]["supporting_evidence"]=["independent_audio_review"]
    with pytest.raises(ValueError,match=match):
        build_meaning_first_review_queue(request,batch,policy,record,ledger)


def test_private_cli_creates_new_restricted_file_without_gpu_or_overwrite(tmp_path,capsys):
    request,batch,policy,record,ledger=fixture()
    input_names=["request.json","batch.json","policy.json","record.json","ledger.json"]
    for name,value in zip(input_names,(request,batch,policy,record,ledger)):
        (tmp_path/name).write_text(json.dumps(value),encoding="utf-8")
    out=tmp_path/"review-queue.json"
    args=["prepare-semantic-review-queue",
          "--request",str(tmp_path/"request.json"),"--batch",str(tmp_path/"batch.json"),
          "--policy",str(tmp_path/"policy.json"),"--original-recording",str(tmp_path/"record.json"),
          "--source-evidence",str(tmp_path/"ledger.json"),"--output-json",str(out)]
    assert main(args)==0
    report=json.loads(capsys.readouterr().out)
    assert report["status"]=="private_review_queue_created"
    assert report["source_or_subtitles_modified"] is False
    assert out.exists()
    assert json.loads(out.read_text())["advisories"]==1
    assert out.stat().st_mode & 0o077==0
    frozen=out.read_bytes()
    assert main(args)==1
    assert "ERROR" in capsys.readouterr().err
    assert out.read_bytes()==frozen
