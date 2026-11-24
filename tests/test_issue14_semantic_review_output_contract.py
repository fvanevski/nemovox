"""Review JSON output contract: no live inference or source modification."""
from __future__ import annotations
import json
from dataclasses import dataclass
import pytest
from nvidia_transcribe.semantic_review import (
    build_semantic_review_messages, plan_boundary_semantic_reviews,
    run_configured_tiered_text_reviews, semantic_review_json_schema,
)

@dataclass
class Chunk:
    chunk_index: int
    core_start_s: float
    core_end_s: float
    infer_start_s: float
    infer_end_s: float

POLICY={
    "schema":"nemovox-semantic-review-policy/v1",
    "policy_id":"meaning-first-wire-format-regression",
    "reviewer_guidance":"Preserve intended meaning, polarity, readability, and uncertainty.",
    "model_review":{"case_limit":1,"token_budget":1792,
        "light_tokens":512,"reasoned_tokens":1792,
        "light_thinking_budget":0,"reasoned_thinking_budget":1024,
        "light_effort":"none","reasoned_effort":"xhigh"},
}

def case_request():
    right=[
        {"text":word,"start":355+i*.28,"end":355.18+i*.28,
         "valid_timing":True,"unit_type":"word","_unit_idx":i,
         "_chunk_idx":1,"speaker":"right_local"}
        for i,word in enumerate(("C'est","pas","notre","nacelle","qui","aurait","pensé","à","ça"))
    ]
    req=plan_boundary_semantic_reviews(
        [[],right],[Chunk(0,270,360,260,370),Chunk(1,360,450,350,460)],
        policy=POLICY,activity_by_chunk={0:[{"start_s":355,"end_s":359,
                                            "possible_unrepresented_speech":True}]},
    )
    assert len(req["cases"])==1
    return req

def decision(req):
    return {"schema":"nemovox-semantic-review-decision/v1",
            "request_sha256":req["request_sha256"],
            "case_id":req["cases"][0]["case_id"],
            "decision":"possible_duplicate",
            "rationale":"The source phrase preserves negation; the earlier displaced ASR may be the same event, but the audio is unverified.",
            "supporting_evidence":["right_context_words","left_lexical_correspondence"],
            "contrary_evidence":["unrepresented_activity_intervals"],
            "unresolved":["original_audio_same_occurrence"]}

class Client:
    def __init__(self, content):
        self.content=content;self.calls=[]
    def complete(self,**kwargs):
        self.calls.append(kwargs)
        return {"choices":[{"finish_reason":"stop","message":{"content":self.content}}],
                "usage":{"completion_tokens":410,
                         "completion_tokens_details":{"reasoning_tokens":120}}}

def test_prompt_and_schema_have_identical_evidence_vocabulary():
    req=case_request()
    schema=semantic_review_json_schema(req,req["cases"][0])
    assert schema["properties"]["request_sha256"]["enum"]==[req["request_sha256"]]
    assert schema["properties"]["case_id"]["enum"]==[req["cases"][0]["case_id"]]
    assert set(schema["required"])=={
        "schema","request_sha256","case_id","decision","rationale",
        "supporting_evidence","contrary_evidence","unresolved"}
    keys=schema["properties"]["supporting_evidence"]["items"]["enum"]
    assert keys==schema["properties"]["contrary_evidence"]["items"]["enum"]
    assert len(keys)==6
    prompt=build_semantic_review_messages(req,req["cases"][0])[0]["content"]
    assert "FIELD NAME KEYS" in prompt and "NO markdown fences" in prompt
    for key in keys:assert key in prompt

def test_constrained_json_is_a_valid_advisory_never_source_authority():
    req=case_request();stub=Client(json.dumps(decision(req)))
    out=run_configured_tiered_text_reviews(req,client=stub,policy=POLICY)
    assert out["outcomes"][0]["status"]=="advisory_only"
    assert out["outcomes"][0]["assessment"]["decision"]=="possible_duplicate"
    assert out["outcomes"][0]["assessment"]["source_application_authorized"] is False
    fmt=stub.calls[0]["response_format"]
    assert fmt["type"]=="json_schema"
    assert fmt["schema"]["properties"]["schema"]["enum"]==["nemovox-semantic-review-decision/v1"]
    assert stub.calls[0]["thinking_budget"]==1024
    assert out["source_or_subtitles_modified"] is False

def test_exact_json_fence_without_surrounding_prose_is_tolerated():
    req=case_request();fence=chr(96)*3
    raw=fence+"json\n"+json.dumps(decision(req))+"\n"+fence
    out=run_configured_tiered_text_reviews(req,client=Client(raw),policy=POLICY)
    assert out["outcomes"][0]["status"]=="advisory_only"

@pytest.mark.parametrize("mutation,reason",[
    ("missing_schema","Wrong semantic review schema"),
    ("prose_citation","Invalid contrary evidence citations"),
    ("wrong_identity","Stale quality-review request"),
    ("unsupported_key","Invalid reviewed evidence citations"),
    ("stray_prose","Expecting value"),
])
def test_observed_output_failures_still_fail_closed(mutation,reason):
    req=case_request();d=decision(req)
    if mutation=="missing_schema":d.pop("schema")
    if mutation=="prose_citation":d["contrary_evidence"]=["This phrase seems displaced by 18 seconds."]
    if mutation=="wrong_identity":d["request_sha256"]="0"*64
    if mutation=="unsupported_key":d["supporting_evidence"]=["audio_verified_by_model"]
    raw=json.dumps(d)
    if mutation=="stray_prose":
        f=chr(96)*3;raw="Probably this:\n"+f+"json\n"+raw+"\n"+f
    out=run_configured_tiered_text_reviews(req,client=Client(raw),policy=POLICY)
    assert out["outcomes"][0]["status"]=="unaccepted_model_response"
    assert reason in out["outcomes"][0]["error"]

def test_truncated_completed_json_is_not_accepted():
    req=case_request()
    class Truncated(Client):
        def complete(self,**kwargs):
            r=super().complete(**kwargs);r["choices"][0]["finish_reason"]="length";return r
    out=run_configured_tiered_text_reviews(
        req,client=Truncated(json.dumps(decision(req))),policy=POLICY)
    assert out["outcomes"][0]["status"]=="unaccepted_model_response"
    assert "output limit" in out["outcomes"][0]["error"]
