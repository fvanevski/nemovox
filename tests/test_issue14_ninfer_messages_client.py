"""NInfer Messages protocol and owned lifecycle tests, no live models."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import copy
import json

import pytest

from nvidia_transcribe.ninfer_messages_client import NInferMessagesTextClient
from nvidia_transcribe.resource_lifecycle import managed_text_client

JSON_OBJECT_SCHEMA = {"type": "json_schema", "schema": {
    "type": "object", "properties": {"decision": {"type": "string"}},
    "required": ["decision"], "additionalProperties": False,
}}


class Reply:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self.body = body if body is not None else _finished_response()
    def json(self):
        return copy.deepcopy(self.body)
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {self.body}")


def _finished_response(*, thinking=True, reason="end_turn"):
    blocks = []
    if thinking:
        blocks.append({"type": "thinking", "thinking": "Consider the contrary evidence."})
    blocks.append({"type": "text", "text": json.dumps({
        "schema": "nemovox-semantic-review-decision/v1",
        "decision": "unresolved",
        "rationale": "No independent review of the audio occurrence",
        "supporting_evidence": [], "contrary_evidence": [],
    })})
    return {"type": "message", "role": "assistant", "model": "chat",
            "content": blocks, "stop_reason": reason,
            "usage": {"output_tokens": 390 if thinking else 90,
                      "output_tokens_details": {"thinking_tokens": 240 if thinking else 0}}}


def _client_transport(monkeypatch, *, responses=None):
    import httpx
    requests = []
    supplied = list(responses or [])
    class Transport:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, json):
            requests.append((url, copy.deepcopy(json)))
            return supplied.pop(0) if supplied else Reply()
    monkeypatch.setattr(httpx, "Client", Transport)
    return requests


def _messages():
    return [{"role": "system", "content": "Be precise"},
            {"role": "user", "content": "Assess source ambiguity"}]


def test_t2_sends_real_anthropic_thinking_budget_and_normalizes_content(monkeypatch):
    requests = _client_transport(monkeypatch)
    client = NInferMessagesTextClient(base_url="http://127.0.0.1:8001/v1", model="chat")
    result = client.complete(messages=_messages(), max_tokens=1792,
        thinking_budget=1024, reasoning_effort="xhigh",
        response_format=JSON_OBJECT_SCHEMA, strict_reasoning_effort=True)
    url, body = requests[0]
    assert url == "http://127.0.0.1:8001/v1/messages"
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert body["output_config"] == {
        "effort":"xhigh",
        "format":{"type":"json_schema", "schema":JSON_OBJECT_SCHEMA["schema"]},
    }
    assert body["max_tokens"] == 1792
    assert body["system"] == "Be precise"
    assert body["messages"] == [{"role":"user","content":"Assess source ambiguity"}]
    assert result["choices"][0]["finish_reason"] == "stop"
    assert json.loads(result["choices"][0]["message"]["content"])["decision"] == "unresolved"
    assert "contrary" in result["choices"][0]["message"]["reasoning_content"]
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 240
    diag = result["_client_request_diagnostics"]
    assert diag["protocol"] == "ninfer_anthropic_messages"
    assert diag["thinking_budget_sent"] == 1024
    assert diag["json_validation_required"] is True
    assert diag["reasoning_effort_fallback"] is False
    assert len(requests) == 1


def test_t1_disables_hidden_thinking_instead_of_inheriting_server_default(monkeypatch):
    requests = _client_transport(monkeypatch, responses=[Reply(body=_finished_response(thinking=False))])
    result = NInferMessagesTextClient(base_url="http://localhost:8001/v1", model="chat").complete(
        messages=_messages(), max_tokens=512, thinking_budget=0, reasoning_effort="none",
        response_format=JSON_OBJECT_SCHEMA, strict_reasoning_effort=True)
    assert requests[0][1]["thinking"] == {"type": "disabled"}
    assert requests[0][1]["system"] == "Be precise"
    assert requests[0][1]["messages"][0]["role"] == "user"
    assert requests[0][1]["output_config"] == {
        "format":{"type":"json_schema", "schema":JSON_OBJECT_SCHEMA["schema"]},
    }
    assert result["_client_request_diagnostics"]["thinking_mode"] == "disabled"
    assert result["usage"]["completion_tokens_details"]["reasoning_tokens"] == 0


def test_messages_unsupported_or_invalid_thinking_is_not_silently_relaxed(monkeypatch):
    requests = _client_transport(monkeypatch, responses=[Reply(400, {"error":{"message":"invalid thinking budget"}})])
    client = NInferMessagesTextClient(base_url="http://localhost:8001/v1", model="chat")
    with pytest.raises(RuntimeError, match="invalid thinking budget"):
        client.complete(messages=_messages(), max_tokens=1792, thinking_budget=1024,
                        reasoning_effort="xhigh", strict_reasoning_effort=True)
    assert len(requests) == 1
    for budget,tokens,effort in ((512,1792,"xhigh"),(1024,1152,"xhigh"),
                                 (0,512,"low"),(None,512,"none")):
        with pytest.raises(ValueError):
            client.complete(messages=_messages(), max_tokens=tokens,
                            thinking_budget=budget, reasoning_effort=effort)
    assert len(requests) == 1


def test_malformed_anthropic_response_or_truncation_preserves_failure(monkeypatch):
    requests = _client_transport(monkeypatch, responses=[
        Reply(body={**_finished_response(), "content":[{"type":"tool_use","name":"unexpected"}]}),
        Reply(body=_finished_response(reason="max_tokens"))])
    client = NInferMessagesTextClient(base_url="http://localhost:8001/v1", model="chat")
    kw = {"messages":_messages(),"max_tokens":1792,"thinking_budget":1024,"reasoning_effort":"xhigh"}
    with pytest.raises(ValueError, match="Unexpected Anthropic Messages content block"):
        client.complete(**kw)
    result=client.complete(**kw)
    assert result["choices"][0]["finish_reason"]=="length"
    assert len(requests)==2


def test_lifecycle_selects_messages_only_when_explicitly_requested(monkeypatch):
    import nvidia_transcribe.ninfer_messages_client as mod
    calls=[]
    class Fake:
        def __init__(self,**kw):
            calls.append(("init",kw))
        def complete(self,**kw):
            calls.append(("complete",kw))
            return {"choices":[{"message":{"content":"{}"}}]}
    monkeypatch.setattr(mod, "NInferMessagesTextClient", Fake)
    class Mgr:
        @contextmanager
        def endpoint(self,resource):
            calls.append(("enter",resource.name))
            try: yield
            finally: calls.append(("exit",resource.name))
    resource=SimpleNamespace(name="text_local",endpoint="http://localhost:8001/v1",
            model="chat",credential=lambda:None)
    with managed_text_client(Mgr(),resource,wire_protocol="ninfer_messages") as c:
        assert calls == []
        c.complete(messages=_messages(),max_tokens=1792,thinking_budget=1024)
    assert [x[0] for x in calls]==["enter","init","complete","exit"]
    with pytest.raises(ValueError,match="Unsupported managed"):
        with managed_text_client(Mgr(),resource,wire_protocol="invalid"):
            pass


def test_messages_adapter_integrates_with_pinned_semantic_review_contract(monkeypatch):
    from nvidia_transcribe.chunking import ChunkSpec
    from nvidia_transcribe.semantic_review import (
        plan_boundary_semantic_reviews, run_configured_tiered_text_reviews,
    )
    policy = {
        "schema": "nemovox-semantic-review-policy/v1",
        "policy_id": "test-review-transport",
        "reviewer_guidance": "Prioritize polarity, clarity and evidence; abstain on unsupported audio.",
        "model_review": {
            "case_limit": 1, "token_budget": 1792, "light_tokens": 512,
            "reasoned_tokens": 1792, "light_thinking_budget": 0,
            "reasoned_thinking_budget": 1024, "light_effort": "none",
            "reasoned_effort": "xhigh",
        },
    }
    right = [{"text":t, "start":80+i*.25, "end":80.2+i*.25,
              "valid_timing":True,"unit_type":"word","speaker":"local_A",
              "_unit_idx":i,"_chunk_idx":1}
             for i,t in enumerate(("The","door","is","not","open"))]
    spec = [ChunkSpec(0,0,90,0,100),ChunkSpec(1,90,180,80,190)]
    request = plan_boundary_semantic_reviews([[],right], spec, policy=policy,
        activity_by_chunk={0:[{"start_s":80,"end_s":82,
                               "possible_unrepresented_speech":True}]})
    assert len(request["cases"])==1
    statement = {
        "schema":"nemovox-semantic-review-decision/v1",
        "request_sha256":request["request_sha256"],
        "case_id":request["cases"][0]["case_id"],
        "decision":"unresolved",
        "rationale":"The audio occurrence is not yet independently reviewed",
        "supporting_evidence":["right_context_words"],
        "contrary_evidence":[], "unresolved":["acoustic_occurrence"],
    }
    backend = _finished_response()
    backend["content"][-1]["text"] = json.dumps(statement)
    requests = _client_transport(monkeypatch, responses=[Reply(body=backend)])
    model = NInferMessagesTextClient(base_url="http://localhost:8001/v1",model="chat")
    result = run_configured_tiered_text_reviews(request,client=model,policy=policy)
    assert result["outcomes"][0]["status"]=="advisory_only"
    assert result["outcomes"][0]["assessment"]["source_application_authorized"] is False
    assert result["model_calls"]==1
    assert result["source_or_subtitles_modified"] is False
    assert requests[0][1]["thinking"]=={"type":"enabled","budget_tokens":1024}
    fmt = requests[0][1]["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"]["request_sha256"]["enum"] == [request["request_sha256"]]
    assert fmt["schema"]["properties"]["case_id"]["enum"] == [request["cases"][0]["case_id"]]
    assert set(fmt["schema"]["properties"]["supporting_evidence"]["items"]["enum"]) == {
        "right_context_words", "left_core_overlap_words", "left_lexical_correspondence",
        "unrepresented_activity_intervals", "independent_audio_review", "independent_visual_review",
    }
    assert "system" in requests[0][1]
    assert requests[0][1]["messages"][0]["role"]=="user"


def test_anthropic_system_prefix_is_preserved_in_top_level_without_reordering_history(monkeypatch):
    requests = _client_transport(monkeypatch, responses=[
        Reply(body=_finished_response(thinking=False)),
        Reply(body=_finished_response(thinking=False)),
    ])
    client = NInferMessagesTextClient(base_url="http://localhost:8001/v1", model="chat")
    client.complete(messages=[
        {"role":"system","content":"Preserve intended meaning."},
        {"role":"system","content":"Never invent dialogue."},
        {"role":"user","content":"First turn."},
        {"role":"assistant","content":"Acknowledged."},
        {"role":"user","content":"Second turn."},
    ],max_tokens=512,thinking_budget=0,reasoning_effort="none")
    body=requests[0][1]
    assert body["system"]=="Preserve intended meaning.\n\nNever invent dialogue."
    assert [m["role"] for m in body["messages"]]==["user","assistant","user"]
    assert [m["content"] for m in body["messages"]]==["First turn.","Acknowledged.","Second turn."]
    # Only the initial system prefix is hoisted: legitimate later system
    # turns stay at their original position in the message history.
    client.complete(messages=[
        {"role":"system","content":"Top-level instruction."},
        {"role":"user","content":"A question."},
        {"role":"system","content":"Scoped context."},
        {"role":"assistant","content":"A response."},
    ],max_tokens=512,thinking_budget=0,reasoning_effort="none")
    second=requests[1][1]
    assert second["system"]=="Top-level instruction."
    assert [m["role"] for m in second["messages"]]==["user","system","assistant"]


def test_system_only_messages_fail_before_wire_request(monkeypatch):
    requests=_client_transport(monkeypatch)
    client=NInferMessagesTextClient(base_url="http://localhost:8001/v1",model="chat")
    with pytest.raises(ValueError,match="at least one non-system"):
        client.complete(messages=[
            {"role":"system","content":"No user turn."}
        ],max_tokens=512,thinking_budget=0,reasoning_effort="none")
    assert requests==[]


def test_messages_json_schema_is_required_and_never_downgraded(monkeypatch):
    requests = _client_transport(monkeypatch)
    client = NInferMessagesTextClient(base_url="http://localhost:8001/v1",model="chat")
    bad = [{"type":"json_object"}, {"type":"json_schema","schema":"not a schema"},
           {"type":"json_schema","schema":{"type":"object"},"unexpected":True}]
    for fmt in bad:
        with pytest.raises(ValueError,match="explicit JSON Schema"):
            client.complete(messages=_messages(),max_tokens=512,thinking_budget=0,
                            reasoning_effort="none",response_format=fmt)
    assert requests==[]
    # With no response format explicitly requested, prior relaxed transport
    # retains its behavior; it must not CLAIM to have constrained the output.
    out=client.complete(messages=_messages(),max_tokens=512,thinking_budget=0,
                        reasoning_effort="none")
    assert "output_config" not in requests[-1][1]
    assert out["_client_request_diagnostics"]["response_format_sent"] is False
