"""NInfer Messages protocol and owned lifecycle tests, no live models."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
import copy
import json

import pytest

from nvidia_transcribe.ninfer_messages_client import NInferMessagesTextClient
from nvidia_transcribe.resource_lifecycle import managed_text_client


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
        response_format={"type":"json_object"}, strict_reasoning_effort=True)
    url, body = requests[0]
    assert url == "http://127.0.0.1:8001/v1/messages"
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert body["output_config"] == {"effort":"xhigh"}
    assert body["max_tokens"] == 1792
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
        response_format={"type":"json_object"}, strict_reasoning_effort=True)
    assert requests[0][1]["thinking"] == {"type": "disabled"}
    assert "output_config" not in requests[0][1]
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
