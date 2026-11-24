"""NInfer-compatible, request-scoped thinking/output contract without a server."""
from __future__ import annotations

import copy
import json
import pytest

from nvidia_transcribe.refinement import OpenAIChatTextClient


class Reply:
    def __init__(self, status, content):
        self.status_code = status
        self.content = content
        self.text = json.dumps(content)
    def json(self):
        return copy.deepcopy(self.content)
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text)


def test_chat_completions_refuses_unenforceable_thinking_budget(monkeypatch):
    import httpx
    requests=[]
    class Transport:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def post(self,url,json):
            requests.append(copy.deepcopy(json))
            return Reply(200, {"choices":[{"finish_reason":"stop","message":{"content":"{}"}}],
                               "usage":{"completion_tokens":121,
                                        "completion_tokens_details":{"reasoning_tokens":48}}})
    monkeypatch.setattr(httpx,"Client",Transport)
    client=OpenAIChatTextClient(base_url="http://localhost:8001/v1",model="chat")
    for effort,budget in [("low",128),("xhigh",512)]:
        with pytest.raises(ValueError,match="cannot enforce thinking_budget"):
            client.complete(messages=[{"role":"user","content":"review"}],
                    max_tokens=1280,response_format={"type":"json_object"},
                    reasoning_effort=effort,thinking_budget=budget,
                    strict_reasoning_effort=True)
    assert requests==[]
    # The earlier unbounded field is no longer advertised as a real cap.
    out=client.complete(messages=[{"role":"user","content":"review"}],
            max_tokens=1280,response_format={"type":"json_object"},
            reasoning_effort="low",strict_reasoning_effort=True)
    assert out["_client_request_diagnostics"]["thinking_budget_sent"] is None
    assert requests[0]["reasoning_effort"]=="low"


def test_rejected_effort_does_not_latch_all_efforts_or_strip_strict_profile(monkeypatch):
    import httpx
    requests=[]
    class Transport:
        def __init__(self,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def post(self,url,json):
            requests.append(copy.deepcopy(json))
            if json.get("reasoning_effort")=="high":
                return Reply(400, {"error":{"message":"invalid prompt | messages 2"}})
            return Reply(200,{"choices":[{"message":{"content":"{}"}}]})
    monkeypatch.setattr(httpx,"Client",Transport)
    client=OpenAIChatTextClient(base_url="http://localhost:8001/v1",model="chat")
    with pytest.raises(RuntimeError,match="invalid prompt"):
        client.complete(messages=[{"role":"user","content":"hi"}],
          reasoning_effort="high",response_format={"type":"json_object"},
          max_tokens=512,strict_reasoning_effort=True)
    assert len(requests)==1
    client.complete(messages=[{"role":"user","content":"hi"}],reasoning_effort="low",
         max_tokens=512,strict_reasoning_effort=True)
    assert requests[-1]["reasoning_effort"]=="low"
    # Legacy tolerant clients retry only the rejected effort, not all future tiers.
    out=client.complete(messages=[],reasoning_effort="high",max_tokens=512)
    assert out["_client_request_diagnostics"]["reasoning_effort_fallback"] is True
    assert requests[-2]["reasoning_effort"]=="high"
    assert "reasoning_effort" not in requests[-1]
    out=client.complete(messages=[],reasoning_effort="low",max_tokens=512)
    assert out["_client_request_diagnostics"]["reasoning_effort_sent"]=="low"
