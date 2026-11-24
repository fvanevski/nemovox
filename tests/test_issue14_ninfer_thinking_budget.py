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


def test_thinking_cap_and_explicit_tier_survive_host_default(monkeypatch):
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
        out=client.complete(messages=[{"role":"user","content":"review"}],
                max_tokens=1280,response_format={"type":"json_object"},
                reasoning_effort=effort,thinking_budget=budget,
                strict_reasoning_effort=True)
        assert out["_client_request_diagnostics"]["thinking_budget_sent"]==budget
        assert out["_client_request_diagnostics"]["reasoning_effort_sent"]==effort
    assert len(requests)==2
    assert [r["reasoning_effort"] for r in requests]==["low","xhigh"]
    assert [r["thinking_budget"] for r in requests]==[128,512]
    with pytest.raises(ValueError,match="final-answer"):
        client.complete(messages=[],max_tokens=512,thinking_budget=500)


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
          thinking_budget=128,max_tokens=512,strict_reasoning_effort=True)
    assert len(requests)==1
    client.complete(messages=[{"role":"user","content":"hi"}],reasoning_effort="low",
         thinking_budget=128,max_tokens=512,strict_reasoning_effort=True)
    assert requests[-1]["reasoning_effort"]=="low"
    # Legacy tolerant clients retry only the rejected effort, not all future tiers.
    out=client.complete(messages=[],reasoning_effort="high",max_tokens=512)
    assert out["_client_request_diagnostics"]["reasoning_effort_fallback"] is True
    assert requests[-2]["reasoning_effort"]=="high"
    assert "reasoning_effort" not in requests[-1]
    out=client.complete(messages=[],reasoning_effort="low",max_tokens=512)
    assert out["_client_request_diagnostics"]["reasoning_effort_sent"]=="low"
