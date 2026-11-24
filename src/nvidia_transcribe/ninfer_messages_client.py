"""NInfer Anthropic Messages transport for explicitly bounded text reasoning.

This adapter is opt-in; the ordinary OpenAI-compatible translation/review
client is unchanged. NInfer's pinned Chat Completions parser ignores the
nonstandard thinking_budget key, whereas /v1/messages enforces
thinking.budget_tokens. No silent downgrade to an unbounded request.
"""

from __future__ import annotations

from typing import Any


class NInferMessagesTextClient:
    """NInfer /v1/messages facade returning the existing chat-completion shape."""

    def __init__(
        self, *, base_url: str, model: str, api_key: str | None = None,
        timeout_s: float = 180.0, reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort

    def complete(
        self, *, messages: list[dict[str, str]], max_tokens: int = 2048,
        temperature: float = 0.0, response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
        strict_reasoning_effort: bool = False,
    ) -> dict[str, Any]:
        import httpx

        if (not isinstance(max_tokens, int) or isinstance(max_tokens, bool)
                or max_tokens < 256):
            raise ValueError("Invalid Messages completion token limit")
        if thinking_budget is None:
            raise ValueError("NInfer Messages requires an explicit thinking budget")
        if not isinstance(thinking_budget, int) or isinstance(thinking_budget, bool):
            raise ValueError("Thinking budget must be an integer")
        if response_format is not None and response_format != {"type": "json_object"}:
            raise ValueError("Only prompt-validated JSON-object responses are supported")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Messages must be a nonempty list")
        for message in messages:
            if (not isinstance(message, dict)
                    or message.get("role") not in ("system", "user", "assistant")
                    or not isinstance(message.get("content"), str)):
                raise ValueError("NInfer Messages QC supports only text role/content pairs")
        # Anthropic Messages puts introductory system instructions at the
        # top level. NInfer rejects a leading role=system inside messages.
        # Only move the initial contiguous system prefix; any later system
        # turn retains its original position and provider semantics.
        prefix_end = 0
        while prefix_end < len(messages) and messages[prefix_end]["role"] == "system":
            prefix_end += 1
        if prefix_end == len(messages):
            raise ValueError("NInfer Messages requires at least one non-system message")
        body: dict[str, Any] = {
            "model": self.model, "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [dict(message) for message in messages[prefix_end:]],
        }
        if prefix_end:
            # The accepted client input is plain text, not Anthropic metadata.
            # Keep all initial instructions, preserving their text order.
            body["system"] = "\n\n".join(message["content"] for message in messages[:prefix_end])
        effort = reasoning_effort if reasoning_effort is not None else self.reasoning_effort
        if thinking_budget == 0:
            if effort not in (None, "none"):
                raise ValueError("Disabled thinking must not request positive reasoning effort")
            body["thinking"] = {"type": "disabled"}
            sent_effort = "none"
        elif thinking_budget >= 1024 and thinking_budget <= max_tokens - 256:
            if effort in (None, "none"):
                raise ValueError("Enabled thinking requires explicit positive effort")
            body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            body["output_config"] = {"effort": effort}
            sent_effort = effort
        else:
            raise ValueError("NInfer Messages enabled thinking needs >=1024 tokens and >=256 answer tokens")

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(headers=headers, timeout=self.timeout_s) as client:
            resp = client.post(f"{self.base_url}/messages", json=body)
            # No fallback: an unsupported field/effort cannot silently
            # reinstate the server's unbounded thinking default.
            resp.raise_for_status()
            answer = resp.json()
        if not isinstance(answer, dict) or answer.get("type") != "message":
            raise ValueError("Expected an Anthropic Messages response object")
        blocks = answer.get("content")
        if not isinstance(blocks, list):
            raise ValueError("Invalid Anthropic Messages content blocks")
        texts: list[str] = []
        thoughts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError("Malformed Anthropic Messages content block")
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                texts.append(block["text"])
            elif kind == "thinking" and isinstance(block.get("thinking"), str):
                thoughts.append(block["thinking"])
            else:
                raise ValueError("Unexpected Anthropic Messages content block in text QC")
        usage = answer.get("usage") or {}
        if not isinstance(usage, dict):
            raise ValueError("Invalid Anthropic Messages usage")
        output_tokens = usage.get("output_tokens")
        details = usage.get("output_tokens_details") or {}
        if not isinstance(details, dict):
            raise ValueError("Invalid Anthropic Messages output token details")
        reasoning_tokens = details.get("thinking_tokens")
        if not isinstance(output_tokens, int) or isinstance(output_tokens, bool) or output_tokens < 0:
            raise ValueError("NInfer Messages did not report output token usage")
        if (reasoning_tokens is not None and
                (not isinstance(reasoning_tokens, int) or isinstance(reasoning_tokens, bool)
                 or reasoning_tokens < 0 or reasoning_tokens > output_tokens)):
            raise ValueError("Invalid Anthropic Messages reasoning token usage")
        reasons = {"end_turn": "stop", "max_tokens": "length",
                   "stop_sequence": "stop", "model_context_window_exceeded": "length"}
        stop = answer.get("stop_reason")
        if stop not in reasons:
            raise ValueError("Unexpected Anthropic Messages stop reason")
        return {
            "model": answer.get("model"), "choices": [{
                "finish_reason": reasons[stop],
                "message": {"role": "assistant", "content": "".join(texts),
                            "reasoning_content": "".join(thoughts)},
            }],
            "usage": {"completion_tokens": output_tokens,
                      "completion_tokens_details": {"reasoning_tokens": reasoning_tokens}},
            "_client_request_diagnostics": {
                "protocol": "ninfer_anthropic_messages",
                "reasoning_effort_requested": effort,
                "reasoning_effort_sent": sent_effort,
                "reasoning_effort_fallback": False,
                "thinking_budget_sent": thinking_budget,
                "thinking_mode": body["thinking"]["type"],
                "response_format_sent": False,
                "json_validation_required": response_format is not None,
                "strict_reasoning_effort": strict_reasoning_effort,
            },
        }
