"""NInfer Anthropic Messages transport for bounded text and image reasoning.

This adapter is opt-in; the ordinary OpenAI-compatible translation/review
client is unchanged. NInfer's pinned Chat Completions parser ignores the
nonstandard thinking_budget key, whereas /v1/messages enforces
thinking.budget_tokens. No silent downgrade to an unbounded request.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any


# Bound request size before decoding; media decoding and vision memory
# admission remain NInfer's responsibility. Only inline still images are
# supported: no URLs, remote fetching, implicit file reads or video.
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
# Generous but bounded production quality-call completion allowances.
QUALITY_COMPLETION_TOKENS = 8192
QUALITY_RETRY_TOKENS = 16384
QUALITY_TIMEOUT_S = 600.0

_MAX_IMAGES = 12
_MAX_TOTAL_IMAGE_BYTES = 24 * 1024 * 1024
_IMAGE_SIGNATURES = {"image/png": b"\x89PNG\r\n\x1a\n", "image/jpeg": b"\xff\xd8\xff"}


def image_content_block(data: bytes, *, media_type: str = "image/png") -> dict[str, Any]:
    """Encode explicit caller-supplied image bytes as native Messages content."""
    if not isinstance(data, bytes) or not 0 < len(data) <= _MAX_IMAGE_BYTES:
        raise ValueError("Invalid inline image byte length")
    if media_type not in _IMAGE_SIGNATURES or not data.startswith(_IMAGE_SIGNATURES[media_type]):
        raise ValueError("Unsupported or mismatched inline image format")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                       "data": base64.b64encode(data).decode("ascii")}}


def _normalize_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages must be a nonempty list")
    normalized = []
    images = []
    total = 0
    for message_index, message in enumerate(messages):
        if (not isinstance(message, dict)
                or message.get("role") not in ("system", "user", "assistant")
                or set(message) != {"role", "content"}):
            raise ValueError("Invalid Messages role/content pair")
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            normalized.append(dict(message))
            continue
        if role != "user" or not isinstance(content, list) or not content:
            raise ValueError("Content blocks require a nonempty user message; system and assistant content must be text")
        parts = []
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                raise ValueError("Invalid Messages content block")
            if (block.get("type") == "text" and set(block) == {"type", "text"}
                    and isinstance(block.get("text"), str)):
                parts.append(dict(block))
                continue
            if block.get("type") != "image" or set(block) != {"type", "source"}:
                raise ValueError("Only text and inline image content blocks are supported")
            source = block["source"]
            if (not isinstance(source, dict)
                    or set(source) != {"type", "media_type", "data"}
                    or source.get("type") != "base64"
                    or not isinstance(source.get("media_type"), str)
                    or source["media_type"] not in _IMAGE_SIGNATURES
                    or not isinstance(source.get("data"), str)):
                raise ValueError("Only inline base64 PNG or JPEG image sources are supported")
            encoded = source["data"]
            if not encoded or len(encoded) > 4 * ((_MAX_IMAGE_BYTES + 2) // 3):
                raise ValueError("Invalid inline image byte length")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("Invalid inline image base64") from exc
            canonical = image_content_block(data, media_type=source["media_type"])
            total += len(data)
            if len(images) >= _MAX_IMAGES or total > _MAX_TOTAL_IMAGE_BYTES:
                raise ValueError("Inline image request exceeds bounded media budget")
            images.append({"message_index": message_index, "block_index": block_index,
                           "media_type": source["media_type"], "size_bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()})
            parts.append(canonical)
        normalized.append({"role": role, "content": parts})
    return normalized, images


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
        self, *, messages: list[dict[str, Any]], max_tokens: int = 2048,
        temperature: float = 0.0, response_format: dict[str, Any] | None = None,
        reasoning_effort: str | None = None,
        thinking_budget: int | None = None,
        strict_reasoning_effort: bool = False,
        retry_max_tokens: int | None = None,
    ) -> dict[str, Any]:
        import httpx

        if (not isinstance(max_tokens, int) or isinstance(max_tokens, bool)
                or max_tokens < 256):
            raise ValueError("Invalid Messages completion token limit")
        if retry_max_tokens is not None and (
                not isinstance(retry_max_tokens,int) or isinstance(retry_max_tokens,bool)
                or retry_max_tokens <= max_tokens or retry_max_tokens > 32768):
            raise ValueError("Invalid bounded completion retry limit")
        if thinking_budget is None:
            raise ValueError("NInfer Messages requires an explicit thinking budget")
        if not isinstance(thinking_budget, int) or isinstance(thinking_budget, bool):
            raise ValueError("Thinking budget must be an integer")
        if response_format is not None:
            if (not isinstance(response_format, dict)
                    or response_format.get("type") != "json_schema"
                    or not isinstance(response_format.get("schema"), dict)
                    or set(response_format) != {"type", "schema"}):
                raise ValueError("NInfer Messages requires an explicit JSON Schema for constrained output")
        messages, input_images = _normalize_messages(messages)
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
        if response_format is not None:
            # Pinned NInfer /v1/messages parses Anthropic output_config.format
            # as type=json_schema with a direct schema object. Never silently
            # downgrade this to unconstrained JSON or prose.
            body.setdefault("output_config", {})["format"] = {
                "type": "json_schema", "schema": response_format["schema"],
            }

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        attempts = []
        with httpx.Client(headers=headers, timeout=self.timeout_s) as client:
            limits = [max_tokens] if retry_max_tokens is None else [max_tokens,retry_max_tokens]
            for limit in limits:
                body["max_tokens"] = limit
                resp = client.post(f"{self.base_url}/messages", json=body)
                # No fallback for transport, admission or unsupported
                # schema/effort. Retry only explicit native output exhaustion.
                resp.raise_for_status()
                answer = resp.json()
                if not isinstance(answer,dict) or answer.get("type")!="message":
                    raise ValueError("Expected an Anthropic Messages response object")
                attempts.append({"max_tokens":limit,"stop_reason":answer.get("stop_reason"),
                    "response_id":answer.get("id"),"usage":answer.get("usage"),
                    "response_sha256":hashlib.sha256(
                        json.dumps(answer,ensure_ascii=False,sort_keys=True).encode()).hexdigest()})
                if answer.get("stop_reason")!="max_tokens":
                    break
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
                "native_stop_reason": stop,
                "completion_attempts": attempts,
                "input_image_count": len(input_images),
                "input_images": input_images,
                "reasoning_effort_requested": effort,
                "reasoning_effort_sent": sent_effort,
                "reasoning_effort_fallback": False,
                "thinking_budget_sent": thinking_budget,
                "thinking_mode": body["thinking"]["type"],
                "response_format_sent": response_format is not None,
                "json_validation_required": response_format is not None,
                "strict_reasoning_effort": strict_reasoning_effort,
            },
        }
