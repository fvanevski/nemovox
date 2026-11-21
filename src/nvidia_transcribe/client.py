"""Reusable OpenAI-compatible vLLM client for Nemotron 3.5 Transcribe."""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from nvidia_transcribe.checkpoint import DEFAULT_MODEL_PATH
from nvidia_transcribe.contract import build_transcript_contract
from nvidia_transcribe.parser import CTCMetadataError
from nvidia_transcribe.prompts import build_instruction, cleanup_generation_loops

AUDIO_MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".webm": "audio/webm",
}


@dataclass(frozen=True)
class InferenceRequestConfig:
    mode: str = "verbatim"
    source_language: str | None = None
    target_language: str = "English"
    context_terms: str | None = None
    custom_instruction: str | None = None
    capture_ctc_timestamps: bool = True
    require_ctc_timestamps: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 16384
    seed: int = 0
    enable_thinking: bool = False
    skip_special_tokens: bool = False
    spaces_between_special_tokens: bool = False
    source_offset_s: float = 0.0
    apply_loop_guardrail: bool = True


def encode_audio_data_url(audio_path: str | Path) -> str:
    """Read a local audio file and return an RFC-2397 data URL."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    mime = AUDIO_MIME_BY_SUFFIX.get(path.suffix.lower(), "audio/wav")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_chat_completion_payload(
    *,
    model: str,
    audio_data_url: str,
    config: InferenceRequestConfig,
) -> dict[str, Any]:
    """Construct the exact vLLM /v1/chat/completions payload for Nemotron 3.5 Transcribe."""
    instruction = config.custom_instruction or build_instruction(
        config.mode,
        source_language=config.source_language,
        target_language=config.target_language,
        context_terms=config.context_terms,
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "audio_url", "audio_url": {"url": audio_data_url}},
                ],
            }
        ],
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens,
        "stream": False,
        "seed": config.seed,
        "chat_template_kwargs": {"enable_thinking": config.enable_thinking},
        "skip_special_tokens": config.skip_special_tokens,
        "spaces_between_special_tokens": config.spaces_between_special_tokens,
    }
    if config.capture_ctc_timestamps:
        payload["mm_processor_kwargs"] = {"capture_ctc_timestamps": True}
    return payload


class NemotronVLLMClient:
    """HTTP client talking to a local vLLM server running Nemotron 3.5 Transcribe."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = str(DEFAULT_MODEL_PATH),
        api_key: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else (os.environ.get("VLLM_API_KEY") or "EMPTY")
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def list_models(self) -> dict[str, Any]:
        with httpx.Client(timeout=30.0, headers=self._headers()) as client:
            resp = client.get(f"{self.base_url}/models")
            resp.raise_for_status()
            return resp.json()

    def transcribe(
        self,
        audio_path: str | Path,
        config: InferenceRequestConfig | None = None,
    ) -> dict[str, Any]:
        cfg = config or InferenceRequestConfig()
        data_url = encode_audio_data_url(audio_path)
        payload = build_chat_completion_payload(
            model=self.model,
            audio_data_url=data_url,
            config=cfg,
        )
        t0 = time.perf_counter()
        with httpx.Client(timeout=self.timeout_s, headers=self._headers()) as client:
            resp = client.post(f"{self.base_url}/chat/completions", json=payload)
        latency_s = time.perf_counter() - t0
        if resp.status_code != 200:
            raise RuntimeError(f"vLLM request failed with HTTP {resp.status_code}: {resp.text}")

        raw_response = resp.json()
        choice = (raw_response.get("choices") or [{}])[0]
        raw_text = (choice.get("message") or {}).get("content") or ""
        if cfg.apply_loop_guardrail:
            raw_text = cleanup_generation_loops(raw_text)

        ctc_payload = raw_response.get("ctc_timestamps")
        if cfg.capture_ctc_timestamps and cfg.require_ctc_timestamps:
            if ctc_payload is None:
                raise CTCMetadataError("Server response omitted ctc_timestamps despite opt-in")
            if isinstance(ctc_payload, dict) and ctc_payload.get("error") is not None:
                raise CTCMetadataError(f"CTC timestamp middleware returned error: {ctc_payload['error']}")

        provenance = {
            "served_model": raw_response.get("model", self.model),
            "request_mode": cfg.mode,
            "capture_ctc_timestamps": cfg.capture_ctc_timestamps,
            "finish_reason": choice.get("finish_reason"),
            "usage": raw_response.get("usage"),
            "latency_s": round(latency_s, 4),
            "audio_path": str(audio_path),
        }

        contract = build_transcript_contract(
            raw_text=raw_text if cfg.mode != "translate" else "",
            source_language=cfg.source_language or ("auto" if cfg.mode != "translate" else None),
            ctc_payload=ctc_payload,
            source_offset_s=cfg.source_offset_s,
            model_provenance=provenance,
            english_translation=raw_text if cfg.mode == "translate" else None,
            uncertainty=None,
        )
        return {
            "contract": contract,
            "raw_response": raw_response,
            "latency_s": round(latency_s, 4),
        }
