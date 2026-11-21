"""CPU-only unit tests for prompts, request formatting, and vLLM client error paths."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from nvidia_transcribe.client import (
    InferenceRequestConfig,
    NemotronVLLMClient,
    build_chat_completion_payload,
    encode_audio_data_url,
)
from nvidia_transcribe.parser import CTCMetadataError
from nvidia_transcribe.prompts import build_instruction, cleanup_generation_loops


def test_build_instruction_does_not_default_to_french() -> None:
    prompt = build_instruction("verbatim")
    assert "French" not in prompt
    assert "Language: as spoken" in prompt

    zh_prompt = build_instruction("verbatim", source_language="Mandarin Chinese")
    assert "Spoken language: Mandarin Chinese." in zh_prompt

    tr_prompt = build_instruction("translate", target_language="English")
    assert tr_prompt == "Please translate the given speech to English."


def test_cleanup_generation_loops() -> None:
    assert cleanup_generation_loops("a" * 25 + " hello") == "a hello"
    assert cleanup_generation_loops(("bonjour " * 25) + "fin") == "bonjour fin"


def test_build_chat_completion_payload(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFFtest")
    data_url = encode_audio_data_url(wav)
    assert data_url.startswith("data:audio/wav;base64,")

    cfg = InferenceRequestConfig(mode="speaker", capture_ctc_timestamps=True)
    payload = build_chat_completion_payload(model="/workspace/models/local", audio_data_url=data_url, config=cfg)

    assert payload["temperature"] == 0.0
    assert payload["stream"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["skip_special_tokens"] is False
    assert payload["mm_processor_kwargs"] == {"capture_ctc_timestamps": True}
    assert "<spk:0>" in payload["messages"][0]["content"][0]["text"]


def test_client_fails_clearly_when_ctc_metadata_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "sample.wav"
    wav.write_bytes(b"RIFFtest")

    def fake_post(self, url, json=None):
        req = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=req,
            json={
                "model": "/workspace/models/local",
                "choices": [{"message": {"content": "Bonjour"}, "finish_reason": "stop"}],
                "ctc_timestamps": {
                    "words": [],
                    "diarization": [],
                    "speaker_tag_to_diarization_speaker": {},
                    "error": "not_enabled",
                },
            },
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    client = NemotronVLLMClient(base_url="http://127.0.0.1:8000/v1", model="/workspace/models/local")
    with pytest.raises(CTCMetadataError, match="not_enabled"):
        client.transcribe(wav, InferenceRequestConfig(capture_ctc_timestamps=True, require_ctc_timestamps=True))

    # Non-strict mode preserves the error cleanly without inventing timestamps.
    res = client.transcribe(wav, InferenceRequestConfig(capture_ctc_timestamps=True, require_ctc_timestamps=False))
    assert res["contract"]["timestamp_provenance"]["error"] == "not_enabled"
    assert res["contract"]["utterances"][0]["source_start"] is None
    assert json.loads(json.dumps(res["contract"]))["schema_version"] == "nemotron-transcript-contract/v1"
