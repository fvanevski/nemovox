"""Issue 14: ASR exhaustion cannot silently become a completed subtitle source."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from nvidia_transcribe.asr_integrity import IncompleteASRGeneration, audit_asr_generation
from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media


def _response(text: str, finish: str = "stop") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": finish}],
        "usage": {"completion_tokens": 16384 if finish == "length" else 20},
        "ctc_timestamps": {
            "words": [{"word": "Bonjour", "speaker": "0", "start": 0.5, "end": 0.9}],
            "diarization": [{"speaker": 0, "start": 0.3, "end": 1.0}],
            "speaker_tag_to_diarization_speaker": {"0": 0},
            "error": None,
        },
    }


def test_length_finish_is_incomplete_even_with_valid_ctc_and_short_cleaned_fragment():
    original = _response("<spk:0> Ah" + ("!" * 21000), "length")
    exact = json.dumps(original)
    audit = audit_asr_generation(original)
    assert audit["status"] == "incomplete"
    assert audit["reason"] == "generation_limit_exhausted"
    assert audit["finish_reason"] == "length"
    assert audit["ctc_metadata_present"] is True
    assert json.dumps(original) == exact


def test_stop_with_true_repeated_interjections_is_not_loop_blocked():
    audit = audit_asr_generation(_response("<spk:0> Non! Non! Non!", "stop"))
    assert audit["status"] == "complete"
    assert audit["loop_guard_removed_chars"] == 0


def test_long_pathological_loop_is_incomplete_even_if_server_says_stop():
    audit = audit_asr_generation(_response("<spk:0> " + ("a" * 4096), "stop"))
    assert audit["status"] == "incomplete"
    assert audit["reason"] == "pathological_generation_loop"


def test_failed_generation_preserves_raw_and_does_not_mark_resume_completed(tmp_path, monkeypatch):
    wav = tmp_path / "source.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 16000 * 3)

    calls = []
    bad = _response("<spk:0> Ah" + ("!" * 21000), "length")
    good = _response("<spk:0> Bonjour.", "stop")

    def fake_transcribe(self, audio_path, config=None):
        calls.append(Path(audio_path).name)
        raw = bad if len(calls) == 1 else good
        return {"raw_response": raw, "contract": {"raw_text": "Bonjour."}, "latency_s": 0.01}

    monkeypatch.setattr(NemotronVLLMClient, "transcribe", fake_transcribe)
    cfg = RecordingPipelineConfig(source_language="fr", speaker_embeddings="none", resume=True)
    root = tmp_path / "private-run"
    with pytest.raises(IncompleteASRGeneration, match="Chunk 0 ASR generation incomplete"):
        process_media(wav, output_dir=root, client=NemotronVLLMClient(model="stub"), config=cfg)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["chunks"][0]["status"] == "incomplete_generation"
    assert manifest["chunks"][0]["generation_integrity"]["reason"] == "generation_limit_exhausted"
    assert manifest["chunks"][0]["raw_response_sha256"]
    rejected = Path(manifest["chunks"][0]["rejected_raw_response_path"])
    assert rejected.parent.name == "rejected" and rejected.is_file()
    assert json.loads(rejected.read_text()) == bad
    assert json.loads((root / "chunks" / "chunk_000.raw.json").read_text()) == bad
    assert not (root / "recording.contract.json").exists()

    result = process_media(wav, output_dir=root, client=NemotronVLLMClient(model="stub"), config=cfg)
    assert len(calls) == 2
    assert result["contract"]["model_provenance"]["resumed_chunks"] == 0
    assert result["contract"]["utterances"]
    accepted = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert accepted["chunks"][0]["status"] == "completed"
    assert accepted["chunks"][0]["generation_integrity"]["status"] == "complete"
    assert json.loads(rejected.read_text()) == bad
