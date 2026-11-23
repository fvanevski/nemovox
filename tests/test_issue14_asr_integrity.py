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


def test_legacy_completed_checkpoint_cannot_hide_length_finish(tmp_path, monkeypatch):
    """A previously accepted but exhausted response must fail on replay too."""
    import hashlib

    wav = tmp_path / "short.wav"
    with wave.open(str(wav), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 16000 * 2)
    invocations = []

    def fake(self, path, config=None):
        invocations.append(Path(path).name)
        return {
            "raw_response": _response("<spk:0> Bonjour.", "stop"),
            "contract": {"raw_text": "Bonjour."}, "latency_s": 0.01,
        }

    monkeypatch.setattr(NemotronVLLMClient, "transcribe", fake)
    cfg = RecordingPipelineConfig(source_language="fr", speaker_embeddings="none")
    root = tmp_path / "run"
    client = NemotronVLLMClient(model="stub")
    process_media(wav, output_dir=root, client=client, config=cfg)
    assert len(invocations) == 1
    response_path = root / "chunks" / "chunk_000.raw.json"
    suspect = _response("<spk:0> Ah" + ("!" * 3000), "length")
    rendered = json.dumps(suspect, ensure_ascii=False, indent=2)
    response_path.write_text(rendered, encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunks"][0]["raw_response_sha256"] = hashlib.sha256(
        rendered.encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # The original unrelated finished contract is not rewritten on failure.
    original_contract = (root / "recording.contract.json").read_bytes()
    with pytest.raises(IncompleteASRGeneration):
        process_media(wav, output_dir=root, client=client, config=cfg)
    assert len(invocations) == 1  # existing response was audited, not trusted
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["chunks"][0]["status"] == "incomplete_generation"
    assert (root / "recording.contract.json").read_bytes() == original_contract
    rejected = Path(updated["chunks"][0]["rejected_raw_response_path"])
    assert json.loads(rejected.read_text()) == suspect


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
        raw = bad if len(calls) <= 2 else good
        return {"raw_response": raw, "contract": {"raw_text": "Bonjour."}, "latency_s": 0.01}

    monkeypatch.setattr(NemotronVLLMClient, "transcribe", fake_transcribe)
    cfg = RecordingPipelineConfig(source_language="fr", speaker_embeddings="none", resume=True)
    root = tmp_path / "private-run"
    with pytest.raises(IncompleteASRGeneration, match="Chunk 0 ASR generation incomplete"):
        process_media(wav, output_dir=root, client=NemotronVLLMClient(model="stub"), config=cfg)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["asr_run_status"] == "incomplete_generation"
    assert manifest["chunks"][0]["status"] == "incomplete_generation"
    assert manifest["chunks"][0]["generation_integrity"]["reason"] == "generation_limit_exhausted"
    assert manifest["chunks"][0]["generation_limit_retry_attempted"] is True
    assert len(manifest["chunks"][0]["generation_attempts"]) == 2
    assert manifest["chunks"][0]["raw_response_sha256"]
    rejected = Path(manifest["chunks"][0]["rejected_raw_response_path"])
    assert rejected.parent.name == "rejected" and rejected.is_file()
    assert json.loads(rejected.read_text()) == bad
    assert json.loads((root / "chunks" / "chunk_000.raw.json").read_text()) == bad
    assert not (root / "recording.contract.json").exists()

    result = process_media(wav, output_dir=root, client=NemotronVLLMClient(model="stub"), config=cfg)
    assert len(calls) == 3
    assert result["contract"]["model_provenance"]["resumed_chunks"] == 0
    assert result["contract"]["utterances"]
    accepted = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert accepted["asr_run_status"] == "completed"
    assert accepted["chunks"][0]["status"] == "completed"
    assert accepted["chunks"][0]["generation_integrity"]["status"] == "complete"
    assert json.loads(rejected.read_text()) == bad
