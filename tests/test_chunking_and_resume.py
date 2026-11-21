"""CPU-only unit tests for chunk planning, boundary reconciliation, and resume."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from nvidia_transcribe.chunking import ChunkConfig, plan_chunks, reconcile_chunk_boundaries
from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.media import MediaInspectionError, probe_media
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media


def test_plan_chunks_short_and_exact_coverage() -> None:
    short = plan_chunks(45.0, ChunkConfig(chunk_seconds=90.0, context_seconds=10.0, min_tail_seconds=15.0))
    assert len(short) == 1
    assert short[0].core_start_s == 0.0
    assert short[0].core_end_s == 45.0
    assert short[0].infer_start_s == 0.0
    assert short[0].infer_end_s == 45.0

    # 192s with 90+15 tail -> chunks [0, 90] and [90, 192] (merges 12s tail cleanly)
    specs = plan_chunks(192.0, ChunkConfig(chunk_seconds=90.0, context_seconds=10.0, min_tail_seconds=15.0))
    assert len(specs) == 2
    assert specs[0].core_start_s == 0.0 and specs[0].core_end_s == 90.0
    assert specs[0].infer_start_s == 0.0 and specs[0].infer_end_s == 100.0
    assert specs[1].core_start_s == 90.0 and specs[1].core_end_s == 192.0
    assert specs[1].infer_start_s == 80.0 and specs[1].infer_end_s == 192.0
    # Exact contiguous partition
    assert specs[0].core_end_s == specs[1].core_start_s


def test_boundary_deduplication_and_simultaneous_repetition_preserved() -> None:
    chunks = plan_chunks(180.0, ChunkConfig(chunk_seconds=90.0, context_seconds=10.0, min_tail_seconds=15.0))
    # Same speaker says "bonjour" spanning 90.0 boundary in both chunks -> deduplicate!
    # Distinct concurrent speaker SPK_01 also says "bonjour" at 90.1 -> MUST NOT discard!
    c0_units = [
        {"text": "bonjour", "start": 89.70, "end": 90.10, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 0},
        {"text": "monde", "start": 90.20, "end": 90.60, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 0},
    ]
    c1_units = [
        {"text": "bonjour", "start": 89.75, "end": 90.12, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 1},
        {"text": "bonjour", "start": 89.80, "end": 90.25, "global_speaker_id": "SPK_01", "valid_timing": True, "chunk_index": 1},
        {"text": "monde", "start": 90.22, "end": 90.62, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 1},
    ]
    merged, diag = reconcile_chunk_boundaries([c0_units, c1_units], chunks)
    assert diag["deduplicated_units"] == 2
    texts_by_spk = [(u["global_speaker_id"], u["text"]) for u in merged]
    assert texts_by_spk.count(("SPK_00", "bonjour")) == 1
    assert texts_by_spk.count(("SPK_00", "monde")) == 1
    assert texts_by_spk.count(("SPK_01", "bonjour")) == 1


def test_boundary_recovery_when_next_chunk_misses_initial_word() -> None:
    chunks = plan_chunks(180.0, ChunkConfig(chunk_seconds=90.0, context_seconds=10.0, min_tail_seconds=15.0))
    c0_units = [
        {"text": "alors", "start": 89.6, "end": 89.9, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 0},
        {"text": "oui", "start": 90.1, "end": 90.4, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 0},
    ]
    c1_units = [
        {"text": "d'accord", "start": 90.8, "end": 91.2, "global_speaker_id": "SPK_00", "valid_timing": True, "chunk_index": 1},
    ]
    merged, diag = reconcile_chunk_boundaries([c0_units, c1_units], chunks)
    assert diag["recovered_boundary_units"] == 1
    assert [u["text"] for u in merged] == ["alors", "oui", "d'accord"]


def test_resume_after_partial_run_and_invalid_media(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Invalid missing media raises clearly
    with pytest.raises(MediaInspectionError):
        probe_media(tmp_path / "missing.wav")

    # Create a real 3s silent WAV using python wave module (no ffmpeg needed for fixture creation,
    # and mock probe/extract for deterministic fast CPU test)
    import wave

    wav = tmp_path / "input.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 3)

    calls: list[str] = []

    def fake_transcribe(self, audio_path, config=None):
        calls.append(Path(audio_path).name)
        off = config.source_offset_s if config else 0.0
        if config and config.mode == "lid":
            return {
                "contract": {"raw_text": "fr"},
                "raw_response": {"choices": [{"message": {"content": "fr"}}]},
                "latency_s": 0.01,
            }
        raw_resp = {
            "choices": [{"message": {"content": "<spk:0> Bonjour"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 10},
            "ctc_timestamps": {
                "words": [{"word": "Bonjour", "start": 0.5, "end": 0.9, "speaker": "0"}],
                "diarization": [{"speaker": 0, "start": 0.4, "end": 1.0}],
                "speaker_tag_to_diarization_speaker": {"0": 0},
                "error": None,
            },
        }
        return {"contract": {"raw_text": "<spk:0> Bonjour"}, "raw_response": raw_resp, "latency_s": 0.05}

    monkeypatch.setattr(NemotronVLLMClient, "transcribe", fake_transcribe)

    out_dir = tmp_path / "out"
    client = NemotronVLLMClient(base_url="http://127.0.0.1:8190/v1", model="test-model")
    res1 = process_media(
        wav,
        output_dir=out_dir,
        client=client,
        config=RecordingPipelineConfig(source_language="fr", speaker_embeddings="none", resume=True),
    )
    assert len(calls) == 1
    assert res1["contract"]["schema_version"] == "nemotron-recording-contract/v1"
    assert res1["contract"]["source_language"] == "fr"
    assert res1["contract"]["language_provenance"] == "explicit"

    # Second run with resume=True reuses completed chunk (0 new transcribe calls!)
    calls.clear()
    res2 = process_media(
        wav,
        output_dir=out_dir,
        client=client,
        config=RecordingPipelineConfig(source_language="fr", speaker_embeddings="none", resume=True),
    )
    assert calls == []
    assert res2["contract"]["model_provenance"]["resumed_chunks"] == 1
    assert json.loads((out_dir / "recording.contract.json").read_text(encoding="utf-8"))["utterances"][0]["source_text"] == "Bonjour"
