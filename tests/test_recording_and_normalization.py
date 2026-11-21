"""Additional CPU-only tests for language resolution, events, code-switching, and archived FR regressions."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest

from nvidia_transcribe.client import NemotronVLLMClient
from nvidia_transcribe.contract import (
    build_transcript_contract,
    extract_events_from_raw_text,
    normalize_language_code,
)
from nvidia_transcribe.recording import (
    RecordingPipelineConfig,
    _resolve_recording_language,
    process_media,
)
from nvidia_transcribe.utterances import join_multilingual_units


def test_language_resolution_explicit_detected_and_ambiguous(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lang, prov, _ = _resolve_recording_language(
        explicit_language="French",
        client=NemotronVLLMClient(model="m"),
        chunk_wavs=[],
        raw_texts=[],
    )
    assert lang == "fr" and prov == "explicit"

    wav = tmp_path / "c.wav"
    wav.write_bytes(b"RIFFstub")

    # Conflicting LID votes -> ambiguous, never silently French/Mandarin/English
    votes = iter(["fr", "es", "de"])
    monkeypatch.setattr(
        NemotronVLLMClient,
        "transcribe",
        lambda self, p, cfg=None: {"contract": {"raw_text": next(votes)}, "raw_response": {}, "latency_s": 0.0},
    )
    lang2, prov2, _ = _resolve_recording_language(
        explicit_language=None,
        client=NemotronVLLMClient(model="m"),
        chunk_wavs=[wav, wav, wav],
        raw_texts=["hello"],
    )
    assert lang2 == "ambiguous" and prov2 == "ambiguous"


def test_code_switching_and_event_isolation() -> None:
    units = [
        {"text": "我", "unit_type": "character"},
        {"text": "用", "unit_type": "character"},
        {"text": "Python", "unit_type": "word"},
        {"text": "写", "unit_type": "character"},
        {"text": "代码", "unit_type": "character"},
    ]
    assert join_multilingual_units(units) == "我用 Python 写代码"

    raw = "<spk:0> Bonjour <ev:laughter> tout le monde <sty:stt:vocal_intensity_pitch> Alerte! <sty:end:vocal_intensity_pitch>"
    events = extract_events_from_raw_text(raw, chunk_index=0)
    tags = [e["tag"] for e in events]
    assert "<ev:laughter>" in tags
    assert "<sty:stt:vocal_intensity_pitch>" in tags
    contract = build_transcript_contract(raw_text=raw, source_language="fr")
    assert "<ev:" not in contract["source_text"]
    assert "<sty:" not in contract["source_text"]
    assert "Bonjour tout le monde Alerte!" == contract["source_text"]


def test_archived_french_overlap_and_child_voices_regressions() -> None:
    fr_ov = Path("/workspace/nvidia-transcribe/outputs/responses/overlap/fr_2spk_overlap_sot.raw.json")
    raw_ov = json.loads(fr_ov.read_text(encoding="utf-8"))
    c_ov = build_transcript_contract(
        raw_text=raw_ov["choices"][0]["message"]["content"],
        source_language="fr",
        ctc_payload=raw_ov["ctc_timestamps"],
    )
    # Two concurrent French speakers reconstructed cleanly (not word-by-word)
    assert len(c_ov["utterances"]) <= 4
    assert {u["speaker_id"] for u in c_ov["utterances"]} == {"0", "1"}

    fr_child = Path("/workspace/nvidia-transcribe/outputs/responses/fr/fr_0316_0355_child_voices_speaker.raw.json")
    raw_ch = json.loads(fr_child.read_text(encoding="utf-8"))
    c_ch = build_transcript_contract(
        raw_text=raw_ch["choices"][0]["message"]["content"],
        source_language="fr",
        ctc_payload=raw_ch["ctc_timestamps"],
        source_offset_s=196.0,
    )
    assert {u["speaker_id"] for u in c_ch["utterances"]} == {"0", "1"}
    assert c_ch["utterances"][0]["source_start"] >= 196.0


def test_partial_interrupted_run_resumes_remaining_chunk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wav = tmp_path / "long.wav"
    with wave.open(str(wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 25)  # 25s -> 2 chunks with chunk=12s, min_tail=5s

    call_log: list[str] = []
    fail_second = {"active": True}

    def fake_transcribe(self, audio_path, config=None):
        name = Path(audio_path).name
        call_log.append(name)
        if fail_second["active"] and name == "chunk_001.wav":
            raise RuntimeError("Simulated mid-run server disconnect")
        idx = 0 if "000" in name else 1
        return {
            "contract": {"raw_text": f"<spk:0> mot{idx}"},
            "raw_response": {
                "choices": [{"message": {"content": f"<spk:0> mot{idx}"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 5},
                "ctc_timestamps": {
                    "words": [{"word": f"mot{idx}", "start": 1.0, "end": 1.4, "speaker": "0"}],
                    "diarization": [{"speaker": 0, "start": 0.9, "end": 1.5}],
                    "speaker_tag_to_diarization_speaker": {"0": 0},
                    "error": None,
                },
            },
            "latency_s": 0.02,
        }

    monkeypatch.setattr(NemotronVLLMClient, "transcribe", fake_transcribe)
    out_dir = tmp_path / "run"
    client = NemotronVLLMClient(model="m")
    cfg = RecordingPipelineConfig(
        source_language="fr",
        chunk_seconds=12.0,
        context_seconds=2.0,
        min_tail_seconds=5.0,
        speaker_embeddings="none",
        resume=True,
    )

    with pytest.raises(RuntimeError, match="Simulated"):
        process_media(wav, output_dir=out_dir, client=client, config=cfg)

    assert call_log == ["chunk_000.wav", "chunk_001.wav"]
    # chunk_000 is persisted in manifest.json; retrying runs ONLY chunk_001!
    call_log.clear()
    fail_second["active"] = False
    res = process_media(wav, output_dir=out_dir, client=client, config=cfg)
    assert call_log == ["chunk_001.wav"]
    assert res["contract"]["model_provenance"]["resumed_chunks"] == 1
    assert len(res["contract"]["utterances"]) >= 1
