"""CPU unit tests for selective Qwen3-ForcedAligner-0.6B fallback (Phase B)."""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

from nvidia_transcribe.alignment import recover_chunk_alignment
from nvidia_transcribe.cli import build_parser
from nvidia_transcribe.qwen_aligner import (
    QwenAlignerConfig,
    QwenForcedAlignerAdapter,
    map_language_for_qwen_aligner,
    validate_qwen_items,
)
from nvidia_transcribe.recording import RecordingPipelineConfig, process_media
from nvidia_transcribe.utterances import reconstruct_speaker_utterances


def _write_silence_wav(path: Path, duration_s: float = 4.0, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(duration_s * rate))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n_frames)


def _make_ctc_payload(words: list[dict[str, Any]], diar: list[dict[str, Any]], error: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"words": words, "diarization": diar}
    if error is not None:
        payload["error"] = error
    return payload


def test_map_language_for_qwen_aligner() -> None:
    assert map_language_for_qwen_aligner("zh") == "Chinese"
    assert map_language_for_qwen_aligner("zh-CN") == "Chinese"
    assert map_language_for_qwen_aligner("fr") == "French"
    assert map_language_for_qwen_aligner("yue") == "Cantonese"
    assert map_language_for_qwen_aligner("Cantonese") == "Cantonese"
    assert map_language_for_qwen_aligner("en") == "English"
    assert map_language_for_qwen_aligner("xx-unknown") is None
    assert map_language_for_qwen_aligner(None) is None


def test_validate_qwen_items_rejects_zero_duration_and_invalid_bounds() -> None:
    raw = [
        {"text": "钙", "start": 0.4, "end": 0.64},
        {"text": "钾", "start": 0.64, "end": 0.64},  # zero duration -> must reject, never clamp
        {"text": "等", "start": 0.9, "end": 0.8},  # inverted
        {"text": "金", "start": 0.75, "end": 0.95},  # monotonic relative to 0.4 -> valid
        {"text": "属", "start": 0.70, "end": 0.98},  # non-monotonic relative to 0.75
        {"text": "越", "start": 3.8, "end": 4.5},  # past audio duration (4.0 + 0.05)
    ]
    units, stats = validate_qwen_items(raw, audio_duration_s=4.0, source_offset_s=10.0)
    assert stats["total_items"] == 6
    assert stats["valid_items"] == 2
    assert stats["zero_duration_count"] == 1
    assert stats["invalid_bounds_count"] == 2
    assert stats["non_monotonic_count"] == 1

    assert units[0]["valid_timing"] is True
    assert units[0]["start"] == pytest.approx(10.4)
    assert units[0]["end"] == pytest.approx(10.64)

    assert units[1]["valid_timing"] is False
    assert units[1]["timing_error"] == "zero_duration"
    assert units[1]["start"] is None
    assert units[1]["end"] is None

    assert units[2]["timing_error"] == "negative_or_inverted"
    assert units[2]["start"] is None
    assert units[4]["timing_error"] == "non_monotonic"
    assert units[4]["start"] is None
    assert units[5]["timing_error"] == "past_audio_duration"
    assert units[5]["start"] is None


def test_qwen_adapter_unavailable_and_unsupported_language(tmp_path: Path) -> None:
    wav = tmp_path / "audio.wav"
    _write_silence_wav(wav, duration_s=2.0)

    adapter_missing = QwenForcedAlignerAdapter(
        QwenAlignerConfig(
            model_path=str(tmp_path / "missing_model"),
            worker_python=str(tmp_path / "missing_python"),
        )
    )
    res_unsupported = adapter_missing.align(audio_path=wav, text="bonjour", language="klingon", source_offset_s=0.0)
    assert res_unsupported["units"] == []
    assert res_unsupported["diagnostic"]["status"] == "unsupported_language"

    res_unavail = adapter_missing.align(audio_path=wav, text="bonjour", language="fr", source_offset_s=0.0)
    assert res_unavail["units"] == []
    assert res_unavail["diagnostic"]["status"] == "unavailable"


def test_qwen_adapter_session_zero_duration_filtered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The adapter retains strict validation with the reusable session protocol."""
    wav = tmp_path / "seg.wav"
    _write_silence_wav(wav, duration_s=3.0)
    fake_py = tmp_path / "python"
    fake_py.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_model = tmp_path / "model"
    fake_model.mkdir()
    adapter = QwenForcedAlignerAdapter(QwenAlignerConfig(
        model_path=str(fake_model), worker_python=str(fake_py), device="cpu",
    ))
    calls: list[dict[str, Any]] = []

    def fake_session_align(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "status": "ok", "device_used": "cpu", "dtype": "torch.float32",
            "items": [
                {"text": "钙", "start": 0.2, "end": 0.44},
                {"text": "钾", "start": 0.56, "end": 0.56},
                {"text": "金属", "start": 0.64, "end": 1.04},
            ],
        }

    monkeypatch.setattr(adapter.session, "align", fake_session_align)
    res = adapter.align(audio_path=wav, text="钙钾金属", language="zh", source_offset_s=5.0)
    assert len(calls) == 1
    assert res["diagnostic"]["status"] == "ok"
    assert res["diagnostic"]["device_used"] == "cpu"
    assert res["diagnostic"]["zero_duration_count"] == 1
    assert res["units"][0]["start"] == pytest.approx(5.2)
    assert res["units"][1]["start"] is None
    assert res["units"][1]["timing_error"] == "zero_duration"
    assert res["units"][2]["start"] == pytest.approx(5.64)


class _FakeExternalAligner:
    def __init__(self, mapping: dict[str, tuple[float, float] | None]) -> None:
        self.mapping = mapping
        self.calls: list[dict[str, Any]] = []

    def align(
        self,
        *,
        audio_path: Path,
        text: str,
        language: str | None,
        source_offset_s: float,
    ) -> dict[str, Any]:
        self.calls.append({"audio_path": str(audio_path), "text": text, "language": language, "offset": source_offset_s})
        out_units: list[dict[str, Any]] = []
        for ch in text:
            if ch in {"。", "，", "、"} or ch.isspace():
                continue
            span = self.mapping.get(ch)
            if span is None:
                out_units.append({"text": ch, "start": None, "end": None, "valid_timing": False, "timing_error": "zero_duration"})
            else:
                out_units.append(
                    {
                        "text": ch,
                        "start": source_offset_s + span[0],
                        "end": source_offset_s + span[1],
                        "valid_timing": True,
                        "timing_error": None,
                    }
                )
        return {
            "units": out_units,
            "diagnostic": {"backend": "qwen3_forced_aligner", "status": "ok", "valid_items": sum(1 for u in out_units if u["valid_timing"])},
        }


def test_selective_qwen_fallback_preserves_ctc_and_recovers_oov(tmp_path: Path) -> None:
    wav = tmp_path / "chunk.wav"
    _write_silence_wav(wav, duration_s=6.0)
    raw_text = "<spk:0>你好。钾金属。"
    ctc_payload = _make_ctc_payload(
        words=[],
        diar=[
            {"speaker": "spk_0", "start": 0.4, "end": 1.4},
            {"speaker": "spk_0", "start": 3.0, "end": 4.8},
        ],
        error="alignment_failed",
    )

    def _subinterval_runner(win_start: float, win_end: float) -> tuple[str, Any]:
        # First window recovers "你好。" via native CTC; second still fails due to '钾'.
        if win_start < 12.0:
            return (
                "<spk:0>你好。",
                _make_ctc_payload(
                    words=[
                        {"word": "你", "start": 0.2, "end": 0.45},
                        {"word": "好。", "start": 0.45, "end": 0.8},
                    ],
                    diar=[{"speaker": "spk_0", "start": 0.2, "end": 0.8}],
                ),
            )
        return ("<spk:0>钾金属。", _make_ctc_payload([], [{"speaker": "spk_0", "start": 0.1, "end": 1.4}], error="oov"))

    def _wav_extractor(win_start: float, win_end: float) -> Path:
        dst = tmp_path / "sub" / f"{win_start:.2f}_{win_end:.2f}.wav"
        _write_silence_wav(dst, duration_s=max(0.5, win_end - win_start))
        return dst

    aligner = _FakeExternalAligner({"钾": (0.2, 0.48), "金": (0.48, 0.76), "属": (0.76, 1.04)})
    recovered = recover_chunk_alignment(
        raw_text=raw_text,
        ctc_payload=ctc_payload,
        source_offset_s=10.0,
        chunk_index=0,
        subinterval_runner=_subinterval_runner,
        external_aligner=aligner,
        chunk_wav_path=wav,
        source_language="zh",
        subinterval_wav_extractor=_wav_extractor,
    )

    assert len(aligner.calls) == 1
    assert "钾" in aligner.calls[0]["text"]
    assert "你" not in aligner.calls[0]["text"]
    assert recovered["alignment_status"] == "hybrid_ctc_and_external_recovered"

    units = recovered["units"]
    # Native subinterval CTC units keep original subinterval timing provenance!
    assert units[0]["text"] == "你"
    assert units[0]["alignment_status"] == "subinterval_ctc_recovered"
    assert units[1]["text"] == "好。"
    assert units[1]["alignment_status"] == "subinterval_ctc_recovered"
    # OOV window recovered via selective Qwen fallback
    assert units[2]["text"] == "钾"
    assert units[2]["alignment_status"] == "external_qwen_recovered"
    assert units[2]["start"] is not None
    assert units[4]["text"] == "属。"
    assert units[4]["alignment_status"] == "external_qwen_recovered"

    utts = reconstruct_speaker_utterances(units, source_language="zh")
    assert len(utts) == 2
    assert utts[0]["source_text"] == "你好。"
    assert utts[0]["timed_unit_count"] == 2
    assert utts[0]["timestamp_provenance"]["timing_status"] == "full"
    assert utts[1]["source_text"] == "钾金属。"
    assert utts[1]["timed_unit_count"] == 3
    assert utts[1]["untimed_unit_count"] == 0
    assert utts[1]["timestamp_provenance"]["timing_status"] == "full"
    assert utts[1]["timestamp_provenance"]["backend"] == "external_forced_aligner"


def test_qwen_fallback_skips_mixed_speaker_overlap(tmp_path: Path) -> None:
    wav = tmp_path / "overlap.wav"
    _write_silence_wav(wav, duration_s=5.0)
    raw_text = "<spk:0>钾好 <spk:1>对对"
    ctc_payload = _make_ctc_payload(
        words=[],
        diar=[
            {"speaker": "spk_0", "start": 0.2, "end": 2.5},
            {"speaker": "spk_1", "start": 0.8, "end": 2.6},
        ],
        error="alignment_failed",
    )

    def _wav_extractor(win_start: float, win_end: float) -> Path:
        dst = tmp_path / "sub" / f"{win_start:.2f}_{win_end:.2f}.wav"
        _write_silence_wav(dst, duration_s=max(0.5, win_end - win_start))
        return dst

    aligner = _FakeExternalAligner({"钾": (0.2, 0.5), "好": (0.5, 0.8), "对": (0.8, 1.1)})
    recovered = recover_chunk_alignment(
        raw_text=raw_text,
        ctc_payload=ctc_payload,
        source_offset_s=0.0,
        chunk_index=0,
        subinterval_runner=None,
        external_aligner=aligner,
        chunk_wav_path=wav,
        source_language="zh",
        subinterval_wav_extractor=_wav_extractor,
    )
    assert aligner.calls == []
    assert all(u["start"] is None for u in recovered["units"])
    assert recovered["diagnostic"]["external_aligner"]["status"] == "skipped_mixed_speaker_overlap"


def test_cli_and_config_alignment_fallback(tmp_path: Path) -> None:
    parser = build_parser()
    args_off = parser.parse_args(["process-media", "foo.wav", "--output-dir", str(tmp_path / "o1")])
    assert args_off.alignment_fallback == "off"
    args_qwen = parser.parse_args(
        ["process-media", "foo.wav", "--output-dir", str(tmp_path / "o2"), "--alignment-fallback", "qwen"]
    )
    assert args_qwen.alignment_fallback == "qwen"

    cfg = RecordingPipelineConfig()
    assert cfg.alignment_fallback == "off"

    src = tmp_path / "tiny.wav"
    _write_silence_wav(src, duration_s=1.0)
    with pytest.raises(ValueError, match="Unsupported alignment_fallback"):
        process_media(
            src,
            output_dir=tmp_path / "out",
            client=object(),  # type: ignore[arg-type]
            config=RecordingPipelineConfig(alignment_fallback="invalid"),
        )


def test_qwen_fallback_localizes_oov_span_and_never_migrates_to_earlier_utterance(tmp_path: Path) -> None:
    """Regression for zh_multichunk_synthetic bug where isolated '钾' migrated from utt 1 into utt 0."""
    wav = tmp_path / "chunk_full.wav"
    _write_silence_wav(wav, duration_s=20.0)
    raw_text = "<spk:0>这并不是告别。 钙、钾等元素属于金属。"
    ctc_payload = _make_ctc_payload(
        words=[],
        diar=[
            {"speaker": "spk_0", "start": 1.8, "end": 7.3},
            {"speaker": "spk_0", "start": 11.9, "end": 17.5},
        ],
        error="alignment_failed",
    )

    def _subinterval_runner(win_start: float, win_end: float) -> tuple[str, Any]:
        if win_start < 10.0:
            # Earlier utterance recovers cleanly via native subinterval CTC.
            return (
                "<spk:0>这并不是告别。",
                _make_ctc_payload(
                    words=[
                        {"word": ch, "start": 0.2 + i * 0.4, "end": 0.5 + i * 0.4}
                        for i, ch in enumerate(["这", "并", "不", "是", "告", "别。"])
                    ],
                    diar=[{"speaker": "spk_0", "start": 0.2, "end": 3.0}],
                ),
            )
        # Second subinterval substitutes '甲' for '钾' (no ctc_error, leaving failed_windows empty!).
        return (
            "<spk:0>钙、甲等元素属于金属。",
            _make_ctc_payload(
                words=[
                    {"word": "钙、", "start": 0.3, "end": 0.55},
                    {"word": "甲", "start": 0.55, "end": 0.8},
                    {"word": "等", "start": 0.85, "end": 1.1},
                    {"word": "元", "start": 1.1, "end": 1.35},
                    {"word": "素", "start": 1.35, "end": 1.6},
                    {"word": "属", "start": 1.6, "end": 1.85},
                    {"word": "于", "start": 1.85, "end": 2.1},
                    {"word": "金", "start": 2.1, "end": 2.35},
                    {"word": "属。", "start": 2.35, "end": 2.65},
                ],
                diar=[{"speaker": "spk_0", "start": 0.2, "end": 2.8}],
            ),
        )

    def _wav_extractor(win_start: float, win_end: float) -> Path:
        dst = tmp_path / "sub" / f"{win_start:.2f}_{win_end:.2f}.wav"
        _write_silence_wav(dst, duration_s=max(0.5, win_end - win_start))
        return dst

    # 1. Aligner sees localized window around [11.5..15.0] and full contiguous span text, NEVER full chunk or isolated '钾'.
    class _LocalizedSpanAligner:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def align(self, *, audio_path: Path, text: str, language: str | None, source_offset_s: float) -> dict[str, Any]:
            self.calls.append({"audio_path": str(audio_path), "text": text, "offset": source_offset_s})
            items = []
            t = 0.25
            for ch in text:
                if ch in {"、", "。"} or ch.isspace():
                    continue
                items.append(
                    {
                        "text": ch,
                        "start": source_offset_s + t,
                        "end": source_offset_s + t + 0.2,
                        "valid_timing": True,
                        "timing_error": None,
                    }
                )
                t += 0.25
            return {"units": items, "diagnostic": {"backend": "qwen3_forced_aligner", "status": "ok", "valid_items": len(items)}}

    good_aligner = _LocalizedSpanAligner()
    rec_good = recover_chunk_alignment(
        raw_text=raw_text,
        ctc_payload=ctc_payload,
        source_offset_s=0.0,
        chunk_index=0,
        subinterval_runner=_subinterval_runner,
        external_aligner=good_aligner,
        chunk_wav_path=wav,
        source_language="zh",
        subinterval_wav_extractor=_wav_extractor,
    )
    assert len(good_aligner.calls) == 1
    call = good_aligner.calls[0]
    assert call["offset"] > 10.0  # Localized to 2nd sentence window, NOT chunk 0.0s!
    assert "钙" in call["text"] and "钾" in call["text"] and "金属" in call["text"]
    assert "告别" not in call["text"]

    utts = reconstruct_speaker_utterances(rec_good["units"], source_language="zh")
    assert len(utts) == 2
    assert utts[0]["source_text"] == "这并不是告别。"
    assert utts[1]["source_text"] == "钙、钾等元素属于金属。"
    jia_unit = [u for u in utts[1]["source_characters"] if u["text"] == "钾"][0]
    assert jia_unit["valid_timing"] is True
    assert jia_unit["start"] >= 11.5
    assert jia_unit["alignment_status"] == "external_qwen_recovered"

    # 2. Adversarial aligner returning bogus timestamp 1.92..2.08 (earlier utterance) must be rejected WITHOUT moving '钾'!
    class _MigratingBugAligner:
        def align(self, *, audio_path: Path, text: str, language: str | None, source_offset_s: float) -> dict[str, Any]:
            return {
                "units": [
                    {"text": "钾", "start": 1.92, "end": 2.08, "valid_timing": True, "timing_error": None}
                ],
                "diagnostic": {"backend": "qwen3_forced_aligner", "status": "ok", "valid_items": 1},
            }

    rec_bad = recover_chunk_alignment(
        raw_text=raw_text,
        ctc_payload=ctc_payload,
        source_offset_s=0.0,
        chunk_index=0,
        subinterval_runner=_subinterval_runner,
        external_aligner=_MigratingBugAligner(),
        chunk_wav_path=wav,
        source_language="zh",
        subinterval_wav_extractor=_wav_extractor,
    )
    utts_bad = reconstruct_speaker_utterances(rec_bad["units"], source_language="zh")
    assert len(utts_bad) == 2
    assert utts_bad[0]["source_text"] == "这并不是告别。"
    assert utts_bad[1]["source_text"] == "钙、钾等元素属于金属。"
    jia_bad = [u for u in utts_bad[1]["source_characters"] if u["text"] == "钾"][0]
    assert jia_bad["valid_timing"] is False
    assert jia_bad["start"] is None
    assert jia_bad["timing_error"] == "external_bounds_violation"

