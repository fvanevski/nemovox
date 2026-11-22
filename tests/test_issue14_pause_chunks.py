"""Issue 14: pause-aware inference planning and resume invariants."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nvidia_transcribe.cli import build_parser
from nvidia_transcribe.chunking import ChunkConfig, plan_chunks
from nvidia_transcribe.speech_chunks import (
    detect_silence_intervals, plan_pause_aware_chunks,
)
from nvidia_transcribe.recording import RecordingPipelineConfig, _config_signature
from nvidia_transcribe.media import MediaInfo


def test_pause_aware_cores_choose_real_pauses_and_cover_entire_episode():
    pauses = [(38.5, 40.2), (79.0, 80.2), (119.5, 120.8), (159.0, 160.2)]
    chunks, diag = plan_pause_aware_chunks(
        200.0, pauses, max_requested_core_s=90, requested_context_s=10,
    )
    assert len(chunks) >= 4
    assert chunks[0].core_start_s == 0
    assert chunks[-1].core_end_s == 200
    assert all(a.core_end_s == b.core_start_s for a, b in zip(chunks, chunks[1:]))
    assert all(0 <= c.infer_start_s < c.infer_end_s <= 200 for c in chunks)
    assert all(c.infer_start_s <= c.core_start_s < c.core_end_s <= c.infer_end_s for c in chunks)
    assert diag["selected_pause_count"] >= 3
    assert diag["max_actual_infer_s"] < 65
    assert diag["effective_context_s"] == 4
    assert any(39 <= d["at_s"] <= 40 for d in diag["boundary_decisions"])


def test_undetected_pauses_force_cuts_but_do_not_fake_speech_boundaries():
    chunks, diag = plan_pause_aware_chunks(
        140.0, [], max_requested_core_s=90, requested_context_s=10,
    )
    assert len(chunks) >= 3
    assert diag["selected_pause_count"] == 0
    assert diag["forced_cut_count"] >= 2
    assert all(d["reason"] == "forced_max_core_no_eligible_pause"
               for d in diag["boundary_decisions"])
    assert chunks[-1].core_end_s == 140


def test_short_media_and_fixed_compatibility():
    specs, diag = plan_pause_aware_chunks(
        10.2, [], max_requested_core_s=90, requested_context_s=10,
    )
    assert len(specs) == 1 and specs[0].infer_end_s == 10.2
    assert diag["forced_cut_count"] == 0
    fixed = plan_chunks(200, ChunkConfig(chunk_seconds=90, context_seconds=10))
    assert len(fixed) == 2
    args = build_parser().parse_args([
        "generate-subtitles", "--output-dir", "unused", "--chunk-policy", "fixed",
    ])
    assert args.chunk_policy == "fixed"


def test_detector_parses_stream_bound_silence_intervals_without_file_writes(monkeypatch):
    observed = []

    def fake_run(cmd, **kwargs):
        observed.append(cmd)
        return SimpleNamespace(returncode=0, stderr="\n".join([
            "[silencedetect] silence_start: 35.1",
            "[silencedetect] silence_end: 37.2 | silence_duration: 2.1",
            "[silencedetect] silence_start: 96.0",
            "[silencedetect] silence_end: 97.4 | silence_duration: 1.4",
        ]))

    monkeypatch.setattr("nvidia_transcribe.speech_chunks.subprocess.run", fake_run)
    pauses, diag = detect_silence_intervals(
        "/synthetic/video.webm", audio_stream_index=1, duration_s=200,
    )
    assert pauses == [(35.1, 37.2), (96.0, 97.4)]
    assert observed[0][observed[0].index("-map") + 1] == "0:a:1"
    assert diag["status"] == "complete"


def test_detector_failure_is_reported_not_disguised_as_silence(monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stderr="bad stream")
    monkeypatch.setattr("nvidia_transcribe.speech_chunks.subprocess.run", fake_run)
    pauses, diag = detect_silence_intervals(
        "/synthetic/video.webm", audio_stream_index=0, duration_s=120,
    )
    assert pauses == []
    assert diag["status"] == "failed"
    assert diag["exit_code"] == 1


def test_plan_fingerprint_changes_on_chunk_policy_and_boundaries():
    info = MediaInfo(
        path="/synthetic/input.wav", duration_s=200, format_name="wav",
        size_bytes=300, audio_stream_count=1, selected_audio_stream=0,
        audio_codec="pcm_s16le", sample_rate_hz=16000, channels=1,
        fingerprint_sha256="stable",
    )
    fixed_cfg = RecordingPipelineConfig(chunk_policy="fixed")
    adaptive_cfg = RecordingPipelineConfig(chunk_policy="pause_aware")
    fixed_plan = plan_chunks(200)
    adaptive_plan, diag = plan_pause_aware_chunks(200, [(39, 40), (80, 81)])
    kwargs = dict(media=info, model_name="m", model_endpoint="local")
    from nvidia_transcribe.recording import _config_signature
    fixed_sig = _config_signature(**kwargs, cfg=fixed_cfg, chunks=fixed_plan,
                                  plan_metadata={"schema": "fixed/v1"})
    adaptive_sig = _config_signature(**kwargs, cfg=adaptive_cfg, chunks=adaptive_plan,
                                     plan_metadata=diag)
    assert fixed_sig != adaptive_sig
    altered = _config_signature(**kwargs, cfg=adaptive_cfg, chunks=adaptive_plan,
                                plan_metadata={**diag, "forced_cut_count": 77})
    assert adaptive_sig != altered


def test_invalid_policy_fails_before_any_processing(tmp_path):
    from nvidia_transcribe.recording import process_media
    from nvidia_transcribe.client import NemotronVLLMClient
    with pytest.raises(ValueError, match="chunk_policy"):
        process_media(
            tmp_path / "absent.wav", output_dir=tmp_path / "out",
            client=NemotronVLLMClient(base_url="http://127.0.0.1:8190/v1", model="dummy"),
            config=RecordingPipelineConfig(chunk_policy="invalid"),
        )
