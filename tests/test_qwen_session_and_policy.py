"""CPU-only regression coverage for Qwen admission, persistent session and ownership safety."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from nvidia_transcribe import qwen_aligner, gpu_handoff
from nvidia_transcribe.qwen_aligner import QwenAlignerConfig, QwenAlignerSession, QwenWorkerError
from nvidia_transcribe.selective_alignment import (
    AdmissionPolicy, apply_selective_recovery, plan_selective_recovery,
)


def _native(texts: list[str], diar: list[dict], timed: set[int] | None = None) -> dict:
    timed = timed or set()
    units = [
        {"text": text, "speaker": "spk_0",
         "start": (i * 0.4 + 0.5 if i in timed else None),
         "end": (i * 0.4 + 0.8 if i in timed else None),
         "valid_timing": i in timed,
         "alignment_status": "ctc_aligned" if i in timed else "untimed",
         "timing_error": None if i in timed else "alignment_failed"}
        for i, text in enumerate(texts)
    ]
    return {"units": units, "diarization": diar, "diagnostic": {"status": "untimed"},
            "alignment_status": "untimed", "speaker_mapping": {}}


def _diar(start: float = 0.4, end: float = 3.0, speaker: str = "spk_0") -> dict:
    return {"start": start, "end": end, "speaker": speaker, "valid_timing": True}


def test_admission_recovers_anchored_untimed_utterance_and_preserves_words() -> None:
    native = _native(["你", "好", "朋友。"], [_diar()])
    candidates, decision = plan_selective_recovery(
        native=native, raw_text="<spk:0>你好朋友。", chunk_start_s=0,
        chunk_end_s=5, policy=AdmissionPolicy(),
    )
    assert len(candidates) == 1
    assert decision[0]["status"] == "admitted"
    native_copy = [dict(u) for u in native["units"]]

    class Aligner:
        calls = 0
        def align(self, **kwargs: object) -> dict:
            self.calls += 1
            return {"units": [
                {"text": t, "start": kwargs["source_offset_s"] + i * .25,
                 "end": kwargs["source_offset_s"] + i * .25 + .2,
                 "valid_timing": True}
                for i, t in enumerate(["你", "好", "朋友。"])
            ], "diagnostic": {"status": "ok"}}

    aligner = Aligner()
    out = apply_selective_recovery(
        native=native, raw_text="<spk:0>你好朋友。",
        chunk_start_s=0, chunk_end_s=5, chunk_index=0, source_language="zh",
        aligner=aligner, extract_wav=lambda a, b: Path("/synthetic/window.wav"),
        policy=AdmissionPolicy(),
    )
    assert aligner.calls == 1
    assert out["diagnostic"]["external_aligner"]["recovered_units"] == 3
    assert [u["text"] for u in out["units"]] == [u["text"] for u in native_copy]
    assert all(u["valid_timing"] for u in out["units"])


def test_admission_default_skips_small_gap_in_usable_cue() -> None:
    native = _native(["我", "们", "在", "这", "里", "等。"], [_diar()], timed={0, 1, 2, 3, 5})
    candidates, decisions = plan_selective_recovery(
        native=native, raw_text="<spk:0>我们在这里等。",
        chunk_start_s=0, chunk_end_s=5, policy=AdmissionPolicy(),
    )
    assert not candidates
    assert decisions[0]["status"] == "skipped_low_subtitle_value"
    finer, _ = plan_selective_recovery(
        native=native, raw_text="<spk:0>我们在这里等。",
        chunk_start_s=0, chunk_end_s=5, policy=AdmissionPolicy(mode="fine"),
    )
    assert finer


def test_admission_skips_unanchored_ambiguous_multiwindow_and_overlap() -> None:
    native = _native(["你", "好", "呀。"], [_diar(0.2, 1.2), _diar(2.0, 3.0)])
    candidates, decisions = plan_selective_recovery(
        native=native, raw_text="<spk:0>你好呀。",
        chunk_start_s=0, chunk_end_s=5, policy=AdmissionPolicy(),
    )
    assert not candidates
    assert decisions[0]["status"] == "skipped_unanchored_or_ambiguous"
    native["diarization"] = [_diar(0.2, 2.0), _diar(1.0, 3.0, speaker="spk_1")]
    candidates, decisions = plan_selective_recovery(
        native=native, raw_text="<spk:0>你好呀。<spk:1>哦",
        chunk_start_s=0, chunk_end_s=5, policy=AdmissionPolicy(),
    )
    assert not candidates
    assert decisions[0]["status"] == "skipped_mixed_speaker_overlap"


def test_no_eligible_spans_does_not_touch_aligner() -> None:
    native = _native(["你", "好", "呀。"], [_diar()], timed={0, 1, 2})
    class NoTouch:
        def align(self, **_kwargs: object) -> dict:
            raise AssertionError("No aligner should run")
    out = apply_selective_recovery(
        native=native, raw_text="<spk:0>你好呀。", chunk_start_s=0,
        chunk_end_s=5, chunk_index=0, source_language="zh",
        aligner=NoTouch(), extract_wav=lambda a, b: Path("/nonexistent"),
        policy=AdmissionPolicy(),
    )
    assert out["diagnostic"]["external_aligner"]["candidate_count"] == 0


def _fake_worker(tmp_path: Path, *, return_device: str = "cuda:0", delay_s: float = 0) -> Path:
    path = tmp_path / "worker.py"
    path.write_text(
        "import json,sys,time\n"
        "loaded=0\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        " a=r.get('action')\n"
        " if a=='init':\n"
        "  loaded+=1\n"
        "  out=dict(status='ok',device_used="+repr(return_device)+",dtype='torch.bfloat16',model_load_count=loaded,model_load_wall_s=0.01)\n"
        " elif a=='align':\n"
        "  time.sleep("+repr(delay_s)+")\n"
        "  out=dict(status='ok',items=[dict(text='钾',start=0.1,end=0.4)],device_used="+repr(return_device)+",dtype='torch.bfloat16',inference_wall_s=0.01)\n"
        " elif a=='shutdown':\n"
        "  out=dict(status='ok')\n"
        " else:\n"
        "  out=dict(status='worker_error')\n"
        " out.update(protocol=1,id=r.get('id'))\n"
        " print(json.dumps(out),flush=True)\n"
        " if a=='shutdown':break\n",
        encoding="utf-8",
    )
    return path


def test_one_model_load_and_one_pid_for_multiple_spans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _fake_worker(tmp_path)
    monkeypatch.setattr(qwen_aligner, "WORKER_SCRIPT", worker)
    model = tmp_path / "model"
    model.mkdir()
    cfg = QwenAlignerConfig(worker_python=sys.executable, model_path=str(model), device="cuda:0")
    session = QwenAlignerSession(cfg)
    assert session.process is None and session.metrics["model_load_count"] == 0
    with session:
        one = session.align(audio_path=worker, text="钾", language="Chinese", source_offset_s=0)
        pid = session.process.pid
        two = session.align(audio_path=worker, text="钾", language="Chinese", source_offset_s=0)
        assert session.process.pid == pid
        assert session.metrics["model_load_count"] == 1
        assert session.metrics["align_count"] == 2
        assert one["items"] == two["items"]
    assert session.process is None


def test_gpu_required_rejects_misreported_cpu_and_reaps_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen_aligner, "WORKER_SCRIPT", _fake_worker(tmp_path, return_device="cpu"))
    model = tmp_path / "model"
    model.mkdir()
    session = QwenAlignerSession(QwenAlignerConfig(
        worker_python=sys.executable, model_path=str(model), device="cuda:0",
    ))
    with pytest.raises(QwenWorkerError, match="Expected cuda:0"):
        session.start()
    assert session.failed and session.process is None


def test_explicit_cpu_is_not_a_silent_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen_aligner, "WORKER_SCRIPT", _fake_worker(tmp_path, return_device="cpu"))
    model = tmp_path / "model"
    model.mkdir()
    with QwenAlignerSession(QwenAlignerConfig(
        worker_python=sys.executable, model_path=str(model), device="cpu",
    )) as session:
        response = session.align(audio_path=model, text="钾", language="Chinese", source_offset_s=0)
        assert response["device_used"] == "cpu"


def test_worker_timeout_is_not_retried_or_converted_to_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen_aligner, "WORKER_SCRIPT", _fake_worker(tmp_path, delay_s=0.4))
    model = tmp_path / "model"
    model.mkdir()
    session = QwenAlignerSession(QwenAlignerConfig(
        worker_python=sys.executable, model_path=str(model), timeout_s=0.05,
    ))
    with pytest.raises(QwenWorkerError) as err:
        session.align(audio_path=model, text="钾", language="Chinese", source_offset_s=0)
    assert err.value.status == "worker_timeout"
    assert session.failed and session.process is None


def _managed_cfg(tmp_path: Path) -> gpu_handoff.ManagedASRConfig:
    return gpu_handoff.ManagedASRConfig(
        unit="nemovox-asr.service",
        lease_path=tmp_path / "asr.lock",
        base_url="http://127.0.0.1:8190/v1",
        served_model="test-nemotron",
        journal_path=tmp_path / "journal.json",
    )


def test_managed_gpu_busy_lease_never_stops_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _managed_cfg(tmp_path)
    monkeypatch.setattr(gpu_handoff, "_free_vram_mb", lambda: 400)
    monkeypatch.setattr(gpu_handoff, "_check_lease", lambda path: (_ for _ in ()).throw(
        gpu_handoff.ManagedASRBlocked("asr_busy")))
    monkeypatch.setattr(gpu_handoff, "_run_systemctl", lambda *a: (_ for _ in ()).throw(
        AssertionError("must not control shared service")))
    with pytest.raises(gpu_handoff.ManagedASRBlocked) as err:
        with gpu_handoff.managed_asr_handoff(cfg):
            pass
    assert err.value.status == "asr_busy"


def test_managed_gpu_restores_after_failure_and_journals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _managed_cfg(tmp_path)
    cfg.lease_path.write_text("", encoding="utf-8")
    cfg.lease_path.chmod(0o600)
    state = {"active": True, "invocation": "original", "free": 400}
    calls: list[str] = []

    def fake_systemctl(*args: str) -> str:
        calls.append(args[0])
        if args[0] == "stop":
            state.update(active=False, free=8000)
        if args[0] == "start":
            state.update(active=True, invocation="restored")
        return ""

    def properties(unit: str) -> dict[str, str]:
        return {"ActiveState": "active" if state["active"] else "inactive",
                "MainPID": "99" if state["active"] else "0",
                "InvocationID": state["invocation"]}

    monkeypatch.setattr(gpu_handoff, "_free_vram_mb", lambda: state["free"])
    monkeypatch.setattr(gpu_handoff, "_verified_owner", lambda cfg: {
        "ActiveState": "active", "InvocationID": "original",
    })
    monkeypatch.setattr(gpu_handoff, "_run_systemctl", fake_systemctl)
    monkeypatch.setattr(gpu_handoff, "_wait_unit",
                        lambda unit, active, deadline: properties(unit))
    monkeypatch.setattr(gpu_handoff, "_unit_properties", properties)
    monkeypatch.setattr(gpu_handoff, "_models_healthy", lambda *a, **k: state["active"])
    with pytest.raises(ValueError, match="synthetic"):
        with gpu_handoff.managed_asr_handoff(cfg) as telemetry:
            assert telemetry["released"] is True
            raise ValueError("synthetic failure")
    assert calls == ["stop", "start"]
    assert state["active"]
    assert json.loads(cfg.journal_path.read_text())["phase"] == "restored"


def test_interrupted_managed_journal_blocks_reentrant_takeover(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _managed_cfg(tmp_path)
    cfg.journal_path.write_text('{"phase":"pending_restore"}', encoding="utf-8")
    monkeypatch.setattr(gpu_handoff, "_free_vram_mb", lambda: 200)
    monkeypatch.setattr(gpu_handoff, "_run_systemctl", lambda *a: (_ for _ in ()).throw(
        AssertionError("must not control unit before recovery")))
    with pytest.raises(gpu_handoff.ManagedASRBlocked) as err:
        with gpu_handoff.managed_asr_handoff(cfg):
            pass
    assert err.value.status == "prior_handoff_requires_recovery"


def test_qwen_pipeline_finishes_all_native_chunks_before_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration regression: optional GPU aligner must not run before ASR completes."""
    import wave
    from nvidia_transcribe import recording
    from nvidia_transcribe.recording import RecordingPipelineConfig, process_media

    source = tmp_path / "two_chunks.wav"
    with wave.open(str(source), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 25)

    events: list[str] = []

    class Client:
        model = "test-nemotron"
        base_url = "http://127.0.0.1:8190/v1"

        def transcribe(self, path: Path, _request: object) -> dict:
            events.append("native")
            raw = {
                "choices": [{"message": {"content": "<spk:0>你好朋友。"}}],
                "ctc_timestamps": {
                    "words": [], "error": "alignment_failed",
                    "diarization": [
                        {"speaker": 0, "start": 0.4, "end": 3.0},
                    ],
                    "speaker_tag_to_diarization_speaker": {"0": 0},
                },
            }
            return {"raw_response": raw, "contract": {"raw_text": "<spk:0>你好朋友。"},
                    "latency_s": 0.01}

    class Aligner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.metrics = {"model_load_count": 1, "align_count": 0}
        def __enter__(self) -> "Aligner":
            return self
        def __exit__(self, *_args: object) -> None:
            return None
        def align(self, **_kwargs: object) -> dict:
            events.append("qwen")
            self.metrics["align_count"] += 1
            return {"units": [], "diagnostic": {"status": "no_valid_units"}}

    monkeypatch.setattr(recording, "QwenForcedAlignerAdapter", Aligner)
    res = process_media(
        source, output_dir=tmp_path / "out", client=Client(),
        config=RecordingPipelineConfig(
            source_language="zh", speaker_embeddings="none",
            enable_subinterval_recovery=False, alignment_fallback="qwen",
            chunk_seconds=12, min_tail_seconds=5,
        ),
    )
    assert events.count("native") >= 2
    assert events.count("qwen") >= 1
    first_qwen = events.index("qwen")
    assert events[:first_qwen] == ["native"] * first_qwen
    assert (tmp_path / "out" / "native-alignment.checkpoint.json").is_file()
    assert res["contract"]["model_provenance"]["qwen_recovery"]["eligible_candidates"] >= 1


def test_qwen_cli_policy_defaults_and_explicit_managed_options() -> None:
    from nvidia_transcribe.cli import build_parser
    parser = build_parser()
    off = parser.parse_args(["process-media", "dummy.wav", "--output-dir", "out"])
    assert off.alignment_fallback == "off"
    assert off.qwen_device == "cuda:0"
    assert off.qwen_policy == "subtitle"
    assert off.qwen_gpu_mode == "available"
    controlled = parser.parse_args([
        "generate-subtitles", "dummy.wav", "--output-dir", "out",
        "--alignment-fallback", "qwen", "--qwen-policy", "fine",
        "--qwen-gpu-mode", "managed", "--qwen-owned-asr-unit", "nemovox-asr.service",
        "--qwen-asr-lease-path", "/run/user/1000/nemovox-asr.lock",
    ])
    assert controlled.qwen_policy == "fine"
    assert controlled.qwen_owned_asr_unit == "nemovox-asr.service"
    assert parser.parse_args([
        "recover-qwen-asr", "--asr-unit", "nemovox-asr.service",
        "--base-url", "http://127.0.0.1:8190/v1", "--model", "m",
        "--lease-path", "/tmp/lease", "--journal", "/tmp/journal",
    ]).command == "recover-qwen-asr"


def test_managed_sigterm_restores_original_asr_and_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal Studio SIGTERM unwinds the managed ASR context before exit."""
    import signal

    cfg = _managed_cfg(tmp_path)
    cfg.lease_path.write_text("", encoding="utf-8")
    cfg.lease_path.chmod(0o600)
    state = {"active": True, "free": 300}
    commands: list[str] = []
    prior_handler = signal.getsignal(signal.SIGTERM)

    def control(*args: str) -> str:
        commands.append(args[0])
        if args[0] == "stop":
            state.update(active=False, free=5000)
        if args[0] == "start":
            state["active"] = True
        return ""

    monkeypatch.setattr(gpu_handoff, "_free_vram_mb", lambda: state["free"])
    monkeypatch.setattr(gpu_handoff, "_verified_owner", lambda cfg: {
        "ActiveState": "active", "InvocationID": "original",
        "ExecStart": "{ path=/bin/true ; argv[]=/bin/true ; }",
        "FragmentPath": "/synthetic/nemovox-asr.service",
    })
    monkeypatch.setattr(gpu_handoff, "_run_systemctl", control)
    monkeypatch.setattr(gpu_handoff, "_wait_unit",
                        lambda unit, active, deadline: {
                            "ActiveState": "active" if state["active"] else "inactive",
                            "MainPID": "123" if state["active"] else "0",
                            "InvocationID": "restored" if state["active"] else "original",
                        })
    monkeypatch.setattr(gpu_handoff, "_unit_properties",
                        lambda unit: {"ActiveState": "active" if state["active"] else "inactive",
                                      "InvocationID": "restored"})
    monkeypatch.setattr(gpu_handoff, "_models_healthy", lambda *a, **kw: state["active"])

    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        with gpu_handoff.managed_asr_handoff(cfg) as telemetry:
            assert telemetry["released"] is True
            active_handler = signal.getsignal(signal.SIGTERM)
            assert active_handler is not prior_handler
            active_handler(signal.SIGTERM, None)
    assert commands == ["stop", "start"]
    assert state["active"]
    assert signal.getsignal(signal.SIGTERM) is prior_handler
    assert json.loads(cfg.journal_path.read_text())["phase"] == "restored"


def test_available_gpu_never_stops_asr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _managed_cfg(tmp_path)
    monkeypatch.setattr(gpu_handoff, "_free_vram_mb", lambda: 6000)
    monkeypatch.setattr(gpu_handoff, "_check_lease",
                        lambda path: (_ for _ in ()).throw(AssertionError("no lease needed")))
    monkeypatch.setattr(gpu_handoff, "_run_systemctl",
                        lambda *args: (_ for _ in ()).throw(AssertionError("never control ASR")))
    with gpu_handoff.managed_asr_handoff(cfg) as report:
        assert report["status"] == "cuda_already_available"
        assert not report["released"]


def test_pending_journal_blocks_even_when_gpu_has_capacity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _managed_cfg(tmp_path)
    cfg.journal_path.write_text('{"phase":"pending_restore"}', encoding="utf-8")
    monkeypatch.setattr(gpu_handoff, "_free_vram_mb", lambda: 9000)
    monkeypatch.setattr(gpu_handoff, "_run_systemctl",
                        lambda *args: (_ for _ in ()).throw(AssertionError("no takeover")))
    with pytest.raises(gpu_handoff.ManagedASRBlocked) as error:
        with gpu_handoff.managed_asr_handoff(cfg):
            pass
    assert error.value.status == "prior_handoff_requires_recovery"


def test_recovery_refuses_changed_asr_unit_without_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvidia_transcribe.gpu_handoff import recover_pending_managed_asr
    cfg = _managed_cfg(tmp_path)
    cfg.lease_path.write_text("", encoding="utf-8")
    cfg.lease_path.chmod(0o600)
    cfg.journal_path.write_text(json.dumps({
        "phase": "pending_restore",
        "unit": cfg.unit,
        "served_model": cfg.served_model,
        "base_url": cfg.base_url,
        "originally_active": True,
        "unit_exec_sha256": "f" * 64,
        "unit_fragment_path": "/synthetic/old-asr.service",
    }), encoding="utf-8")
    monkeypatch.setattr(gpu_handoff, "_unit_properties", lambda u: {
        "LoadState": "loaded", "ActiveState": "inactive",
        "Environment": "NEMOVOX_MANAGED_ASR=1 NEMOVOX_ASR_LEASE_PATH=" + str(cfg.lease_path),
        "FragmentPath": "/synthetic/new-asr.service",
        "ExecStart": "changed",
    })
    monkeypatch.setattr(gpu_handoff, "_run_systemctl",
                        lambda *a: (_ for _ in ()).throw(AssertionError("must not restart changed unit")))
    with pytest.raises(gpu_handoff.ManagedASRBlocked) as err:
        recover_pending_managed_asr(cfg)
    assert err.value.status == "asr_unit_changed_since_handoff"


def test_default_off_never_loads_worker_or_touches_managed_services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wave
    from nvidia_transcribe import recording
    from nvidia_transcribe.recording import RecordingPipelineConfig, process_media

    source = tmp_path / "oov.wav"
    with wave.open(str(source), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 5)

    class Client:
        model = "test-nemotron"
        base_url = "http://127.0.0.1:8190/v1"

        def transcribe(self, path: Path, _request: object) -> dict:
            return {
                "raw_response": {
                    "choices": [{"message": {"content": "<spk:0>钙、钾等元素属于金属。"}}],
                    "ctc_timestamps": {
                        "words": [],
                        "error": "alignment_failed",
                        "diarization": [{"speaker": 0, "start": 0.5, "end": 4.0}],
                        "speaker_tag_to_diarization_speaker": {},
                    },
                },
                "contract": {"raw_text": "<spk:0>钙、钾等元素属于金属。"},
                "latency_s": 0.01,
            }

    monkeypatch.setattr(
        recording,
        "QwenForcedAlignerAdapter",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("off must never load Qwen worker")),
    )
    monkeypatch.setattr(
        recording,
        "managed_asr_handoff",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("off must never manage GPU services")),
    )
    res = process_media(
        source,
        output_dir=tmp_path / "out",
        client=Client(),
        config=RecordingPipelineConfig(
            source_language="zh",
            speaker_embeddings="none",
            enable_subinterval_recovery=False,
            alignment_fallback="off",
        ),
    )
    assert res["contract"]["model_provenance"]["qwen_recovery"] == {"status": "disabled"}
    assert not (tmp_path / "out" / "native-alignment.checkpoint.json").exists()


def test_admission_merges_short_intra_utterance_diar_pauses_and_cross_speaker_neighbor() -> None:
    # 1. Single untimed sentence split into 3 Sortformer pauses <=0.6s merges into 1 speech window.
    native_pauses = _native(
        ["钙、", "钾", "等", "金", "属。"],
        [_diar(1.12, 3.32), _diar(3.75, 5.08), _diar(5.47, 6.63)],
    )
    cands, decs = plan_selective_recovery(
        native=native_pauses,
        raw_text="<spk:0>钙、钾等金属。",
        chunk_start_s=0.0,
        chunk_end_s=8.28,
        policy=AdmissionPolicy(),
    )
    assert len(cands) == 1
    assert decs[0]["status"] == "admitted"
    assert cands[0].start_s == pytest.approx(0.77, abs=0.01)
    assert cands[0].end_s == pytest.approx(6.98, abs=0.01)

    # 2. Cross-speaker right boundary + unique grouped diarization localizes middle OOV sentence.
    units = [
        {"text": "好", "speaker": "0", "start": 1.8, "end": 2.2, "valid_timing": True},
        {"text": "啊。", "speaker": "0", "start": 2.2, "end": 7.28, "valid_timing": True},
        {"text": "钙、", "speaker": "0", "start": None, "end": None, "valid_timing": False},
        {"text": "钾", "speaker": "0", "start": None, "end": None, "valid_timing": False},
        {"text": "金属。", "speaker": "0", "start": None, "end": None, "valid_timing": False},
        {"text": "下", "speaker": "1", "start": 22.61, "end": 23.0, "valid_timing": True},
    ]
    diar = [
        _diar(1.9, 7.3, speaker="0"),
        _diar(12.03, 14.16, speaker="0"),
        _diar(14.64, 15.91, speaker="0"),
        _diar(16.39, 17.49, speaker="0"),
        _diar(22.41, 27.06, speaker="1"),
    ]
    cands2, decs2 = plan_selective_recovery(
        native={"units": units, "diarization": diar},
        raw_text="<spk:0>好啊。钙、钾金属。<spk:1>下",
        chunk_start_s=0.0,
        chunk_end_s=35.0,
        policy=AdmissionPolicy(),
    )
    assert len(cands2) == 1
    assert decs2[0]["status"] == "admitted"
    assert cands2[0].start_s == pytest.approx(11.68, abs=0.01)
    assert cands2[0].end_s == pytest.approx(17.84, abs=0.01)


def test_worker_protocol_id_mismatch_reaps_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_worker = tmp_path / "bad_id_worker.py"
    bad_worker.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        " r = json.loads(line)\n"
        " print(json.dumps({'protocol': 1, 'id': 999999, 'status': 'ok'}), flush=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(qwen_aligner, "WORKER_SCRIPT", bad_worker)
    model = tmp_path / "model"
    model.mkdir()
    session = QwenAlignerSession(QwenAlignerConfig(
        worker_python=sys.executable, model_path=str(model), device="cpu",
    ))
    with pytest.raises(QwenWorkerError) as err:
        session.start()
    assert err.value.status == "malformed_worker_output"
    assert session.failed and session.process is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux PR_SET_PDEATHSIG check")
def test_linux_parent_death_sigkill_reaps_harmless_worker(tmp_path: Path) -> None:
    import signal
    import subprocess
    import time

    pid_file = tmp_path / "worker.pid"
    fake_worker = tmp_path / "pdeath_worker.py"
    fake_worker.write_text(
        "import ctypes, os, signal, sys, time\n"
        "from pathlib import Path\n"
        "orig_parent = int(os.environ['NEMOVOX_QWEN_PARENT_PID'])\n"
        "libc = ctypes.CDLL(None, use_errno=True)\n"
        "assert libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0\n"
        "if os.getppid() != orig_parent:\n"
        "    sys.exit(3)\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    parent_script = tmp_path / "parent.py"
    parent_script.write_text(
        "import os, subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, {str(fake_worker)!r}], "
        "start_new_session=True, env={**os.environ, 'NEMOVOX_QWEN_PARENT_PID': str(os.getpid())})\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    parent = subprocess.Popen([sys.executable, str(parent_script)])
    worker_pid: int | None = None
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not pid_file.is_file():
            time.sleep(0.05)
        assert pid_file.is_file(), "Fake worker did not arm PR_SET_PDEATHSIG in time"
        worker_pid = int(pid_file.read_text(encoding="utf-8"))
        assert Path(f"/proc/{worker_pid}").exists()
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(timeout=5)
        reap_deadline = time.monotonic() + 5.0
        while time.monotonic() < reap_deadline:
            try:
                os.kill(worker_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"Worker {worker_pid} survived parent SIGKILL despite PR_SET_PDEATHSIG")
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)
        if worker_pid is not None:
            try:
                os.kill(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_managed_config_and_owner_rejects_studio_unmarked_and_live_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Studio"):
        gpu_handoff.ManagedASRConfig(
            unit="nemovox-studio.service",
            lease_path=tmp_path / "l",
            base_url="http://127.0.0.1:8190/v1",
            served_model="m",
            journal_path=tmp_path / "j",
        )
    with pytest.raises(ValueError, match="nemovox-"):
        gpu_handoff.ManagedASRConfig(
            unit="other-vllm.service",
            lease_path=tmp_path / "l",
            base_url="http://127.0.0.1:8190/v1",
            served_model="m",
            journal_path=tmp_path / "j",
        )
    cfg = _managed_cfg(tmp_path)
    monkeypatch.setattr(gpu_handoff, "_unit_properties", lambda u: {
        "LoadState": "loaded",
        "ActiveState": "active",
        "FragmentPath": "/synthetic/nemovox-asr.service",
        "InvocationID": "abc",
        "MainPID": str(os.getpid()),
        "Environment": "",
        "ExecStart": "vllm serve nemotron --port 8190",
    })
    with pytest.raises(gpu_handoff.ManagedASRBlocked) as err:
        gpu_handoff._verified_owner(cfg)
    assert err.value.status == "asr_not_cooperatively_owned"

    monkeypatch.setattr(gpu_handoff, "_unit_properties", lambda u: {
        "LoadState": "loaded",
        "ActiveState": "active",
        "FragmentPath": "/synthetic/nemovox-asr.service",
        "InvocationID": "abc",
        "MainPID": str(os.getpid()),
        "Environment": f"NEMOVOX_MANAGED_ASR=1 NEMOVOX_ASR_LEASE_PATH={cfg.lease_path}",
        "ExecStart": "vllm serve nemotron --port 8190",
    })
    monkeypatch.setattr(Path, "read_bytes", lambda self: b"vllm\x00serve\x00nemotron\x00--port\x008190")
    monkeypatch.setattr(gpu_handoff, "_models_healthy", lambda *a, **kw: True)
    import subprocess
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, stdout="127.0.0.1:8190 127.0.0.1:54321\n", stderr=""),
    )
    with pytest.raises(gpu_handoff.ManagedASRBlocked) as err2:
        gpu_handoff._verified_owner(cfg)
    assert err2.value.status == "asr_has_live_connections"



def test_unique_diar_window_respects_adjacent_native_timed_speech() -> None:
    """The sole untimed sentence must not consume an already-timed neighbor."""
    units = [
        {"text": "你", "speaker": "0", "start": 0.5, "end": 0.9, "valid_timing": True},
        {"text": "好。", "speaker": "0", "start": 1.0, "end": 1.7, "valid_timing": True},
        {"text": "世", "speaker": "0", "start": None, "end": None, "valid_timing": False},
        {"text": "界", "speaker": "0", "start": None, "end": None, "valid_timing": False,
         "join_no_space_prev": True},
        {"text": "啊。", "speaker": "0", "start": None, "end": None, "valid_timing": False,
         "join_no_space_prev": True},
        {"text": "再", "speaker": "0", "start": 6.5, "end": 6.8, "valid_timing": True},
        {"text": "见。", "speaker": "0", "start": 6.9, "end": 7.2, "valid_timing": True},
    ]
    # A single, continuous same-speaker diarization interval covers three
    # sentences; native CTC provides trustworthy bounds for the outer two.
    native = {"units": units, "diarization": [_diar(0.2, 8.0, speaker="0")]}
    cands, decisions = plan_selective_recovery(
        native=native, raw_text="<spk:0>你好。世界啊。再见。",
        chunk_start_s=0.0, chunk_end_s=9.0, policy=AdmissionPolicy(),
    )
    assert len(cands) == 1
    assert decisions[0]["status"] == "admitted"
    assert cands[0].lo == 2 and cands[0].hi == 5
    assert cands[0].start_s >= 1.65
    assert cands[0].end_s <= 6.55
    assert cands[0].text == "世界啊。"


def test_neighbor_bounds_fail_closed_if_candidate_overlaps_native_timing() -> None:
    units = [
        {"text": "你。", "speaker": "0", "start": 1.0, "end": 3.8, "valid_timing": True},
        {"text": "你好朋友。", "speaker": "0", "start": None, "end": None, "valid_timing": False},
        {"text": "再见。", "speaker": "1", "start": 3.9, "end": 5.0, "valid_timing": True},
    ]
    native = {"units": units, "diarization": [_diar(0.5, 3.85, speaker="0"),
                                               _diar(3.9, 5.0, speaker="1")]}
    cands, decisions = plan_selective_recovery(
        native=native, raw_text="<spk:0>你。你好朋友。<spk:1>再见。",
        chunk_start_s=0.0, chunk_end_s=6.0, policy=AdmissionPolicy(),
    )
    assert not cands
    assert decisions[0]["status"] == "skipped_conflicting_or_long_window"


@pytest.mark.parametrize("marker", [
    "NEMOVOX_MANAGED_ASR=1 NEMOVOX_ASR_LEASE_PATH={lease}.other",
    "NEMOVOX_MANAGED_ASR=10 NEMOVOX_ASR_LEASE_PATH={lease}",
    "NEMOVOX_MANAGED_ASR=1 NEMOVOX_ASR_LEASE_PATH={lease} "
    "NEMOVOX_ASR_LEASE_PATH={lease}.other",
    "NEMOVOX_MANAGED_ASR=1 NEMOVOX_ASR_LEASE_PATH={lease}.other "
    "NEMOVOX_ASR_LEASE_PATH={lease}",
])
def test_managed_owner_rejects_nonexact_or_duplicate_lease_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str,
) -> None:
    cfg = _managed_cfg(tmp_path)
    monkeypatch.setattr(gpu_handoff, "_unit_properties", lambda u: {
        "LoadState": "loaded", "ActiveState": "active",
        "FragmentPath": "/synthetic/nemovox-asr.service",
        "InvocationID": "abc", "MainPID": str(os.getpid()),
        "Environment": marker.format(lease=cfg.lease_path),
        "ExecStart": "vllm serve nemotron --port 8190",
    })
    monkeypatch.setattr(Path, "read_bytes", lambda self: (
        _ for _ in ()
    ).throw(AssertionError("reject ownership before inspecting process")))
    with pytest.raises(gpu_handoff.ManagedASRBlocked) as error:
        gpu_handoff._verified_owner(cfg)
    assert error.value.status == "asr_not_cooperatively_owned"


def test_cooperative_marker_parser_accepts_exact_quoted_assignment(tmp_path: Path) -> None:
    lease = tmp_path / "space in name.lock"
    assert gpu_handoff._has_exact_cooperative_markers(
        f'NEMOVOX_MANAGED_ASR=1 "NEMOVOX_ASR_LEASE_PATH={lease}"', lease,
    )
    assert not gpu_handoff._has_exact_cooperative_markers(
        f'NEMOVOX_MANAGED_ASR=1 "NEMOVOX_ASR_LEASE_PATH={lease}.suffix"', lease,
    )
    assert not gpu_handoff._has_exact_cooperative_markers(
        'NEMOVOX_MANAGED_ASR=1 "NEMOVOX_ASR_LEASE_PATH=unterminated', lease,
    )


def test_managed_crash_recovery_rejects_prefix_lease_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    cfg = _managed_cfg(tmp_path)
    cfg.lease_path.write_text("", encoding="utf-8")
    cfg.lease_path.chmod(0o600)
    command = "vllm serve nemotron --port 8190"
    fragment = "/synthetic/nemovox-asr.service"
    cfg.journal_path.write_text(json.dumps({
        "phase": "pending_restore", "unit": cfg.unit,
        "served_model": cfg.served_model, "base_url": cfg.base_url,
        "originally_active": True,
        "unit_exec_sha256": hashlib.sha256(command.encode()).hexdigest(),
        "unit_fragment_path": fragment,
    }), encoding="utf-8")
    monkeypatch.setattr(gpu_handoff, "_unit_properties", lambda u: {
        "LoadState": "loaded", "ActiveState": "inactive",
        "Environment": f"NEMOVOX_MANAGED_ASR=1 NEMOVOX_ASR_LEASE_PATH={cfg.lease_path}.other",
        "FragmentPath": fragment, "ExecStart": command,
    })
    monkeypatch.setattr(gpu_handoff, "_run_systemctl", lambda *args: (
        _ for _ in ()
    ).throw(AssertionError("must not start mismatched ASR")))
    with pytest.raises(gpu_handoff.ManagedASRBlocked) as error:
        gpu_handoff.recover_pending_managed_asr(cfg)
    assert error.value.status == "asr_not_cooperatively_owned"
    assert json.loads(cfg.journal_path.read_text())["phase"] == "pending_restore"


@pytest.mark.parametrize("leading", [True, False])
def test_partly_timed_span_covers_diarized_untimed_edge(leading: bool) -> None:
    untimed = [
        {"text": word, "speaker": "spk_0", "start": None, "end": None,
         "valid_timing": False}
        for word in ("好", "久", "啊")
    ]
    timed = {"text": "回来了", "speaker": "spk_0", "start": 7.0, "end": 8.0,
             "valid_timing": True}
    if leading:
        units = untimed + [timed]
        diar = [_diar(0.5, 8.0)]
    else:
        timed = {**timed, "start": 2.0, "end": 3.0}
        units = [timed] + untimed
        diar = [_diar(2.0, 11.0)]
    candidates, decisions = plan_selective_recovery(
        native={"units": units, "diarization": diar},
        raw_text="<spk:0>好久啊回来了",
        chunk_start_s=0.0, chunk_end_s=12.0, policy=AdmissionPolicy(),
    )
    assert len(candidates) == 1
    assert decisions[0]["status"] == "admitted"
    if leading:
        assert candidates[0].start_s <= 0.5
        assert candidates[0].end_s >= 8.0
    else:
        assert candidates[0].start_s <= 2.0
        assert candidates[0].end_s >= 11.0


def test_partly_timed_untimed_edge_without_unique_diarization_is_skipped() -> None:
    units = [
        {"text": "好", "speaker": "spk_0", "start": None, "end": None,
         "valid_timing": False},
        {"text": "久", "speaker": "spk_0", "start": None, "end": None,
         "valid_timing": False},
        {"text": "回来了", "speaker": "spk_0", "start": 7.0, "end": 8.0,
         "valid_timing": True},
    ]
    for diar in ([], [_diar(0.2, 2.0), _diar(6.5, 8.2)]):
        candidates, decisions = plan_selective_recovery(
            native={"units": units, "diarization": diar},
            raw_text="<spk:0>好久回来了", chunk_start_s=0.0,
            chunk_end_s=10.0, policy=AdmissionPolicy(),
        )
        assert not candidates
        assert decisions[0]["status"] == "skipped_partial_span_unanchored_edge"


def test_restoration_systemctl_transport_error_is_fatal_and_journal_stays_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _managed_cfg(tmp_path)
    cfg.lease_path.write_text("", encoding="utf-8")
    cfg.lease_path.chmod(0o600)
    state = {"active": True, "free": 100}
    def control(*args: str) -> str:
        if args[0] == "stop":
            state.update(active=False, free=8000)
            return ""
        if args[0] == "start":
            raise gpu_handoff.ManagedASRBlocked("systemctl_unavailable", "TimeoutExpired")
        raise AssertionError(args)

    monkeypatch.setattr(gpu_handoff, "_free_vram_mb", lambda: state["free"])
    monkeypatch.setattr(gpu_handoff, "_verified_owner", lambda config: {
        "ActiveState": "active", "InvocationID": "original",
        "ExecStart": "vllm serve nemotron --port 8190",
        "FragmentPath": "/synthetic/nemovox-asr.service",
    })
    monkeypatch.setattr(gpu_handoff, "_run_systemctl", control)
    monkeypatch.setattr(gpu_handoff, "_unit_properties", lambda unit: {
        "ActiveState": "active" if state["active"] else "inactive",
        "InvocationID": "original",
    })
    monkeypatch.setattr(gpu_handoff, "_wait_unit",
                        lambda unit, active, deadline: {
                            "ActiveState": "active" if state["active"] else "inactive",
                        })
    with pytest.raises(gpu_handoff.ManagedASRLifecycleError) as error:
        with gpu_handoff.managed_asr_handoff(cfg) as telemetry:
            assert telemetry["released"]
    assert error.value.status == "asr_restore_unverified"
    assert isinstance(error.value.__cause__, gpu_handoff.ManagedASRBlocked)
    assert error.value.__cause__.status == "systemctl_unavailable"
    assert state["active"] is False
    assert json.loads(cfg.journal_path.read_text())["phase"] == "pending_restore"


def test_recording_propagates_typed_post_stop_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import contextmanager
    import wave
    from nvidia_transcribe import recording
    from nvidia_transcribe.recording import RecordingPipelineConfig, process_media

    source = tmp_path / "clip.wav"
    with wave.open(str(source), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 16000 * 5)

    class Client:
        model = "test-nemotron"
        base_url = "http://127.0.0.1:8190/v1"
        def transcribe(self, path: Path, request: object) -> dict:
            raw = {
                "choices": [{"message": {"content": "<spk:0>你好朋友。"}}],
                "ctc_timestamps": {
                    "words": [], "error": "alignment_failed",
                    "diarization": [{"speaker": 0, "start": 0.4, "end": 3.0}],
                    "speaker_tag_to_diarization_speaker": {"0": 0},
                },
            }
            return {
                "raw_response": raw,
                "contract": {"raw_text": "<spk:0>你好朋友。"},
                "latency_s": 0.01,
            }

    class Aligner:
        metrics = {"model_load_count": 1, "align_count": 0}
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass
        def __enter__(self) -> "Aligner":
            return self
        def __exit__(self, *args: object) -> None:
            return None
        def align(self, **kwargs: object) -> dict:
            self.metrics["align_count"] += 1
            return {"units": [], "diagnostic": {"status": "no_valid_units"}}

    @contextmanager
    def interrupted_restore(config: object):
        yield {"released": True, "restored": False}
        raise gpu_handoff.ManagedASRLifecycleError("systemctl_unavailable")

    monkeypatch.setattr(recording, "managed_asr_handoff", interrupted_restore)
    monkeypatch.setattr(recording, "QwenForcedAlignerAdapter", Aligner)
    with pytest.raises(RuntimeError, match="Managed ASR lifecycle could not be verified"):
        process_media(
            source, output_dir=tmp_path / "out", client=Client(),
            config=RecordingPipelineConfig(
                source_language="zh", speaker_embeddings="none",
                enable_subinterval_recovery=False, alignment_fallback="qwen",
                qwen_gpu_mode="managed", qwen_owned_asr_unit="nemovox-asr.service",
                qwen_asr_lease_path=str(tmp_path / "lease"),
            ),
        )
