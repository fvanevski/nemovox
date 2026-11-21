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
