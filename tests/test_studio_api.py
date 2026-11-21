"""CPU-only Studio API checks; no model, FFmpeg, or network service needed."""
from __future__ import annotations

import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nvidia_transcribe.studio_api import JobRunner, StudioConfig, _remote_url, _srt_cues, create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    media = tmp_path / "allowed"
    media.mkdir()
    (media / "sample.webm").write_bytes(b"synthetic media placeholder")
    (media / "recording.contract.json").write_text('{"utterances": []}', encoding="utf-8")
    secret = tmp_path / "outside"
    secret.mkdir()
    (secret / "private.webm").write_bytes(b"not allowed")
    settings = StudioConfig(job_root=tmp_path / "jobs", upload_root=tmp_path / "uploads",
                            media_roots=(media,), python="python")
    app = create_app(settings)
    monkeypatch.setattr(app.state.runner.executor, "submit", lambda *a, **kw: None)
    with TestClient(app) as c:
        yield c, media, secret, settings
    app.state.runner.executor.shutdown(wait=False, cancel_futures=True)


def test_health_and_media_roots(client):
    c, media, _secret, _ = client
    assert c.get("/api/health").json()["status"] == "ok"
    assert c.get("/api/media").json()["items"][0]["path"] == str(media)
    paths = {x["name"] for x in c.get("/api/media", params={"path": str(media)}).json()["items"]}
    assert {"sample.webm", "recording.contract.json"}.issubset(paths)


def test_path_traversal_is_denied(client):
    c, _media, secret, _ = client
    assert c.get("/api/media", params={"path": str(secret)}).status_code == 403
    assert c.post("/api/jobs", json={"source": {"kind": "path", "value": str(secret / "private.webm")},
                                      "task": "transcription"}).status_code == 403


def test_job_submission_and_preview(client):
    c, media, _secret, settings = client
    response = c.post("/api/jobs", json={
        "source": {"kind": "contract", "value": str(media / "recording.contract.json")},
        "task": "subtitles",
        "options": {"source_language": "zh", "refinement": "balanced", "translator": "swiftu"},
    })
    assert response.status_code == 202, response.text
    job = response.json()
    assert job["state"] == "queued"
    root = settings.job_root / job["id"]
    output = root / "output"
    output.mkdir()
    (output / "demo.en.srt").write_text(
        "1\n00:00:01,200 --> 00:00:03,400\nHello from the demo!\n", encoding="utf-8"
    )
    cues = c.get("/api/jobs/" + job["id"] + "/preview").json()["cues"]
    assert cues == [{"start": 1.2, "end": 3.4, "text": "Hello from the demo!"}]
    artifacts = c.get("/api/jobs/" + job["id"] + "/artifacts").json()
    assert any(x["name"] == "demo.en.srt" for x in artifacts)
    assert c.get("/api/jobs/" + job["id"] + "/artifact/output/demo.en.srt").status_code == 200
    assert c.get("/api/jobs/" + job["id"] + "/artifact/../../outside/private.webm").status_code in (403, 404)


def test_invalid_options_and_bad_url(client):
    c, media, *_ = client
    assert c.post("/api/jobs", json={
        "source": {"kind": "path", "value": str(media / "sample.webm")},
        "task": "subtitles", "options": {"translator": "none"},
    }).status_code == 422
    assert c.post("/api/jobs", json={
        "source": {"kind": "url", "value": "http://127.0.0.1:8000/admin"},
        "task": "transcription",
    }).status_code == 422


def test_parse_srt():
    assert _srt_cues("1\n00:00:02,000 --> 00:00:04,125\nLine one\nLine two\n")[0] == {
        "start": 2.0, "end": 4.125, "text": "Line one\nLine two",
    }


def test_upload_and_extension_rejection(client):
    c, _media, _secret, settings = client
    bad = c.post("/api/media/upload", files={"media_file": ("notes.txt", b"hello", "text/plain")})
    assert bad.status_code == 422

    ok = c.post("/api/media/upload", files={"media_file": ("clip.webm", b"synthetic-webm-bytes", "video/webm")})
    assert ok.status_code == 200
    payload = ok.json()
    uploaded = Path(payload["path"])
    assert uploaded.is_file()
    assert uploaded.parent == settings.upload_root.resolve()
    assert payload["size_bytes"] == len(b"synthetic-webm-bytes")

    sub = c.post("/api/jobs", json={"source": {"kind": "upload", "value": str(uploaded)}, "task": "transcription"})
    assert sub.status_code == 202


def test_interrupted_job_persistence_and_contract_media(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    media = tmp_path / "allowed"
    media.mkdir()
    clip = media / "clip.webm"
    clip.write_bytes(b"synthetic-clip")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret_clip = outside / "secret.webm"
    secret_clip.write_bytes(b"secret-clip")

    contract_ok = media / "recording.contract.json"
    contract_ok.write_text(json.dumps({"source": {"path": str(clip)}, "utterances": []}), encoding="utf-8")
    contract_bad_dir = media / "sub"
    contract_bad_dir.mkdir()
    contract_bad = contract_bad_dir / "recording.contract.json"
    contract_bad.write_text(json.dumps({"source": {"path": str(secret_clip)}, "utterances": []}), encoding="utf-8")

    job_root = tmp_path / "jobs"
    for jid, state in [("a" * 32, "running"), ("b" * 32, "queued"), ("c" * 32, "complete")]:
        jdir = job_root / jid
        jdir.mkdir(parents=True)
        (jdir / "job.json").write_text(json.dumps({
            "id": jid,
            "state": state,
            "stage": "Working",
            "progress": 50,
            "request": {"source": {"kind": "path", "value": str(clip)}, "task": "transcription", "options": {}},
            "source_path": str(clip),
            "created_at": 100.0,
            "updated_at": 100.0,
            "error": None,
        }), encoding="utf-8")

    settings = StudioConfig(job_root=job_root, upload_root=tmp_path / "uploads", media_roots=(media,), python="python")
    app = create_app(settings)
    monkeypatch.setattr(app.state.runner.executor, "submit", lambda *a, **kw: None)
    try:
        with TestClient(app) as c:
            assert c.get("/api/jobs/" + ("a" * 32)).json()["state"] == "interrupted"
            assert c.get("/api/jobs/" + ("b" * 32)).json()["state"] == "interrupted"
            assert c.get("/api/jobs/" + ("c" * 32)).json()["state"] == "complete"
            assert len(c.get("/api/jobs").json()) == 3

            r1 = c.post("/api/jobs", json={"source": {"kind": "contract", "value": str(contract_ok)}, "task": "subtitles"})
            assert r1.status_code == 202
            j1 = r1.json()["id"]
            assert c.get(f"/api/jobs/{j1}/media").status_code == 200

            (job_root / j1 / "asr").mkdir()
            (job_root / j1 / "asr" / "transcript.txt").write_text("hello", encoding="utf-8")
            (job_root / j1 / "asr" / "recording.contract.json").write_text("{}", encoding="utf-8")
            art_names = {x["name"] for x in c.get(f"/api/jobs/{j1}/artifacts").json()}
            assert {"transcript.txt", "recording.contract.json"}.issubset(art_names)

            r2 = c.post("/api/jobs", json={"source": {"kind": "contract", "value": str(contract_bad)}, "task": "subtitles"})
            assert r2.status_code == 202
            assert c.get(f"/api/jobs/{r2.json()['id']}/media").status_code == 404
    finally:
        app.state.runner.executor.shutdown(wait=False, cancel_futures=True)


def test_cancel_process_group_and_env_forwarding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import os
    import signal
    import sys
    import time

    media = tmp_path / "allowed"
    media.mkdir()
    clip = media / "clip.webm"
    clip.write_bytes(b"synthetic-clip")

    # Fake CLI script that spawns a child sleeper process, captures args, and sleeps until SIGTERM.
    fake_cli_dir = tmp_path / "fake_pkg" / "nvidia_transcribe"
    fake_cli_dir.mkdir(parents=True)
    (fake_cli_dir / "__init__.py").write_text("", encoding="utf-8")
    pid_file = tmp_path / "child.pid"
    args_file = tmp_path / "args.json"
    (fake_cli_dir / "cli.py").write_text(
        "import json, os, pathlib, subprocess, sys, time\n"
        f"pathlib.Path({str(args_file)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "if sys.argv[1] == 'process-media':\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"    pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "    time.sleep(30)\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake_pkg"))
    monkeypatch.setenv("ASR_BASE_URL", "http://127.0.0.1:9190/v1")
    monkeypatch.setenv("SWIFTU_BASE_URL", "http://127.0.0.1:9788/v1")

    settings = StudioConfig(
        job_root=tmp_path / "jobs",
        upload_root=tmp_path / "uploads",
        media_roots=(media,),
        python=sys.executable,
    )
    app = create_app(settings)
    try:
        with TestClient(app) as c:
            res = c.post("/api/jobs", json={"source": {"kind": "path", "value": str(clip)}, "task": "transcription"})
            assert res.status_code == 202
            job_id = res.json()["id"]

            for _ in range(40):
                if pid_file.exists():
                    break
                time.sleep(0.05)
            assert pid_file.exists()
            child_pid = int(pid_file.read_text())
            recorded_args = json.loads(args_file.read_text())
            assert "--base-url" in recorded_args and "http://127.0.0.1:9190/v1" in recorded_args

            cancel_res = c.post(f"/api/jobs/{job_id}/cancel")
            assert cancel_res.status_code == 200

            state = None
            for _ in range(40):
                state = c.get(f"/api/jobs/{job_id}").json()["state"]
                if state == "cancelled":
                    break
                time.sleep(0.1)
            assert state == "cancelled"

            # Verify child process group member was terminated (no orphan left behind).
            alive = True
            for _ in range(20):
                try:
                    os.kill(child_pid, 0)
                    time.sleep(0.05)
                except ProcessLookupError:
                    alive = False
                    break
            assert not alive
            log_lines = c.get(f"/api/jobs/{job_id}/logs").json()["lines"]
            assert any("STAGE: Transcribing audio and speakers" in line for line in log_lines)
    finally:
        app.state.runner.executor.shutdown(wait=False, cancel_futures=True)


def test_contract_transcription_import_and_empty_contract(client):
    c, media, _secret, settings = client
    contract = media / "recording.contract.json"
    original = {
        "source": {"path": str(media / "sample.webm")},
        "source_language": "zh",
        "language_provenance": "explicit",
        "utterances": [
            {"id": "utt-1", "source_text": "你好", "speaker_id": "SPK_0",
             "source_start": 1.0, "source_end": 2.5},
            {"id": "utt-2", "source_text": "没有时间", "speaker_id": "SPK_1",
             "source_start": None, "source_end": None},
        ],
    }
    contract.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    response = c.post("/api/jobs", json={
        "source": {"kind": "contract", "value": str(contract)}, "task": "transcription"
    })
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    c.app.state.runner._run(job_id)
    assert c.get(f"/api/jobs/{job_id}").json()["state"] == "complete"
    imported = settings.job_root / job_id / "asr" / "recording.contract.json"
    assert json.loads(imported.read_text(encoding="utf-8")) == original
    transcript = (settings.job_root / job_id / "asr" / "transcript.txt").read_text(encoding="utf-8")
    assert "你好" in transcript and "没有时间" in transcript and "[untimed -> untimed]" in transcript
    preview = c.get(f"/api/jobs/{job_id}/preview").json()
    assert preview["kind"] == "transcript"
    assert [(x["utterance_id"], x["text"]) for x in preview["cues"]] == [
        ("utt-1", "你好"), ("utt-2", "没有时间")
    ]
    artifacts = c.get(f"/api/jobs/{job_id}/artifacts").json()
    assert {x["name"] for x in artifacts} >= {"recording.contract.json", "transcript.txt"}
    for item in artifacts:
        assert c.get(item["url"]).status_code == 200
    assert json.loads(contract.read_text(encoding="utf-8")) == original

    contract.write_text('{"utterances": []}', encoding="utf-8")
    empty_response = c.post("/api/jobs", json={
        "source": {"kind": "contract", "value": str(contract)}, "task": "transcription"
    })
    assert empty_response.status_code == 202
    empty_job = empty_response.json()["id"]
    c.app.state.runner._run(empty_job)
    assert c.get(f"/api/jobs/{empty_job}").json()["state"] == "complete"
    assert c.get(f"/api/jobs/{empty_job}/preview").json() == {"kind": "transcript", "cues": []}
    assert "No utterances in source contract" in (
        settings.job_root / empty_job / "asr" / "transcript.txt"
    ).read_text(encoding="utf-8")

    contract.write_text('{"utterances": [{"id":"broken"}]}', encoding="utf-8")
    invalid = c.post("/api/jobs", json={
        "source": {"kind": "contract", "value": str(contract)}, "task": "transcription"
    })
    assert invalid.status_code == 422


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="Linux process-group inspection required")
def test_cancel_cleans_descendant_after_group_leader_exits(tmp_path: Path):
    """A reaped Popen leader is not proof that its SIGTERM-resistant child exited."""
    ready = tmp_path / "ready.pid"
    child_script = (
        "import os, pathlib, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    leader_script = (
        "import subprocess, sys\n"
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}],"
        " stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,"
        " stderr=subprocess.DEVNULL)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", leader_script],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.is_file() and time.monotonic() < deadline:
            time.sleep(0.025)
        assert ready.is_file()
        process.wait(timeout=3)
        assert process.poll() is not None  # Regression: the old code returned early.
        assert JobRunner._group_has_live_members(process.pid)
        JobRunner._terminate_process_group(process)
        assert not JobRunner._group_has_live_members(process.pid)
    finally:
        if JobRunner._group_has_live_members(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=3)
