"""CPU-only Studio API checks; no model, FFmpeg, or network service needed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nvidia_transcribe.studio_api import StudioConfig, _remote_url, _srt_cues, create_app


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
