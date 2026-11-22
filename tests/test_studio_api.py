"""CPU-only Studio API checks; no model, FFmpeg, or network service needed."""
from __future__ import annotations

import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
from urllib.parse import unquote, urlsplit

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nvidia_transcribe.studio_api import JobRunner, StudioConfig, _remote_url, _srt_cues, create_app
from nvidia_transcribe.media import MediaInspectionError


@pytest.fixture(autouse=True)
def fake_media_probe(monkeypatch: pytest.MonkeyPatch):
    # The existing Studio tests use synthetic files. Explicit media-probe tests
    # below independently exercise the real ffprobe parser and API validation.
    monkeypatch.setattr(
        "nvidia_transcribe.studio_api.inspect_audio_streams",
        lambda path: [{"index": 0, "codec": "aac", "channels": 2, "sample_rate_hz": 48000,
                       "container_index": 1, "language": "eng", "title": "Main"}],
    )


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
        "options": {"source_language": "zh", "refinement": "off", "translator": "swiftu"},
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


@pytest.mark.parametrize("extension", [".srt", ".ass"])
def test_artifact_links_url_encode_reserved_and_unicode_filenames(client, extension):
    """Export and ASS URLs must survive URL delimiters in original media stems."""
    c, media, _secret, settings = client
    response = c.post("/api/jobs", json={
        "source": {"kind": "contract", "value": str(media / "recording.contract.json")},
        "task": "subtitles",
    })
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    output = settings.job_root / job_id / "output"
    output.mkdir()
    filename = "Episode #1? 50% 日本語.en" + extension
    payload = b"1\n00:00:00,000 --> 00:00:02,000\nSynthetic cue\n"
    (output / filename).write_bytes(payload)

    items = c.get(f"/api/jobs/{job_id}/artifacts").json()
    artifact = next(a for a in items if a["name"] == filename)
    assert artifact["path"] == "output/" + filename  # display/path metadata stays unescaped
    url = artifact["url"]
    parsed = urlsplit(url)
    assert not parsed.query and not parsed.fragment
    assert unquote(parsed.path).endswith("/artifact/output/" + filename)
    for token in ("%20", "%23", "%3F", "%25", "%E6%97%A5"):
        assert token in url
    downloaded = c.get(url)
    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content == payload


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
        f"pathlib.Path({str(args_file)!r}).write_text(json.dumps(dict(args=sys.argv[1:], asr=os.getenv('ASR_BASE_URL'), resource_config=os.getenv('NEMOVOX_MODEL_RESOURCES'))))\n"
        "if sys.argv[1] == 'process-media':\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"    pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "    time.sleep(30)\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "fake_pkg"))
    monkeypatch.setenv("ASR_BASE_URL", "http://127.0.0.1:9190/v1")
    monkeypatch.setenv("SWIFTU_BASE_URL", "http://127.0.0.1:9788/v1")
    monkeypatch.setenv("NEMOVOX_MODEL_RESOURCES", str(tmp_path / "model-resources.local.json"))

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
            assert "--base-url" not in recorded_args["args"]
            assert "--swiftu-url" not in recorded_args["args"]
            assert recorded_args["asr"] == "http://127.0.0.1:9190/v1"
            assert recorded_args["resource_config"] == str(tmp_path / "model-resources.local.json")

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



def test_app_creation_does_not_reconcile_another_running_server(tmp_path: Path):
    """Import/app construction is read-only; exclusive startup owns reconciliation."""
    root = tmp_path / "jobs"
    job_dir = root / ("d" * 32)
    job_dir.mkdir(parents=True)
    record = {
        "id": "d" * 32, "state": "running", "stage": "Transcribing", "progress": 25,
        "request": {"source": {"kind": "path", "value": str(tmp_path / "sample.webm")},
                    "task": "transcription", "options": {}},
        "source_path": str(tmp_path / "sample.webm"),
        "created_at": 1.0, "updated_at": 1.0, "error": None,
    }
    record_file = job_dir / "job.json"
    record_file.write_text(json.dumps(record), encoding="utf-8")
    settings = StudioConfig(
        job_root=root, upload_root=tmp_path / "uploads", media_roots=(tmp_path,),
        python=sys.executable,
    )
    first = create_app(settings)
    second = create_app(settings)
    # Constructing a second app, including during a separate module import,
    # cannot change the running owner's persisted job record.
    assert json.loads(record_file.read_text(encoding="utf-8")) == record
    try:
        with TestClient(first) as owner_client:
            assert owner_client.get(f"/api/jobs/{record['id']}").json()["state"] == "interrupted"
            with pytest.raises(RuntimeError, match="already owned"):
                with TestClient(second):
                    pass
            assert owner_client.get(f"/api/jobs/{record['id']}").json()["state"] == "interrupted"
    finally:
        # A rejected lifespan never entered shutdown, so close its idle pool.
        second.state.runner.executor.shutdown(wait=True)



def test_cancel_and_completion_have_serialized_terminal_state(client):
    c, media, _secret, settings = client
    runner = c.app.state.runner

    def new_running_job():
        response = c.post("/api/jobs", json={
            "source": {"kind": "contract", "value": str(media / "recording.contract.json")},
            "task": "transcription",
        })
        assert response.status_code == 202
        job_id = response.json()["id"]
        runner.update(job_id, state="running", stage="Preparing media")
        return job_id

    cancelled = new_running_job()
    response = c.post(f"/api/jobs/{cancelled}/cancel")
    assert response.status_code == 200
    with pytest.raises(InterruptedError, match="Cancelled"):
        runner._complete_if_not_cancelled(cancelled)
    assert runner.get(cancelled)["state"] == "running"
    assert runner.get(cancelled)["stage"] == "Cancelling"

    completed = new_running_job()
    runner._complete_if_not_cancelled(completed)
    after = c.post(f"/api/jobs/{completed}/cancel")
    assert after.status_code == 200
    assert after.json()["state"] == "complete"
    assert after.json()["stage"] == "Complete"
    assert runner.get(completed)["state"] == "complete"



def test_audio_inspection_authorization_and_stream_index_validation(client, monkeypatch):
    c, media, secret, _ = client
    source = {"kind": "path", "value": str(media / "sample.webm")}
    tracks = [
        {"index": 0, "container_index": 1, "codec": "aac",
         "channels": 2, "sample_rate_hz": 48000, "language": "fra", "title": "French"},
        {"index": 1, "container_index": 3, "codec": "eac3",
         "channels": 6, "sample_rate_hz": 48000, "language": "eng", "title": "English"},
    ]
    monkeypatch.setattr("nvidia_transcribe.studio_api.inspect_audio_streams",
                        lambda path: tracks)
    response = c.post("/api/media/inspect", json=source)
    assert response.status_code == 200, response.text
    assert response.json() == {"audio_streams": tracks}
    assert c.post("/api/media/inspect", json={
        "kind": "path", "value": str(secret / "private.webm")}).status_code == 403
    assert c.post("/api/media/inspect", json={
        "kind": "contract", "value": str(media / "recording.contract.json")}).status_code == 422
    allowed = c.post("/api/jobs", json={
        "source": source, "task": "transcription", "options": {"audio_stream": 1}})
    assert allowed.status_code == 202, allowed.text
    rejected = c.post("/api/jobs", json={
        "source": source, "task": "transcription", "options": {"audio_stream": 2}})
    assert rejected.status_code == 422


def test_glossary_metadata_explains_series_specific_examples(client):
    c, _media, _secret, _settings = client
    entries = c.get("/api/glossaries").json()
    assert any("A Will Eternal" in e["name"] and e["characters"] and e["terms"]
               for e in entries)
    assert any("French example" in e["name"] and e["notes"] for e in entries)
    assert all("path" in e and "notes" in e for e in entries)


def test_audio_stream_parser_uses_audio_only_indices(monkeypatch, tmp_path):
    from nvidia_transcribe import media as media_module
    clip = tmp_path / "episode.mkv"
    clip.write_bytes(b"synthetic media")
    payload = {"streams": [
        {"index": 0, "codec_type": "video", "codec_name": "hevc"},
        {"index": 1, "codec_type": "audio", "codec_name": "aac",
         "channels": 2, "sample_rate": "48000",
         "tags": {"language": "fra", "title": "French"}},
        {"index": 2, "codec_type": "subtitle", "codec_name": "ass"},
        {"index": 3, "codec_type": "audio", "codec_name": "eac3",
         "channels": 6, "sample_rate": "48000", "tags": {"language": "eng"}},
    ]}
    monkeypatch.setattr(media_module.subprocess, "check_output",
                        lambda *args, **kwargs: json.dumps(payload))
    records = media_module.inspect_audio_streams(clip)
    assert [x["index"] for x in records] == [0, 1]
    assert [x["container_index"] for x in records] == [1, 3]
    assert records[0]["language"] == "fra"



def test_studio_rejects_legacy_text_translation_with_riva_only_review(client):
    c, media, _secret, _settings = client
    response = c.post("/api/jobs", json={
        "source": {"kind": "contract", "value": str(media / "recording.contract.json")},
        "task": "subtitles", "options": {
            "translator": "openai_chat", "translation_review": "balanced",
        },
    })
    assert response.status_code == 422



def test_studio_keeps_source_asr_uncorrected(client):
    from nvidia_transcribe.studio_api import Options
    c, media, _secret, _settings = client
    assert Options().refinement == "off"
    payload = {
        "source": {"kind": "contract", "value": str(media / "recording.contract.json")},
        "task": "subtitles",
        "options": {"translator": "openai_chat", "refinement": "balanced"},
    }
    result = c.post("/api/jobs", json=payload)
    assert result.status_code == 422
    assert "original ASR transcript" in result.json()["detail"]
    payload["options"]["refinement"] = "off"
    permitted = c.post("/api/jobs", json=payload)
    assert permitted.status_code == 202



def test_condensation_is_opt_in_for_api_and_persisted_job_options(client):
    """Omitted API options must not enable an unexpected second text-model pass."""
    from nvidia_transcribe.studio_api import JobRequest, Options

    assert Options().condense_high_cps is False
    assert Options(condense_high_cps=True).condense_high_cps is True

    c, media, _secret, settings = client
    request = {
        "source": {"kind": "contract", "value": str(media / "recording.contract.json")},
        "task": "subtitles",
        "options": {"translator": "openai_chat"},
    }
    parsed = JobRequest.model_validate(request)
    assert parsed.options.refinement == "off"
    assert parsed.options.condense_high_cps is False
    response = c.post("/api/jobs", json=request)
    assert response.status_code == 202, response.text
    persisted = json.loads((settings.job_root / response.json()["id"] / "job.json").read_text(encoding="utf-8"))
    assert persisted["request"]["options"]["condense_high_cps"] is False

    # Existing jobs that explicitly opted in must retain that request.
    request["options"]["condense_high_cps"] = True
    assert JobRequest.model_validate(request).options.condense_high_cps is True


def test_yaml_glossary_dependency_is_declared_for_cli_and_studio():
    """Clean installs must supply the YAML reader used by both runtime surfaces."""
    import tomllib

    manifest = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(manifest.read_text(encoding="utf-8"))
    requirements = metadata["project"]["dependencies"]
    assert any(item.lower().replace("-", "").startswith("pyyaml") for item in requirements)


def test_english_studio_subtitles_do_not_require_riva_checkpoint(client, monkeypatch):
    """Same-language subtitles do not depend on an unused Riva resource."""
    c, media, _secret, _settings = client

    def forbidden_resource_probe():
        raise AssertionError("English-to-English job does not require Riva")

    monkeypatch.setattr("nvidia_transcribe.studio_api.load_model_resources", forbidden_resource_probe)
    result = c.post("/api/jobs", json={
        "source": {"kind": "path", "value": str(media / "sample.webm")},
        "task": "subtitles",
        "options": {"source_language": "en", "target_language": "en",
                    "translator": "riva", "translation_review": "balanced"},
    })
    assert result.status_code == 202, result.text


def test_invalid_glossaries_do_not_disable_valid_entries(client, monkeypatch, tmp_path):
    """One malformed optional glossary must not turn all discovery into HTTP 500."""
    from nvidia_transcribe import studio_api
    from nvidia_transcribe.refinement import load_glossary

    root = tmp_path / "configs" / "glossaries"
    root.mkdir(parents=True)
    (root / "good.json").write_text(json.dumps({
        "title": "Valid Example", "characters": ["Esteban"],
        "terms": ["Golden Condor"], "notes": "A valid example",
    }), encoding="utf-8")
    (root / "bad-null.json").write_text("null", encoding="utf-8")
    (root / "bad-scalar.json").write_text("42", encoding="utf-8")
    (root / "bad-list-shape.yaml").write_text("characters: not-a-list\n", encoding="utf-8")
    (root / "bad-syntax.yaml").write_text("characters: [unfinished\n", encoding="utf-8")
    monkeypatch.setattr(studio_api, "PROJECT", tmp_path)

    c, _media, _secret, _settings = client
    response = c.get("/api/glossaries")
    assert response.status_code == 200, response.text
    records = response.json()
    assert len(records) == 1 and records[0]["name"] == "Valid Example"
    assert records[0]["characters"] == ["Esteban"]
    with pytest.raises(ValueError, match="must contain a mapping or list"):
        load_glossary(root / "bad-null.json")
    with pytest.raises(ValueError, match="character and term fields must be lists"):
        load_glossary(root / "bad-list-shape.yaml")



def test_studio_reconciliation_occurs_only_after_owner_start(tmp_path, monkeypatch):
    """Constructing a secondary app must not mutate model service state."""
    observed = []
    monkeypatch.setattr(JobRunner, "_reconcile_idle_models",
                        staticmethod(lambda: observed.append("recover")))
    settings = StudioConfig(job_root=tmp_path / "jobs", upload_root=tmp_path / "uploads",
                            media_roots=(tmp_path,), python=sys.executable)
    app = create_app(settings)
    assert observed == []
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
        assert observed == ["recover"]
    app.state.runner.executor.shutdown(wait=True)


def test_studio_reconciles_owned_units_after_abrupt_cli_exit(tmp_path, monkeypatch):
    """SIGKILLed child leaves its own unit; supervisor must reconcile."""
    from nvidia_transcribe import studio_api

    observed = []
    monkeypatch.setattr(JobRunner, "_reconcile_idle_models",
                        staticmethod(lambda: observed.append("recover")))
    monkeypatch.setattr(JobRunner, "_terminate_process_group",
                        staticmethod(lambda _process: observed.append("group_cleaned")))

    class DeadCLI:
        pid = 424242
        returncode = -signal.SIGKILL
        def poll(self):
            return self.returncode

    monkeypatch.setattr(studio_api.subprocess, "Popen", lambda *a, **kw: DeadCLI())
    settings = StudioConfig(job_root=tmp_path / "jobs", upload_root=tmp_path / "uploads",
                            media_roots=(tmp_path,), python=sys.executable)
    runner = JobRunner(settings)
    job_id = "a" * 32
    root = settings.job_root / job_id
    root.mkdir()
    (root / "job.json").write_text(json.dumps({
        "id": job_id, "state": "running", "stage": "ASR", "progress": 8,
        "error": None, "created_at": 0.0, "updated_at": 0.0,
    }), encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="failed \\(exit -9\\)"):
            runner._command(job_id, ["process-media", "synthetic.wav"],
                            "Transcribing audio and speakers", 8)
        assert observed == ["group_cleaned", "recover"]
        assert runner._processes == {}
    finally:
        runner.executor.shutdown(wait=True)
