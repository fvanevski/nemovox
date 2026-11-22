"""CPU-only tests for Studio export routes, structured stages, logs and history."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nvidia_transcribe.studio_api import JobRunner, StudioConfig, create_app
from nvidia_transcribe.utterances import audit_and_reconstruct_contract


@pytest.fixture
def live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    media = tmp_path / "media"
    media.mkdir()
    clip = media / "test.wav"
    clip.write_bytes(b"mock-audio")
    export_root = tmp_path / "exports"
    export_root.mkdir()
    settings = StudioConfig(
        job_root=tmp_path / "jobs", upload_root=tmp_path / "uploads",
        media_roots=(media,), python="python",
        export_roots=(export_root,),
    )
    monkeypatch.setattr(JobRunner, "_reconcile_idle_models", staticmethod(lambda: None))
    monkeypatch.setattr("nvidia_transcribe.studio_api.inspect_audio_streams",
                        lambda _path: [{"index": 0, "codec": "pcm_s16le"}])
    app = create_app(settings)
    monkeypatch.setattr(app.state.runner.executor, "submit", lambda *args, **kwargs: None)
    with TestClient(app) as client:
        yield client, settings, clip, export_root
    app.state.runner.executor.shutdown(wait=False, cancel_futures=True)


def make_job(client: TestClient, clip: Path, export_root: Path, **plan):
    response = client.post("/api/jobs", json={
        "source": {"kind": "path", "value": str(clip)},
        "task": "transcription",
        "export": {"destination": str(export_root),
                   "artifacts": ["source_transcript"], **plan},
    })
    assert response.status_code == 202, response.text
    return response.json()["id"]


def test_authorized_destinations_and_rejected_paths(live, tmp_path: Path):
    client, _settings, clip, root = live
    listed = client.get("/api/export-directories")
    assert listed.status_code == 200
    assert listed.json()["default"] == str(root)
    child = root / "chosen"
    child.mkdir()
    inside = client.get("/api/export-directories", params={"path": str(root)})
    assert inside.json()["directories"] == [{"name": "chosen", "path": str(child)}]
    outside = tmp_path / "unauthorized"
    outside.mkdir()
    assert client.get("/api/export-directories", params={"path": str(outside)}).status_code == 403
    assert client.post("/api/jobs", json={
        "source": {"kind": "path", "value": str(clip)},
        "task": "transcription",
        "export": {"destination": str(outside), "artifacts": ["source_transcript"]},
    }).status_code == 422
    assert client.post("/api/jobs", json={
        "source": {"kind": "path", "value": str(clip)},
        "task": "transcription",
        "export": {"destination": None, "artifacts": ["source_transcript"]},
    }).status_code == 422


def test_legacy_request_remains_valid_and_default_export_is_disabled(live):
    client, _settings, clip, _root = live
    response = client.post("/api/jobs", json={
        "source": {"kind": "path", "value": str(clip)},
        "task": "transcription",
    })
    assert response.status_code == 202, response.text
    record = response.json()
    assert record["request"]["export"]["destination"] is None
    assert record["export_report"]["status"] == "disabled"


def test_empty_export_selection_is_disabled_even_with_folder(live):
    client, _settings, clip, root = live
    response = client.post("/api/jobs", json={
        "source": {"kind": "path", "value": str(clip)},
        "task": "transcription",
        "export": {"destination": str(root), "artifacts": [], "save_remote_media": False},
    })
    assert response.status_code == 202, response.text
    assert response.json()["export_report"]["status"] == "disabled"


def test_subtitles_imported_contract_has_source_exports(live, tmp_path: Path, monkeypatch):
    client, settings, _clip, root = live
    contract = settings.media_roots[0] / "recording.contract.json"
    contract.write_text(json.dumps({
        "source_language": "fr",
        "utterances": [{"source_text": "Une phrase.", "source_start": 0.2, "source_end": 1.2}],
    }), encoding="utf-8")
    response = client.post("/api/jobs", json={
        "source": {"kind": "contract", "value": str(contract)},
        "task": "subtitles",
        "options": {"translator": "swiftu", "translation_review": "off"},
        "export": {"destination": str(root),
                   "artifacts": ["source_transcript", "recording_contract"]},
    })
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    runner = client.app.state.runner

    def fake_generate(job: str, args: list[str], stage: str, progress: int) -> None:
        assert job == job_id
        output = settings.job_root / job_id / "output"
        output.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(runner, "_command", fake_generate)
    runner._run(job_id)
    result = runner.get(job_id)
    assert result["state"] == "complete", result
    assert result["export_report"]["status"] == "complete"
    saved = root / ("nemovox-" + job_id)
    assert (saved / "transcript.txt").is_file()
    assert "Une phrase." in (saved / "transcript.txt").read_text(encoding="utf-8")
    assert json.loads((saved / "recording.contract.json").read_text())[
        "utterances"][0]["source_text"] == "Une phrase."


@pytest.mark.parametrize("task", ["translation", "subtitles"])
def test_imported_contract_processing_keeps_adjacent_reconstruction_sidecars(
    live, monkeypatch: pytest.MonkeyPatch, task: str,
):
    client, settings, _clip, root = live
    original = settings.media_roots[0] / "recording.contract.json"
    source_contract = {
        "source_language": "fr",
        "utterances": [
            {"utterance_id": "chunk1-untimed", "source_text": "Parole sans temps.",
             "chunk_index": 1, "source_start": None, "source_end": None},
            {"utterance_id": "chunk0-timed", "source_text": "Parole avec temps.",
             "chunk_index": 0, "source_start": 150.0, "source_end": 151.0},
        ],
    }
    original.write_text(json.dumps(source_contract), encoding="utf-8")
    manifest = original.parent / "manifest.json"
    manifest.write_text(json.dumps({"chunks": [
        {"chunk_index": 0, "infer_start_s": 0.0, "infer_end_s": 300.0},
        {"chunk_index": 1, "infer_start_s": 290.0, "infer_end_s": 610.0},
    ]}), encoding="utf-8")
    chunks = original.parent / "chunks"
    chunks.mkdir()
    sidecar = chunks / "chunk_001.raw.json"
    sidecar.write_text(json.dumps({"choices": [{
        "finish_reason": "stop",
        "message": {"content": "<spk:0> Parole sans temps."},
    }]}), encoding="utf-8")
    original_contents = original.read_bytes()
    response = client.post("/api/jobs", json={
        "source": {"kind": "contract", "value": str(original)},
        "task": task,
        "options": {"translator": "swiftu", "translation_review": "off"},
        "export": {"destination": str(root),
                   "artifacts": ["recording_contract", "source_transcript"]},
    })
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    runner = client.app.state.runner
    invoked = []

    def fake_generate(job: str, args: list[str], stage: str, progress: int) -> None:
        assert job == job_id
        assert args[0] == "generate-subtitles"
        path = Path(args[args.index("--recording-contract") + 1])
        invoked.append(path)
        # Untimed-order reconstruction reads these files via path.parent.
        assert path == original
        manifest_chunks = json.loads((path.parent / "manifest.json").read_text())["chunks"]
        assert manifest_chunks[1]["infer_end_s"] == 610.0
        assert (path.parent / "chunks" / sidecar.name).read_bytes() == sidecar.read_bytes()
        # Exercise the real reconstruction algorithm with the sidecars the CLI
        # will see, not a mocked interpretation of chunk-window ordering.
        reconstructed, audit = audit_and_reconstruct_contract(
            json.loads(path.read_text(encoding="utf-8")),
            chunk_raw_dir=path.parent / "chunks", manifest_chunks=manifest_chunks,
        )
        entry = next(item for item in audit
                     if item["original_utterance_id"] == "chunk1-untimed")
        assert entry["chunk_window_s"] == [290.0, 610.0]
        assert entry["estimated_order_s"] > 150.0
        assert reconstructed["utterances"][-1]["original_utterance_id"] == "chunk1-untimed"
        (settings.job_root / job_id / "output").mkdir(exist_ok=True)

    monkeypatch.setattr(runner, "_command", fake_generate)
    runner._run(job_id)
    job = runner.get(job_id)
    assert job["state"] == "complete", job
    assert job["export_report"]["status"] == "complete"
    assert invoked == [original]
    archived = settings.job_root / job_id / "asr"
    assert json.loads((archived / "recording.contract.json").read_text()) == source_contract
    assert (archived / "recording.contract.json").read_bytes() == original_contents
    assert "Parole sans temps." in (archived / "transcript.txt").read_text()
    assert original.read_bytes() == original_contents
    saved = root / ("nemovox-" + job_id)
    assert (saved / "recording.contract.json").read_bytes() == original_contents
    assert (saved / "transcript.txt").is_file()


def test_invalid_task_and_remote_media_export_selection(live):
    client, _settings, clip, root = live
    base = {"source": {"kind": "path", "value": str(clip)},
            "task": "transcription"}
    response = client.post("/api/jobs", json={
        **base, "export": {"destination": str(root), "artifacts": ["ass"]},
    })
    assert response.status_code == 422
    response = client.post("/api/jobs", json={
        **base, "export": {"destination": str(root), "save_remote_media": True},
    })
    assert response.status_code == 422


def test_typed_milestone_lines_and_monotonic_progress(live):
    client, settings, clip, root = live
    job_id = make_job(client, clip, root)
    job_root = settings.job_root / job_id
    journal = job_root / "execution.log"
    payload = "\n".join([
        "ordinary log",
        'NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"translation","status":"running","detail":""}',
        'NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"translation","status":"complete","detail":"uncached"}',
        'NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"review","status":"running","detail":""}',
        'NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"review","status":"complete","detail":"completed"}',
        'NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"render","status":"running","detail":""}',
        'NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"render","status":"complete","detail":""}',
        'NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"arbitrary","status":"running","detail":"no"}',
        "",
    ])
    journal.write_text(payload, encoding="utf-8")
    offset = client.app.state.runner._read_stage_events(job_id, journal, 0)
    assert offset == journal.stat().st_size
    state = client.get("/api/jobs/" + job_id).json()
    assert [(item["id"], item["status"]) for item in state["stage_history"]] == [
        ("translation", "complete"), ("review", "complete"), ("render", "complete"),
    ]
    assert state["progress"] >= 90
    assert all(item.get("started_at") and item.get("finished_at")
               for item in state["stage_history"])
    assert client.app.state.runner._read_stage_events(job_id, journal, offset) == offset


def test_degraded_translation_event_survives_completed_job(live):
    client, settings, clip, root = live
    job_id = make_job(client, clip, root)
    runner = client.app.state.runner
    journal = settings.job_root / job_id / "execution.log"
    detail = "completed_with_errors: 1 translation error(s); fallback source text retained"
    events = [
        {"schema": "nemovox-progress/v1", "stage": "translation", "status": "running"},
        {"schema": "nemovox-progress/v1", "stage": "translation",
         "status": "failed", "detail": detail},
        {"schema": "nemovox-progress/v1", "stage": "render", "status": "running"},
        {"schema": "nemovox-progress/v1", "stage": "render", "status": "complete"},
    ]
    journal.write_text("".join("NEMOVOX_EVENT " + json.dumps(event) + "\n"
                               for event in events), encoding="utf-8")
    assert runner._read_stage_events(job_id, journal, 0) == journal.stat().st_size
    runner.update(job_id, state="complete", progress=100)
    job = client.get("/api/jobs/" + job_id).json()
    assert job["state"] == "complete"
    assert [(event["id"], event["status"]) for event in job["stage_history"]] == [
        ("translation", "failed"), ("render", "complete"),
    ]
    assert job["stage_history"][0]["detail"] == detail
    assert job["stage_history"][0]["finished_at"]
    assert job["stage_history"][1]["finished_at"]


def test_completed_cli_drains_all_stage_events_after_large_final_log(live, monkeypatch):
    client, settings, clip, root = live
    job_id = make_job(client, clip, root)
    runner = client.app.state.runner

    class FinishedProcess:
        pid = 12345
        returncode = 0

        def poll(self):
            return self.returncode

    def fake_popen(_args, **kwargs):
        log = kwargs["stdout"]
        log.write(("large ordinary log line\n" * 15000))
        log.write('NEMOVOX_EVENT {"schema":"not-our-schema","stage":"review","status":"failed"}\n')
        log.write('NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"translation","status":"running"}\n')
        log.write('NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"translation","status":"complete"}\n')
        log.write('NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"review","status":"complete"}\n')
        log.write('NEMOVOX_EVENT {"schema":"nemovox-progress/v1","stage":"render","status":"complete"}\n')
        log.flush()
        return FinishedProcess()

    monkeypatch.setattr("nvidia_transcribe.studio_api.subprocess.Popen", fake_popen)
    monkeypatch.setattr(runner, "_terminate_process_group", lambda _process: None)
    runner._command(job_id, ["generate-subtitles"], "Generating English outputs", 55)
    history = runner.get(job_id)["stage_history"]
    assert [(item["id"], item["status"]) for item in history] == [
        ("translation", "complete"), ("review", "complete"), ("render", "complete")
    ]
    assert (settings.job_root / job_id / "execution.log").stat().st_size > 256 * 1024


def test_failure_marks_active_stage_and_does_not_mark_job_complete(live):
    client, settings, clip, root = live
    job_id = make_job(client, clip, root)
    runner = client.app.state.runner
    runner._stage_event(job_id, "translation", "running")
    runner._finish_active_stage(job_id, "failed")
    event = runner.get(job_id)["stage_history"][0]
    assert event["status"] == "failed"
    assert runner.get(job_id)["state"] == "queued"


def test_large_utf8_log_pages_and_download_survive_completion(live):
    client, settings, clip, root = live
    job_id = make_job(client, clip, root)
    log = settings.job_root / job_id / "execution.log"
    original = ("abcdefg" + "日本語" + "\n") * 6000
    log.write_text(original, encoding="utf-8")
    client.app.state.runner.update(job_id, state="complete", progress=100)
    out, offset = "", 0
    for _ in range(30):
        response = client.get(f"/api/jobs/{job_id}/log/chunk",
                              params={"offset": offset, "limit": 7001})
        assert response.status_code == 200
        page = response.json()
        out += page["text"]
        if page["complete"]:
            break
        assert page["next_offset"] > offset
        offset = page["next_offset"]
    assert out == original
    response = client.get(f"/api/jobs/{job_id}/log/download")
    assert response.status_code == 200
    assert response.content == original.encode("utf-8")
    tail = client.get(f"/api/jobs/{job_id}/logs").json()
    assert tail["lines"]


def test_older_jobs_are_paginated_beyond_legacy_fifty(live):
    client, settings, clip, root = live
    for i in range(56):
        job_id = f"{i:032x}"
        record_path = settings.job_root / job_id / "job.json"
        record_path.parent.mkdir(parents=True)
        record_path.write_text(json.dumps({
            "id": job_id, "state": "complete", "stage": "Complete",
            "created_at": i + 1, "request": {"source": {"kind": "path", "value": str(clip)},
            "task": "transcription", "options": {}},
        }))
    assert len(client.get("/api/jobs").json()) == 50
    page = client.get("/api/job-history", params={"offset": 50, "limit": 20}).json()
    assert page["total"] == 56
    assert len(page["items"]) == 6
    assert page["next_offset"] is None
