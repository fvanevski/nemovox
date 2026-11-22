"""CPU-only output-root authorization and automatic Studio export regression tests."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nvidia_transcribe.studio_exports import (
    browse_destinations, export_roots, export_selected, validate_destination,
)


@pytest.fixture
def store(tmp_path: Path):
    root = tmp_path / "allowed-exports"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    job = tmp_path / "jobs" / ("a" * 32)
    job.mkdir(parents=True)
    (job / "asr").mkdir()
    (job / "output").mkdir()
    (job / "remote").mkdir()
    (job / "asr" / "transcript.txt").write_text("Synthetic source", encoding="utf-8")
    (job / "output" / "final_english_subtitles.txt").write_text("Synthetic English", encoding="utf-8")
    (job / "output" / "show.en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n")
    (job / "output" / "show.en.ass").write_text("[Script Info]\nTitle: synthetic\n")
    return root, outside, job


def plan(root: Path, *artifacts: str, save_remote_media: bool = False) -> dict:
    return {"destination": str(root), "artifacts": list(artifacts),
            "save_remote_media": save_remote_media}


def test_roots_default_and_host_configuration(tmp_path: Path):
    assert export_roots(tmp_path, "") == ((tmp_path / "outputs" / "exports").resolve(),)
    home = tmp_path / "one"
    second = tmp_path / "two"
    assert export_roots(tmp_path, os.pathsep.join([str(home), str(second)])) == (
        home.resolve(), second.resolve())


def test_export_destination_authorization_and_browse(store):
    root, outside, _job = store
    nested = root / "nested"
    nested.mkdir()
    listing = browse_destinations((root,), str(root))
    assert listing["default"] == str(root)
    assert listing["directories"] == [{"name": "nested", "path": str(nested)}]
    assert validate_destination(str(nested / "new-subdirectory"), (root,)) == nested / "new-subdirectory"
    with pytest.raises(ValueError, match="outside"):
        validate_destination(str(outside), (root,))
    with pytest.raises(ValueError, match="absolute"):
        validate_destination("relative/path", (root,))
    with pytest.raises(ValueError, match="outside"):
        browse_destinations((root,), str(outside))


def test_symlink_outside_root_is_rejected(store):
    root, outside, _job = store
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        validate_destination(str(root / "escape" / "surprise"), (root,))
    assert all(item["name"] != "escape" for item in browse_destinations((root,))["directories"])


def test_selected_files_export_while_full_job_files_remain(store):
    root, _outside, job = store
    result = export_selected(job, job.name,
                             plan(root, "srt", "final_english"), (root,))
    assert result["status"] == "complete"
    assert {item["type"] for item in result["items"]} == {"srt", "final_english"}
    exported = root / ("nemovox-" + job.name)
    assert (exported / "show.en.srt").read_bytes() == (job / "output" / "show.en.srt").read_bytes()
    assert (exported / "final_english_subtitles.txt").is_file()
    assert not (exported / "show.en.ass").exists()
    assert (job / "output" / "show.en.ass").is_file()
    assert json.loads((job / "exports.json").read_text())["status"] == "complete"


def test_imported_recording_contract_source_transcript_export(store):
    root, _outside, job = store
    # Subtitle generation from an existing recording contract skips ASR
    # but still generates the faithful source transcript in output/.
    source = job / "asr" / "transcript.txt"
    source.unlink()
    transcript = job / "output" / "raw_source_transcript.txt"
    transcript.write_text("Unchanged source-language dialogue", encoding="utf-8")
    report = export_selected(job, job.name, plan(root, "source_transcript"), (root,))
    assert report["status"] == "complete"
    assert report["items"][0]["source"] == "output/raw_source_transcript.txt"
    assert (root / ("nemovox-" + job.name) / transcript.name).read_bytes() == transcript.read_bytes()
    assert export_selected(job, job.name, plan(root, "source_transcript"), (root,))[
        "items"][0]["status"] == "already_saved"


def test_retry_is_idempotent_and_collision_refuses_overwrite(store):
    root, _outside, job = store
    first = export_selected(job, job.name, plan(root, "srt"), (root,))
    assert first["items"][0]["status"] == "saved"
    second = export_selected(job, job.name, plan(root, "srt"), (root,))
    assert second["items"][0]["status"] == "already_saved"
    destination = root / ("nemovox-" + job.name) / "show.en.srt"
    destination.write_text("Different existing content")
    third = export_selected(job, job.name, plan(root, "srt"), (root,))
    assert third["status"] == "failed"
    assert third["items"][0]["status"] == "failed"
    assert destination.read_text() == "Different existing content"


def test_distinct_jobs_do_not_collide(store):
    root, outside, job = store
    second = job.parent / ("b" * 32)
    (second / "output").mkdir(parents=True)
    (second / "output" / "show.en.srt").write_text("Different synthetic cue")
    for directory in (job, second):
        result = export_selected(directory, directory.name, plan(root, "srt"), (root,))
        assert result["status"] == "complete"
    assert (root / ("nemovox-" + job.name) / "show.en.srt").read_text() != (
        root / ("nemovox-" + second.name) / "show.en.srt").read_text()
    assert outside.is_dir()


def test_remote_media_is_exported_before_later_pipeline_failure(store):
    root, _outside, job = store
    remote = job / "remote" / "source.webm"
    remote.write_bytes(b"synthetic remote file")
    report = export_selected(job, job.name, plan(root, "srt", save_remote_media=True),
                             (root,), remote_media=remote, media_only=True)
    assert report["status"] == "complete"
    assert (root / ("nemovox-" + job.name) / "remote-media.webm").read_bytes() == remote.read_bytes()
    # Simulate ASR failure by never producing an output in a different job.
    after_failure = json.loads((job / "exports.json").read_text())
    assert after_failure["items"][0]["type"] == "remote_media"
    assert remote.is_file()


def test_unselected_remote_media_is_not_exported(store):
    root, _outside, job = store
    remote = job / "remote" / "source.webm"
    remote.write_bytes(b"synthetic")
    report = export_selected(job, job.name, plan(root, "srt"), (root,),
                             remote_media=remote, media_only=True)
    assert report["status"] == "pending"
    assert not (root / ("nemovox-" + job.name)).exists()


def test_missing_artifact_is_reported_without_fabrication(store):
    root, _outside, job = store
    result = export_selected(job, job.name, plan(root, "translated_contract"), (root,))
    assert result["status"] == "failed"
    assert result["items"][0]["type"] == "translated_contract"
    assert result["items"][0]["status"] == "failed"
    assert not (root / ("nemovox-" + job.name) / "translated.contract.json").exists()


def test_atomic_link_failure_does_not_leave_partial_export(monkeypatch, store):
    root, _outside, job = store
    def fail_link(_source, _target):
        raise PermissionError("synthetic permission failure")
    monkeypatch.setattr("nvidia_transcribe.studio_exports.os.link", fail_link)
    result = export_selected(job, job.name, plan(root, "srt"), (root,))
    assert result["status"] == "failed"
    exported = root / ("nemovox-" + job.name)
    assert not list(exported.glob("*.tmp"))
    assert not (exported / "show.en.srt").exists()
    assert (job / "output" / "show.en.srt").is_file()
