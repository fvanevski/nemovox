"""Offline evidence evaluator does not equate model support with truth."""
import hashlib
import io
import tarfile

import pytest



def test_failed_review_never_claims_stale_render_metadata(tmp_path):
    from nvidia_transcribe.render_receipt import current_render_metadata
    failed={"translation_review":{"status":"completed_with_errors"}}
    assert current_render_metadata(tmp_path,failed) is None
    path=tmp_path/"subtitles.metadata.json"
    path.write_text('{"subtitles":{"renderable_cues":10}}')
    assert current_render_metadata(tmp_path,failed) is None
    assert current_render_metadata(tmp_path,{"translation_review":{"status":"completed"}})["subtitles"]["renderable_cues"]==10
    path.unlink()
    with pytest.raises(ValueError,match="Missing render metadata"):
        current_render_metadata(tmp_path,{"translation_review":{"status":"completed"}})


def test_failed_review_salvage_receipt_rejects_stale_or_incomplete_outputs(tmp_path):
    import json
    from nvidia_transcribe.render_receipt import current_render_metadata
    failed = {"translation_review": {"status": "completed_with_errors",
                                      "errors": ["transport"], "resource_blocked_reason": None}}
    contract = tmp_path / "translated.contract.json"
    contract.write_text(json.dumps(failed))
    subtitle = tmp_path / "episode.en.srt"
    subtitle.write_text("retained subtitle")
    metadata = {"render_status": "rendered_with_review_errors",
                "translation_review": {"status": "reused_translated_contract",
                    "retained_review_status": "completed_with_errors",
                    "errors": ["transport"], "resource_blocked_reason": None},
                "files": {"srt": "/old/job/episode.en.srt"},
                "subtitles": {"renderable_cues": 1},
                "salvage_render_receipt": {"schema": "nemovox-failed-review-render/v1",
                    "translated_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
                    "outputs": {"srt": {"name": subtitle.name,
                        "sha256": hashlib.sha256(subtitle.read_bytes()).hexdigest()}}}}
    path = tmp_path / "subtitles.metadata.json"
    path.write_text(json.dumps(metadata))
    assert current_render_metadata(tmp_path, failed) == metadata
    with pytest.raises(ValueError, match="Stale failed-review salvage"):
        current_render_metadata(tmp_path, {"translation_review": {"status": "completed"}})
    for changed in (
        {**metadata, "files": {**metadata["files"], "ass": "episode.en.ass"}},
        {**metadata, "translation_review": {**metadata["translation_review"], "errors": []}},
        {**metadata, "salvage_render_receipt": {**metadata["salvage_render_receipt"],
            "outputs": {"srt": {"name": "../episode.en.srt",
                "sha256": hashlib.sha256(subtitle.read_bytes()).hexdigest()}}}},
    ):
        path.write_text(json.dumps(changed))
        assert current_render_metadata(tmp_path, failed) is None
    path.write_text(json.dumps(metadata))
    subtitle.unlink()
    assert current_render_metadata(tmp_path, failed) is None
