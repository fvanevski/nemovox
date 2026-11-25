"""Offline evidence evaluator does not equate model support with truth."""
import hashlib
import io
import tarfile

import pytest

from scripts.evaluate_final_integration import sealed, compare_rows, arm
from scripts.evaluate_chunk_geometry import compare_reference


def test_sealed_input_rejects_manifest_mismatch(tmp_path):
    archive=tmp_path/"fixture.tar.gz"
    with tarfile.open(archive,"w:gz") as out:
        for name,payload in (("root/data.json",b"{}"),
                             ("root/MANIFEST.sha256",b"wrong  data.json\n")):
            info=tarfile.TarInfo(name)
            info.size=len(payload)
            out.addfile(info,io.BytesIO(payload))
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match="payload mismatch"):
        sealed(archive,digest)
    with pytest.raises(ValueError,match="identity mismatch"):
        sealed(archive,"wrong")


def test_reference_uses_actual_offset_and_distinguishes_word_locality():
    row={"id":"a","source_text":"Bonjour Nadia","source_start":1.,"source_end":3.,
         "speaker_id":"SPK_00","untimed_unit_count":1,"occurrence_span_ids":["span"]}
    ledger={"recognized_occurrence_inventories":[
        {"locality":{"spans":[{"id":"span","locality_status":"disputed"}]},
         "recognized_turns":[{"text":"Bonjour Nadia"}]}]}
    event={"source_text":"Bonjour Nadia","source_start":261.,"source_end":263.,
           "supporting_observation_ids":["ref1","ref2"]}
    result=compare_reference([event],arm([row],ledger,260.))
    assert result[0]["status"]=="same_model_timing_support"
    strict=compare_reference([event],arm([row],ledger,260.,strict=True))
    assert strict[0]["status"]=="no_strict_event_match"
    assert row["source_start"]==1. and row["source_end"]==3.


def test_occurrence_comparison_never_pairs_new_row_by_position():
    prior=[{"id":"a","source_text":"Bonjour","machine_translated_text":"Hello",
            "translated_text":"Hi"}]
    current=[{"id":"b","source_text":"Bonjour","machine_translated_text":"Hello",
              "translated_text":"Hi"}]
    result=compare_rows(prior,current)
    assert result=={"removed_ids":["a"],"added_ids":["b"],"shared":[]}



def test_failed_review_never_claims_stale_render_metadata(tmp_path):
    from scripts.evaluate_final_integration import current_render_metadata
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
    from scripts.evaluate_final_integration import current_render_metadata
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
