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
