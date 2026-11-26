"""Sequence nomination is not source deletion or acoustic verification."""
from copy import deepcopy
from scripts.evaluate_source_robustness import displaced_sequences, validate_plan
import pytest


def ledger(late_start=30):
    parents=[{"id":"a","chunk_index":0,"infer_start_s":0,"infer_end_s":25,"core_end_s":20},
             {"id":"b","chunk_index":1,"infer_start_s":15,"infer_end_s":45,"core_end_s":40}]
    units=[]
    for idx,(parent,start) in enumerate(zip(parents,(16,late_start))):
        for j,word in enumerate("the captain found our secret hidden passage".split()):
            units.append({"id":f"u{idx}-{j}","asr_observation_id":parent["id"],
                          "chunk_index":idx,"source_ordinal":j,"text":word,
                          "raw_recognized_token_ids":[f"raw{idx}-{j}"],
                          "recognized_turn_id":f"turn{idx}",
                          "ctc_start_s":start+j*.4,"ctc_end_s":start+j*.4+.25,
                          "boundary_disposition":"selected_provisional"})
    return {"asr_observations":parents,"lexical_observations":units,
            "recognized_occurrence_inventories":[{"asr_observation_id":p["id"],
                                                "projection_status":"exact"} for p in parents]}


def test_displaced_sequence_nominates_without_mutating_or_asserting_truth():
    data=ledger()
    before=deepcopy(data)
    result=displaced_sequences(data)
    assert len(result)==1 and result[0]["unit_count"]==7
    assert result[0]["acoustic_truth_accepted"] is False
    assert data==before


def test_same_locality_and_incomplete_projection_do_not_nominate():
    assert not displaced_sequences(ledger(late_start=16))
    data=ledger()
    data["recognized_occurrence_inventories"][1]["projection_status"]="partial"
    assert not displaced_sequences(data)


def test_budget_rejects_large_or_uncontained_independent_crops():
    with pytest.raises(ValueError):
        validate_plan({"independent_targets":[{"id":"t","bounds":[10,20],
                    "crops":[[11,25],[0,30]]}]},400)
    with pytest.raises(ValueError):
        validate_plan({"local_probes":[{"id":"p","bounds":[0,61]}]},400)
