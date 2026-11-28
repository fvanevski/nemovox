"""Private evaluation selection is deterministic and connected-unit complete."""
from copy import deepcopy

from scripts.evaluate_semantic_review import select_targets
from nvidia_transcribe import connected_units


def row(i, source, primary, selected, start, end):
    return {"id":f"event-{i:05d}","source_text":source,"machine_translated_text":primary,
            "translated_text":selected,"speaker_id":"S01","source_start":start,"source_end":end,
            "timestamp_provenance":{"backend":"moss_events","native_event_index":i,
                                    "timing_status":"observed_segment"}}


def test_plan_expands_reserved_and_acquisition_targets_to_atomic_candidates():
    rows=[row(0,"拿","Take","Take",1,2),row(1,"杯子。","a cup.","the cup.",2.1,3),
          row(2,"他走了。","He left.","They left.",10,11),
          row(3,"是。","Yes.","Yes.",20,21)]
    for group in connected_units.nominate(rows):
        record={**connected_units.request_record(group,[]),"joint_mt":"Take a cup.","joint_input_tokens":5}
        for r in group:
            r["translation_unit_candidate"]=deepcopy(record)
    ids=[r["id"] for r in rows]
    contract={"source":{"duration_s":30},"utterances":rows,"translation_review":{
        "evidence_assessments":[{"target_id":ids[2],"reference_id":ids[2],"kind":"visual"}]}}
    first=select_targets(contract,[(2.5,2.9)],[ids[3]])
    assert first==select_targets(contract,[(2.5,2.9)],[ids[3]])
    by_id={r["id"]:r["reasons"] for r in first}
    assert by_id[ids[0]]==["connected_allocation_member"]
    assert "reserved_comparison_window_1" in by_id[ids[1]]
    assert {"prior_visual_acquisition","referent_change_candidate"}<=set(by_id[ids[2]])
    assert by_id[ids[3]]==["documented_development_case"]
