"""Deferred native evidence and comparative decisions retain complete primary MT."""
import pytest

from nvidia_transcribe.subtitle_pipeline import validate_deferred_review


def test_mixed_deferred_stages_are_explicit_and_complete():
    rows=[
        {"id":"a","machine_translated_text":"A","translated_text":"A",
         "translation_review_status":"review_deferred_evidence_preserved_machine_translation"},
        {"id":"b","machine_translated_text":"B","translated_text":"B",
         "translation_review_status":"review_deferred_comparative_preserved_machine_translation"},
        {"id":"c","machine_translated_text":"C","translated_text":"Correct C",
         "translation_review_status":"corrected"}]
    report={"status":"completed_with_deferred_comparisons","reviewed":1,
            "deferred_comparison_ids":["b"],"supplementary_proposal_errors":[{"target_id":"a","reason":"incomplete"}]}
    validate_deferred_review(rows,report,{})
    with pytest.raises(RuntimeError):
        validate_deferred_review(rows,{**report,"supplementary_proposal_errors":[]},{})
    with pytest.raises(RuntimeError):
        validate_deferred_review([{**rows[0],"translated_text":"invented"},*rows[1:]],report,{})
    with pytest.raises(RuntimeError):
        validate_deferred_review(rows,report,{"errors":["translation failure"]})
    validate_deferred_review([rows[0],rows[2]],{**report,"status":"completed_with_deferred_evidence",
        "deferred_comparison_ids":[]},{})
