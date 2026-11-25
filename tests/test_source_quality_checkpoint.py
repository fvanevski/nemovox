"""Current source-quality resume cannot treat stale or failed work as a cache hit."""
import json
from copy import deepcopy

import pytest

from nvidia_transcribe.source_quality import _digest
from nvidia_transcribe.source_quality_checkpoint import run_checkpoint


def test_exact_checkpoint_reuses_partial_bounded_result_and_recomputes_on_identity_change(tmp_path):
    rec={"source":{"fingerprint_sha256":"media"},"source_language":"fr","utterances":[{"id":"a"}]}
    calls=[]
    def compute():
        calls.append(1)
        report={"status":"partial","semantic_review_requested":True,"original_recording_modified":False,
                "recording_sha256":_digest(rec),"recording_context":{"attempts":[{"status":"validated_response"}]},
                "scene_evidence":{"errors":[],"deferred_source_ids":["a"]},
                "independent_recovery":{"status":"observed","errors":[]}}
        return {**deepcopy(rec),"source_quality":deepcopy(report)},report
    path=tmp_path/"source-quality.checkpoint.json"
    first=run_checkpoint(path,"identity1",rec,resume=True,run=compute)
    second=run_checkpoint(path,"identity1",rec,resume=True,run=compute)
    assert first[:2]==second[:2] and second[2]=="reused_exact_input" and len(calls)==1
    assert run_checkpoint(path,"identity2",rec,resume=True,run=compute)[2]=="computed"
    doc=json.loads(path.read_bytes())
    doc["payload"]["canonical"]["utterances"][0]["id"]="invented"
    path.write_text(json.dumps(doc))
    assert run_checkpoint(path,"identity2",rec,resume=True,run=compute)[2]=="computed"
    assert len(calls)==3
    assert run_checkpoint(path,"identity2",rec,resume=False,run=compute)[2]=="computed"
    assert len(calls)==4


def test_failed_fresh_run_does_not_resurrect_checkpoint(tmp_path):
    rec={"source":{},"source_language":"fr","utterances":[]}
    report={"status":"partial","semantic_review_requested":True,"original_recording_modified":False,
        "recording_sha256":_digest(rec),"recording_context":{"attempts":[]},
        "scene_evidence":{"errors":[]},"independent_recovery":{"errors":[]}}
    path=tmp_path/"quality.json"
    run_checkpoint(path,"id",rec,resume=True,run=lambda:({**rec,"source_quality":report},report))
    def fail():
        raise RuntimeError("provider failure")
    with pytest.raises(RuntimeError):
        run_checkpoint(path,"id",rec,resume=False,run=fail)
    assert not path.exists()
    bad={**report,"recording_context":{"attempts":[{"status":"failed"}]}}
    assert run_checkpoint(path,"id",rec,resume=True,run=lambda:({**rec,"source_quality":bad},bad))[2]=="computed_not_reusable"
    assert not path.exists()
