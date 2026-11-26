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



def test_identity_binds_complete_media_ledger_and_policy(tmp_path,monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from dataclasses import replace
    from nvidia_transcribe import source_quality_checkpoint as checkpoint
    from nvidia_transcribe.source_quality import SourceQualityPolicy
    rec={"source":{"fingerprint_sha256":"media","selected_audio_stream":0,"duration_s":2.},
         "source_language":"fr","processing":{"chunks":[{"chunk_index":0,"raw_response_sha256":"raw"}]},
         "utterances":[{"id":"a","source_text":"Bonjour"}]}
    ledger={"schema_version":"nemovox-source-evidence/v1","recognized_occurrence_inventories":[],
            "media_fingerprint_sha256":"media","source_language":"fr",
            "asr_observations":[{"chunk_index":0,"raw_response_sha256":"raw"}]}
    recording=tmp_path/"recording.contract.json"
    recording.write_text(json.dumps(rec))
    sidecar=tmp_path/"source-evidence.v1.json"
    sidecar.write_text(json.dumps(ledger))
    (tmp_path/"manifest.json").write_text(json.dumps({"asr_run_status":"completed",
        "recording_contract_sha256":hashlib.sha256(recording.read_bytes()).hexdigest(),
        "source_evidence":{"sha256":hashlib.sha256(sidecar.read_bytes()).hexdigest()}}))
    media=tmp_path/"video.mkv"
    media.write_bytes(b"test media")
    monkeypatch.setattr(checkpoint,"probe_media",lambda *a,**k:SimpleNamespace(
        fingerprint_sha256="media",duration_s=2.))
    policy=SourceQualityPolicy()
    get=lambda p=policy,r={"asr":"one"}:checkpoint.identity(rec,recording,media,p,r)
    key=get()
    assert get()==key
    assert get(replace(policy,max_windows=5))!=key
    assert get(r={"asr":"two"})!=key
    media.write_bytes(b"test other")
    assert get()!=key
    sidecar.write_bytes(sidecar.read_bytes()+b" ")
    with pytest.raises(ValueError,match="source evidence"):
        get()


def test_missing_or_modified_frame_receipt_invalidates_reuse(tmp_path):
    import hashlib
    from nvidia_transcribe.source_quality_checkpoint import frame_receipts_available
    frame=tmp_path/"frame.png"
    frame.write_bytes(b"original")
    report={"scene_evidence":{"decisions":[{"visual":[{
        "path":str(frame),"sha256":hashlib.sha256(frame.read_bytes()).hexdigest()}]}]}}
    assert frame_receipts_available(report,tmp_path)
    frame.write_bytes(b"altered!")
    assert not frame_receipts_available(report,tmp_path)
    frame.unlink()
    assert not frame_receipts_available(report,tmp_path)


def test_checkpoint_digest_is_stable_for_two_digit_anchor_offsets(tmp_path):
    rec={"source":{},"source_language":"zh","utterances":[]}
    calls=[]
    def compute():
        calls.append(1)
        report={"status":"partial","semantic_review_requested":True,"original_recording_modified":False,
            "recording_sha256":_digest(rec),"recording_context":{"attempts":[]},
            "scene_evidence":{"errors":[]},"independent_recovery":{"errors":[]},
            "decisions":[{"constituent_event_evidence":[{"anchors":{2:[1.,2.],10:[3.,4.]},
                "window":(1.,4.)}]}]}
        return {**rec,"source_quality":report},report
    path=tmp_path/"quality.json"
    first=run_checkpoint(path,"id",rec,resume=True,run=compute)
    saved=json.loads(path.read_bytes())
    assert saved["payload_sha256"]==_digest(saved["payload"])
    second=run_checkpoint(path,"id",rec,resume=True,run=compute)
    assert len(calls)==1 and second[2]=="reused_exact_input" and first[:2]==second[:2]


def test_checkpoint_rejects_key_collisions_before_publication(tmp_path):
    rec={"source":{},"source_language":"zh","utterances":[]}
    report={"status":"partial","semantic_review_requested":True,"original_recording_modified":False,
        "recording_sha256":_digest(rec),"recording_context":{"attempts":[]},
        "scene_evidence":{"errors":[]},"independent_recovery":{"errors":[]},
        "anchors":{2:"one","2":"two"}}
    path=tmp_path/"quality.json"
    with pytest.raises(ValueError,match="Duplicate checkpoint JSON key"):
        run_checkpoint(path,"id",rec,resume=True,run=lambda:({**rec,"source_quality":report},report))
    assert not path.exists()
