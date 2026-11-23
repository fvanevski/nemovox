"""General direct-audio unit amendment gates, not episode-specific rescue rules."""
from copy import deepcopy
import pytest
from nvidia_transcribe.evidence_fusion import _digest
from nvidia_transcribe.reviewed_unit_amendment import apply_reviewed_unit_amendments, SCHEMA

MEDIA="m"
H="a"*64

def example():
    recording={"source":{"fingerprint_sha256":MEDIA},"source_language":"fr",
               "utterances":[{"id":"old","source_words":[
                   {"chunk_index":0,"source_unit_index":0,"text":"we"},
                   {"chunk_index":0,"source_unit_index":1,"text":"go"}]}]}
    ledger={"media_fingerprint_sha256":MEDIA,"source_language":"fr",
            "asr_observations":[{"id":"asr0","chunk_index":0,"infer_start_s":0,"infer_end_s":20,
                                 "raw_response_sha256":H},
                                {"id":"asr1","chunk_index":1,"infer_start_s":5,"infer_end_s":25,
                                 "raw_response_sha256":H}],
            "lexical_observations":[{"id":"owned0","chunk_index":0,"source_ordinal":0,"text":"we"},
                                    {"id":"owned1","chunk_index":0,"source_ordinal":1,"text":"go"},
                                    {"id":"other","chunk_index":1,"source_ordinal":0,"text":"we"}]}
    dossier={"schema":"nemovox-source-reconciliation/v1","media_fingerprint_sha256":MEDIA,
             "source_language":"fr",
             "audited_audio_regions":[{"audio_event_id":"event","source_status":"audio_review_attested",
                                        "positive_audio_event_unit_ids":["owned0","other"]}],
             "source_turns_audio_review_attested":[{"audio_event_id":"event",
                "source_status":"audio_review_attested_source","reviewer_has_attested_direct_audio_listening":True,
                "source_start":10,"source_end":12}]}
    actions=[
        {"kind":"assign_owned_unit","source_unit_id":"owned1","audio_event_id":"event"},
        {"kind":"reviewed_alternate_observation","source_unit_id":"other",
         "preferred_owned_source_unit_id":"owned0","audio_event_id":"event",
         "separate_occurrence_excluded_by_listening":True,"competing_occurrence_window_s":[8,13]}]
    for action in actions:
        action.update({"review_method":"direct_audio_listening",
                       "same_audio_event_verified":True,"source_unit_identity_verified":True,
                       "review_evidence_sha256":H,"review_audio_crop_sha256":H,
                       "review_audio_crop_interval_s":[7,14]})
    request={"schema":SCHEMA,"media_fingerprint_sha256":MEDIA,"source_language":"fr",
             "recording_content_sha256":_digest(recording),"ledger_content_sha256":_digest(ledger),
             "dossier_content_sha256":_digest(dossier),"actions":actions}
    return recording,ledger,dossier,request


def test_verified_owned_assignment_and_unowned_alternate_preserve_raw():
    recording,ledger,dossier,request=example()
    frozen=deepcopy((recording,ledger,dossier,request))
    res=apply_reviewed_unit_amendments(recording,ledger,dossier,request)
    region=res["derived_dossier"]["audited_audio_regions"][0]
    assert region["positive_audio_event_unit_ids"]==["owned0","owned1"]
    assert region["reviewed_alternate_source_observations"][0]["source_unit_id"]=="other"
    assert res["report"]["raw_unit_observations_deleted"]==0
    assert res["report"]["assigned_owned_units"]==1
    assert res["report"]["reviewed_raw_alternates"]==1
    assert (recording,ledger,dossier,request)==frozen


def test_stale_and_missing_independent_audio_review_fail():
    r,l,d,q=example()
    q["dossier_content_sha256"]="0"*64
    with pytest.raises(ValueError,match="Stale"):
        apply_reviewed_unit_amendments(r,l,d,q)
    r,l,d,q=example()
    q["actions"][1]["separate_occurrence_excluded_by_listening"]=False
    with pytest.raises(ValueError,match="genuine repeated"):
        apply_reviewed_unit_amendments(r,l,d,q)
    r,l,d,q=example()
    q["actions"][0]["review_method"]="model_guess"
    with pytest.raises(ValueError,match="independent listening"):
        apply_reviewed_unit_amendments(r,l,d,q)


def test_unreviewed_competing_window_and_wrong_unit_role_fail():
    r,l,d,q=example()
    q["actions"][1]["review_audio_crop_interval_s"]=[10,12]
    with pytest.raises(ValueError,match="must include competing"):
        apply_reviewed_unit_amendments(r,l,d,q)
    r,l,d,q=example()
    q["actions"][1]["preferred_owned_source_unit_id"]="owned1"
    # owner1 is NOT positively event linked in the original dossier; it is only
    # a proposed assignment and cannot be trusted as original alternate owner.
    with pytest.raises(ValueError,match="same-event retained owner"):
        apply_reviewed_unit_amendments(r,l,d,q)


def test_conflicting_claims_and_inference_exclusion_fail_closed():
    r,l,d,q=example()
    q["actions"].append(deepcopy(q["actions"][0]))
    with pytest.raises(ValueError,match="repeated"):
        apply_reviewed_unit_amendments(r,l,d,q)
    r,l,d,q=example()
    l["asr_observations"][1]["infer_start_s"]=15
    q["ledger_content_sha256"]=_digest(l)
    with pytest.raises(ValueError,match="outside original"):
        apply_reviewed_unit_amendments(r,l,d,q)
