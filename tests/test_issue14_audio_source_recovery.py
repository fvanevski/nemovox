"""Audio-attested source recovery is not a model-only partition bypass."""
from copy import deepcopy
import hashlib
import io
import wave

import pytest

from nvidia_transcribe.audio_source_recovery import (
    apply_audio_source_recovery, postedited_audio_recovery_translation,
)
from nvidia_transcribe.evidence_fusion import _digest
from nvidia_transcribe.semantic_review_followups import build_meaning_first_review_queue
from nvidia_transcribe.source_fusion_downstream import assemble_revised_translation
from nvidia_transcribe.translation_review import TranslationReviewConfig
from test_issue14_semantic_review_followups import fixture


def setup():
    req, batch, policy, record, ledger = fixture()
    record["schema_version"] = "nemotron-recording-contract/v1"
    record["processing"] = {"chunks": deepcopy(ledger["asr_observations"])}
    for i, unit in enumerate(ledger["lexical_observations"]):
        unit.update(unit_type="word", ctc_start_s=80+i*.3, ctc_end_s=80+i*.3+.2)
    unit = ledger["lexical_observations"][0]
    record["utterances"] = [{"id":"old","source_text":unit["text"],"source_start":80.,
        "source_end":80.2,"speaker_id":"SPK_UNK","source_words":[
            {"text":unit["text"],"unit_type":"word","chunk_index":1,"source_unit_index":0}]}]
    queue = build_meaning_first_review_queue(req,batch,policy,record,ledger)
    buffer = io.BytesIO()
    with wave.open(buffer,"wb") as wav:
        wav.setparams((1,2,8000,0,"NONE","not compressed"))
        wav.writeframes(b"\0\0"*8000*10)
    audio = buffer.getvalue()
    sha = hashlib.sha256(audio).hexdigest()
    def event(eid,bounds,kind,units,text):
        return {"audio_event_id":eid,"event_s":bounds,"source_text":text,
                "source_unit_ids":units,"parent_asr_observation_ids":["asr_0","asr_1"]
                if bounds[0]>=80 else ["asr_0"],
                "recovery_kind":kind,"spoken_source_text_attested":True,
                "event_bounds_attested":True,"same_occurrence_unit_identity_attested":True,
                "absence_from_original_raw_asr_attested":True}
    review = {"schema":"nemovox-audio-source-recovery-review/v1",
              "queue_sha256":queue["queue_sha256"],"recording_content_sha256":_digest(record),
              "source_ledger_content_sha256":_digest(ledger),"media_fingerprint_sha256":"a"*64,
              "source_language":"fr","review_method":"direct_audio_listening",
              "reviewer_attests_original_media_and_crop_extraction":True,
              "superseded_legacy_ids":["old"],"windows":[{"window_s":[76.,86.],
                  "crop_sha256":sha,"case_ids":[queue["items"][0]["case_id"]],
                  "events":[event("missing",[77.,79.],"asr_absent_speech",[],"Attention!"),
                            event("recovered",[80.2,81.5],"raw_unit_recovery",
                                  ["unit_1","unit_2","unit_3","unit_4"],"La porte n'est pas ouverte.")],
                  "rejected_observations":[{"source_unit_id":"unit_0","reason":"not_spoken_asr_artifact",
                                            "direct_audio_disposition_attested":True}]}]}
    return record,ledger,req,batch,policy,queue,review,{sha:audio}


def apply(inputs, enable=True):
    return apply_audio_source_recovery(*inputs,expected_review_sha256=_digest(inputs[6]),enable=enable)


def test_recovery_preserves_original_and_handles_unowned_absent_and_artifact():
    inputs=setup()
    old=deepcopy(inputs)
    assert apply(inputs,False)["new_recording"] is None
    result=apply(inputs)
    rows=result["new_recording"]["utterances"]
    assert [r["audio_event_id"] for r in rows]==["missing","recovered"]
    assert rows[0]["source_unit_lineage_ids"]==[]
    assert rows[1]["source_unit_lineage_count"]==4
    assert result["report"]["source_unit_observations_deleted"]==0
    assert inputs==old


@pytest.mark.parametrize("mutation",["stale_queue","model_only","bad_crop","partial","duplicate",
    "retained_owner","fake_absence","outside_parent","missing_attestation"])
def test_fail_closed(mutation):
    inputs=list(setup())
    review=inputs[6]
    w=review["windows"][0]
    if mutation=="stale_queue":
        inputs[5]["items"][0]["model_decision"]="possible_duplicate"
    elif mutation=="model_only":
        review["review_method"]="model_advisory"
    elif mutation=="bad_crop":
        inputs[7]={}
    elif mutation=="partial":
        w["rejected_observations"]=[]
    elif mutation=="duplicate":
        w["events"][1]["source_unit_ids"].append("unit_0")
    elif mutation=="retained_owner":
        review["superseded_legacy_ids"]=[]
    elif mutation=="fake_absence":
        w["events"][0]["source_unit_ids"]=["unit_1"]
    elif mutation=="outside_parent":
        w["events"][0]["parent_asr_observation_ids"]=["asr_1"]
    elif mutation=="missing_attestation":
        w["events"][0]["spoken_source_text_attested"]=False
    with pytest.raises(ValueError):
        apply(inputs)


def test_translation_bridge_and_selective_polarity_guard():
    inputs=setup()
    record=inputs[0]
    revised=apply(inputs)["new_recording"]
    prior=deepcopy(record)
    prior["target_language"]="en"
    prior["utterances"][0]["translated_text"]="The"
    receipts=[{"utterance_id":u["id"],"source_revision_id":u["source_revision_id"],
               "source_text_sha256":hashlib.sha256(u["source_text"].encode()).hexdigest(),
               "source_language":"fr","target_language":"en","status":"translated",
               "translated_text":"Watch out!" if i==0 else "The door is not open.",
               "translation_evidence_sha256":"e"*64}
              for i,u in enumerate(revised["utterances"])]
    assembled=assemble_revised_translation(record,prior,revised,receipts,enable=True)["translated_contract"]
    import json
    class Client:
        def complete(self,**kwargs):
            targets=json.loads(kwargs["messages"][1]["content"])["target_utterances"]
            rows=[{"utterance_id":u["utterance_id"],"changed":True,
                   "corrected_text":"Look out!" if i==0 else "The door is open.",
                   "reason":"test"} for i,u in enumerate(targets)]
            return {"choices":[{"message":{"content":json.dumps({"reviews":rows})}}]}
    pristine=deepcopy(assembled)
    output=postedited_audio_recovery_translation(assembled,
            config=TranslationReviewConfig(profile="balanced"),text_client=Client(),enable=True)
    rows=output["translated_contract"]["utterances"]
    assert rows[0]["translated_text"]=="Look out!"
    assert rows[1]["translated_text"]=="The door is not open."
    assert output["report"]["rejected_polarity"]==1
    assert assembled==pristine
