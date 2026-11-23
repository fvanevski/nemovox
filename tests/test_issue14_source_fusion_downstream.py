"""Revised event translation bridge: provenance, local voice and stale-MT refusal."""
from copy import deepcopy
import hashlib
import pytest
from nvidia_transcribe.source_fusion_downstream import (
    assemble_revised_translation, attach_local_acoustic_voices, _hash,
)
SHA = "f"*64
EID = "acoustic1"
REV = "revision1"


def fixture():
    old = {"id":"old","source_text":"wrong merged text","source_start":2.0,"source_end":5.0,
           "speaker_id":"SPK_02"}
    keep = {"id":"keep","source_text":"Hello","source_start":10.0,"source_end":11.0,
            "speaker_id":"SPK_03"}
    orig={"source":{"fingerprint_sha256":SHA},"source_language":"fr","utterances":[old,keep]}
    prior=deepcopy(orig)
    prior["target_language"]="en"
    for u,t in zip(prior["utterances"],["wrong old","Hello English"]):
        u["translated_text"]=t
        u["grouped_utterance_ids"]=[u["id"]]
    review={"id":"rev_1","audio_event_id":EID,"source_revision_id":REV,
            "source_text":"Deux voix","source_start":20.0,"source_end":21.0,
            "speaker_id":"SPK_UNK","speaker_identity_accepted":False}
    revised=deepcopy(orig)
    revised["utterances"]=[keep,review]
    revised["source_fusion_revision"]={
        "activation_enabled":True, "recording_content_sha256":_hash(orig),
        "superseded_legacy_ids":["old"],"reviewed_event_ids":[EID]}
    translation={"utterance_id":"rev_1","source_revision_id":REV,
                 "source_text_sha256":hashlib.sha256(b"Deux voix").hexdigest(),
                 "source_language":"fr","target_language":"en","status":"translated",
                 "translated_text":"Two voices","translation_evidence_sha256":"a"*64}
    dossier={"media_fingerprint_sha256":SHA,
             "source_turns_audio_review_attested":[{"audio_event_id":EID,
                "source_revision_id":REV,"source_start":20.0,"source_end":21.0,
                "reviewer_has_attested_direct_audio_listening":True}]}
    voice={"schema":"nemovox-local-acoustic-voice-review/v1",
           "media_fingerprint_sha256":SHA,"review_scope_sha256":"a"*64,
           "review_evidence_sha256":"b"*64,
           "events":[{"audio_event_id":EID,"speaker_evidence_status":"direct_audio_reviewed",
                      "speaker_identity_scope":"local_acoustic_only","review_method":"direct_audio_listening",
                      "review_evidence_sha256":"b"*64,"voice_label":"voice_1"}]}
    return orig,prior,revised,translation,dossier,voice


def test_preview_and_enabled_reuse_only_unchanged_source():
    o,p,r,t,_,_=fixture()
    old=deepcopy((o,p,r))
    preview=assemble_revised_translation(o,p,r,[],enable=False)
    assert preview["translated_contract"] is None
    assert preview["report"]["missing_fresh_translation_ids"]==["rev_1"]
    with pytest.raises(ValueError,match="missing"):
        assemble_revised_translation(o,p,r,[],enable=True)
    out=assemble_revised_translation(o,p,r,[t],enable=True)["translated_contract"]
    assert [u["id"] for u in out["utterances"]]==["keep","rev_1"]
    assert out["utterances"][0]["translated_text"]=="Hello English"
    assert out["utterances"][1]["translated_text"]=="Two voices"
    assert out["utterances"][1]["speaker_id"]=="SPK_UNK"
    assert out["utterances"][1]["translation_review_status"]=="not_reviewed_for_source_revision"
    assert (o,p,r)==old


def test_local_voice_review_preserves_global_speaker_and_requires_scope():
    o,p,r,t,d,v=fixture()
    out=attach_local_acoustic_voices(r,d,v)
    assert out["utterances"][1]["reviewed_local_acoustic_voice"]["label"]=="voice_1"
    assert out["utterances"][1]["speaker_id"]=="SPK_UNK"
    assert out["utterances"][1]["speaker_identity_accepted"] is False
    v["events"][0]["voice_label"]="Tao"
    with pytest.raises(ValueError,match="opaque"):
        attach_local_acoustic_voices(r,d,v)


def test_translation_must_bind_source_revision_and_original():
    o,p,r,t,_,_=fixture()
    t["source_revision_id"]="stale"
    with pytest.raises(ValueError,match="unbound"):
        assemble_revised_translation(o,p,r,[t],enable=True)
    o,p,r,t,_,_=fixture()
    r["source_fusion_revision"]["recording_content_sha256"]="0"*64
    with pytest.raises(ValueError,match="exact original"):
        assemble_revised_translation(o,p,r,[t],enable=True)


def test_stale_retained_translation_and_group_reference_blocked():
    o,p,r,t,_,_=fixture()
    p["utterances"][1]["source_text"]="changed"
    with pytest.raises(ValueError,match="changed retained"):
        assemble_revised_translation(o,p,r,[t],enable=True)
    o,p,r,t,_,_=fixture()
    p["utterances"][1]["grouped_utterance_ids"]=["keep","old"]
    with pytest.raises(ValueError,match="removed legacy"):
        assemble_revised_translation(o,p,r,[t],enable=True)


def test_duplicate_or_unexpected_translation_and_unknown_review_event_blocked():
    o,p,r,t,d,v=fixture()
    with pytest.raises(ValueError,match="Duplicated"):
        assemble_revised_translation(o,p,r,[t,t],enable=True)
    v["events"][0]["audio_event_id"]="not_in_revision"
    with pytest.raises(ValueError,match="unselected"):
        attach_local_acoustic_voices(r,d,v)


def test_voice_metadata_kept_in_translation_without_global_voice_promotion():
    o,p,r,t,d,v=fixture()
    rv=attach_local_acoustic_voices(r,d,v)
    out=assemble_revised_translation(o,p,rv,[t],enable=True)["translated_contract"]
    new=out["utterances"][1]
    assert new["reviewed_local_acoustic_voice"]["scope_sha256"]=="a"*64
    assert new["speaker_id"]=="SPK_UNK"
