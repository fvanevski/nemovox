"""Generic scene and audio-geometry invariants, independent of episode fixtures."""
from nvidia_transcribe.scene_evidence import (
    SceneEvidencePolicy, scene_context, audio_candidates, caption_patch, apply_decision,
)
from nvidia_transcribe.source_quality import SourceQualityPolicy, challenge_geometries


def row(uid,text,a,b):
    return {"id":uid,"source_text":text,"source_start":a,"source_end":b,"speaker_id":"SPK_A"}


def recording(rows):
    return {"source":{"duration_s":400},"source_language":"fr","utterances":rows}


def test_complete_context_includes_distant_setup_reply_and_untimed_dialogue():
    rows=[row("setup","We are hiding from the guards.",20,25),
          row("before","Be careful.",70,73),
          row("untimed","An untimed intervening line.",None,None),
          row("target","I told you not to speak.",100,104),
          row("reply","Sorry, I could not stay quiet.",130,135),
          row("later","They found us.",230,234)]
    small=scene_context(recording(rows),(100,104),60,24000)
    large=scene_context(recording(rows),(100,104),180,24000)
    assert [x["id"] for x in small["rows"]]==["before","untimed","target","reply"]
    assert [x["id"] for x in large["rows"]]==[x["id"] for x in rows]
    assert large["before_available"] and large["after_available"]
    assert large["rows"][0]["source_text"]==rows[0]["source_text"]
    short=scene_context(recording([rows[3]]),(100,104),60,24000)
    assert not short["before_available"] and not short["after_available"]


def test_geometry_changes_with_target_duration_and_failure_type():
    p=SourceQualityPolicy()
    short=challenge_geometries((20,24),400,p,timing_problem=True)
    long=challenge_geometries((20,32),400,p,timing_problem=True)
    assert short[0][0]=="compact" and long[0][0]=="compact"
    assert long[0][1][1]-long[0][1][0]>short[0][1][1]-short[0][1][0]
    assert challenge_geometries((20,24),400,p)[0][0]=="expanded"
    g=dict(short)
    assert g["shift_left"][1]>g["shift_right"][0]
    assert all(0<=a<b<=400 for _,(a,b) in short+long)
    assert all(crop!=short[0][1] for _,crop in
               challenge_geometries((20,24),400,p,timing_problem=True,seen=[short[0][1]]))


def test_multilingual_caption_whole_and_partial_replacement():
    assert caption_patch("Je veux partir demain.","Je veux partir demain.",
                         "Je veux revenir demain.")=="Je veux revenir demain."
    assert caption_patch("我们在古城的权利比赛中获取利益然后离开",
                         "在古城的权利比赛中获取利益",
                         "在古城的权力竞赛中获取利益")=="我们在古城的权力竞赛中获取利益然后离开"
    assert caption_patch("She left. She left.","She left.","She stayed.") is None
    assert caption_patch("He hid the key.","He hid the key.","Unrelated title") is None
    assert caption_patch("Lost source words","Lost source words","A visible line",
                         [{"text":"A visible line"}])=="A visible line"


def test_acoustic_selection_preserves_event_identity_and_timing():
    base=row("t","I told you not to leave.",100,104)
    base["source_words"]=[{"text":"leave","start":103,"end":104}]
    alt={**row("p","I told you not to speak.",100.1,103.9),"observation_id":"p"}
    obs=[{"audio_sha256":"a","utterances":[alt]}]
    candidates=audio_candidates(base,[base],obs,[])
    assert len(candidates)==1
    d={"decision":{"action":"acoustic","candidate_id":candidates[0]["id"],
                   "materially_resolved":True}}
    assert not apply_decision(base,d,candidates,[])
    candidates=audio_candidates(base,[base],obs,
        [{"audio_sha256":"b","rows":[{**alt,"observation_id":"q"}]}])
    assert apply_decision(base,d,candidates,[])
    assert (base["source_start"],base["source_end"],base["speaker_id"])==(100,104,"SPK_A")
    assert base["source_text"]==alt["source_text"]
    assert base["source_words"]==[]
    assert base["prior_scene_source_hypothesis"]["source_words"]
    assert not apply_decision(base,{"decision":{"action":"acoustic","candidate_id":"invented",
                                                "materially_resolved":True}},candidates,[])


def test_no_candidate_crosses_another_accepted_turn():
    base=row("t","Please open the door.",100,104)
    other=row("o","Wait!",103.9,105)
    alt={**row("p","Please close the door.",100.1,104.1),"observation_id":"p"}
    assert audio_candidates(base,[base,other],[{"audio_sha256":"a","utterances":[alt]}],[])==[]



def test_current_hypothesis_corroboration_is_visible_without_counting_votes():
    base=row("t","I told you not to breathe.",100,104)
    a={**base,"observation_id":"p"}
    b={**base,"observation_id":"q"}
    c={**row("alt","I told you not to speak.",100.1,103.9),"observation_id":"r"}
    candidates=audio_candidates(base,[base],
        [{"audio_sha256":"one","utterances":[a,c]}],
        [{"audio_sha256":"two","rows":[b]}],include_current=True)
    current=next(x for x in candidates if x["id"]=="current")
    assert {x["backend"] for x in current["support"]}=={"nemotron","independent_asr"}
    assert len(candidates)==2



def test_partial_caption_edit_preserves_uncertain_remainder_and_prior():
    original=row("t","I will find the old gate. Unknown tail.",100,104)
    item=dict(original)
    visual=[{"id":"frame","caption":"I will find the hidden gate.","inside_event":True,"uncertainty":"",
             "text_kind":"dialogue_caption","literal_observation_status":"model_observed",
             "visible_text":"I will find the hidden gate."}]
    assessment={"decision":{"action":"caption","candidate_id":"frame",
                             "replace_span":"I will find the old gate.",
                             "materially_resolved":True}}
    assert not apply_decision(item,assessment,[],visual)
    alt={**row("p","I will find the hidden gate. Unknown tail.",100,104),"observation_id":"p"}
    candidates=audio_candidates(item,[item],[{"audio_sha256":"a","utterances":[alt]}],
        [{"audio_sha256":"b","rows":[{**alt,"observation_id":"q"}]}])
    assert apply_decision(item,assessment,candidates,visual)
    assert item["source_text"]=="I will find the hidden gate. Unknown tail."
    assert item["source_semantic_resolution"]=="partial"
    assert item["source_quality_status"]=="partially_supported_caption_hypothesis"
    assert item["prior_scene_source_hypothesis"]==original
    assert item["source_caption_evidence"][0]["replaced_span"]=="I will find the old gate."
    assert (item["source_start"],item["source_end"])==(100,104)



def test_context_nominations_and_visual_literals_are_row_bound():
    from nvidia_transcribe.semantic_context import row_nominations, accepted_row_context, current_row_context
    base=row("t","Nadia ferme la porte.",100,104)
    base["source_unit_ids"]=["unit1"]
    prop={"proposal_id":"p","kind":"entity_variant","citations":[
        {"source_id":"t","quote":"Nadia","character_span":[0,5],"source_row_unit_ids":["unit1"]},
        {"source_id":"missing","quote":"Nadya","character_span":[0,5],"source_row_unit_ids":["unit2"]}]}
    nominations,unavailable=row_nominations([base],{"proposals":[prop]})
    assert nominations["t"][0]["status"]=="advisory_only"
    assert unavailable[0]["source_id"]=="missing"
    frame={"id":"f","inside_event":True,"visible_text":"Nadia","text_kind":"entity_label",
           "literal_observation_status":"model_observed",
           "uncertainty":"","sha256":"a"*64,"requested_seek_s":101.}
    context=accepted_row_context(base,[],[frame])
    assert context["visual_literals"][0]["literal"]=="Nadia"
    assert context["entity_identity_accepted"] is False
    assert accepted_row_context(base,[],[{**frame,"visible_text":"Nadya"}]) is None
    assert accepted_row_context(base,[],[{**frame,"inside_event":False}]) is None
    assert accepted_row_context(base,[],[{**frame,"uncertainty":"unreadable"}]) is None
    base["semantic_context"]=context
    assert current_row_context(base)==context
    base["source_text"]="Nadya ferme la porte."
    assert current_row_context(base) is None
    assert not row_nominations([base],{"proposals":[prop]})[0]



def test_competing_cross_model_wordings_remain_unresolved():
    base=row("t","I told you not to leave.",100,104)
    alt={**row("alt","I told you not to speak.",100,104),"observation_id":"a"}
    current={**base,"observation_id":"b"}
    candidates=audio_candidates(base,[base],[{"audio_sha256":"one","utterances":[alt,current]}],
        [{"audio_sha256":"two","rows":[{**alt,"observation_id":"c"},{**current,"observation_id":"d"}]}],
        include_current=True)
    chosen=next(x for x in candidates if x["id"]!="current")
    assert not apply_decision(base,{"decision":{"action":"acoustic","candidate_id":chosen["id"],
        "materially_resolved":True}},candidates,[])
    assert base["source_text"]=="I told you not to leave."
    from nvidia_transcribe.semantic_context import accepted_row_context
    assert accepted_row_context(base,candidates,[]) is None



def test_visual_description_and_branding_are_not_dialogue_captions():
    from nvidia_transcribe.scene_evidence import normalize_visual_observation
    frame={"id":"f","inside_event":True,"sha256":"a"*64,"requested_seek_s":101.}
    def observation(kind,text,description="A door in a room.",uncertainty=""):
        return normalize_visual_observation(frame,{"id":"f","visible_text":text,
            "text_kind":kind,"description":description,"uncertainty":uncertainty})
    assert observation("logo","OKOO")["caption"]==""
    assert observation("title","Nadia")["caption"]==""
    assert observation("none","A door in a room.")["visible_text"]==""
    assert observation("dialogue_caption","A door in a room.")["caption"]==""
    assert observation("dialogue_caption","Bonjour",uncertainty="blurred")["caption"]==""
    assert observation("dialogue_caption","Bonjour")["caption"]=="Bonjour"
    assert observation("entity_label","Nadia")["visible_text"]=="Nadia"


def test_scene_rejected_response_retained_and_one_correction_per_case():
    import json
    import pytest
    from nvidia_transcribe.scene_evidence import (
        _decision_schema, assess_case_payload, SceneDecisionError,
    )
    payload={"scene":{"rows":[{"id":"s"}]},
             "acoustic_alternatives":[{"id":"current"}],"visual_observations":[]}
    schema=_decision_schema(["current"],[],["s"])
    good={"action":"retain","candidate_id":"","replace_span":"","meaning":"m",
          "reason":"r","materially_resolved":False,"context_ids":["s"],
          "next_evidence":"none"}
    bad={**good,"context_ids":["s","s"]}
    class Client:
        def __init__(self):
            self.responses=iter([bad,good,bad])
            self.calls=[]
        def complete(self,**kw):
            self.calls.append(kw)
            return {"choices":[{"finish_reason":"stop","message":{
                "content":json.dumps(next(self.responses))}}]}
    client=Client()
    case={}
    result=assess_case_payload(client,payload,schema,case)
    assert result["decision"]==good
    assert case["assessment_recovery_attempts"]==1
    assert case["failed_assessments"][0]["decision"]==bad
    assert "duplicate_context_ids" in case["failed_assessments"][0]["violations"]
    assert "duplicate_context_ids" in client.calls[1]["messages"][0]["content"]
    assert client.calls[0]["messages"][1]==client.calls[1]["messages"][1]
    assert client.calls[0]["response_format"]==client.calls[1]["response_format"]
    with pytest.raises(SceneDecisionError):
        assess_case_payload(client,payload,schema,case)
    assert len(client.calls)==3
    assert len(case["failed_assessments"])==2


def test_scene_bad_references_and_malformed_types_fail_without_selection():
    from nvidia_transcribe.scene_evidence import _decision_schema, decision_violations
    payload={"scene":{"rows":[{"id":"s"}]},
             "acoustic_alternatives":[{"id":"a"}],
             "visual_observations":[{"id":"f","caption":"text"}]}
    schema=_decision_schema(["a"],["f"],["s"])
    good={"action":"acoustic","candidate_id":"a","replace_span":"","meaning":"m",
          "reason":"r","materially_resolved":True,"context_ids":["s"],
          "next_evidence":"none"}
    assert not decision_violations(good,payload,schema)
    assert "acoustic_candidate_reference" in decision_violations(
        {**good,"candidate_id":"f"},payload,schema)
    assert "context_id_type" in decision_violations({**good,"context_ids":[{}]},payload,schema)
    assert "unknown_context_ids" in decision_violations({**good,"context_ids":["fake"]},payload,schema)
    assert "materially_resolved_type" in decision_violations(
        {**good,"materially_resolved":"true"},payload,schema)
    assert decision_violations([],payload,schema)==["decision_not_object"]
