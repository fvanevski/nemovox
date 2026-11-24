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
