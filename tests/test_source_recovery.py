"""Synthetic independent-source recovery invariants; no model or media required."""
from nvidia_transcribe.source_recovery import (
    apply_independent, plan_recovery,
)


def row(uid,text,a,b,chunk=0):
    return {"id":uid,"source_text":text,"source_start":a,"source_end":b,
            "speaker_id":"SPK_A","chunk_indices":[chunk],"observation_id":uid}


def rec(rows,language="fr"):
    return {"source":{"fingerprint_sha256":"synthetic","duration_s":40},
            "source_language":language,"utterances":rows}


def obs(uid,a,b,rows):
    return {"id":uid,"crop":[a,b],"audio_sha256":uid,"rows":rows}


def test_insert_cross_model_supported_omission_keeps_uncertainty():
    r=rec([row("before","Bonjour",1,2),row("after","Au revoir",15,16)])
    nem=[{"audio_sha256":str(i),"chunks":[0],
          "utterances":[row("n"+str(i),"Je vous avais dit de ne pas partir",5,9)]}
          for i in range(2)]
    targets=plan_recovery(r,nem)
    w=[obs("challenge-0-0",3,11,[row("a","Je vous avais dit de ne pas parler",5.4,8.9)]),
       obs("challenge-0-1",0,14,[row("b","Je vous avais dit de ne pas parler",5.2,8.8)])]
    rows,decisions=apply_independent(r,targets,nem,w)
    assert len(decisions)==1 and len(rows)==3
    assert rows[1]["source_text"]=="Je vous avais dit de ne pas parler"
    assert rows[1]["speaker_id"]=="SPK_UNK"
    assert rows[1]["source_quality_status"].startswith("unresolved")
    assert rows[1]["source_words"]==[]
    assert r["utterances"]==[rows[0],rows[2]]


def test_no_new_occurrence_if_original_competes():
    original=row("old","Je vous avais dit de ne pas partir",20,22)
    r=rec([original])
    nem=[{"audio_sha256":str(i),"chunks":[0],"utterances":[row("n"+str(i),original["source_text"],5,9)]} for i in range(2)]
    w=[obs("challenge-0-0",3,11,[row("a",original["source_text"],5.4,8.9)]),
       obs("challenge-0-1",0,14,[row("b",original["source_text"],5.2,8.8)])]
    rows,decisions=apply_independent(r,plan_recovery(r,nem),nem,w)
    assert rows==r["utterances"] and decisions==[]


def test_untimed_run_duplicate_fragments_across_chunks_only():
    original=[row("left","你好",1,2),row("fragment","为了回归家乡",None,None,0),
              row("main","为了回归家乡再次开始新的生活",None,None,1),
              row("later","最终成功完成任务",None,None,1),row("right","再见",20,21)]
    r=rec(original,"zh")
    w=[obs("challenge-0-0",0,22,[row("a","为了回归家乡再次开始新的生活",4,10),row("b","最终成功完成任务",11,15)]),
       obs("challenge-0-1",0,24,[row("c","为了回归家乡再次开始新的生活",4.1,10.1),row("d","最终成功完成任务",11.1,15.1)])]
    rows,decisions=apply_independent(r,plan_recovery(r,[]),[],w)
    assert len(decisions)==1 and len(rows)==4
    assert rows[1]["source_start"]==4.1 and rows[2]["source_end"]==15.1
    assert rows[1]["prior_source_hypotheses"]==original[1:4]
    original[1]["chunk_indices"]=[1]
    rows,decisions=apply_independent(r,plan_recovery(r,[]),[],w)
    assert rows==original and decisions==[]


def test_noisy_interior_segment_does_not_invalidate_enclosing_timed_event():
    original=[row("left","你好",1,2),
              row("main","从山城出发穿过古老森林最后回到家乡",None,None,1),
              row("right","再见",20,21)]
    r=rec(original,"zh")
    segments=[row("a","从山城出发",4,6),
              row("b","穿过顾老深林",6,9),
              row("c","最后回到家乡",9,12)]
    # Same source occurrence, different independently extracted crops.
    w=[obs("challenge-0-0",0,22,segments),
       obs("challenge-0-1",0,24,[{**x,"observation_id":"other-"+x["id"]} for x in segments])]
    rows,decisions=apply_independent(r,plan_recovery(r,[]),[],w)
    assert len(decisions)==1
    assert rows[1]["source_text"]=="从山城出发穿过顾老深林最后回到家乡"
    assert rows[1]["source_quality_status"].startswith("unresolved")



def test_crop_edge_is_only_corroboration_when_interior_sequence_agrees():
    r=rec([row("before","Hello",1,2),row("after","Goodbye",15,16)],"en")
    text="I told you not to speak"
    nem=[{"audio_sha256":str(i),"chunks":[0],
          "utterances":[row("n"+str(i),text,5,9)]} for i in range(2)]
    targets=plan_recovery(r,nem)
    pair=[obs("challenge-0-0",5,11,[row("a",text,5,8.9)]),
          obs("challenge-0-1",0,14,[row("b",text,5.1,8.8)])]
    rows,decisions=apply_independent(r,targets,nem,pair)
    assert len(decisions)==1 and rows[1]["source_start"]==5.1
    assert decisions[0]["edge_observation_used_as_corroboration"]
    assert rows[1]["timestamp_provenance"]["timing_source_audio_sha256"]=="challenge-0-1"
    # Reversing edge/interior positions still selects the interior timing.
    swapped=[{**pair[1],"id":"challenge-0-0"},{**pair[0],"id":"challenge-0-1"}]
    rows,decisions=apply_independent(r,targets,nem,swapped)
    assert len(decisions)==1 and rows[1]["source_start"]==5.1
    both_edge=[pair[0],{**pair[1],"crop":[5.1,14]}]
    assert apply_independent(r,targets,nem,both_edge)==(r["utterances"],[])
    conflict=[pair[0],{**pair[1],"rows":[row("b","I told you not to leave",5.1,8.8)]}]
    assert apply_independent(r,targets,nem,conflict)==(r["utterances"],[])


def test_low_information_residual_does_not_block_supported_run():
    original=[row("left","你好",1,2),
              row("a","从古城出发然后回到家乡",None,None,1),
              row("b","最后成功完成任务",None,None,1),
              row("fragment","最",None,None,1),
              row("right","再见",20,21)]
    r=rec(original,"zh")
    segs=[row("p","从古城出发然后回到家乡",4,9),
          row("q","最后成功完成任务",10,15)]
    pair=[obs("challenge-0-0",0,22,segs),
          obs("challenge-0-1",0,24,[{**x,"observation_id":"other-"+x["id"]} for x in segs])]
    rows,decisions=apply_independent(r,plan_recovery(r,[]),[],pair)
    assert len(decisions)==1
    assert decisions[0]["residual_untimed_source_ids"]==["fragment"]
    assert [x["source_start"] for x in rows]==[1,4,10,None,20]
    assert rows[3]=={**original[3],"source_quality_status":"unresolved_fragment_timing"}
    assert original[1]["source_start"] is None
    original[3]["source_text"]="一段完全不同的实质话语"
    assert apply_independent(r,plan_recovery(r,[]),[],pair)==(original,[])
