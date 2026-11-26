"""Successor occurrence controls; no private episode fixture."""
from copy import deepcopy

from nvidia_transcribe.targeted_occurrences import (
    adjudicate_sequences, displaced_sequences, sequence_id, recover_unlocalized_events,
)
from nvidia_transcribe.semantic_context import current_row_context
from nvidia_transcribe.recovery_planner import plan_recovery_windows
from nvidia_transcribe.source_quality import SourceQualityPolicy


def row(uid,text,a,b,chunk,units=()):
    return {"id":uid,"utterance_id":uid,"source_text":text,
            "source_start":a,"source_end":b,"chunk_indices":[chunk],
            "source_unit_ids":list(units),"observation_id":uid,"speaker_id":"SPK_A"}


def observation(uid,crop,rows,key,task):
    return {"id":uid,"audio_sha256":("%064x"%sum(map(ord,uid))),
            "crop":crop,key:rows,"task_ids":[task],
            "integrity":{"status":"complete","finish_reason":"stop"}}


def fixture():
    words="the captain found our secret hidden passage"
    parents=[{"id":"a","chunk_index":0,"infer_start_s":0,"infer_end_s":25,
              "core_start_s":0,"core_end_s":20},
             {"id":"b","chunk_index":1,"infer_start_s":15,"infer_end_s":65,
              "core_start_s":20,"core_end_s":60}]
    units=[]
    for i,start in enumerate((16,35)):
        for k,word in enumerate(words.split()):
            units.append({"id":f"{i}-{k}","asr_observation_id":parents[i]["id"],
                "raw_recognized_token_ids":[f"raw{i}-{k}"],"chunk_index":i,
                "source_ordinal":k,"recognized_turn_id":f"turn{i}","text":word,
                "ctc_start_s":start+k*.4,"ctc_end_s":start+k*.4+.25,
                "boundary_disposition":"selected_provisional"})
    ledger={"asr_observations":parents,"lexical_observations":units,
            "recognized_occurrence_inventories":[{"asr_observation_id":p["id"],
                "projection_status":"exact"} for p in parents]}
    left=row("left",words,16,18.65,0,[u["id"] for u in units[:7]])
    right=row("right",words,35,37.65,1,[u["id"] for u in units[7:]])
    rec={"source":{"duration_s":70,"fingerprint_sha256":"a"*64},
         "source_language":"en","utterances":[left,right]}
    n=displaced_sequences(ledger)[0]
    uid=sequence_id(n)
    windows=[{"task_id":k,"evidence_ids":[uid],"reasons":[reason],"state":"observed"}
             for k,reason in (("early","sequence_origin"),("late","sequence_displacement"))]
    alt=row("other","we have never seen such a remarkable landscape",34.9,38,1)
    native=[observation("native-a",[14,21],[left],"utterances","early"),
            observation("native-b",[33,40],[alt],"utterances","late"),
            observation("native-arm",[20,65],[alt],"utterances","late")]
    native[-1]["context_removed_task_ids"]=["late"]
    native[-1]["raw_response"]={"choices":[{"message":{"content":alt["source_text"]}}]}
    windows[1]["context_removed_crop"]=[20,65]
    targets=[{"target_id":k,"task_ids":[k]} for k in ("early","late")]
    ind=[observation(k+"-"+str(i),crop,[r],"rows",k)
         for k,r,crops in (("early",left,([14,21],[13,22])),
                           ("late",alt,([33,40],[32,41])))
         for i,crop in enumerate(crops)]
    return rec,ledger,windows,native,targets,ind


def test_displacement_requires_positive_earlier_later_and_core_support():
    inputs=fixture()
    before=deepcopy(inputs)
    rows,decisions=adjudicate_sequences(*inputs)
    assert [r["id"] for r in rows]==["left"]
    assert rows[0]["prior_source_hypotheses"][0]["id"]=="right"
    assert decisions[0]["status"]=="retired_displaced_context_hypothesis"
    assert decisions[0]["acoustic_truth_accepted"] is False
    assert inputs==before
    # A real repeated line in an independently observed later crop vetoes
    # retirement even though the earlier exact lexical correspondence remains.
    inputs[-1][-1]["rows"]=[inputs[0]["utterances"][1]]
    rows,decisions=adjudicate_sequences(*inputs)
    assert len(rows)==2 and decisions[0]["status"].startswith("unresolved")


def test_displacement_without_core_arm_or_positive_competing_speech_is_retained():
    inputs=fixture()
    inputs[3].pop()
    assert len(adjudicate_sequences(*inputs)[0])==2
    inputs=fixture()
    inputs[3][1]["utterances"]=[]
    assert len(adjudicate_sequences(*inputs)[0])==2


def test_scheduler_preserves_complete_sequence_as_shared_bounded_tasks():
    rec,ledger,*_=fixture()
    windows,_=plan_recovery_windows(rec,ledger,{"cases":[]},None,SourceQualityPolicy())
    pair=[w for w in windows if w["reason"].startswith("sequence_")]
    assert len(pair)==2
    assert next(w for w in pair if w["reason"]=="sequence_displacement")["context_removed_crop"]==(20.,65.)
    assert set(pair[0]["source_ids"]+pair[1]["source_ids"])=={"left","right"}


def test_broad_event_recovery_preserves_unknown_voice_and_original_lineage():
    before=row("before","The messenger has arrived",10,12,0)
    first=row("first","we must protect the hidden archive",None,None,1,["u1"])
    second=row("second","before the enemy reaches the city",None,None,1,["u2"])
    after=row("after","Where is the key",20,22,1)
    text=first["source_text"]+" "+second["source_text"]
    rec={"source":{"duration_s":60},"source_language":"en",
         "utterances":[before,first,second,after]}
    native=[observation("n"+str(i),crop,[row("nrow"+str(i),text,13,19,1)],
                        "utterances","run") for i,crop in enumerate(([8,24],[7,25]))]
    ind=[observation("run-"+str(i),crop,[row("irow"+str(i),text,12.8,19.1,1)],
                     "rows","run") for i,crop in enumerate(([8,24],[7,25]))]
    window={"task_id":"run","state":"observed","reasons":["untimed_event_context"],
            "window":[8,24],"source_ids":["first","second"],"evidence_ids":["run"]}
    targets=[{"target_id":"run","task_ids":["run"]}]
    rows,decisions=recover_unlocalized_events(rec,[window],native,targets,ind)
    assert len(decisions)==1 and len(rows)==3
    recovered=rows[1]
    assert recovered["source_start"]==13 and recovered["source_end"]==19
    assert recovered["speaker_identity_accepted"] is False
    assert recovered["speaker_id"]=="SPK_UNK" and not recovered["source_words"]
    assert {r["id"] for r in recovered["prior_source_hypotheses"]}=={"first","second"}
    assert current_row_context(recovered)["entity_identity_accepted"] is False
    assert current_row_context({**recovered,"source_text":"different"}) is None
    # Unrelated overlapping timed dialogue is not consumed.
    rec["utterances"].insert(3,row("overlap","Unrelated interjection",15,16,2))
    assert not recover_unlocalized_events(rec,[window],native,targets,ind)[1]


def test_core_contrast_is_generation_evidence_not_ctc_locality():
    inputs=fixture()
    inputs[3][-1]["utterances"]=[]  # rejected core CTC projection
    assert len(adjudicate_sequences(*inputs)[0])==1
    inputs[3][-1]["raw_response"]["choices"][0]["message"]["content"]=inputs[0]["utterances"][0]["source_text"]
    assert len(adjudicate_sequences(*inputs)[0])==2


def test_incomplete_later_native_coverage_is_not_a_repeat_veto():
    inputs=fixture()
    rec,ledger,windows,native,targets,ind=inputs
    partial=observation("native-partial",[32,41],[],"utterances","late")
    partial["raw_response"]={"choices":[{"message":{"content":"unrelated competing words"}}]}
    native.append(partial)
    rows,decisions=adjudicate_sequences(*inputs)
    assert len(rows)==1
    assert any(x["status"]=="insufficient_coverage" for x in decisions[0]["later_observation_dispositions"])
    partial["raw_response"]["choices"][0]["message"]["content"]=rec["utterances"][1]["source_text"]
    rows,decisions=adjudicate_sequences(*inputs)
    assert len(rows)==2
    assert any(x["status"]=="repeat_veto" for x in decisions[0]["later_observation_dispositions"])


def test_corroborated_cross_chunk_prefix_variant_requires_two_families():
    from nvidia_transcribe.targeted_occurrences import _corroborated_prefix_variant
    from nvidia_transcribe.source_quality import _lex
    before=row("prior","Le guide",10,12,0)
    before["recognized_turn_ids"]=["previous-turn"]
    prefix=row("prefix","Cet Alaric, il",13.05,14.30,0)
    prefix["recognized_turn_ids"]=["previous-turn"]
    origins=[row("untimed","Il cherchait la carte du royaume",None,None,1)]
    ns=[]
    for i,name in enumerate(("Alarik","Alaric")):
        r=row("n"+str(i),f"C'est {name}. Il cherchait la carte du royaume",13,18,1)
        r["timed_lexical_units"]=[{"lex":token,"start":a,"end":b}
            for token,a,b in (("cest",13,13.3),(name.lower(),13.3,14.1),("il",14.1,14.3))]
        o=observation("native"+str(i),[8-i,22+i],[r],"utterances","run")
        ns.append((o,([r],(13,18),_lex(r["source_text"]))))
    ins=[]
    for i in range(2):
        first=row("i"+str(i)+"-0","Le guide, c'est Alaric",11.1,14.0,1)
        second=row("i"+str(i)+"-1","Il cherchait la carte du royaume",14.0,18,1)
        o=observation("ind"+str(i),[8-i,22+i],[first,second],"rows","run")
        # Whole-event matching can pick only the second segment.
        ins.append((o,([second],(14,18),_lex(second["source_text"]))))
    receipt=_corroborated_prefix_variant(prefix,before,ns,ins,origins,(13,18))
    assert receipt and not receipt["entity_identity_accepted"] and not receipt["speaker_identity_accepted"]
    assert _corroborated_prefix_variant(prefix,before,ns,ins[:1],origins,(13,18)) is None
    # A second broad event's name can differ substantially while a separate
    # previously acquired local crop corroborates the actual timed prefix.
    broad=ns[1][1][0][0]
    broad["timed_lexical_units"][1]["lex"]="alexander"
    assert _corroborated_prefix_variant(prefix,before,ns,ins,origins,(13,18)) is None
    local=row("local","Cet Alaric, il cherchait",13.02,15,0)
    local["timed_lexical_units"]=[{"lex":w,"start":a,"end":b}
        for w,a,b in (("cet",13.02,13.3),("alaric",13.3,14.1),("il",14.1,14.3))]
    aux=observation("aux-local",[11,16],[local],"utterances","other-task")
    recovered=_corroborated_prefix_variant(prefix,before,ns,ins,origins,(13,18),[aux])
    assert recovered and "local" in recovered["native_observation_ids"]
    prefix["recognized_turn_ids"]=["unrelated-turn"]
    assert _corroborated_prefix_variant(prefix,before,ns,ins,origins,(13,18)) is None
