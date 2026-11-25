"""Shared recovery planning, acoustic provenance and canonical occurrence controls."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest

from nvidia_transcribe.acoustic_observations import reconstruct_acoustic_observation
from nvidia_transcribe.occurrence_resolution import (
    cross_model_events, resolve_occurrences, resolution_receipts,
)
from nvidia_transcribe.recovery_planner import plan_recovery_windows, schedule_local_alignment
from nvidia_transcribe.selective_alignment import (
    AdmissionPolicy, RecoveryCandidate, apply_selective_recovery,
)
from nvidia_transcribe.source_quality import (
    SourceQualityPolicy, acquire_acoustic_evidence, select_source_events,
)
from nvidia_transcribe.source_recovery import schedule_independent
from nvidia_transcribe.translation import _apply_safe_adjacent_regrouping


def row(uid,text,a,b,span=None):
    r={"id":uid,"source_text":text,"source_start":a,"source_end":b,
       "speaker_id":"SPK_A","chunk_indices":[0]}
    if span:
        r.update(source_unit_ids=[uid+"-unit"],recognized_turn_ids=["turn"],
                 occurrence_span_ids=[span])
    return r


def recording(rows=(),duration=100.):
    return {"source":{"fingerprint_sha256":"media","duration_s":duration},
            "source_language":"fr","utterances":list(rows)}


def span(uid,bounds,status="disputed",context=False):
    return {"id":uid,"source_unit_ids":[uid+"-unit"],"locality_status":status,
            "candidate_bounds_s":bounds,"selection_dispositions":[
                {"source_unit_id":uid+"-unit","status":"context_candidate_unresolved" if context else "selected_unlocalized"}]}


def inventory(spans,findings=()):
    return {"asr_observation_id":"asr","projection_status":"exact",
            "native_ctc_observations":[],
            "locality":{"spans":list(spans),"findings":list(findings)}}


def ledger(spans,findings=(),duration=100.):
    return {"asr_observations":[{"id":"asr","chunk_index":0,"infer_start_s":0.,"infer_end_s":duration}],
            "recognized_occurrence_inventories":[inventory(spans,findings)]}


def event(text,a,b,ids=("n1","n2"),**kwargs):
    return {"source_text":text,"source_start":a,"source_end":b,
            "chunks":[0],"supporting_observation_ids":list(ids),
            "audio_sha256":["a","b"],"raw_response_sha256":["raw1","raw2"],**kwargs}


def test_disputed_late_occurrence_gets_admission_before_early_backlog():
    spans=[span("early"+str(i),[i*6.,i*6.+4.]) for i in range(4)]
    spans.append(span("late",[94.,96.]))
    w,d=plan_recovery_windows(recording(),ledger(spans),{"cases":[]},None,
                              SourceQualityPolicy(max_windows=2))
    assert len(w)==2 and any("late" in x["evidence_ids"] for x in w)
    assert {uid for x in w+d for uid in x["evidence_ids"]}=={x["id"] for x in spans}
    assert all(x["state"]=="deferred" for x in d)
    assert all(x["window"][1]-x["window"][0]<=14.000001 for x in w+d)


def test_short_brackets_and_competing_hypotheses_are_not_fabricated_times():
    rows=[row("left","before",1.,2.),row("missing","unknown wording",None,None),
          row("right","after",2.013,4.)]
    findings=[{"id":"conflict","reason":"intra_turn_order_conflict",
               "source_unit_ids":["u"],"candidate_windows_s":[[30.,31.],[85.,86.]]}]
    rec=recording(rows)
    before=deepcopy(rec)
    w,d=plan_recovery_windows(rec,ledger([],findings),{},None,SourceQualityPolicy(max_windows=6))
    assert rec==before
    assert all(x["window"][1]-x["window"][0]>=3.-1e-6 for x in w+d if x["window"])
    conflict=[x["window"] for x in w+d if "conflict" in x["evidence_ids"]]
    assert len(conflict)==2 and conflict[0][1]<conflict[1][0]
    assert any("missing" in x["source_ids"] for x in w+d)


def test_coalesced_context_and_dispute_keep_all_unit_references():
    w,d=plan_recovery_windows(recording(),ledger([
        span("a",[10.,13.]),span("b",[11.,14.],"provisional",True)]),{},None,SourceQualityPolicy())
    assert len(w)==1 and not d
    assert w[0]["evidence_ids"]==["a","b"]
    assert w[0]["source_unit_ids"]==["a-unit","b-unit"]
    assert w[0]["reasons"]==["disputed_occurrence","unselected_context"]


def test_semantic_request_requires_real_row_and_locality():
    rec=recording([row("known","Bonjour",10.,11.)])
    requests={"occurrence_requests":[
        {"case_id":"unknown","source_ids":["invented"],"context_window_s":[10.,11.]},
        {"case_id":"wrong_time","source_ids":["known"],"context_window_s":[90.,95.]},
        {"case_id":"supported","source_ids":["known"],"context_window_s":[9.,12.]}]}
    w,d=plan_recovery_windows(rec,{},requests,None,SourceQualityPolicy())
    assert [x["evidence_ids"] for x in w]==[["supported"]]
    assert {x["reason"] for x in d}=={"invalid_semantic_reference","invalid_semantic_window"}


def test_unlocalized_without_any_real_window_stays_visible():
    w,d=plan_recovery_windows(recording([row("untimed","unknown",None,None)]),{}, {},None,SourceQualityPolicy())
    assert not w and len(d)==1
    assert d[0]["state"]=="unlocalized" and d[0]["source_ids"]==["untimed"]


def test_primary_pairs_precede_extras_and_reuse_equal_crops(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from nvidia_transcribe import source_quality as quality
    calls=[]
    def observe(client,media,crop,root,index):
        calls.append(crop)
        return {"crop":crop,"audio_sha256":str(index),"raw_response_sha256":str(index),"utterances":[]}
    monkeypatch.setattr(quality,"_observe",observe)
    windows=[{"task_id":uid,"window":bounds,"evidence_ids":[uid],"chunks":[0],"reason":"disputed_occurrence"}
             for uid,bounds in [("a",(10.,13.)),("b",(10.,13.)),("c",(70.,73.))]]
    observations,events,unresolved,metrics=acquire_acoustic_evidence(
        recording(),SimpleNamespace(duration_s=100.),windows,asr_scope=lambda:nullcontext(object()),
        output_dir=tmp_path,policy=SourceQualityPolicy(max_extra_contexts=1,max_acoustic_audio_s=45.))
    assert calls[:4]==[(8.,18.),(5.,15.),(68.,78.),(65.,75.)]
    assert metrics["reused_requests"]==2 and metrics["requests"]==len(calls)
    assert metrics["audio_seconds"]<=45.
    assert all(w["state"]=="observed" for w in windows)
    assert set(observations[0]["task_ids"])=={"a","b"}
    assert not events and len(unresolved)==3


def test_independent_queue_uses_only_observed_tasks_and_explicit_budget():
    windows=[{"task_id":uid,"window":bounds,"evidence_ids":[uid],"chunks":[0],"priority":0,"state":state}
             for uid,bounds,state in [("a",(5.,10.),"observed"),("b",(15.,20.),"deferred"),
                                      ("c",(85.,90.),"observed")]]
    admitted,deferred=schedule_independent(recording(),[],windows=windows,
                                         policy=SourceQualityPolicy(max_independent_targets=1))
    assert admitted[0]["task_ids"]==["a"]
    assert deferred[0]["task_ids"]==["c"] and deferred[0]["defer_reason"]=="independent_target_budget"


def test_localized_native_reconstruction_rejects_compression_but_preserves_words():
    words=["marvellous","mountains","ancient","gardens","extraordinary","journeys",
           "familiar","travelers","captivating","beautiful","children"]
    raw={"choices":[{"message":{"content":"<spk:0> "+" ".join(words)},"finish_reason":"stop"}],
         "ctc_timestamps":{"words":[{"word":w,"speaker":"0","start":5+i*.105,
                                    "end":5+i*.105+.2} for i,w in enumerate(words)]}}
    rows,inv=reconstruct_acoustic_observation(raw,audio_sha256="a",crop=(100.,120.),language="fr")
    assert any(s["locality_status"]=="disputed" for s in inv["locality"]["spans"])
    assert all(r["source_start"] is None for r in rows)
    assert {uid for r in rows for uid in r["source_unit_ids"]}==set(inv["working_source_unit_ids"])
    again,_=reconstruct_acoustic_observation(raw,audio_sha256="a",crop=(100.,120.),language="fr")
    assert rows==again


def test_cross_model_positive_support_relocates_only_explicit_disputed_origin():
    text="Je ne sais pas"
    original=row("old",text,10.,11.,"span")
    original["source_words"]=[{"text":"Je","valid_timing":True,"locality_status":"disputed"},
                             {"text":"ne","valid_timing":False,"locality_status":"disputed"}]
    target={"target_id":"t","task_ids":["task"],"evidence_ids":["span"],"chunks":[0],
            "kind":"occurrence","ids":["old"],"bounds":(40.,46.)}
    primary={"audio_sha256":"native","raw_response_sha256":"raw","crop":[35.,50.],
             "task_ids":["task"],"chunks":[0],"utterances":[
                 {**row("n",text,41.,43.),"observation_id":"n","local_voice":"spk0"}]}
    independent=[{"id":"t-0","crop":[38.,48.],"audio_sha256":"w1","rows":[
                    {**row("w1",text,41.1,43.1),"observation_id":"w1"}]},
                 {"id":"t-1","crop":[37.,49.],"audio_sha256":"w2","rows":[
                    {**row("w2",text,41.2,43.2),"observation_id":"w2"}]}]
    report={"observations":independent,"status":"observed","receipt_sha256":"receipt"}
    rec=recording([original])
    before=deepcopy(rec)
    rows,ds=resolve_occurrences(rec,[],[primary],[target],report,SourceQualityPolicy())
    assert rec==before and len(rows)==1
    assert rows[0]["id"]=="old" and rows[0]["source_start"]==41.
    assert rows[0]["source_unit_ids"]==original["source_unit_ids"]
    assert rows[0]["prior_source_hypothesis"]==original
    assert ds[0]["status"]=="relocated_disputed_occurrence"
    assert set(ds[0]["independent_observation_ids"])=={"w1","w2"}
    # Same text and new acoustic location cannot retire a valid native row.
    clean={**original,"source_words":[{"text":"Je","valid_timing":True,"locality_status":"provisional"}]}
    rows,ds=resolve_occurrences(recording([clean]),[],[primary],[target],deepcopy(report),SourceQualityPolicy())
    assert rows==[clean] and ds[0]["status"].startswith("unresolved")
    # Different same-text original occurrence is not an identity proof.
    duplicate=row("repeat",text,60.,61.,"other")
    rows,ds=resolve_occurrences(recording([original,duplicate]),[],[primary],[target],deepcopy(report),SourceQualityPolicy())
    assert rows==[original,duplicate] and ds[0]["status"].startswith("unresolved")


def test_competing_event_support_never_reuses_same_observation_as_two_events():
    events=[event("Je ne sais pas",10.,12.,("shared","other")),
            event("Je sais",10.,12.,("shared","third"))]
    rows,ds=resolve_occurrences(recording(),events,[],[],{"status":"no_targets","observations":[]},SourceQualityPolicy())
    assert not rows and ds[0]["status"]=="unresolved_conflicting_observation_correspondence"


def test_one_untimed_origin_and_two_positive_repeats_remain_unresolved():
    original=row("old","Come back",None,None,"span")
    rows,ds=select_source_events(recording([original]),[
        event("Come back",4.,5.),event("Come back",7.,8.,("n3","n4"))],SourceQualityPolicy())
    assert rows==[original]
    assert all(d["status"].startswith("unresolved") for d in ds)


def test_context_receipt_needs_unique_exact_supported_occurrence():
    sp=span("ctx",[10.,12.],"provisional",True)
    led=ledger([sp])
    led["lexical_observations"]=[{"id":"ctx-unit","text":"Bonjour mon ami"}]
    rows=[row("new","Bonjour mon ami",10.1,12.1)]
    ds=[{"source_id":"new","status":"recovered_corroborated_turn","trigger_evidence_ids":["ctx"]}]
    receipts=resolution_receipts(recording(),led,rows,ds,SourceQualityPolicy())
    assert receipts[0]["status"]=="represented_provisional"
    assert receipts[0]["canonical_source_ids"]==["new"]
    rows[0]["source_text"]="Unrelated speech"
    assert resolution_receipts(recording(),led,rows,ds,SourceQualityPolicy())[0]["status"]=="unresolved"


def test_translation_does_not_collapse_distinct_occurrences_in_one_turn():
    first={**row("a","Reviens",10.,11.,"span_a"),"translated_text":"Come back!"}
    second={**row("b","Reviens",12.,13.,"span_b"),"translated_text":"Come back!","_grouped_with_prev":True}
    rows=[first,second]
    assert _apply_safe_adjacent_regrouping(rows)==0
    assert [r["translated_text"] for r in rows]==["Come back!","Come back!"]
    assert not rows[1].get("merged_into")


@pytest.mark.parametrize("mode",["ordered","wrong_order","zero_duration"])
def test_qwen_admission_and_lexical_order_negative_controls(mode):
    texts=["bonjour","mon","ami"]
    native={"units":[{"text":t,"speaker":"0","start":None,"end":None,"valid_timing":False} for t in texts],
            "diarization":[{"speaker":"0","start":1.,"end":4.,"valid_timing":True}]}
    class Aligner:
        def align(self,**kwargs):
            words=texts if mode!="wrong_order" else list(reversed(texts))
            return {"units":[{"text":t,"start":1.+i*.5,
                              "end":1.+i*.5+(0. if mode=="zero_duration" else .3),
                              "valid_timing":True} for i,t in enumerate(words)],
                    "diagnostic":{"status":"ok"}}
    result=apply_selective_recovery(native=native,raw_text="<spk:0>bonjour mon ami",
        chunk_start_s=0.,chunk_end_s=6.,chunk_index=0,source_language="fr",aligner=Aligner(),
        extract_wav=lambda a,b:Path("synthetic.wav"),policy=AdmissionPolicy())
    assert [u["text"] for u in result["units"]]==texts
    assert sum(u["valid_timing"] for u in result["units"])==(3 if mode=="ordered" else 0)
    if mode!="ordered":
        assert result["diagnostic"]["external_aligner"]["localized_spans"][0]["rejection_reason"]


def test_global_alignment_budget_retains_late_candidate_and_defers_others():
    plans=[]
    for offset in (0.,80.):
        candidates=[RecoveryCandidate(i,i+1,"word",offset+i,offset+i+1,1) for i in range(3)]
        plans.append((candidates,[{"span_range":[c.lo,c.hi],"status":"admitted"} for c in candidates]))
    admitted,receipt=schedule_local_alignment(plans,max_candidates=2,max_audio_s=5.)
    assert [len(x[0]) for x in admitted]==[1,1]
    assert receipt["admitted_candidates"]==2
    assert sum(d["status"]=="skipped_recording_alignment_budget" for _,ds in admitted for d in ds)==4


def test_independent_worker_admission_binds_receipts_and_cpu_device(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from nvidia_transcribe.model_resources import ModelResource
    from nvidia_transcribe import source_recovery as recovery
    python=tmp_path/"python"
    python.touch()
    checkpoint=tmp_path/"checkpoint"
    checkpoint.mkdir()
    resource=ModelResource("test","whisper_cpu",checkpoint_path=str(checkpoint),
                           worker_python=str(python),device="cpu")
    monkeypatch.setattr(recovery,"extract_wav_segment",lambda *a,**kw:None)
    monkeypatch.setattr(recovery,"validate_wav_file",lambda p:(10.,p.stem))
    target={"target_id":"target","crops":[(0.,10.),(2.,12.)]}
    bad=False
    def run(cmd,**kw):
        req=json.loads(kw["input"])
        assert kw["env"]["CUDA_VISIBLE_DEVICES"]=="" and kw["env"]["HF_HUB_OFFLINE"]=="1"
        assert 0<kw["timeout"]<=900
        result={"device":"cpu","backend":"faster-whisper","compute_type":"int8",
                "observations":[{"id":c["id"],"audio_sha256":c["audio_sha256"] if not bad else "wrong",
                "crop":[c["start"],c["end"]],"rows":[]} for c in req["crops"]]}
        return SimpleNamespace(returncode=0,stdout=json.dumps(result),stderr="")
    monkeypatch.setattr(recovery.subprocess,"run",run)
    media=SimpleNamespace(path="media",selected_audio_stream=0,duration_s=100.)
    kwargs=dict(challenger_resource=resource,output_dir=tmp_path,policy=SourceQualityPolicy())
    report=recovery.acquire_independent_evidence(recording(),[target],media,**kwargs)
    assert report["status"]=="observed" and report["requests"]==2
    assert report["resource"]["resource_fingerprint"]==resource.fingerprint()
    bad=True
    report=recovery.acquire_independent_evidence(recording(),[target],media,**kwargs)
    assert report["status"]=="error" and not report["observations"]


def test_bound_untimed_target_is_prioritized_over_earlier_parent_scans():
    # Mirrors the retained boundary failure: an explicit bracket appears
    # only in the third observed task, after two broad parent scans.
    rec=recording([row("first","Come back",91.01,91.535),
                   row("missing","Come back",None,None,"missing"),
                   row("next","Look",93.57,93.999)],110.)
    windows=[{"task_id":uid,"window":bounds,"source_ids":["missing"],
              "evidence_ids":[uid],"chunks":[0],"priority":1,"state":"observed"}
             for uid,bounds in [("broad_a",(79.25,90.)),
                               ("broad_b",(99.215,110.)),
                               ("specific",(88.5195,100.))]]
    admitted,deferred=schedule_independent(rec,[],windows=windows,
        policy=SourceQualityPolicy(max_independent_targets=2,max_independent_audio_s=100.))
    assert admitted[0]["kind"]=="untimed"
    assert admitted[0]["bounds"]==(91.535,93.57)
    assert admitted[0]["ids"]==["missing"]
    assert admitted[0]["task_ids"]==["specific"]
    assert len(admitted)==2 and len(deferred)==1
    assert rec["utterances"][1]["source_start"] is None


def test_planner_uses_bracket_and_local_native_hypotheses_before_blind_scans():
    rows=[row("left","Hello",91.,91.535),
          row("missing","Come back",None,None,"missing"),
          row("right","Look",93.57,94.)]
    led=ledger([span("missing",None,"unlocalized"),span("other",None,"unlocalized")],duration=110.)
    led["asr_observations"][0].update(infer_start_s=80.,infer_end_s=110.)
    led["recognized_occurrence_inventories"][0]["native_ctc_observations"]=[
        {"id":"native-local","timing":{"start":102.,"end":102.4,"valid_timing":True}},
        {"id":"native-unknown","timing":{"start":None,"end":None,"valid_timing":False}}]
    w,d=plan_recovery_windows(recording(rows,110.),led,{},None,
                              SourceQualityPolicy(max_windows=2))
    assert "missing" in w[0]["evidence_ids"]
    assert w[0]["window"][0]<=91.535<93.57<=w[0]["window"][1]
    assert w[0]["window"][1]-w[0]["window"][0]<5.
    assert "native-local" in w[1]["evidence_ids"]
    assert w[1]["window"][0]<=102.<102.4<=w[1]["window"][1]
    assert all(x["priority"]>=3 for x in d)
    assert {"other","native-unknown"}<={uid for x in d for uid in x["evidence_ids"]}


def test_cross_model_event_support_does_not_claim_missing_word_time():
    sp=span("part",None,"unlocalized")
    rec=recording([row("part","On peut descendre par là",100.2,101.23,"part")],110.)
    before=deepcopy(rec)
    ds=[{"source_id":"part","status":"already_represented","window":[100.135,101.533],
         "observations":["n1","n2","w1","w2"],"independent_observation_ids":["w1","w2"]}]
    receipts=resolution_receipts(rec,ledger([sp],duration=110.),rec["utterances"],ds,SourceQualityPolicy())
    assert receipts[0]["status"]=="event_supported_word_timing_unresolved"
    assert receipts[0]["canonical_source_ids"]==["part"]
    assert receipts[0]["supporting_observation_ids"]==["n1","n2","w1","w2"]
    assert rec==before and not receipts[0]["acoustic_truth_accepted"]
    ds[0]["independent_observation_ids"]=[]
    assert resolution_receipts(rec,ledger([sp],duration=110.),rec["utterances"],ds,SourceQualityPolicy())[0]["status"]=="unresolved"
