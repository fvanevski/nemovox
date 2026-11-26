"""Whole-primary CTC failure does not erase independently localized speech."""
from copy import deepcopy
from nvidia_transcribe.failed_ctc_recovery import recover_failed_ctc_events, failed_ctc_parents
from nvidia_transcribe.recovery_planner import plan_recovery_windows
from nvidia_transcribe.source_quality import SourceQualityPolicy
from nvidia_transcribe.source_recovery import schedule_independent


def fixture():
    words="we must protect the hidden archive before sunset".split()
    units=[{"id":f"u{i}","asr_observation_id":"primary","chunk_index":0,
            "source_ordinal":i,"text":w,"recognized_turn_id":"turn",
            "timing_reason":"alignment_failed","ctc_association_status":"native_error",
            "raw_recognized_token_ids":[f"token{i}"],"ctc_start_s":None,"ctc_end_s":None,
            "boundary_disposition":"selected_unlocalized"} for i,w in enumerate(words)]
    parent={"id":"primary","chunk_index":0,"infer_start_s":0,"infer_end_s":40,
            "core_start_s":0,"core_end_s":40}
    ledger={"asr_observations":[parent],"lexical_observations":units,
        "recognized_occurrence_inventories":[{"asr_observation_id":"primary","projection_status":"exact"}]}
    row={"id":"original","source_text":" ".join(words),"source_start":None,"source_end":None,
         "source_words":[{"text":w,"unit_type":"word","source_unit_id":f"u{i}","valid_timing":False,
                          "occurrence_span_id":"span"} for i,w in enumerate(words)],
         "source_characters":[],"source_unit_ids":[u["id"] for u in units],
         "recognized_turn_ids":["turn"],"occurrence_span_ids":["span"],"chunk_indices":[0],
         "speaker_id":"SPK_00","event_refs":[]}
    rec={"source":{"duration_s":40,"fingerprint_sha256":"media"},"source_language":"en",
         "utterances":[row]}
    windows=[{"task_id":"task","window":[10,13],"chunks":[0],"state":"observed",
              "evidence_ids":["primary"],"source_ids":["original"],"reasons":["failed_ctc_core"]}]
    target={"target_id":"test","task_ids":["task"],"chunks":[0],"bounds":[10,13],
            "evidence_ids":["primary"]}
    native=[{"task_ids":["task"],"crop":[7,16],"audio_sha256":"native",
             "raw_response_sha256":"raw","integrity":{"status":"complete","finish_reason":"stop"},
             "raw_response":{"choices":[{"message":{"content":"<spk:0> "+" ".join(words)}}]}}]
    ind=[{"id":f"test-{i}","audio_sha256":f"ind{i}","crop":[8-i,15+i],
          "rows":[{"observation_id":f"obs{i}","source_text":" ".join(words[:6]),
                   "source_start":10-.1*i,"source_end":13-.1*i}]} for i in range(2)]
    return rec,ledger,windows,native,[target],ind,SourceQualityPolicy()


def test_independent_event_recovers_exact_subspan_without_neighbor_or_fabricated_word_times():
    args=fixture()
    original=deepcopy(args[0])
    rows,decisions=recover_failed_ctc_events(*args)
    assert args[0]==original
    assert len(rows)==2 and len(decisions)==1
    assert rows[0]["source_text"]=="we must protect the hidden archive"
    assert rows[0]["source_start"]==9.9 and rows[0]["source_end"]==12.9
    assert rows[0]["speaker_id"]=="SPK_UNK" and not rows[0]["source_words"]
    assert rows[0]["timestamp_provenance"]["word_timing_verified"] is False
    assert rows[1]["source_text"]=="before sunset" and rows[1]["source_start"] is None
    assert set(rows[0]["source_unit_ids"]).isdisjoint(rows[1]["source_unit_ids"])
    assert set(rows[0]["source_unit_ids"]+rows[1]["source_unit_ids"])==set(original["utterances"][0]["source_unit_ids"])


def test_failed_ctc_recovery_rejects_repeats_wrong_core_and_missing_positive_native_support():
    args=fixture()
    repeat={**args[0]["utterances"][0],"id":"repeat","speaker_id":"SPK_01"}
    args[0]["utterances"].append(repeat)
    assert not recover_failed_ctc_events(*args)[1]
    args=fixture()
    args[1]["asr_observations"][0]["core_end_s"]=9
    assert not recover_failed_ctc_events(*args)[1]
    args=fixture()
    args[3][0]["raw_response"]["choices"][0]["message"]["content"]="unrelated dialogue"
    assert not recover_failed_ctc_events(*args)[1]
    args=fixture()
    args[5][1]["audio_sha256"]=args[5][0]["audio_sha256"]
    assert not recover_failed_ctc_events(*args)[1]


def test_failed_primary_core_coverage_is_scheduled_once_under_existing_budget():
    rec,ledger,*_=fixture()
    assert list(failed_ctc_parents(ledger))==[0]
    policy=SourceQualityPolicy()
    admitted,deferred=plan_recovery_windows(rec,ledger,{"cases":[]},None,policy)
    core=[w for w in admitted if w["reason"]=="failed_ctc_core"]
    assert [w["window"] for w in core]==[(0.,20.),(20.,40.)]
    assert not deferred
    for w in core:
        w["state"]="observed"
    targets,deferred=schedule_independent(rec,[],windows=core,policy=policy)
    assert len(targets)==2 and all(t["kind"]=="failed_ctc_core" for t in targets)
    assert sum(b-a for t in targets for a,b in t["crops"])<=policy.max_independent_audio_s


def test_failed_ctc_recovery_rejects_raw_repeat_absent_from_canonical_rows():
    args=fixture()
    ledger=args[1]
    for i,u in enumerate(list(ledger["lexical_observations"])[:6]):
        ledger["lexical_observations"].append({**u,"id":f"repeat{i}",
            "recognized_turn_id":"repeat-turn","source_ordinal":8+i,
            "raw_recognized_token_ids":[f"repeat-token{i}"]})
    assert not recover_failed_ctc_events(*args)[1]


def _timed_segments(words,groups,offset=10.):
    result=[]
    for i,(lo,hi) in enumerate(groups):
        items=[{"text":words[k],"start":offset+k*.4,"end":offset+(k+1)*.4} for k in range(lo,hi)]
        result.append({"observation_id":f"seg{i}","source_text":" ".join(words[lo:hi]),
                       "source_start":items[0]["start"],"source_end":items[-1]["end"],"words":items})
    return result


def test_ordered_event_anchors_accept_internal_wording_and_segmentation_variants():
    args=fixture()
    rec,ledger,windows,native,targets,ind,policy=args
    words="we must protect the secret archive before sunset".split()
    native[0]["raw_response"]["choices"][0]["message"]["content"]=" ".join(words)
    for i,groups in enumerate(([(0,4),(4,8)],[(0,2),(2,6),(6,8)])):
        ind[i]["rows"]=_timed_segments(words,groups,10.-.05*i)
        for r in ind[i]["rows"]:
            r["observation_id"]=ind[i]["id"]+":"+r["observation_id"]
    original=deepcopy(rec)
    rows,ds=recover_failed_ctc_events(*args)
    assert rec==original and len(rows)==len(ds)==1
    assert rows[0]["source_text"]==original["utterances"][0]["source_text"]
    assert rows[0]["wording_status"]=="model_variants_unresolved"
    assert rows[0]["source_unit_ids"]==original["utterances"][0]["source_unit_ids"]
    assert not rows[0]["source_words"] and not rows[0]["source_characters"]
    assert rows[0]["timestamp_provenance"]["word_timing_verified"] is False
    assert ds[0]["constituent_event_evidence"] and not ds[0]["wording_exact"]


def test_repeated_local_recognition_does_not_assign_one_origin_arbitrarily():
    args=fixture()
    for o in args[5]:
        repeat={**o["rows"][0],"observation_id":o["id"]+":repeat",
                "source_start":13.2,"source_end":14.9}
        o["rows"].append(repeat)
    # The two possible local realizations must not be assigned to one
    # primary row merely by the sequence matcher's first best match.
    assert not recover_failed_ctc_events(*args)[1]


def test_ambiguous_cross_chunk_origin_localizes_speech_without_consuming_hypotheses():
    args=fixture()
    rec,ledger,windows,native,targets,ind,policy=args
    previous=deepcopy(rec["utterances"][0])
    repeat=deepcopy(previous)
    repeat.update(id="second",chunk_indices=[1],recognized_turn_ids=["second-turn"])
    lookup={}
    for u in list(ledger["lexical_observations"]):
        new={**u,"id":"second-"+u["id"],"asr_observation_id":"second-parent","chunk_index":1,
             "recognized_turn_id":"second-turn","raw_recognized_token_ids":["second-"+u["id"]]}
        lookup[u["id"]]=new["id"]
        ledger["lexical_observations"].append(new)
    repeat["source_unit_ids"]=[lookup[x] for x in previous["source_unit_ids"]]
    for u in repeat["source_words"]:
        u["source_unit_id"]=lookup[u["source_unit_id"]]
    rec["utterances"].append(repeat)
    ledger["asr_observations"][0]["core_end_s"]=20
    ledger["asr_observations"].append({"id":"second-parent","chunk_index":1,"infer_start_s":10,
        "infer_end_s":40,"core_start_s":20,"core_end_s":40})
    ledger["recognized_occurrence_inventories"].append({"asr_observation_id":"second-parent","projection_status":"exact"})
    before=deepcopy(rec)
    rows,ds=recover_failed_ctc_events(*args)
    assert rec==before and len(rows)==3 and len(ds)==1
    assert rows[:2]==before["utterances"]
    event=rows[-1]
    assert event["source_start"]==9.9 and event["source_end"]==12.9
    assert not event["source_unit_ids"] and not event["recognized_turn_ids"]
    assert event["source_lineage"]["primary_origin_accepted"] is False
    assert set(ds[0]["candidate_origin_ids"])=={"original","second"}
    assert ds[0]["status"]=="localized_event_ambiguous_primary_origin"


def test_short_wholly_failed_recording_covers_tail_within_same_audio_and_target_budgets():
    args=fixture()
    rec,ledger,*_=args
    rec["source"]["duration_s"]=119.
    ledger["asr_observations"][0].update(infer_end_s=100.,core_end_s=90.)
    ledger["asr_observations"].append({"id":"later","chunk_index":1,"infer_start_s":80.,
        "infer_end_s":119.,"core_start_s":90.,"core_end_s":119.})
    ledger["recognized_occurrence_inventories"].append({"asr_observation_id":"later","projection_status":"exact"})
    for i,u in enumerate(list(ledger["lexical_observations"])):
        ledger["lexical_observations"].append({**u,"id":f"later{i}","chunk_index":1,
            "asr_observation_id":"later","recognized_turn_id":"later-turn","raw_recognized_token_ids":[f"later-token{i}"]})
    policy=SourceQualityPolicy()
    admitted,_=plan_recovery_windows(rec,ledger,{"cases":[]},None,policy)
    core=sorted((w for w in admitted if w["reason"]=="failed_ctc_core"),key=lambda w:w["window"])
    assert len(core)==6 and core[0]["window"][0]==0 and core[-1]["window"][1]==119
    assert all(x["window"][1]==y["window"][0] for x,y in zip(core,core[1:]))
    for w in core:
        w["state"]="observed"
    targets,deferred=schedule_independent(rec,[],windows=core,policy=policy)
    assert len(targets)==6 and not deferred
    assert all(len(t["crops"])==2 and t["crops"][0]!=t["crops"][1] for t in targets)
    assert sum(b-a for t in targets for a,b in t["crops"])<=policy.max_independent_audio_s


def test_independent_word_envelope_survives_coarser_segment_disagreement():
    from nvidia_transcribe.ordered_event_anchors import _atoms,_streams
    r={"source_start":73.46,"source_end":74.72,"source_text":"将于一月后",
       "observation_id":"a","words":[{"text":x,"start":a,"end":b} for x,a,b in (
           ("将",73.28,73.68),("于",73.68,73.94),("一",73.94,74.18),
           ("月",74.18,74.46),("后",74.46,74.72))]}
    atoms=_atoms(r,False)
    assert atoms and atoms[0][1]==73.28 and atoms[-1][2]==74.72
    assert list(_streams({"crop":[72.,75.],"rows":[r]},False))
    assert not list(_streams({"crop":[73.4,75.],"rows":[r]},False))
    assert not _atoms({**r,"source_start":74.5},False)
    assert not _atoms({**r,"source_text":"不同文字"},False)


def test_native_edge_variant_requires_two_independent_timed_streams():
    from nvidia_transcribe.ordered_event_anchors import _raw_support
    from nvidia_transcribe.source_quality import _lex
    words=_lex("we must protect the hidden archive before sunset")
    o={"crop":[0.,20.],"raw_response":{"choices":[{"message":{
        "content":"<spk:0> we must protect the hidden archive before sunrise"}}]}}
    assert not _raw_support([o],words,(2.,8.))
    assert _raw_support([o],words,(2.,8.),independently_localized=True)
    assert not _raw_support([o],_lex("we must destroy the secret house after midnight"),(2.,8.),
                            independently_localized=True)
