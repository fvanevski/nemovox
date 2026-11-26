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
