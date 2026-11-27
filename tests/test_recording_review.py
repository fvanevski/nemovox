"""Regressions for the comprehensive native dialogue review workflow."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from nvidia_transcribe.recording_context import RecordingContext, supported_terms, digest
from nvidia_transcribe.review_evidence import ReviewEvidenceProvider
from nvidia_transcribe.model_resources import load_model_resources
from nvidia_transcribe import connected_units
from nvidia_transcribe.translation_review import TranslationReviewConfig, review_translated_utterances


def row(i, source, english, speaker="S01", start=None, end=None):
    start = 10*i if start is None else start
    end = start+1 if end is None else end
    return {"id":f"event-{i:05d}", "source_text":source, "speaker_id":speaker,
            "source_start":start, "source_end":end, "translated_text":english,
            "machine_translated_text":english, "timestamp_provenance":
            {"backend":"moss_events","native_event_index":i,"timing_status":"observed_segment"}}


def config(**kwargs):
    return TranslationReviewConfig(**{"profile":"balanced", "wire_protocol":"ninfer_messages",
        "thinking_budget":2048, "reasoning_effort":"xhigh", "max_tokens":4096, **kwargs})


def reply(body):
    return {"choices":[{"finish_reason":"stop","message":{"content":json.dumps(body)}}]}


def exhausted():
    return {"choices":[{"finish_reason":"length","message":{"content":" "*2048}}],
            "_client_request_diagnostics":{"native_stop_reason":"max_tokens"}}


def ids(payload):
    return [uid for c in payload["comparisons"] for uid in c["target_ids"]]


def decisions(payload,choice="proposal"):
    return reply({"decisions":{c["key"]:{"choice":choice,"reason":"specific source meaning"}
                               for c in payload["comparisons"]}})


class Model:
    def __init__(self, propose, compare=None):
        self.propose = propose
        self.compare = compare
        self.calls = []
    def complete(self, *, messages, **kwargs):
        p = json.loads(messages[-1]["content"])
        self.calls.append(p)
        if "target_utterances" in p:
            schema=kwargs["response_format"]["schema"]["properties"]["reviews"]["items"]
            assert "discourse_terms" in schema["required"]
            supplemented="supplementary_observations" in p
            assert ("caption_evidence_ids" if supplemented else "evidence_request") in schema["required"]
            return reply({"reviews":[{"utterance_id":t["utterance_id"], "reason":"source evidence",
                "discourse_terms":[], **({"caption_evidence_ids":[]} if supplemented else {
                    "evidence_request":{"kind":"none","reference_id":"","question":"","reason":"text sufficient","focus":"none"}}),
                **self.propose(t,p)} for t in p["target_utterances"]]})
        if self.compare:
            return self.compare(p,kwargs)
        return decisions(p)


def run(rows, client, **kw):
    return review_translated_utterances(rows, source_language="zh",
        config=kw.pop("cfg",config()), text_client=client, **kw)


def test_full_context_in_both_stages_and_mode_is_cache_bound(tmp_path):
    rows = [row(i, f"明皇{i}。", f"Emperor {i}.") for i in range(15)]
    ckpt = tmp_path/"review.json"
    def propose(t,p):
        return {"corrected_text":"Ming Emperor."}
    client = Model(propose)
    target = {rows[7]["id"]}
    _, report = run(rows,client,target_utterance_ids=target,checkpoint_path=ckpt)
    assert report["status"] == "completed"
    assert len(client.calls) == 2
    assert client.calls[0]["recording_context"] == client.calls[1]["recording_context"]
    ctx = client.calls[0]["recording_context"]
    assert ctx["coverage"] == "complete" and ctx["included_occurrences"] == 15
    assert ctx["dialogue"][0][-2:] == ["明皇0。","Emperor 0."]
    assert ctx["dialogue"][-1][-2:] == ["明皇14。","Emperor 14."]
    assert "machine_translation" not in client.calls[0]["target_utterances"][0]
    replay = Model(propose)
    _, summary = run(rows,replay,target_utterance_ids=target,checkpoint_path=ckpt)
    assert summary["cache_hits"] == 1 and not replay.calls
    english = Model(propose)
    _, summary = run(rows,english,cfg=config(context_mode="english"),
        target_utterance_ids=target,checkpoint_path=ckpt)
    assert not summary["cache_hits"]
    assert "source" not in english.calls[0]["recording_context"]["columns"]
    assert english.calls[0]["target_utterances"][0]["source_text"] == rows[7]["source_text"]
    changed = deepcopy(rows)
    changed[-1]["machine_translated_text"] = "Distant context changed."
    replay = Model(propose)
    _, summary = run(changed,replay,target_utterance_ids=target,checkpoint_path=ckpt)
    assert summary["cache_hits"] == 0


def test_context_fallback_whole_observations_explicit():
    rows = [row(i,"甲"*100,"English "*40) for i in range(40)]
    ctx = RecordingContext(rows,"bilingual",2500).for_targets(20,21)
    assert ctx["coverage"] == "bounded"
    assert len(json.dumps(ctx,ensure_ascii=False,separators=(",",":")).encode()) <= 2500
    assert any(r[0]==rows[20]["id"] for r in ctx["dialogue"])
    assert [r[0] for r in ctx["dialogue"]] == sorted(r[0] for r in ctx["dialogue"])


def test_advisory_term_only_after_valid_accepted_source_citations():
    rows=[row(0,"明皇来了。","Bright Emperor came."),row(1,"明皇走了。","Emperor left.")]
    def propose(t,p):
        item={"corrected_text":"Ming Emperor."}
        if t["utterance_id"]==rows[0]["id"]:
            item["discourse_terms"]=[{"source":"明皇","english":"Ming Emperor",
                                     "reference_ids":[rows[1]["id"]]}]
        return item
    client=Model(propose)
    out,report=run(rows,client,cfg=config(batch_size=1))
    assert report["status"]=="completed"
    assert report["discourse_terms"][0]["reference_ids"] == [rows[1]["id"]]
    assert client.calls[2]["advisory_discourse_terms"] == [{**report["discourse_terms"][0],
        "accepted_target_ids":[rows[0]["id"]]}]
    assert out[0]["translation_review_discourse_terms"]
    invalid={"corrected_text":"Ming Emperor","discourse_terms":[
        {"source":"明皇","english":"Ming Emperor","reference_ids":["missing"]}]}
    target={"utterance_id":rows[0]["id"],"source_text":rows[0]["source_text"]}
    assert not supported_terms(invalid,target,{})


class Provider:
    identity={"policy":"test","available":True}
    def __init__(self):
        self.requests=[]
    def collect(self, uid, request):
        self.requests.append((uid,request))
        return {"target_id":uid,"request":request,"status":"observed","provider":self.identity,
                "observations":[{"description":"A character points to a cup."}]}


def test_conditional_evidence_refines_only_nominated_atomic_unit_and_replays(tmp_path):
    rows=[row(0,"拿", "Take",start=1,end=2),row(1,"那个", "that.",start=2.1,end=3),
          row(2,"好了。","All right.",start=6,end=7)]
    for group in connected_units.nominate(rows):
        record={**connected_units.request_record(group,[]),"joint_mt":"Take that.","joint_input_tokens":8}
        for r in group:
            r["translation_unit_candidate"]=deepcopy(record)
    assert connected_units.groups(rows)==[[0,1],[2]]
    p=Provider()
    def propose(t,payload):
        uid=t["utterance_id"]
        if "supplementary_observations" in payload:
            return {"corrected_text":"Take" if uid==rows[0]["id"] else "that cup."}
        result={"corrected_text":t["source_text"] if uid!=rows[2]["id"] else "All right."}
        if uid==rows[1]["id"]:
            result["evidence_request"]={"kind":"visual","reference_id":rows[1]["id"],
                                        "question":"What object is indicated?","reason":"Object reference ambiguous.","focus":"interaction"}
        return result
    client=Model(propose)
    cp=tmp_path/"review.json"
    out,report=run(rows,client,evidence_provider=p,checkpoint_path=cp)
    assert report["status"]=="completed"
    assert len(p.requests)==1 and len(client.calls)==3
    assert [x["kind"] for x in report["evidence_assessments"]]==["none","visual","none"]
    assert out[1]["translation_review_initial_proposal"]["corrected_text"]==rows[1]["source_text"]
    assert [t["utterance_id"] for t in client.calls[1]["target_utterances"]]==[r["id"] for r in rows[:2]]
    assert client.calls[2]["supplementary_observations"][0]["status"]=="observed"
    assert out[1]["translated_text"]=="that cup."
    assert out[2]["translated_text"]=="All right."
    assert out[0]["translation_unit"]==out[1]["translation_unit"]
    again=Model(propose)
    replay,_=run(rows,again,evidence_provider=p,checkpoint_path=cp)
    assert not again.calls and len(p.requests)==1 and replay==out


def test_recovery_preserves_other_decisions_and_never_regenerates_proposals():
    rows=[row(0,"一。","One."),row(1,"二。","Two."),row(2,"三。","Three.")]
    def propose(t,p):
        return {"corrected_text":"Correct "+t["utterance_id"]}
    def compare(p,kw):
        owned=ids(p)
        if len(owned)>1 or owned==[rows[1]["id"]]:
            return exhausted()
        return decisions(p)
    model=Model(propose,compare)
    out,summary=run(rows,model)
    assert summary["status"]=="completed_with_deferred_comparisons"
    assert summary["deferred_comparison_ids"]==[rows[1]["id"]]
    assert len([c for c in model.calls if "target_utterances" in c])==1
    assert summary["changed"]==2
    assert out[1]["translated_text"]=="Two."
    assert out[0]["translated_text"]=="Correct "+rows[0]["id"]
    assert out[2]["translated_text"]=="Correct "+rows[2]["id"]
    assert len(summary["comparison_recoveries"])==5
    assert [len(p["comparisons"]) for p in model.calls if "comparisons" in p]==[3,2,1,1,1]


def test_visual_role_inherits_or_is_explicit(tmp_path):
    reg=load_model_resources(environ={})
    assert reg.resolve("visual_inspection")==reg.resolve("translation_review")
    path=tmp_path/"models.json"
    path.write_text(json.dumps({"version":1,"resources":{"vision":{"adapter":"openai_chat",
       "endpoint":"http://127.0.0.1:9999/v1","model":"vision"}},"roles":{"visual_inspection":"vision"}}))
    reg=load_model_resources(path,environ={})
    assert not reg.visual_inherits_review
    assert reg.resolve("visual_inspection").model=="vision"
    assert reg.resolve("translation_review").model!="vision"


def test_provider_missing_media_and_invalid_reference():
    registry=load_model_resources(environ={})
    rows=[row(0,"甲","A")]
    p=ReviewEvidenceProvider(media_path=None,contract={"utterances":rows},lifecycle=None,
        review_resource=registry.resolve("translation_review"),visual_resource=registry.resolve("visual_inspection"),
        acoustic_resource=None,review_client=None)
    request={"kind":"visual","reference_id":rows[0]["id"],"question":"What is visible?"}
    assert p.collect(rows[0]["id"],request)["status"]=="media_unavailable"
    assert p.collect(rows[0]["id"],{**request,"reference_id":"forged"})["status"]=="invalid_nomination"


def test_shared_vision_submits_frame_hashes_without_switch(tmp_path,monkeypatch):
    from nvidia_transcribe import review_evidence
    media=tmp_path/"media.mkv"
    media.write_bytes(b"test media")
    rows=[row(0,"甲","A",start=1,end=2)]
    registry=load_model_resources(environ={})
    resource=registry.resolve("translation_review")
    captured=[]
    class Vision:
        def complete(self,**kw):
            captured.append(kw)
            return reply({"observations":[{"frame_ids":["frame-1"],
                          "category":"environment","description":"A cup","interpretation":"","uncertainty":"None visible"}],
                          "visible_text":[{"frame_ids":["frame-2"],"literal_text":"杯子",
                              "text_kind":"dialogue_caption","language":"zh","uncertainty":""}]})
        def close(self):
            raise AssertionError("shared resource must not switch")
    monkeypatch.setattr(review_evidence.subprocess,"run",
        lambda *a,**k: SimpleNamespace(stdout=b"\xff\xd8\xfftest-frame"))
    p=ReviewEvidenceProvider(media_path=media,contract={"utterances":rows,
        "source":{"content_sha256":hashlib.sha256(media.read_bytes()).hexdigest()},
        "native_dialogue":{"decoded_duration_s":10},"source_language":"zh"},lifecycle=None,
        review_resource=resource,visual_resource=resource,acoustic_resource=None,review_client=Vision())
    request={"kind":"visual","reference_id":rows[0]["id"],"question":"What is visible?"}
    result=p.collect(rows[0]["id"],request)
    assert result["status"]=="observed"
    assert len(result["frames"])>=3
    assert all(len(f["sha256"])==64 for f in result["frames"])
    assert len([b for b in captured[0]["messages"][1]["content"] if b["type"]=="image"])==len(result["frames"])
    assert captured[0]["reasoning_effort"]=="xhigh"
    caption,=result["visible_text"]
    assert caption["literal_text"]=="杯子" and caption["source_language_match"] is True
    assert caption["related_occurrence_ids"]==[rows[0]["id"]]
    assert caption["frame_sha256"]==[result["frames"][1]["sha256"]]
    assert result["observation_sha256"]==digest({k:result[k] for k in ("frames","observations","visible_text","focus")})
    assert p.collect(rows[0]["id"],request)==result and len(captured)==1


def test_distinct_visual_resource_releases_review_before_acquisition(tmp_path,monkeypatch):
    from contextlib import contextmanager
    from nvidia_transcribe import review_evidence
    from nvidia_transcribe.model_resources import ModelResource
    media=tmp_path/"media.mkv"
    media.write_bytes(b"test media")
    rows=[row(0,"甲","A",start=1,end=2)]
    review=load_model_resources(environ={}).resolve("translation_review")
    vision=ModelResource(name="vision",adapter="openai_chat",
                         endpoint="http://127.0.0.1:9999/v1",model="vision")
    events=[]
    class Review:
        def close(self):
            events.append("review_released")
        def complete(self,**kw):
            raise AssertionError("distinct visual resource must not call review")
    class Vision:
        def complete(self,**kw):
            events.append("vision_request")
            return reply({"observations":[{"frame_ids":["frame-1"],
                          "category":"environment","description":"A cup","interpretation":"","uncertainty":""}],"visible_text":[]})
    @contextmanager
    def managed(manager,resource,**kw):
        assert resource==vision and kw["wire_protocol"]=="ninfer_messages"
        events.append("vision_acquired")
        yield Vision()
        events.append("vision_released")
    monkeypatch.setattr(review_evidence,"managed_text_client",managed)
    monkeypatch.setattr(review_evidence.subprocess,"run",
        lambda *a,**k: SimpleNamespace(stdout=b"\xff\xd8\xfftest-frame"))
    p=ReviewEvidenceProvider(media_path=media,contract={"utterances":rows,
        "source":{"content_sha256":hashlib.sha256(media.read_bytes()).hexdigest()},
        "native_dialogue":{"decoded_duration_s":10}},lifecycle=object(),
        review_resource=review,visual_resource=vision,acoustic_resource=None,review_client=Review())
    result=p.collect(rows[0]["id"],{"kind":"visual","reference_id":rows[0]["id"],
        "question":"What is visible?"})
    assert result["status"]=="observed"
    assert events==["review_released","vision_acquired","vision_request","vision_released"]


def test_acoustic_crop_preserves_observations_and_original_stream(tmp_path,monkeypatch):
    from nvidia_transcribe import review_evidence
    media=tmp_path/"media.mkv"
    media.write_bytes(b"original")
    rows=[row(0,"甲","A",start=4,end=5)]
    registry=load_model_resources(environ={})
    review=registry.resolve("translation_review")
    asr=registry.resolve("asr")
    events=[]
    class Review:
        def close(self):
            events.append("review_released")
    def crop(path,**kw):
        assert kw["audio_stream_index"]==2 and kw["start_s"]==2
        kw["out_wav"].write_bytes(b"crop")
        events.append("crop")
        return 5
    def recognize(path,**kw):
        assert kw["resource"]==asr and kw["source_language"]=="zh"
        assert kw["audio_stream_index"]==0
        events.append("recognize")
        return {"contract":{"utterances":[row(0,"乙","B",start=1,end=2)],
                "model_provenance":{"adapter":"moss_events"},"native_dialogue":{"event_count":1}}}
    monkeypatch.setattr(review_evidence,"extract_wav_segment",crop)
    monkeypatch.setattr(review_evidence,"process_moss_media",recognize)
    p=ReviewEvidenceProvider(media_path=media,contract={"utterances":rows,
        "source":{"content_sha256":hashlib.sha256(media.read_bytes()).hexdigest(),
                  "selected_audio_stream":2},"source_language":"zh",
        "native_dialogue":{"decoded_duration_s":10}},lifecycle=object(),
        review_resource=review,visual_resource=review,acoustic_resource=asr,review_client=Review())
    result=p.collect(rows[0]["id"],{"kind":"acoustic","reference_id":rows[0]["id"],
        "question":"Re-observe the spoken word."})
    assert result["status"]=="observed" and not result["independently_confirmed"]
    assert result["observations"][0]["absolute_interval"]==[3,4]
    assert events==["crop","review_released","recognize"]
    assert rows[0]["source_text"]=="甲"


def test_explicit_advisory_decisions_are_required_in_direct_and_keyed_proposals():
    from nvidia_transcribe.translation_review import _native_proposals, ProposalDecisionError
    target={"utterance_id":"u","machine_translation":"A"}
    base={"utterance_id":"u","corrected_text":"B","reason":"meaning"}
    none={"kind":"none","reference_id":"","question":"","reason":"text sufficient","focus":"none"}
    for item in (base,{**base,"discourse_terms":[]},{**base,"evidence_request":none},
                 {**base,"discourse_terms":[],"evidence_request":{**none,"reference_id":"u"}}):
        with pytest.raises(ProposalDecisionError):
            _native_proposals(reply({"reviews":[item]}),[target],require_advisory=True)
    item={**base,"discourse_terms":[],"evidence_request":none}
    result=_native_proposals(reply({"reviews":[item]}),[target],require_advisory=True)
    assert result["u"]["evidence_request"]==none
    keyed={k:v for k,v in item.items() if k!="utterance_id"}
    with pytest.raises(ProposalDecisionError):
        _native_proposals(reply({"reviews":{"r01":{k:v for k,v in keyed.items() if k!="evidence_request"}}}),
                          [target],{"r01":"u"},require_advisory=True)
    assert _native_proposals(reply({"reviews":{"r01":keyed}}),[target],{"r01":"u"},
                             require_advisory=True)==result
    visual={**item,"evidence_request":{"kind":"visual","reference_id":"u","question":"Read caption",
                                      "reason":"Ambiguous source","focus":"text"}}
    with pytest.raises(ProposalDecisionError):
        _native_proposals(reply({"reviews":[visual]}),[target],require_advisory=True,allow_evidence=False)


def test_explicit_none_avoids_acquisition_and_extra_calls():
    rows=[row(0,"十一。","Eleven.")]
    p=Provider()
    out,report=run(rows,Model(lambda t,p:{"corrected_text":"Eleven."}),evidence_provider=p)
    assert report["status"]=="completed" and report["api_calls"]==1
    assert report["evidence_assessments"][0]["kind"]=="none"
    assert report["discourse_assessments"]==[{"target_id":rows[0]["id"],"nominated":[]}]
    assert p.requests==[] and out[0]["translated_text"]=="Eleven."


@pytest.mark.parametrize("reject", [False,True])
def test_caption_interpretation_is_bound_to_source_and_comparative_selection(tmp_path,reject):
    rows=[row(0,"大人打不过。","The adults cannot beat us."),row(1,"走吧。","Let's go.")]
    class CaptionProvider(Provider):
        def collect(self,uid,request):
            self.requests.append((uid,request))
            body={"frames":[{"id":"frame-1","sha256":"a"*64}],
                  "observations":[{"description":"A character speaks."}],
                  "visible_text":[{"caption_id":"visible-test","literal_text":"打又打不过",
                      "text_kind":"dialogue_caption","source_language_match":True,
                      "frame_ids":["frame-1"],"related_occurrence_ids":[uid],
                      "independently_confirmed":False}],"focus":"text"}
            return {"target_id":uid,"request":request,"provider":self.identity,
                "status":"observed","kind":"visual",**body,"observation_sha256":digest(body)}
    p=CaptionProvider()
    def propose(t,payload):
        if t["utterance_id"]==rows[1]["id"]:
            return {"corrected_text":"Let's go."}
        if "supplementary_observations" in payload:
            return {"corrected_text":"I can't even beat him.","caption_evidence_ids":["visible-test"]}
        return {"corrected_text":"My lord cannot win.",
                "evidence_request":{"kind":"visual","reference_id":rows[0]["id"],
                "question":"What literal dialogue caption is visible?","reason":"Possible homophone error.","focus":"text"}}
    def compare(payload,kw):
        assert payload["caption_interpretations"][0]["caption_ids"]==["visible-test"]
        assert payload["supplementary_observations"][0]["visible_text"][0]["literal_text"]=="打又打不过"
        return decisions(payload,"original" if reject else "proposal")
    cp=tmp_path/"review.json"
    client=Model(propose,compare)
    out,report=run(rows,client,evidence_provider=p,checkpoint_path=cp)
    assert report["status"]=="completed"
    assert out[0]["source_text"]==rows[0]["source_text"]
    assert out[0]["machine_translated_text"]==rows[0]["machine_translated_text"]
    interpretation=out[0]["translation_review_caption_interpretation"]
    assert interpretation["canonical_source_changed"] is False
    assert interpretation["status"]==("proposed_not_selected" if reject else "selected_interpretation")
    assert out[0]["translated_text"]==(rows[0]["translated_text"] if reject else "I can't even beat him.")
    assert out[0]["translation_review_initial_proposal"]["corrected_text"]=="My lord cannot win."
    assert out[1]["translated_text"]==rows[1]["translated_text"]
    replay=Model(propose,compare)
    again,summary=run(rows,replay,evidence_provider=p,checkpoint_path=cp)
    assert not replay.calls and len(p.requests)==1 and again==out
    assert summary["cache_hits"]==2


def test_caption_references_reject_wrong_occurrence_wrong_language_and_logo():
    from nvidia_transcribe.translation_review import _caption_references, ProposalDecisionError
    caption={"caption_id":"v1","text_kind":"dialogue_caption","source_language_match":True,
             "related_occurrence_ids":["u1"]}
    item={"caption_evidence_ids":["v1"]}
    for c in ({**caption,"related_occurrence_ids":["u2"]},
              {**caption,"source_language_match":False},
              {**caption,"text_kind":"other_text"}):
        with pytest.raises(ProposalDecisionError):
            _caption_references(item,"u1",[{"kind":"visual","status":"observed","visible_text":[c]}])
    assert _caption_references(item,"u1",[{"kind":"visual","status":"observed","visible_text":[caption]}])==[caption]


def test_rejected_term_never_enters_later_batches():
    rows=[row(0,"明皇来了。","Emperor came."),row(1,"明皇走了。","Emperor left.")]
    def propose(t,p):
        if t["utterance_id"]==rows[0]["id"]:
            return {"corrected_text":"Bright Emperor came.","discourse_terms":[
                {"source":"明皇","english":"Bright Emperor","reference_ids":[rows[1]["id"]]}]}
        assert p["advisory_discourse_terms"]==[]
        return {"corrected_text":"Emperor left."}
    def compare(p,kw):
        return decisions(p,"original")
    out,report=run(rows,Model(propose,compare),cfg=config(batch_size=1))
    assert report["status"]=="completed"
    assert report["discourse_terms"]==[]
    assert report["discourse_assessments"][0]["nominated"]
    assert not out[0].get("translation_review_discourse_terms")


def test_early_deferred_retry_freezes_completed_downstream_context(tmp_path):
    rows=[row(i,"明皇来了。",f"Emperor {i}.") for i in range(5)]
    def propose(t,p):
        uid=t["utterance_id"]
        return {"corrected_text":"Ming Emperor "+uid,
                "discourse_terms":[{"source":"明皇","english":"Ming Emperor",
                    "reference_ids":[rows[(int(uid[-5:])+1)%len(rows)]["id"]]}]}
    def first_compare(p,kw):
        return exhausted() if rows[0]["id"] in ids(p) else decisions(p)
    cp=tmp_path/"progress.json"
    first=Model(propose,first_compare)
    old,summary=run(rows,first,cfg=config(batch_size=1),checkpoint_path=cp)
    assert summary["deferred_comparison_ids"]==[rows[0]["id"]]
    assert all(r["translated_text"]!=r["machine_translated_text"] for r in old[1:])
    second=Model(lambda t,p:pytest.fail("must not regenerate proposals"))
    new,report=run(rows,second,cfg=config(batch_size=1),checkpoint_path=cp)
    assert report["status"]=="completed" and report["api_calls"]==1 and report["cache_hits"]==4
    assert ids(second.calls[0])==[rows[0]["id"]]
    assert new[1:]==old[1:]
    assert len(report["discourse_terms"])==len(rows)
    assert report["recording_context_receipts"][1:]==summary["recording_context_receipts"][1:]
    third=Model(lambda t,p:pytest.fail("must not call model"))
    replay,receipt=run(rows,third,cfg=config(batch_size=1),checkpoint_path=cp)
    assert replay==new and receipt["api_calls"]==0 and receipt["cache_hits"]==5


def test_failed_refinement_retries_only_owned_unit_and_reuses_acquisition(tmp_path):
    rows=[row(0,"拿", "Take",start=1,end=2),row(1,"那个", "that.",start=2.1,end=3),
          row(2,"好了。","All right.",start=6,end=7),row(3,"再见。","Bye.",start=20,end=21)]
    for group in connected_units.nominate(rows):
        record={**connected_units.request_record(group,[]),"joint_mt":"Take that.","joint_input_tokens":8}
        for r in group:
            r["translation_unit_candidate"]=deepcopy(record)
    provider=Provider()
    def propose(t,p):
        uid=t["utterance_id"]
        if "supplementary_observations" in p:
            return {"corrected_text":"Take" if uid==rows[0]["id"] else "that cup."}
        if uid==rows[1]["id"]:
            return {"corrected_text":"the thing.","evidence_request":{"kind":"visual","focus":"interaction",
                "reference_id":uid,"question":"What is indicated?","reason":"Ambiguous object"}}
        return {"corrected_text":"Take" if uid==rows[0]["id"] else "Finished."}
    class FailedSupplement(Model):
        def complete(self,*,messages,**kw):
            payload=json.loads(messages[-1]["content"])
            if "target_utterances" in payload and "supplementary_observations" in payload:
                self.calls.append(payload)
                return exhausted()
            return super().complete(messages=messages,**kw)
    cp=tmp_path/"progress.json"
    first=FailedSupplement(propose)
    old,report=run(rows,first,evidence_provider=provider,checkpoint_path=cp)
    assert report["status"]=="completed_with_deferred_evidence"
    assert {x["target_id"] for x in report["supplementary_proposal_errors"]}=={r["id"] for r in rows[:2]}
    assert [r["translated_text"] for r in old[:2]]==["Take","that."]
    assert len(provider.requests)==1
    # New provider instance proves durable, not just in-memory, reuse.
    again_provider=Provider()
    second=Model(propose)
    new,report=run(rows,second,evidence_provider=again_provider,checkpoint_path=cp)
    assert report["status"]=="completed" and report["cache_hits"]==2
    assert report["review_execution"]["checkpoint_acquisition_reuses"]==1
    assert len(again_provider.requests)==0
    assert len(second.calls)==2
    assert [t["utterance_id"] for t in second.calls[0]["target_utterances"]]==[r["id"] for r in rows[:2]]
    assert ids(second.calls[1])==[r["id"] for r in rows[:2]]
    assert new[2:]==old[2:]
    assert new[1]["translated_text"]=="that cup."
    assert new[0]["translation_unit"]==new[1]["translation_unit"]
    final,receipt=run(rows,Model(propose),evidence_provider=Provider(),checkpoint_path=cp)
    assert final==new and receipt["api_calls"]==0


def test_keyed_comparison_rejects_duplicate_and_unknown_keys():
    from nvidia_transcribe.review_execution import keyed_schema,keyed_decisions
    aliases,schema=keyed_schema(["passage:"+"a"*64,"event-00001"])
    assert schema["properties"]["decisions"]["required"]==["d01","d02"]
    assert not schema["properties"]["decisions"]["additionalProperties"]
    for raw in ('{"decisions":{"d01":{"choice":"proposal","reason":"ok"},"d01":{"choice":"original","reason":"no"}}}',
                '{"decisions":{"d01":{"choice":"proposal","reason":"ok"},"d03":{"choice":"original","reason":"no"}}}'):
        with pytest.raises(ValueError):
            keyed_decisions({"choices":[{"finish_reason":"stop","message":{"content":raw}}]},aliases)
    valid=keyed_decisions(reply({"decisions":{"d01":{"choice":"proposal","reason":"ok"},
        "d02":{"choice":"original","reason":"no"}}}),aliases)
    assert set(valid)==set(aliases.values())


def test_visual_sampling_and_nearby_scene_projection():
    from nvidia_transcribe.review_evidence import visual_seeks
    from nvidia_transcribe.review_execution import scene_projection
    scene=visual_seeks(5,12,30,6,"scene")
    speech=visual_seeks(5,12,30,6,"speech_activity")
    assert len(scene)<=6 and len(speech)<=6 and scene[0]==speech[0] and scene[-1]==speech[-1]
    assert max(speech[1:-1])-min(speech[1:-1])<=1.31
    assert max(scene[1:-1])-min(scene[1:-1])>1.31
    obs={"observation_id":"scene-a","requested_seek_interval":[5,12],"related_occurrence_ids":["a"],
         "category":"participants","description":"Two children.","interpretation":"","uncertainty":""}
    assert scene_projection([obs],[{"utterance_id":"b","start":13,"end":14}])==[obs]
    assert scene_projection([obs],[{"utterance_id":"c","start":30,"end":31}])==[]


def test_progress_corruption_is_not_a_cache_hit(tmp_path):
    cp=tmp_path/"progress.json"
    rows=[row(0,"一。","One.")]
    run(rows,Model(lambda t,p:{"corrected_text":"First."}),checkpoint_path=cp)
    doc=json.loads(cp.read_text())
    next(iter(doc["records"].values()))["record"]["snapshot"]["discourse_terms"]=[{"fake":True}]
    cp.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="digest mismatch"):
        run(rows,Model(lambda t,p:pytest.fail("corrupted checkpoint")),checkpoint_path=cp)


def test_structured_scene_observation_reaches_later_batch_without_new_acquisition(tmp_path):
    rows=[row(0,"拿那个。","Take that.",start=1,end=2),
          row(1,"放那里。","Put it there.",start=4,end=5)]
    class SceneProvider(Provider):
        def collect(self,uid,request):
            self.requests.append((uid,request))
            body={"frames":[{"id":"frame-1","sha256":"a"*64,"requested_seek_s":1.5}],
                "observations":[{"observation_id":"scene-cup","frame_ids":["frame-1"],
                    "category":"interactions","description":"A child points at a cup beside a shelf.",
                    "interpretation":"The cup may be the intended object.","uncertainty":"No speech attribution.",
                    "requested_seek_interval":[1.5,1.5],"related_occurrence_ids":[uid],
                    "independently_confirmed":False}],
                "visible_text":[],"focus":"interaction"}
            return {"target_id":uid,"request":request,"provider":self.identity,"status":"observed",
                "kind":"visual",**body,"observation_sha256":digest(body)}
    p=SceneProvider()
    def propose(t,payload):
        if t["utterance_id"]==rows[0]["id"]:
            if "supplementary_observations" in payload:
                return {"corrected_text":"Take the cup."}
            return {"corrected_text":"Take that.","evidence_request":{"kind":"visual","focus":"interaction",
                "reference_id":rows[0]["id"],"question":"What is indicated?","reason":"Unclear object"}}
        prior,=payload["advisory_scene_observations"]
        assert prior["observation_id"]=="scene-cup"
        return {"corrected_text":"Put the cup there."}
    cp=tmp_path/"progress.json"
    out,report=run(rows,Model(propose),cfg=config(batch_size=1),evidence_provider=p,checkpoint_path=cp)
    assert report["status"]=="completed" and len(p.requests)==1
    assert report["recording_context_receipts"][1]["advisory_scene_observations"]==report["scene_observations"]
    assert out[1]["translated_text"]=="Put the cup there."
    new_provider=SceneProvider()
    again,receipt=run(rows,Model(lambda t,p:pytest.fail("unexpected model request")),cfg=config(batch_size=1),
        evidence_provider=new_provider,checkpoint_path=cp)
    assert again==out and receipt["api_calls"]==0 and new_provider.requests==[]


def test_retryable_transport_does_not_fan_out_comparisons(tmp_path):
    import httpx
    rows=[row(i,"一。",f"Number {i}.") for i in range(4)]
    def fail(p,kw):
        raise httpx.ReadTimeout("bounded test transport")
    cp=tmp_path/"progress.json"
    out,report=run(rows,Model(lambda t,p:{"corrected_text":"One."},fail),checkpoint_path=cp)
    assert report["status"]=="completed_with_deferred_comparisons"
    assert report["api_calls"]==2
    assert report["review_execution"]["recovery_comparison_requests"]==0
    assert set(report["deferred_comparison_ids"])=={r["id"] for r in rows}
    fixed,receipt=run(rows,Model(lambda t,p:pytest.fail("proposal regenerated")),checkpoint_path=cp)
    assert receipt["api_calls"]==1 and receipt["status"]=="completed"
    assert all(r["translated_text"]=="One." for r in fixed)


def evidence_request(reference_id, kind="visual"):
    return {"kind":kind,"reference_id":reference_id,"question":"What is the relevant evidence?",
            "reason":"Ambiguous source","focus":"scene" if kind=="visual" else "none"}


class NominationModel(Model):
    """Exercise the real direct and keyed proposal parsers, not a validator mock."""
    def complete(self, *, messages, **kwargs):
        payload=json.loads(messages[-1]["content"])
        if "response_keys" not in payload:
            return super().complete(messages=messages, **kwargs)
        self.calls.append(payload)
        schema=kwargs["response_format"]["schema"]["properties"]["reviews"]
        assert set(schema["required"])==set(payload["response_keys"])
        targets={t["utterance_id"]:t for t in payload["target_utterances"]}
        return reply({"reviews":{key:{"reason":"source evidence","discourse_terms":[],
            "evidence_request":{"kind":"none","reference_id":"","question":"",
                                "reason":"text sufficient","focus":"none"},
            **self.propose(targets[uid],payload)}
            for key,uid in payload["response_keys"].items()}})


@pytest.mark.parametrize("persistent", [False, True])
def test_invalid_reference_is_recovered_or_rejected_before_durable_admission(tmp_path,persistent):
    rows=[row(0,"甲。","A."),row(1,"乙。","B.")]
    a,b=[r["id"] for r in rows]
    provider=Provider()
    def propose(t,payload):
        uid=t["utterance_id"]
        if uid!=a:
            return {"corrected_text":"B."}
        if "supplementary_observations" in payload:
            return {"corrected_text":"A."}
        ref=b if "response_keys" in payload and not persistent else "missing-event"
        return {"corrected_text":"A.","evidence_request":evidence_request(ref)}
    cp=tmp_path/"progress.json"
    model=NominationModel(propose)
    out,report=run(rows,model,cfg=config(batch_size=1),evidence_provider=provider,checkpoint_path=cp)
    assert len(report["proposal_recoveries"])==1
    assert "unknown_evidence_reference" in json.dumps(report["proposal_recoveries"])
    assert not report["supplementary_proposal_errors"]
    records=[w["record"] for w in json.loads(cp.read_text())["records"].values()]
    if persistent:
        assert report["status"]=="completed_with_errors"
        assert provider.requests==[]
        assert out[0]["translated_text"]=="A."
        assert out[0]["translation_review_status"]=="review_error_preserved_machine_translation"
        assert all(a not in (r["entries"] or {}) for r in records)
        # The failed proposal never becomes durable successful work; retry
        # can actually regenerate it while the completed neighbor is reused.
        def corrected(t,payload):
            assert t["utterance_id"]==a
            return {"corrected_text":"A.",**({} if "supplementary_observations" in payload else
                {"evidence_request":evidence_request(b)})}
        final,receipt=run(rows,Model(corrected),cfg=config(batch_size=1),
            evidence_provider=provider,checkpoint_path=cp)
        assert receipt["status"]=="completed" and receipt["cache_hits"]==1
        assert final[1]==out[1]
    else:
        assert report["status"]=="completed"
        assert report["proposal_recoveries"][0]["outcome"]=="complete"
        final=out
    # A valid reference may name a different real occurrence, not only the
    # current target or the current packed batch.
    assert provider.requests==[
        (a,{k:v for k,v in evidence_request(b).items() if k!="reason"})]
    assert final[0]["translation_review_evidence"]["status"]=="observed"
    fresh_provider=Provider()
    again,receipt=run(rows,Model(lambda t,p:pytest.fail("completed proposal regenerated")),
        cfg=config(batch_size=1),evidence_provider=fresh_provider,checkpoint_path=cp)
    assert again==final and receipt["api_calls"]==0 and fresh_provider.requests==[]


def test_zero_budget_uses_text_only_schema_and_resumes(tmp_path):
    rows=[row(0,"甲。","A.")]
    class DisabledModel(NominationModel):
        def complete(self,*,messages,**kw):
            p=json.loads(messages[-1]["content"])
            if "target_utterances" in p:
                schema=kw["response_format"]["schema"]["properties"]["reviews"]
                items=schema["properties"]["r01"] if "response_keys" in p else schema["items"]
                assert items["properties"]["evidence_request"]["properties"]["kind"]["enum"]==["none"]
                assert "Acquisition is disabled" in messages[0]["content"]
            return super().complete(messages=messages,**kw)
    def propose(t,p):
        if "response_keys" not in p:
            return {"corrected_text":"A.","evidence_request":evidence_request(rows[0]["id"])}
        return {"corrected_text":"A."}
    p=Provider()
    cp=tmp_path/"progress.json"
    cfg=config(max_evidence_requests_per_batch=0)
    out,report=run(rows,DisabledModel(propose),cfg=cfg,evidence_provider=p,checkpoint_path=cp)
    assert report["status"]=="completed" and report["api_calls"]==2
    assert report["proposal_recoveries"][0]["outcome"]=="complete"
    assert report["evidence_assessments"][0]["kind"]=="none"
    assert report["supplementary_proposal_errors"]==[] and p.requests==[]
    assert report["review_execution"]["provider_requests"]==0
    again,receipt=run(rows,Model(lambda t,p:pytest.fail("regenerated text-only proposal")),
        cfg=cfg,evidence_provider=Provider(),checkpoint_path=cp)
    assert again==out and receipt["api_calls"]==0 and receipt["cache_hits"]==1


@pytest.mark.parametrize("kind,status,end,available,visual,acoustic",[
    ("visual","media_unavailable",2,False,True,True),
    ("visual","reference_untimed",2,True,True,True),
    ("visual","visual_resource_unavailable",2,True,False,True),
    ("acoustic","acoustic_resource_unavailable",2,True,True,False),
    ("acoustic","reference_exceeds_acoustic_budget",40,True,True,True),
])
def test_permanent_evidence_limitation_completes_and_reuses_receipt(
        tmp_path,monkeypatch,kind,status,end,available,visual,acoustic):
    from nvidia_transcribe import review_evidence
    media=tmp_path/"media.mkv"
    media.write_bytes(b"authenticated test media")
    rows=[row(0,"甲。","A.",start=1,end=end)]
    if status=="reference_untimed":
        rows[0]["source_start"]=rows[0]["source_end"]=None
    registry=load_model_resources(environ={})
    review=registry.resolve("translation_review")
    def forbidden(*a,**kw):
        pytest.fail("terminal limitation must not invoke media or model acquisition")
    monkeypatch.setattr(review_evidence.subprocess,"run",forbidden)
    monkeypatch.setattr(review_evidence,"extract_wav_segment",forbidden)
    monkeypatch.setattr(review_evidence,"process_moss_media",forbidden)
    def provider():
        return ReviewEvidenceProvider(media_path=media if available else None,
            contract={"utterances":rows,"source":{"content_sha256":hashlib.sha256(media.read_bytes()).hexdigest(),
                "selected_audio_stream":0},"native_dialogue":{"decoded_duration_s":60},"source_language":"zh"},
            lifecycle=None,review_resource=review,visual_resource=review if visual else None,
            acoustic_resource=registry.resolve("asr") if acoustic else None,review_client=None)
    def propose(t,p):
        assert "supplementary_observations" not in p
        return {"corrected_text":"Text-supported A.","evidence_request":evidence_request(rows[0]["id"],kind)}
    def compare(p,kw):
        assert p["supplementary_observations"][0]["status"]==status
        return decisions(p,"original")
    cp=tmp_path/"progress.json"
    out,report=run(rows,Model(propose,compare),evidence_provider=provider(),checkpoint_path=cp)
    assert report["status"]=="completed" and report["api_calls"]==2
    assert report["supplementary_evidence_unresolved_count"]==1
    assert report["supplementary_proposal_errors"]==[]
    assert report["supplementary_evidence"][0]["status"]==status
    assert out[0]["translated_text"]=="A."
    p2=provider()
    again,receipt=run(rows,Model(lambda t,p:pytest.fail("unexpected proposal")),
        evidence_provider=p2,checkpoint_path=cp)
    assert again==out and receipt["api_calls"]==0 and receipt["cache_hits"]==1
    assert receipt["review_execution"]["provider_requests"]==0
    assert receipt["review_execution"]["checkpoint_acquisition_reuses"]==0


@pytest.mark.parametrize("failure",["acquisition_failed","batch_budget_exhausted"])
def test_retryable_evidence_resolves_only_pending_unit(tmp_path,failure):
    rows=[row(0,"甲。","A."),row(1,"乙。","B."),row(2,"丙。","C.")]
    a,b,c=[r["id"] for r in rows]
    class TransientProvider(Provider):
        def collect(self,uid,request):
            obs=super().collect(uid,request)
            if uid==b:
                obs["status"]="acquisition_failed"
            return obs
    def propose(t,p):
        uid=t["utterance_id"]
        if "supplementary_observations" in p:
            return {"corrected_text":"Refined "+uid}
        if uid in (a,b):
            return {"corrected_text":"Proposed "+uid,"evidence_request":evidence_request(uid)}
        return {"corrected_text":"Revised C."}
    cfg=config(max_evidence_requests_per_batch=1 if failure=="batch_budget_exhausted" else 3)
    cp=tmp_path/"progress.json"
    first=Provider() if failure=="batch_budget_exhausted" else TransientProvider()
    old,report=run(rows,Model(propose),cfg=cfg,evidence_provider=first,checkpoint_path=cp)
    assert report["status"]=="completed_with_deferred_evidence"
    assert report["supplementary_proposal_errors"]==[{"target_id":b,"reason":"unresolved_evidence_acquisition"}]
    assert old[1]["translation_review_evidence"]["status"]==failure
    assert old[1]["translated_text"]=="B."
    p2=Provider()
    second=Model(propose)
    final,receipt=run(rows,second,cfg=cfg,evidence_provider=p2,checkpoint_path=cp)
    assert receipt["status"]=="completed" and receipt["cache_hits"]==2
    assert receipt["api_calls"]==2 and receipt["review_execution"]["checkpoint_acquisition_reuses"]==1
    assert [uid for uid,_ in p2.requests]==[b]
    assert final[0]==old[0] and final[2]==old[2]
    assert final[1]["translated_text"]=="Refined "+b
    assert [t["utterance_id"] for t in second.calls[0]["target_utterances"]]==[b]
    assert ids(second.calls[1])==[b]


def test_invalid_legacy_checkpoint_nomination_fails_closed(tmp_path):
    rows=[row(0,"甲。","A.")]
    uid=rows[0]["id"]
    cp=tmp_path/"progress.json"
    p=Provider()
    def propose(t,payload):
        return {"corrected_text":"A.",**({} if "supplementary_observations" in payload else
            {"evidence_request":evidence_request(uid)})}
    _,report=run(rows,Model(propose),evidence_provider=p,checkpoint_path=cp)
    assert report["status"]=="completed"
    # Emulate a structurally valid pre-fix checkpoint with an invalid
    # admitted nomination, including coherent request and envelope digests.
    doc=json.loads(cp.read_text())
    wrapped=next(iter(doc["records"].values()))
    record=wrapped["record"]
    item=record["entries"][uid]
    ref=record["proposal_request_sha256"]
    receipt=record["requests"].pop(ref)
    response=json.loads(receipt["response"][0]["message"]["content"])
    response["reviews"][0]["evidence_request"]["reference_id"]="missing"
    receipt["response"][0]["message"]["content"]=json.dumps(response)
    ref=digest(receipt)
    record["requests"][ref]=receipt
    record["proposal_request_sha256"]=ref
    item["evidence_request"]["reference_id"]="missing"
    wrapped["sha256"]=digest(record)
    cp.write_text(json.dumps(doc))
    p2=Provider()
    _,report=run(rows,Model(lambda t,p:pytest.fail("invalid checkpoint must fail visibly")),
        evidence_provider=p2,checkpoint_path=cp)
    assert report["status"]=="completed_with_errors"
    assert any("unknown_evidence_reference" in e for e in report["errors"])
    assert p2.requests==[] and report["supplementary_proposal_errors"]==[]


def visual_fixture(tmp_path, monkeypatch, rows, client):
    """Real provider with only frame extraction and model transport replaced."""
    from nvidia_transcribe import review_evidence
    media = tmp_path / "visual-fixture.mkv"
    media.write_bytes(b"authenticated fixture media")
    resource = load_model_resources(environ={}).resolve("translation_review")
    monkeypatch.setattr(review_evidence.subprocess, "run",
        lambda *a, **kw: SimpleNamespace(stdout=b"\xff\xd8\xfftest-frame"))
    return ReviewEvidenceProvider(media_path=media,
        contract={"utterances":rows,
            "source":{"content_sha256":hashlib.sha256(media.read_bytes()).hexdigest()},
            "native_dialogue":{"decoded_duration_s":60},"source_language":"zh"},
        lifecycle=None,review_resource=resource,visual_resource=resource,
        acoustic_resource=None,review_client=client)


def caption_only_body():
    return {"observations":[], "visible_text":[{
        "frame_ids":["frame-2"], "literal_text":"杯子", "text_kind":"dialogue_caption",
        "language":"zh", "uncertainty":""}]}


class FixtureVision:
    def __init__(self, body=None, error=None):
        self.body = body
        self.error = error
        self.calls = []
    def complete(self, **kw):
        self.calls.append(kw)
        if self.error is not None:
            raise self.error
        return reply(self.body)
    def close(self):
        pytest.fail("shared visual resource must not switch")


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 408, 429, 500, 503, None])
def test_visual_http_failure_respects_shared_retry_policy(tmp_path, monkeypatch, status):
    import httpx
    from nvidia_transcribe.review_execution import retryable_review_transport
    rows = [row(0,"杯子。","Cup.",start=1,end=2),
            row(1,"走吧。","Go.",start=20,end=21)]
    uid = rows[0]["id"]
    request = httpx.Request("POST","http://127.0.0.1:8001/v1/messages")
    error = (httpx.HTTPStatusError("fixture status",request=request,
                response=httpx.Response(status,request=request))
             if status is not None else httpx.ReadTimeout("fixture timeout",request=request))
    retryable = status in (408,429,500,503,None)
    assert retryable_review_transport(error) is retryable
    vision = FixtureVision(error=error)
    provider = visual_fixture(tmp_path,monkeypatch,rows,vision)
    def propose(t,p):
        if t["utterance_id"] != uid:
            return {"corrected_text":"Let's go."}
        if "supplementary_observations" in p:
            caption, = p["supplementary_observations"][0]["visible_text"]
            return {"corrected_text":"The cup.","caption_evidence_ids":[caption["caption_id"]]}
        return {"corrected_text":"The object.","evidence_request":{
            **evidence_request(uid),"focus":"text"}}
    cp = tmp_path/"progress.json"
    cfg = config(batch_size=1)
    old,report = run(rows,Model(propose),cfg=cfg,evidence_provider=provider,checkpoint_path=cp)
    assert len(vision.calls)==1 and provider.cache=={}
    assert old[0]["translated_text"]=="Cup."
    assert old[1]["translated_text"]=="Let's go."
    if retryable:
        assert report["status"]=="completed_with_deferred_evidence"
        assert report["errors"]==[]
        assert old[0]["translation_review_evidence"]["status"]=="acquisition_failed"
        assert old[0]["translation_review_evidence"].get("http_status")==status
        assert report["supplementary_proposal_errors"]==[
            {"target_id":uid,"reason":"unresolved_evidence_acquisition"}]
    else:
        assert report["status"]=="completed_with_errors"
        assert any("HTTPStatusError" in reason for reason in report["errors"])
        assert report["supplementary_proposal_errors"]==[]
        assert old[0]["translation_review_status"]=="review_error_preserved_machine_translation"
        assert "translation_review_evidence" not in old[0]
        record=next(w["record"] for w in json.loads(cp.read_text())["records"].values()
                    if uid in (w["record"]["entries"] or {}))
        assert "supplementary_observation" not in record["entries"][uid]
    # Corrected endpoint (or recovered transient service) can finish the
    # admitted proposal; no successful downstream work is reopened.
    next_vision = FixtureVision(caption_only_body())
    next_provider = visual_fixture(tmp_path,monkeypatch,rows,next_vision)
    model=Model(propose)
    new,receipt=run(rows,model,cfg=cfg,evidence_provider=next_provider,checkpoint_path=cp)
    assert receipt["status"]=="completed" and receipt["cache_hits"]==1
    assert receipt["api_calls"]==2 and len(next_vision.calls)==1
    assert new[1]==old[1] and new[0]["translated_text"]=="The cup."
    assert len(model.calls[0]["target_utterances"])==1
    assert "supplementary_observations" in model.calls[0]
    assert ids(model.calls[1])==[uid]


def test_caption_only_real_provider_refinement_and_durable_reuse(tmp_path,monkeypatch):
    rows=[row(0,"大人打不过。","The adults cannot win.",start=1,end=2)]
    uid=rows[0]["id"]
    body=caption_only_body()
    body["visible_text"][0]["literal_text"]="打又打不过"
    vision=FixtureVision(body)
    provider=visual_fixture(tmp_path,monkeypatch,rows,vision)
    def propose(t,p):
        if "supplementary_observations" in p:
            obs,=p["supplementary_observations"]
            assert obs["observations"]==[]
            caption,=obs["visible_text"]
            assert caption["related_occurrence_ids"]==[uid]
            assert caption["source_language_match"] is True
            return {"corrected_text":"I can't even beat him.","caption_evidence_ids":[caption["caption_id"]]}
        return {"corrected_text":"My lord cannot win.","evidence_request":{
            **evidence_request(uid),"focus":"text"}}
    def compare(p,kw):
        obs,=p["supplementary_observations"]
        caption,=obs["visible_text"]
        assert p["caption_interpretations"][0]["caption_ids"]==[caption["caption_id"]]
        assert obs["observations"]==[]
        return decisions(p)
    cp=tmp_path/"progress.json"
    out,report=run(rows,Model(propose,compare),evidence_provider=provider,checkpoint_path=cp)
    assert report["status"]=="completed" and report["api_calls"]==3
    assert out[0]["source_text"]==rows[0]["source_text"]
    assert out[0]["machine_translated_text"]==rows[0]["machine_translated_text"]
    assert out[0]["translated_text"]=="I can't even beat him."
    obs=out[0]["translation_review_evidence"]
    assert obs["status"]=="observed" and obs["observations"]==[]
    assert report["scene_observations"]==[]
    caption,=obs["visible_text"]
    assert caption["frame_sha256"]==[obs["frames"][1]["sha256"]]
    assert caption["reference_occurrence_id"]==uid
    assert caption["independently_confirmed"] is False
    assert obs["observation_sha256"]==digest({k:obs[k] for k in ("frames","observations","visible_text","focus")})
    assert out[0]["translation_review_caption_interpretation"]["status"]=="selected_interpretation"
    assert provider.collect(uid,obs["request"])==obs and len(vision.calls)==1
    no_vision=FixtureVision(error=AssertionError("unexpected vision inference"))
    p2=visual_fixture(tmp_path,monkeypatch,rows,no_vision)
    again,receipt=run(rows,Model(lambda t,p:pytest.fail("unexpected text inference")),
        evidence_provider=p2,checkpoint_path=cp)
    assert again==out and receipt["api_calls"]==0 and receipt["cache_hits"]==1
    assert receipt["review_execution"]["checkpoint_acquisition_reuses"]==1
    assert receipt["review_execution"]["provider_requests"]==0
    assert p2.collect(uid,obs["request"])==obs and no_vision.calls==[]


@pytest.mark.parametrize("case",["both_empty","bad_scene","bad_caption","blank_caption",
    "wrong_collection_type","too_many_scenes","too_many_captions"])
def test_visual_collections_reject_empty_or_malformed_evidence(tmp_path,monkeypatch,case):
    rows=[row(0,"杯子。","Cup.",start=1,end=2)]
    uid=rows[0]["id"]
    body=caption_only_body()
    scene={"frame_ids":["frame-2"],"category":"environment","description":"A cup.",
           "interpretation":"","uncertainty":""}
    if case=="both_empty":
        body["visible_text"]=[]
    elif case=="bad_scene":
        body["observations"]=[{**scene,"frame_ids":["forged"]}]
    elif case=="bad_caption":
        body["observations"]=[scene]
        body["visible_text"][0]["frame_ids"]=["forged"]
    elif case=="blank_caption":
        body["visible_text"][0]["literal_text"]=" "
    elif case=="wrong_collection_type":
        body["observations"]={}
    elif case=="too_many_scenes":
        body["observations"]=[scene]*13
    else:
        body["visible_text"]*=17
    vision=FixtureVision(body)
    provider=visual_fixture(tmp_path,monkeypatch,rows,vision)
    request={k:v for k,v in evidence_request(uid).items() if k!="reason"}
    request["focus"]="text"
    obs=provider.collect(uid,request)
    assert obs["status"]=="acquisition_failed"
    assert obs["error_type"]=="ValueError"
    assert "observation_sha256" not in obs and provider.cache=={}
    assert len(vision.calls)==1
