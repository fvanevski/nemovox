"""Regressions for the comprehensive native dialogue review workflow."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from nvidia_transcribe.recording_context import RecordingContext, supported_terms
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
    return [c["utterance_id"] for c in payload["comparisons"]+payload["passage_checks"]]


class Model:
    def __init__(self, propose, compare=None):
        self.propose = propose
        self.compare = compare
        self.calls = []
    def complete(self, *, messages, **kwargs):
        p = json.loads(messages[-1]["content"])
        self.calls.append(p)
        if "target_utterances" in p:
            return reply({"reviews":[{"utterance_id":t["utterance_id"], "reason":"source evidence",
                **self.propose(t,p)} for t in p["target_utterances"]]})
        if self.compare:
            return self.compare(p,kwargs)
        return reply({"decisions":[{"utterance_id":uid,"choice":"proposal","reason":"source supports this"}
                                   for uid in ids(p)]})


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
    assert client.calls[2]["advisory_discourse_terms"] == report["discourse_terms"]
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
                                        "question":"What object is indicated?"}
        return result
    client=Model(propose)
    cp=tmp_path/"review.json"
    out,report=run(rows,client,evidence_provider=p,checkpoint_path=cp)
    assert report["status"]=="completed"
    assert len(p.requests)==1 and len(client.calls)==3
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
        return reply({"reason":"Specific source fact","verdicts":[owned[0]+"=proposal"]})
    model=Model(propose,compare)
    out,summary=run(rows,model)
    assert summary["status"]=="completed_with_deferred_comparisons"
    assert summary["deferred_comparison_ids"]==[rows[1]["id"]]
    assert len([c for c in model.calls if "target_utterances" in c])==1
    assert summary["changed"]==2
    assert out[1]["translated_text"]=="Two."
    assert out[0]["translated_text"]=="Correct "+rows[0]["id"]
    assert out[2]["translated_text"]=="Correct "+rows[2]["id"]
    assert len(summary["comparison_recoveries"])==3


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
                          "description":"A cup","uncertainty":"None visible"}]})
        def close(self):
            raise AssertionError("shared resource must not switch")
    monkeypatch.setattr(review_evidence.subprocess,"run",
        lambda *a,**k: SimpleNamespace(stdout=b"\xff\xd8\xfftest-frame"))
    p=ReviewEvidenceProvider(media_path=media,contract={"utterances":rows,
        "source":{"content_sha256":hashlib.sha256(media.read_bytes()).hexdigest()},
        "native_dialogue":{"decoded_duration_s":10}},lifecycle=None,
        review_resource=resource,visual_resource=resource,acoustic_resource=None,review_client=Vision())
    request={"kind":"visual","reference_id":rows[0]["id"],"question":"What is visible?"}
    result=p.collect(rows[0]["id"],request)
    assert result["status"]=="observed"
    assert len(result["frames"])>=3
    assert all(len(f["sha256"])==64 for f in result["frames"])
    assert len([b for b in captured[0]["messages"][1]["content"] if b["type"]=="image"])==len(result["frames"])
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
                          "description":"A cup","uncertainty":""}]})
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
