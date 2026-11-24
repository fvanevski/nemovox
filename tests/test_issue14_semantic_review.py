"""Meaning-first overlap triage: diagnostics are not accepted source changes."""
from dataclasses import dataclass
from copy import deepcopy
import pytest

from nvidia_transcribe.semantic_review import (
    plan_boundary_semantic_reviews, validate_semantic_review,
)
from nvidia_transcribe.chunking import ChunkSpec, reconcile_chunk_boundaries

@dataclass
class C:
    chunk_index: int
    core_start_s: float
    core_end_s: float
    infer_start_s: float
    infer_end_s: float

P={"schema":"nemovox-semantic-review-policy/v1","policy_id":"draft-meaning-first-v1",
   "min_phrase_words":3,"max_interword_gap_s":1.2,"max_nominal_core_coverage_ratio":.45,
   "reviewer_guidance":"Assess meaning, polarity, narrative context and timing; cite provenance and abstain on acoustic uncertainty."}

def w(t, time, idx, chunk, voice="A"):
    return {"text":t,"start":time,"end":time+.18,"_mid":time+.09,"valid_timing":True,
            "unit_type":"word","_unit_idx":idx,"_chunk_idx":chunk,"_in_core":chunk==0,
            "_drop":False,"speaker":voice}

def chunks(boundary=360):
    return [C(0,boundary-90,boundary,boundary-100,boundary+10),
            C(1,boundary,boundary+90,boundary-10,boundary+100)]

def test_left_context_omission_and_displaced_duplicate_plan_review_only():
    left=[w(x,304+i*.27,i,0) for i,x in enumerate(["This","may","not","be","possible"])]
    right=[w(x,355+i*.27,11+i,1,"B") for i,x in enumerate(["This","may","not","be","possible"])]
    original=deepcopy((left,right))
    activity={0:[{"start_s":355,"end_s":357,"possible_unrepresented_speech":True}]}
    result=plan_boundary_semantic_reviews([left,right],chunks(),policy=P,activity_by_chunk=activity)
    assert len(result['cases'])==1
    candidate=result['cases'][0]
    assert candidate['suggested_review_tier']=='reasoned_text'
    assert candidate['left_lexical_correspondence']['matched_words']==5
    assert candidate['proposed_source_action'].startswith('none')
    assert (left,right)==original

def test_unrepresented_neighbor_with_no_lexical_match_is_light_review():
    right=[w(x,80+i*.3,i,1) for i,x in enumerate(["turn","the","light","on"])]
    case=plan_boundary_semantic_reviews([[],right],chunks(90),policy=P)['cases'][0]
    assert case['suggested_review_tier']=='light_text'
    assert case['nominal_core_word_coverage_ratio']==0

def test_existing_core_coverage_and_short_or_character_controls_are_not_promoted():
    left=[w(x,80+i*.25,i,0) for i,x in enumerate(["the","river","is","deep"])]
    right=[w(x,80+i*.25,i,1) for i,x in enumerate(["the","river","is","deep"])]
    assert not plan_boundary_semantic_reviews([left,right],chunks(90),policy=P)['cases']
    for unit in right:unit['unit_type']='character'
    assert not plan_boundary_semantic_reviews([[],right],chunks(90),policy=P)['cases']
    assert not plan_boundary_semantic_reviews([[],right[:1]],chunks(90),policy=P)['cases']

def test_reasoner_cannot_automatically_promote_or_claim_audio_from_text_only():
    right=[w(x,80+i*.25,i,1) for i,x in enumerate(["there","is","no","exit"])]
    q=plan_boundary_semantic_reviews([[],right],chunks(90),policy=P)
    r={"schema":"nemovox-semantic-review-decision/v1","request_sha256":q['request_sha256'],
       "case_id":q['cases'][0]['case_id'],"decision":"adopt_neighbor_context",
       "rationale":"Looks plausible, needs audio.","supporting_evidence":["right_context_words"],
       "contrary_evidence":[]}
    with pytest.raises(ValueError,match='Cannot certify neighbor speech'):
        validate_semantic_review(q,r)
    r['independent_acoustic_support']=True
    r['supporting_evidence'].append('independent_audio_review')
    r['independent_audio_review_sha256']='a'*64
    assert validate_semantic_review(q,r)['source_application_authorized'] is False
    r['request_sha256']='0'*64
    with pytest.raises(ValueError,match='Stale'):validate_semantic_review(q,r)

def test_bad_policy_or_untrusted_review_evidence_fails():
    with pytest.raises(ValueError,match='thresholds'):
        plan_boundary_semantic_reviews([[],[]],chunks(),policy={**P,'max_interword_gap_s':0})
    q=plan_boundary_semantic_reviews([[],[w(x,80+i*.25,i,1) for i,x in enumerate(['do','not','go'])]],chunks(90),policy=P)
    r={"schema":"nemovox-semantic-review-decision/v1","request_sha256":q['request_sha256'],
       "case_id":q['cases'][0]['case_id'],"decision":"retain_core",
       "rationale":"review","supporting_evidence":["fabricated_source"],"contrary_evidence":[]}
    with pytest.raises(ValueError,match='evidence citations'):validate_semantic_review(q,r)


def test_boundary_pipeline_surfaces_review_candidate_without_auto_speech_insertion():
    specs=[ChunkSpec(0,0,90,0,100),ChunkSpec(1,90,180,80,190)]
    right=[{"text":x,"start":80+i*.30,"end":80.18+i*.30,
            "speaker":"SPK_RIGHT","valid_timing":True,
            "alignment_status":"ctc_aligned","unit_type":"word"} for i,x in
           enumerate(("there","is","no","known","passage"))]
    merged, diag = reconcile_chunk_boundaries([[],right],specs)
    assert merged == []  # No automatic word invention or insertion
    review=diag["semantic_review_candidates"]
    assert review["cases"] and review["cases"][0]["status"]=="review_required_not_auto_adopted"
    assert review["raw_source_modified"] is False


def test_injected_tiered_text_client_is_advisory_and_bounded():
    from nvidia_transcribe.semantic_review import run_tiered_text_reviews
    import json
    right=[w(x,80+i*.25,i,1) for i,x in enumerate(['you','should','not','go'])]
    q=plan_boundary_semantic_reviews([[],right],chunks(90),policy=P)
    class Stub:
        def __init__(self):self.calls=[]
        def complete(self,**kwargs):
            self.calls.append(kwargs)
            return {'choices':[{'message':{'content':json.dumps({
                'schema':'nemovox-semantic-review-decision/v1',
                'request_sha256':q['request_sha256'],'case_id':q['cases'][0]['case_id'],
                'decision':'unresolved','rationale':'No verified audio, cannot infer source',
                'supporting_evidence':[],'contrary_evidence':[], 'unresolved':['acoustics']})}}]}
    client=Stub()
    out=run_tiered_text_reviews(q,client=client,light_tokens=256,reasoned_tokens=512,token_budget=800)
    assert out['model_calls']==1
    assert out['outcomes'][0]['assessment']['source_application_authorized'] is False
    assert client.calls[0]['reasoning_effort']=='low'
    assert out['source_or_subtitles_modified'] is False
    with pytest.raises(ValueError,match='budgets'):
        run_tiered_text_reviews(q,client=client,token_budget=100,light_tokens=512)


def test_interleaved_left_voice_is_matched_in_its_own_lexical_order():
    # A's coherent cross-chunk phrase is interleaved in CTC lexical order
    # with a genuinely separate B speaker. Never build a fake global stream.
    words=["nothing","important","should","be","lost"]
    left=[]
    for i,txt in enumerate(words):
        left.append(w(txt,303+i*.32,2*i,0,"A"))
        left.append(w("interjection"+str(i),304+i*.32,2*i+1,0,"B"))
    right=[w(txt,355+i*.28,i,1,"RIGHT") for i,txt in enumerate(words)]
    case=plan_boundary_semantic_reviews([left,right],chunks(),policy=P)["cases"][0]
    match=case["left_lexical_correspondence"]
    assert match["matched_words"]==len(words)
    assert match["left_local_voice_hypothesis"]=="A"
    assert [unit["text"] for unit in match["left_words"]]==words
    assert case["proposed_source_action"].startswith("none")


def test_reasoned_model_uses_supported_effort_and_enforced_final_headroom():
    from nvidia_transcribe.semantic_review import run_tiered_text_reviews
    import json
    right=[w(x,355+i*.3,i,1) for i,x in enumerate(["It","does","not","work"])]
    request=plan_boundary_semantic_reviews([[],right],chunks(),policy=P,
        activity_by_chunk={0:[{"start_s":355.0,"end_s":356.5,
                               "possible_unrepresented_speech":True}]})
    class Stub:
        def __init__(self): self.calls=[]
        def complete(self,**kwargs):
            self.calls.append(kwargs)
            return {"choices":[{"finish_reason":"stop","message":{"content":json.dumps({
                "schema":"nemovox-semantic-review-decision/v1",
                "request_sha256":request["request_sha256"],
                "case_id":request["cases"][0]["case_id"],
                "decision":"unresolved","rationale":"Acoustic event not independently verified",
                "supporting_evidence":["right_context_words"],"contrary_evidence":[],
                "unresolved":["acoustic_occurrence"]})}}],
                "usage":{"completion_tokens":440,"completion_tokens_details":{"reasoning_tokens":240}}}
    stub=Stub()
    batch=run_tiered_text_reviews(request,client=stub,token_budget=1280,
        light_tokens=512,reasoned_tokens=1280,
        light_thinking_budget=128,reasoned_thinking_budget=512)
    assert batch["outcomes"][0]["status"]=="advisory_only"
    assert batch["outcomes"][0]["telemetry"]["reserved_answer_headroom_tokens"]==768
    assert stub.calls[0]["reasoning_effort"]=="xhigh"
    assert stub.calls[0]["thinking_budget"]==512
    assert stub.calls[0]["strict_reasoning_effort"] is True
    assert batch["source_or_subtitles_modified"] is False


def test_thinking_starvation_cannot_be_reported_as_accepted_review():
    from nvidia_transcribe.semantic_review import run_tiered_text_reviews
    right=[w(x,355+i*.3,i,1) for i,x in enumerate(["No","speech","was","verified"])]
    req=plan_boundary_semantic_reviews([[],right],chunks(),policy=P)
    class Starved:
        def complete(self, **kw):
            return {"choices":[{"finish_reason":"length","message":{
                "content":"","reasoning_content":"unfinished analysis"}}],
                "usage":{"completion_tokens":512,
                         "completion_tokens_details":{"reasoning_tokens":512}}}
    outcome=run_tiered_text_reviews(req,client=Starved(),
        token_budget=512,light_tokens=512,reasoned_tokens=512,
        light_thinking_budget=128,reasoned_thinking_budget=128)["outcomes"][0]
    assert outcome["status"]=="unaccepted_model_response"
    assert "exhausted by reasoning" in outcome["error"]
    assert outcome["telemetry"]["reasoning_tokens"]==512


def test_execution_config_is_bound_to_candidate_policy_hash():
    from nvidia_transcribe.semantic_review import run_configured_tiered_text_reviews
    import json
    cfg={**P,"model_review":{"case_limit":1,"token_budget":512,
            "light_tokens":512,"reasoned_tokens":512,
            "light_thinking_budget":128,"reasoned_thinking_budget":128,
            "light_effort":"low","reasoned_effort":"xhigh"}}
    right=[w(x,80+i*.20,i,1) for i,x in enumerate(["Speech","may","be","missing"])]
    req=plan_boundary_semantic_reviews([[],right],chunks(90),policy=cfg)
    class Fake:
        def complete(self,**kw):
            return {"choices":[{"message":{"content":json.dumps({
                "schema":"nemovox-semantic-review-decision/v1",
                "case_id":req["cases"][0]["case_id"],
                "request_sha256":req["request_sha256"],"decision":"unresolved",
                "rationale":"Need original audio","supporting_evidence":[],
                "contrary_evidence":[],"unresolved":["audio"]})}}]}
    batch=run_configured_tiered_text_reviews(req,client=Fake(),policy=cfg)
    assert batch["outcomes"][0]["status"]=="advisory_only"
    with pytest.raises(ValueError,match="pinned candidate plan"):
        run_configured_tiered_text_reviews(req,client=Fake(),
            policy={**cfg,"policy_id":"another-version"})


def test_narrow_legacy_reasoned_completion_uses_safe_default_thinking_cap():
    from nvidia_transcribe.semantic_review import run_tiered_text_reviews
    import json
    right=[w(txt,355+i*.25,i,1) for i,txt in enumerate(["Meaning","must","remain","clear"])]
    request=plan_boundary_semantic_reviews([[],right],chunks(),policy=P,
        activity_by_chunk={0:[{"start_s":355,"end_s":356,
                               "possible_unrepresented_speech":True}]})
    class Probe:
        def __init__(self): self.kwargs=None
        def complete(self,**kwargs):
            self.kwargs=kwargs
            return {"choices":[{"finish_reason":"stop","message":{"content":json.dumps({
                "schema":"nemovox-semantic-review-decision/v1",
                "request_sha256":request["request_sha256"],
                "case_id":request["cases"][0]["case_id"],
                "decision":"unresolved","rationale":"Need the real audio",
                "supporting_evidence":[],"contrary_evidence":[],
                "unresolved":["original_audio"]})}}]}
    stub=Probe()
    batch=run_tiered_text_reviews(request,client=stub,token_budget=512,
            light_tokens=512,reasoned_tokens=512)
    assert batch["outcomes"][0]["status"]=="advisory_only"
    assert stub.kwargs["thinking_budget"]==384
    assert stub.kwargs["max_tokens"]==512
    with pytest.raises(ValueError,match="reserve at least"):
        run_tiered_text_reviews(request,client=stub,token_budget=512,
            light_tokens=512,reasoned_tokens=512,reasoned_thinking_budget=512)


def test_completion_limit_blocks_even_valid_structured_but_unfinished_review():
    from nvidia_transcribe.semantic_review import run_tiered_text_reviews
    import json
    right=[w(txt,80+i*.20,i,1) for i,txt in enumerate(["This","may","be","incomplete"])]
    q=plan_boundary_semantic_reviews([[],right],chunks(90),policy=P)
    class Truncated:
        def complete(self,**kwargs):
            return {"choices":[{"finish_reason":"length","message":{"content":json.dumps({
                "schema":"nemovox-semantic-review-decision/v1",
                "request_sha256":q["request_sha256"],"case_id":q["cases"][0]["case_id"],
                "decision":"unresolved","rationale":"Need audio","supporting_evidence":[],
                "contrary_evidence":[],"unresolved":["audio"]})}}]}
    outcome=run_tiered_text_reviews(q,client=Truncated(),token_budget=512,
        light_tokens=512,reasoned_tokens=512)["outcomes"][0]
    assert outcome["status"]=="unaccepted_model_response"
    assert "output limit" in outcome["error"]
