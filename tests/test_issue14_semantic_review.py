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
       "rationale":"Looks plausible, needs audio.","supporting_evidence":["right_context_words"]}
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
       "rationale":"review","supporting_evidence":["fabricated_source"]}
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
