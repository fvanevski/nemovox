"""Synthetic reversible replacement checks; only independent reviewed source turns."""
from copy import deepcopy
import pytest
import nvidia_transcribe.source_fusion_application as a
from nvidia_transcribe.evidence_fusion import _digest

SHA='f'*64

def source_inputs(monkeypatch, *, events=2):
    old={'id':'old','source_text':'unreliable old mixed turn', 'source_start':8,'source_end':9,
         'source_words':[{'text':'Yes','chunk_index':0,'source_unit_index':0},
                         {'text':'No','chunk_index':0,'source_unit_index':1}]}
    recording={'schema_version':'nemotron-recording-contract/v1',
               'source':{'fingerprint_sha256':SHA,'duration_s':90},'source_language':'fr',
               'utterances':[old, {'id':'clean','source_text':'clean','source_start':60,'source_end':61}]}
    ledger={'lexical_observations':[{'id':f'raw{i}','chunk_index':0,'source_ordinal':i,'text':w}
                                    for i,w in enumerate(['Yes','No'])],
            'asr_observations':[{'id':'asr0','chunk_index':0,'infer_start_s':0,'infer_end_s':90}]}
    component='component1'
    eids=['ev1','ev2'] if events==2 else ['ev1']
    groups=[{'audio_event_id':eid,'source_unit_ids':[f'raw{i}']} for i,eid in enumerate(eids)]
    if events==1: groups[0]['source_unit_ids']=['raw0','raw1']
    plan={'schema':'nemovox-reviewed-unit-partitions/v1','media_fingerprint_sha256':SHA,
          'source_language':'fr','recording_content_sha256':None,
          'source_ledger_content_sha256':None,'reconciliation_dossier_content_sha256':None,
          'connected_components':[{'id':component,'reviewed_event_ids':eids,'legacy_utterance_ids':['old'],
                                   'status':'complete_reviewed_component_requires_application_gate'}],
          'legacy_utterances':[{'legacy_utterance_id':'old','reviewed_event_groups':groups}]}
    monkeypatch.setattr(a,'plan_reviewed_source_partitions',lambda *args: deepcopy(plan))
    turns=[];regions=[]
    for i,eid in enumerate(eids):
        t0=float(40+i*2);t1=t0+1;words=['Oui!','Non!'] if events==2 else ['Oui non!']
        text=words[i]; revision=a._canonical_id('src_revision',[SHA,eid,text,'a'*64,t0,t1]);tid=a._canonical_id('src_turn',[SHA,eid])
        turns.append({'schema':'nemovox-source-turn-revision/v1','audio_event_id':eid,
                      'source_turn_id':tid,'source_revision_id':revision,'source_text':text,
                      'source_language':'fr','source_start':t0,'source_end':t1,
                      'source_status':'audio_review_attested_source','selection_basis':'independent_audio_review',
                      'reviewer_has_attested_direct_audio_listening':True,
                      'review_evidence_sha256':'a'*64})
        regions.append({'audio_event_id':eid,'source_status':'audio_review_attested',
                        'positive_audio_event_unit_ids':list(groups[i]['source_unit_ids']),
                        'review_audio_crop_provenance':{'sha256':'b'*64,'start_s':39,'end_s':47,
                                                       'event_fully_covered_by_review_audio':True}})
    dossier={'source_turns_audio_review_attested':turns,'audited_audio_regions':regions}
    plan['recording_content_sha256']=_digest(recording)
    plan['source_ledger_content_sha256']=_digest(ledger)
    plan['reconciliation_dossier_content_sha256']=_digest(dossier)
    return recording,ledger,dossier,plan


def test_opt_in_reversible_split_and_untrusted_old_ctc_not_used(monkeypatch):
    recording,ledger,dossier,plan=source_inputs(monkeypatch)
    original=deepcopy(recording)
    plan_sha=_digest(plan)
    preview=a.build_reviewed_source_revision(recording,ledger,dossier,
        selected_component_ids=['component1'], expected_plan_sha256=plan_sha)
    assert preview['new_recording'] is None
    assert preview['report']['source_unit_observations_deleted']==0
    activated=a.build_reviewed_source_revision(recording,ledger,dossier,
        selected_component_ids=['component1'], expected_plan_sha256=plan_sha,enable=True)
    output=activated['new_recording']
    assert recording==original
    assert [u['id'] for u in output['utterances']] == [a._canonical_id('src_turn',[SHA,'ev1']),a._canonical_id('src_turn',[SHA,'ev2']),'clean']
    assert all(u['speaker_id']=='SPK_UNK' for u in output['utterances'][:2])
    assert all(u['source_words']==[] and u['alignment_status']=='audio_event_bounds_only_word_timing_unverified'
               for u in output['utterances'][:2])
    assert {unit for u in output['utterances'][:2] for unit in u['source_unit_lineage_ids']} == {'raw0','raw1'}
    assert output['utterances'][0]['source_start']==40
    assert output['source_fusion_revision']['source_unit_observations_deleted']==0


def test_one_reviewed_event_can_absorb_two_old_units(monkeypatch):
    recording,ledger,dossier,plan=source_inputs(monkeypatch,events=1)
    result=a.build_reviewed_source_revision(recording,ledger,dossier,selected_component_ids=['component1'],
                                            expected_plan_sha256=_digest(plan),enable=True)
    assert len(result['new_recording']['utterances'])==2
    assert result['new_recording']['utterances'][0]['source_unit_lineage_count']==2


def test_incomplete_or_stale_or_unadmitted_event_cannot_activate(monkeypatch):
    recording,ledger,dossier,plan=source_inputs(monkeypatch)
    with pytest.raises(ValueError,match='Stale'):
        a.build_reviewed_source_revision(recording,ledger,dossier,
            selected_component_ids=['component1'],expected_plan_sha256='0'*64,enable=True)
    plan['connected_components'][0]['status']='blocked_incomplete_reviewed_component'
    monkeypatch.setattr(a,'plan_reviewed_source_partitions',lambda *args:deepcopy(plan))
    with pytest.raises(ValueError,match='incomplete'):
        a.build_reviewed_source_revision(recording,ledger,dossier,
            selected_component_ids=['component1'],expected_plan_sha256=_digest(plan),enable=True)


def test_missing_crop_wrong_event_revision_and_nonoverlapping_parent_fail(monkeypatch):
    recording,ledger,dossier,plan=source_inputs(monkeypatch)
    digest=_digest(plan)
    dossier['audited_audio_regions'][0]['review_audio_crop_provenance']['end_s']=39.5
    plan['reconciliation_dossier_content_sha256']=_digest(dossier)
    digest=_digest(plan)
    with pytest.raises(ValueError,match='crop'):
        a.build_reviewed_source_revision(recording,ledger,dossier,
            selected_component_ids=['component1'],expected_plan_sha256=digest,enable=True)
    dossier['audited_audio_regions'][0]['review_audio_crop_provenance']['end_s']=47
    dossier['source_turns_audio_review_attested'][0]['source_revision_id']='bad'
    plan['reconciliation_dossier_content_sha256']=_digest(dossier)
    digest=_digest(plan)
    with pytest.raises(ValueError,match='revision'):
        a.build_reviewed_source_revision(recording,ledger,dossier,
            selected_component_ids=['component1'],expected_plan_sha256=digest,enable=True)
    dossier['source_turns_audio_review_attested'][0]['source_revision_id']=a._canonical_id('src_revision',[
        SHA,'ev1','Oui!','a'*64,40.0,41.0])
    ledger['asr_observations'][0]['infer_end_s']=25
    plan['source_ledger_content_sha256']=_digest(ledger)
    plan['reconciliation_dossier_content_sha256']=_digest(dossier)
    digest=_digest(plan)
    with pytest.raises(ValueError,match='outside original'):
        a.build_reviewed_source_revision(recording,ledger,dossier,
            selected_component_ids=['component1'],expected_plan_sha256=digest,enable=True)


def test_selected_component_cannot_claim_missing_or_repeated_unit(monkeypatch):
    recording,ledger,dossier,plan=source_inputs(monkeypatch)
    digest=_digest(plan)
    dossier['audited_audio_regions'][0]['positive_audio_event_unit_ids'].append('raw1')
    plan['reconciliation_dossier_content_sha256']=_digest(dossier)
    digest=_digest(plan)
    with pytest.raises(ValueError,match='Incomplete or duplicate'):
        a.build_reviewed_source_revision(recording,ledger,dossier,
            selected_component_ids=['component1'],expected_plan_sha256=digest,enable=True)


def test_reviewed_alternates_are_preserved_as_evidence_not_extra_subtitle_units(monkeypatch):
    recording,ledger,dossier,plan=source_inputs(monkeypatch,events=1)
    ledger['lexical_observations'].append(
        {'id':'raw_alternate','chunk_index':0,'source_ordinal':2,'text':'Yes'})
    ledger['asr_observations'][0]['infer_end_s']=90
    dossier['audited_audio_regions'][0]['reviewed_alternate_source_observations']=[{
        'source_unit_id':'raw_alternate','preferred_owned_source_unit_id':'raw0',
        'status':'same_occurrence_reviewed_alternate_not_an_output_unit',
        'review_evidence_sha256':'a'*64}]
    plan['source_ledger_content_sha256']=_digest(ledger)
    plan['reconciliation_dossier_content_sha256']=_digest(dossier)
    result=a.build_reviewed_source_revision(recording,ledger,dossier,
        selected_component_ids=['component1'],expected_plan_sha256=_digest(plan),enable=True)
    turn=result['new_recording']['utterances'][0]
    assert turn['source_unit_lineage_ids']==['raw0','raw1']
    assert turn['reviewed_alternate_source_observations'][0]['source_unit_id']=='raw_alternate'
    assert turn['source_words']==[]
    assert result['report']['source_unit_observations_deleted']==0
    assert len(recording['utterances'])==2
    dossier['audited_audio_regions'][0]['reviewed_alternate_source_observations'][0][
        'preferred_owned_source_unit_id']='unowned_owner'
    plan['reconciliation_dossier_content_sha256']=_digest(dossier)
    with pytest.raises(ValueError,match='lacks source-unit ownership'):
        a.build_reviewed_source_revision(recording,ledger,dossier,
            selected_component_ids=['component1'],expected_plan_sha256=_digest(plan),enable=True)
