"""Generic evidence-fusion contracts: no model, no media, no case-name routing."""
import copy
import json
from pathlib import Path
import pytest
from nvidia_transcribe.evidence_fusion import (
    adjudicate_review, plan_reviewed_source_partitions, prepare_evidence_review,
)

SHA = 'f' * 64


def observation(oid, modality, family, claim, *, parents=()):
    return {
        'id': oid, 'modality': modality, 'claim_kind': claim, 'subject_id': 'event0',
        'source_family': family, 'media_fingerprint_sha256': SHA,
        'artifact_sha256': 'e' * 64, 'interval_s': [3, 5],
        'method': 'independent_local_observation', 'derived_from': list(parents),
        'assertion': {'visible_or_audible': True},
    }


def case(scene='animated', sync='unverified'):
    return {'schema': 'nemovox-evidence-case/v1', 'case_id': 'sample_01',
            'media_fingerprint_sha256': SHA, 'duration_s': 20,
            'context': {'scene_kind': scene, 'av_sync_status': sync},
            'observations': [
                observation('a', 'audio', 'original_audio', 'independently_reviewed_voice_change'),
                observation('v', 'video', 'original_video', 'mouth_motion_speaker_correspondence'),
                observation('model', 'fusion', 'model_output', 'speaker_change', parents=('a', 'v')),
            ], 'candidates': [{'id': 'split', 'action': 'split_speaker_turn'},
                              {'id': 'retain', 'action': 'retain_unresolved'}]}


def policy():
    return {'schema': 'nemovox-evidence-policy/v1', 'policy_id': 'test',
            'reviewer_guidance': 'Compare counterevidence and abstain.', 'rules': [
                {'match': {'claim_kind': 'mouth_motion_speaker_correspondence', 'scene_kind': 'animated'},
                 'status': 'limited', 'salience': .4, 'rationale': 'Animation need not lip sync.'},
                {'match': {'claim_kind': 'mouth_motion_speaker_correspondence', 'scene_kind': 'live_action', 'av_sync_status': 'verified'},
                 'status': 'consider', 'salience': 1.5, 'rationale': 'Better physical mouth correspondence.'},
                {'match': {'claim_kind': 'mouth_motion_speaker_correspondence', 'av_sync_status': 'unverified'},
                 'priority': 1, 'status': 'limited', 'salience': .1, 'rationale': 'Audio/video sync unverified.'},
            ]}


def test_context_sensitive_policy_and_derivation_not_double_counted():
    c = case()
    original = copy.deepcopy(c)
    p = policy()
    out = prepare_evidence_review(c, p)
    ev = {x['id']: x for x in out['observations']}
    assert ev['v']['review_salience'] == .1
    assert ev['model']['root_source_families'] == ['original_audio', 'original_video']
    assert c == original
    live = prepare_evidence_review(case('live_action', 'verified'), p)
    assert {x['id']: x for x in live['observations']}['v']['review_salience'] == 1.5
    assert {x['id']: x for x in live['observations']}['model']['root_source_families'] == ['original_audio', 'original_video']


def test_model_review_preserves_counterevidence_and_has_no_mutation_authority():
    r = prepare_evidence_review(case(), policy())
    response = {'schema': 'nemovox-evidence-model-review/v1', 'case_id': 'sample_01',
                'case_sha256': r['case_sha256'], 'policy_sha256': r['policy_sha256'],
                'decisions': [
                    {'candidate_id': 'split', 'decision': 'support', 'supporting_evidence_ids': ['a', 'model'],
                     'contrary_evidence_ids': ['v'], 'rationale': 'Audio supports a change, lip motion imperfect.', 'unresolved': ['identity']},
                    {'candidate_id': 'retain', 'decision': 'reject', 'supporting_evidence_ids': [],
                     'contrary_evidence_ids': ['a'], 'rationale': 'Turn distinction likely.', 'unresolved': []},
                ]}
    out = adjudicate_review(r, response)
    assert out['decisions'][0]['independent_root_families'] == ['original_audio', 'original_video']
    assert out['decisions'][0]['correction_application'] == 'not_authorized_evidence_review_only'
    response['policy_sha256'] = '0'*64
    with pytest.raises(ValueError, match='Stale'):
        adjudicate_review(r, response)


def test_cycle_cross_media_and_excluded_evidence_fail_closed():
    c = case()
    c['observations'][0]['derived_from'] = ['model']
    with pytest.raises(ValueError, match='Cyclic'):
        prepare_evidence_review(c, policy())
    c = case()
    c['observations'][0]['media_fingerprint_sha256'] = 'a'*64
    with pytest.raises(ValueError, match='Cross-media'):
        prepare_evidence_review(c, policy())
    p = policy()
    p['rules'].append({'match': {'claim_kind': 'independently_reviewed_voice_change'},
                       'status': 'exclude', 'rationale': 'Test exclusion'})
    r = prepare_evidence_review(case(), p)
    response = {'schema': 'nemovox-evidence-model-review/v1', 'case_id': 'sample_01',
                'case_sha256': r['case_sha256'], 'policy_sha256': r['policy_sha256'],
                'decisions': [{'candidate_id': cid, 'decision': 'support',
                               'supporting_evidence_ids': ['a'], 'contrary_evidence_ids': [],
                               'rationale': 'unsupported claim', 'unresolved': []}
                              for cid in ('split','retain')]}
    with pytest.raises(ValueError, match='policy-excluded'):
        adjudicate_review(r, response)


def test_unknown_dependencies_and_malformed_time_rejected():
    c=case()
    c['observations'][2]['derived_from']=['fictional']
    with pytest.raises(ValueError, match='Unknown/self'):
        prepare_evidence_review(c, policy())
    c=case()
    c['observations'][0]['interval_s']=[15,30]
    with pytest.raises(ValueError, match='outside'):
        prepare_evidence_review(c,policy())


def test_policy_config_example_is_valid_and_animation_rule_evaluable():
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/evidence-fusion.example.json').read_text())
    out=prepare_evidence_review(case(),config)
    assert {x['id']:x for x in out['observations']}['v']['applicability']=='limited'
    assert out['policy_sha256']


def test_complete_reviewed_unit_partition_supports_one_to_many_without_application(monkeypatch):
    import sys
    import types
    dummy = types.ModuleType('nvidia_transcribe.source_unit_lineage')
    def binder(recording, ledger, dossier):
        return {'media_fingerprint_sha256': SHA, 'source_language':'fr',
                'recording_content_sha256':'0'*64, 'source_ledger_content_sha256':'1'*64,
                'reconciliation_dossier_content_sha256':'2'*64,
                'legacy_utterances':[{'legacy_utterance_id':'old',
                                      'status':'partial_or_multiple_reviewed_audio_events'}]}
    dummy.bind_legacy_source_lineage = binder
    monkeypatch.setitem(sys.modules, 'nvidia_transcribe.source_unit_lineage', dummy)
    recording={'utterances':[{'id':'old','source_words':[
        {'chunk_index':0,'source_unit_index':0,'text':'Yes'},
        {'chunk_index':0,'source_unit_index':1,'text':'No'}]}]}
    ledger={'lexical_observations':[
        {'id':'raw0','chunk_index':0,'source_ordinal':0,'text':'Yes'},
        {'id':'raw1','chunk_index':0,'source_ordinal':1,'text':'No'}]}
    dossier={'source_turns_audio_review_attested':[{'audio_event_id':'a'},{'audio_event_id':'b'}],
            'audited_audio_regions':[
                {'audio_event_id':'a', 'positive_audio_event_unit_ids':['raw0']},
                {'audio_event_id':'b', 'positive_audio_event_unit_ids':['raw1']} ]}
    old=copy.deepcopy(recording)
    result=plan_reviewed_source_partitions(recording,ledger,dossier)
    assert result['diagnostics']['fully_partitioned']==1
    assert len(result['legacy_utterances'][0]['reviewed_event_groups'])==2
    assert result['legacy_utterances'][0]['replacement_recording_created'] is False
    assert recording==old
    dossier['audited_audio_regions'][1]['positive_audio_event_unit_ids']=[]
    incomplete=plan_reviewed_source_partitions(recording,ledger,dossier)
    assert incomplete['diagnostics']['fully_partitioned']==0
    assert incomplete['diagnostics']['missing_or_unreviewed']==1
    dossier['audited_audio_regions'][1]['positive_audio_event_unit_ids']=['raw0']
    with pytest.raises(ValueError, match='multiple'):
        plan_reviewed_source_partitions(recording,ledger,dossier)
