"""No live model or media: invariant tests for new opt-in CLI runner."""
import json
from pathlib import Path
import pytest
import nvidia_transcribe.reviewed_revision_runner as runner
from nvidia_transcribe.evidence_fusion import _digest


def setup_case(tmp_path, monkeypatch):
    base = {'utterances': [{'id': 'old', 'source_text': 'one'}]}
    ledger = {'lexical_observations': [{'id': 'raw'}]}
    dossier = {'audited_audio_regions': [{'audio_event_id': 'e'}]}
    plan = {'connected_components': [{'id': 'ready', 'status': 'complete_reviewed_component_requires_application_gate'},
                                      {'id': 'blocked', 'status': 'blocked_incomplete_reviewed_component'}]}
    selection = {'schema': runner.SELECTION_SCHEMA, 'selected_component_ids': ['ready'],
                 'expected_plan_sha256': _digest(plan),
                 'recording_content_sha256': _digest(base), 'ledger_content_sha256': _digest(ledger),
                 'dossier_content_sha256': _digest(dossier)}
    for name, obj in {'original': base, 'ledger': ledger, 'dossier': dossier, 'selection': selection,
                      'previous': {'target_language': 'en'}, 'receipts': {'receipts': [{'id': 'new'}]}}.items():
        (tmp_path / (name + '.json')).write_text(json.dumps(obj))
    monkeypatch.setattr(runner, 'plan_reviewed_source_partitions', lambda *_: plan)
    def revision(*args, **kwargs):
        return {'report': {'some': 'provenance'},
                'new_recording': {'utterances': [{'id':'new', 'source_text': 'two'}]} if kwargs['enable'] else None}
    monkeypatch.setattr(runner, 'build_reviewed_source_revision', revision)
    def translation(*args, **kwargs):
        return {'report': {'missing_fresh_translation_ids': []},
                'translated_contract': {'target_language': 'en', 'utterances': [{'id': 'new', 'translated_text':'Two'}]}
                   if kwargs['enable'] else None}
    monkeypatch.setattr(runner, 'assemble_revised_translation', translation)
    def render(doc, *, output_dir, episode_stem, target_language, formats, **kwargs):
        result={}
        for fmt in formats:
            path=Path(output_dir)/(episode_stem+'.'+target_language+'.'+fmt)
            path.write_text('a review-safe cue')
            result[fmt]=str(path)
        return {'outputs':result, 'stats':{'renderable_cues':1}}
    monkeypatch.setattr(runner, 'render_subtitles', render)
    args=dict(original_recording_path=tmp_path/'original.json', source_evidence_path=tmp_path/'ledger.json',
              review_dossier_path=tmp_path/'dossier.json', selection_path=tmp_path/'selection.json',
              previous_translated_path=tmp_path/'previous.json', translation_receipts_path=tmp_path/'receipts.json',
              episode_stem='episode', output_dir=tmp_path/'new-output')
    return args


def test_preview_never_creates_outputs(tmp_path, monkeypatch):
    args=setup_case(tmp_path, monkeypatch)
    result=runner.run_reviewed_revision(**args)
    assert result['status']=='preview' and result['applied'] is False
    assert not args['output_dir'].exists()
    assert 'blocked' in result['unselected_blocked_components']


def test_apply_private_unique_output_and_exact_inputs(tmp_path, monkeypatch):
    args=setup_case(tmp_path, monkeypatch)
    before=(tmp_path/'original.json').read_bytes()
    out=runner.run_reviewed_revision(**args,apply=True)
    assert out['applied'] is True and out['render_stats']['renderable_cues']==1
    assert (args['output_dir']/'reviewed.recording.contract.json').is_file()
    assert (args['output_dir']/'reviewed.translated.contract.json').is_file()
    assert (args['output_dir']/'episode.en.srt').is_file()
    assert (tmp_path/'original.json').read_bytes()==before
    with pytest.raises(FileExistsError):
        runner.run_reviewed_revision(**args,apply=True)


def test_selection_stale_or_incomplete_blocks_without_output(tmp_path, monkeypatch):
    args=setup_case(tmp_path, monkeypatch)
    path=tmp_path/'selection.json'; s=json.loads(path.read_text());s['expected_plan_sha256']='0'*64;path.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='stale plan'):
        runner.run_reviewed_revision(**args,apply=True)
    s['expected_plan_sha256']=_digest(runner.plan_reviewed_source_partitions({}, {}, {}))
    s['selected_component_ids']=['blocked'];path.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='incomplete'):
        runner.run_reviewed_revision(**args,apply=True)
    assert not args['output_dir'].exists()


def test_missing_fresh_receipt_or_invalid_output_stem_rejected(tmp_path, monkeypatch):
    args=setup_case(tmp_path, monkeypatch)
    def missing(*args, **kwargs):
        return {'report':{'missing_fresh_translation_ids':['new']},'translated_contract':None}
    monkeypatch.setattr(runner,'assemble_revised_translation',missing)
    with pytest.raises(ValueError,match='receipts are missing'):
        runner.run_reviewed_revision(**args,apply=True)
    with pytest.raises(ValueError,match='filename component'):
        runner.run_reviewed_revision(**dict(args,episode_stem='../unsafe'),apply=True)
    assert not args['output_dir'].exists()


def test_render_failure_leaves_no_accepted_output_and_cleans_temporary_root(tmp_path, monkeypatch):
    args=setup_case(tmp_path, monkeypatch)
    def render_fails(*a, **kw):
        (kw['output_dir']/'incomplete.srt').write_text('not accepted')
        raise RuntimeError('simulated renderer failure')
    monkeypatch.setattr(runner, 'render_subtitles', render_fails)
    with pytest.raises(RuntimeError,match='simulated renderer failure'):
        runner.run_reviewed_revision(**args,apply=True)
    assert not args['output_dir'].exists()
    assert not list(tmp_path.glob('.reviewed-revision-*'))
