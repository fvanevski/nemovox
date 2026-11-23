"""No external Whisper model, no private audio and no world-level truth claims."""
import copy
import pytest
from nvidia_transcribe.source_challenger_timing import compare_independent_word_timing

SHA = '1' * 64
MEDIA = 'f' * 64
TOKENS = ['the', 'hidden', 'valley', 'has', 'ancient', 'gardens']

def fixture(*, ctc_start=12.0, word_start=52.0, repeat=False):
    source = {'schema_version':'nemovox-source-evidence/v1',
              'media_fingerprint_sha256':MEDIA,'source_language':'fr',
              'asr_observations':[{'id':'asr0','chunk_index':0,'raw_response_sha256':SHA,'infer_start_s':0,'infer_end_s':100}],
              'lexical_observations':[
                  {'id':f'unit{i}','chunk_index':0,'source_ordinal':i,
                   'asr_observation_id':'asr0','unit_type':'word',
                   'text':tok,'ctc_start_s':ctc_start+i*.3,'ctc_rejected_start_s':None}
                  for i,tok in enumerate(TOKENS*(2 if repeat else 1))]}
    frozen={'windows':[{'case_id':'fr0','language':'fr','media_fingerprint_sha256':MEDIA,
                        'start_s':50,'end_s':58,'wav_sha256':'2'*64}]}
    model={'whisper_large_v3':{'repo_id':'Systran/faster-whisper-large-v3', 'revision':'a'*40,
                              'cases':{'fr0':{'language':'fr','segments':[{'words':[
                                  {'word':tok, 'abs_start_s':word_start+i*.3,
                                   'abs_end_s':word_start+i*.3+.2}
                                  for i,tok in enumerate(TOKENS)]}]}}}}
    return source,frozen,model


def test_distinctive_long_drift_yields_review_only_without_mutating_raw_source():
    src,win,model=fixture()
    original=copy.deepcopy(src)
    result=compare_independent_word_timing(src,win,model)
    assert len(result['candidates'])==1
    c=result['candidates'][0]
    assert c['classification']=='displaced_ctc_candidate'
    assert c['median_ctc_minus_challenger_s']==pytest.approx(-40.0)
    assert c['source_unit_ids']==[f'unit{i}' for i in range(len(TOKENS))]
    assert c['source_raw_response_sha256']==SHA
    assert c['challenger_crop_sha256']=='2'*64
    assert c['timing_accepted'] is False and c['speaker_identity_accepted'] is False
    assert c['requires_positive_acoustic_review_before_promotion'] is True
    assert src == original


def test_clean_timing_is_only_concordant_and_unmatched_music_hallucination_no_promotion():
    src,win,model=fixture(ctc_start=52.0)
    out=compare_independent_word_timing(src,win,model)
    assert out['candidates'][0]['classification']=='concordant_timestamp_observations'
    model['whisper_large_v3']['cases']['fr0']['segments'][0]['words']=[
        {'word':w,'abs_start_s':52+i*.3,'abs_end_s':52+i*.3+.2}
        for i,w in enumerate(['subtitles','provided','by','community','credits','here'])]
    out=compare_independent_word_timing(src,win,model)
    assert not out['candidates']
    assert out['unmatched_cases'][0]['reason']=='no_unique_corresponding_source_phrase'


def test_repeated_identical_phrase_does_not_mint_unique_anchor():
    src,win,model=fixture(repeat=True)
    assert not compare_independent_word_timing(src,win,model)['candidates']


def test_refuses_wrong_media_or_clip_span_and_cross_language_review():
    src,win,model=fixture()
    win['windows'][0]['media_fingerprint_sha256']='different'
    assert compare_independent_word_timing(src,win,model)['candidates']==[]
    src,win,model=fixture()
    model['whisper_large_v3']['cases']['fr0']['segments'][0]['words'][0]['abs_start_s']=38
    with pytest.raises(ValueError,match='outside authenticated WAV'):
        compare_independent_word_timing(src,win,model)
    src,win,model=fixture()
    src['lexical_observations'][0]['asr_observation_id']='forged'
    with pytest.raises(ValueError,match='unmatched raw ASR owner'):
        compare_independent_word_timing(src,win,model)


def test_unlocalized_raw_ctc_stays_provisional_not_word_time_replacement():
    src,win,model=fixture()
    for word in src['lexical_observations']:
        word['ctc_start_s']=None
        word['ctc_rejected_start_s']=None
    out=compare_independent_word_timing(src,win,model)
    assert out['candidates'][0]['classification']=='source_ctc_unlocalized'
    assert out['candidates'][0]['timing_accepted'] is False
