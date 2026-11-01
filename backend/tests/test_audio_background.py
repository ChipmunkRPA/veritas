from copy import deepcopy
import hashlib
import io

import numpy as np
from PIL import Image
import pytest

from app.core.canonical import canonical_json
from app.detectors.audio.decode import SAMPLE_RATE, read_wav
from app.evidence.audio_background import build_content, measurements, spectrogram_values, verify_source_content
from app.evidence.audio_packets import _chart


def build(pcm):
    pcm = (np.rint(np.asarray(pcm) * 32768).clip(-32768, 32767).astype('<i2').astype(np.float32) / 32768)
    sha = hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
    source = {'sha256': 'a'*64, 'pcm_sha256': sha, 'samples': len(pcm), 'sample_rate': SAMPLE_RATE, 'coordinate_space': 'analysed-mono-pcm'}
    split = len(pcm)//2
    mapping = {'samples': len(pcm), 'timeline_time_base':'1/16000', 'timeline': [
        {'start_sample': 0, 'end_sample': split, 'source_pts': 16000},
        {'start_sample': split, 'end_sample': len(pcm), 'source_pts': split+32000}]}
    assets = {}
    def asset(data, media_type, **metadata):
        sha = hashlib.sha256(data).hexdigest()
        assets[sha] = data
        return {'sha256': sha, 'media_type': media_type, 'url': '/assets/'+sha, **metadata}
    packets = [{'id':'A01', 'interval':{'start_sample':0, 'end_sample':len(pcm)}}]
    document = build_content(pcm, mapping, source, packets, asset, _chart)
    return document, assets, pcm, source, mapping, packets


def verify(fixture):
    doc, assets, pcm, source, mapping, packets = fixture
    verify_source_content(doc, pcm=pcm, mapping=mapping, source=source, voice_packets=packets, read_asset=assets.__getitem__, waveform=_chart)


def test_stable_tone_has_no_acoustic_event_and_tail_is_measured():
    time = np.arange(4*SAMPLE_RATE+17)/SAMPLE_RATE
    fixture = build(np.sin(2*np.pi*200*time)*.25)
    doc, assets, pcm, *_ = fixture
    assert doc['events'] == []
    assert doc['bins'][-1]['end_sample'] == len(pcm)
    assert sum(b['end_sample']-b['start_sample'] for b in doc['bins']) == len(pcm)
    assert doc['summary']['measured_samples'] == len(pcm)
    assert doc['bins'][1]['low_band_fraction'] > .99
    assert doc['bins'][1]['high_band_fraction'] < .001
    assert Image.open(io.BytesIO(assets[doc['overview']['energy']['sha256']])).size == (960,280)
    assert Image.open(io.BytesIO(assets[doc['overview']['spectrogram']['sha256']])).size == (960,320)
    verify(fixture)


def test_background_level_step_records_exact_bins_without_ai_claim():
    random = np.random.default_rng(241)
    pcm = random.normal(size=4*SAMPLE_RATE)*.002
    pcm[2*SAMPLE_RATE:] *= 12
    fixture = build(pcm)
    doc, assets, pcm, _, mapping, _ = fixture
    events = [x for x in doc['events'] if x['category']=='energy_change']
    assert len(events)==1
    event = events[0]
    assert event['change_sample'] == 32000
    assert event['interval'] == {'start_sample': 24000, 'end_sample': 40000}
    assert event['start_s']==1.5 and event['end_s']==2.5
    assert len(event['source_timestamp_ranges'])==2  # Actual source-clock gap survives.
    assert event['source_timestamp_ranges'][1]['source_start_pts']==64000
    assert 20 < event['observations'][0]['value'] < 24
    assert event['evidence_type']=='descriptive'
    assert event['related_voice_packet_ids']==['A01']
    np.testing.assert_array_equal(read_wav(assets[event['artifacts']['selected_clip']['sha256']]), pcm[24000:40000])
    assert 'same pattern' in event['interpretation']
    assert 'not been separated' in doc['limitations'][0]
    verify(fixture)


def test_silence_and_clipping_are_distinct_recording_context():
    fixture = build(np.concatenate([np.zeros(SAMPLE_RATE), np.ones(SAMPLE_RATE), np.zeros(SAMPLE_RATE)]))
    doc = fixture[0]
    assert [event['category'] for event in doc['events']]==['near_silence','clipping','near_silence']
    assert doc['summary']['clipped_samples']==SAMPLE_RATE
    assert doc['summary']['near_silence_samples']==2*SAMPLE_RATE
    assert doc['events'][1]['observations'][0]['value']==1
    verify(fixture)


def test_silence_has_finite_floor_and_no_synthesis_result():
    fixture = build(np.zeros(2*SAMPLE_RATE))
    doc = fixture[0]
    assert all(row['rms_dbfs']==-120 and row['spectral_flatness']==0 for row in doc['bins'])
    assert len(doc['events'])==1 and doc['events'][0]['category']=='near_silence'
    assert all('score' not in row for row in doc['bins'])
    assert b'NaN' not in canonical_json(doc)
    verify(fixture)


@pytest.mark.parametrize('mutation',['number','time','clip','chart','overlap','source'])
def test_recomputation_refuses_fabricated_context(mutation):
    fixture = build(np.concatenate([np.zeros(SAMPLE_RATE), np.ones(SAMPLE_RATE)*.2]))
    doc = fixture[0]
    if mutation=='number':
        doc['bins'][0]['rms_dbfs']=-20
    elif mutation=='time':
        doc['events'][0]['source_timestamp_ranges'][0]['source_start_pts']+=1
    elif mutation=='clip':
        doc['events'][0]['artifacts']['selected_clip']['sha256']='b'*64
    elif mutation=='chart':
        fixture[1][doc['overview']['energy']['sha256']]=b'changed pixels'
    elif mutation=='overlap':
        doc['events'][0]['related_voice_packet_ids']=[]
    else:
        doc['source']=deepcopy(doc['source'])
        doc['source']['sha256']='c'*64
    with pytest.raises(ValueError, match='Acoustic context'):
        verify(fixture)


def test_many_transitions_have_bounded_selection_and_record_omission():
    time = np.arange(SAMPLE_RATE//2)/SAMPLE_RATE
    tone = np.sin(2*np.pi*440*time)
    fixture = build(np.concatenate([tone*(.01 if i%2 else .4) for i in range(30)]))
    doc=fixture[0]
    assert len(doc['events'])==8 and doc['omitted_events']==21
    assert [row['start_s'] for row in doc['events']]==sorted(row['start_s'] for row in doc['events'])
    verify(fixture)


@pytest.mark.parametrize('length', [1, 255, 256, 257, 8001])
def test_spectrogram_retains_first_and_last_samples_in_partial_windows(length):
    for position in (0, length-1):
        pcm=np.zeros(length)
        pcm[position]=.5
        values=spectrogram_values(pcm, 0, length)
        assert values.shape[0]==257 and values.shape[1]<=880
        assert values.max() > .001  # A full-coverage window gives the edge sample nonzero weight.


def test_long_spectrogram_preserves_brief_sound_between_old_sampled_slices():
    pcm=np.zeros(60*SAMPLE_RATE)
    pcm[546]=.5  # Between 0 and ~1092: a sparse 512-sample slice scheme missed this.
    values=spectrogram_values(pcm, 0, len(pcm))
    assert values.shape==(257,880)
    assert values.max()>.001
