"""Controlled AV input correspondence fixtures; these are not accuracy benchmarks."""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.canonical import canonical_hash, canonical_json
from app.api.public_projection import alias_signal
from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes
from app.detectors.video import lipsync, sampling, xcorr
from app.detectors.video.adapters.lipsync import AVInputs, source_audio_origin
from app.detectors.video.av_evidence import capture_correlation
from app.detectors.video.decode import probe_frames
from app.detectors.video.frame_store import FrameStore
from app.evidence import video_audio, video_av
from app.evidence.av_binding import InvalidAV, validate_av_channel
from app.evidence.image_packets import PacketIntegrityError
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Base
from app.storage.objects import ObjectStore


@pytest.fixture(scope='module')
def av_source(tmp_path_factory):
    path = tmp_path_factory.mktemp('av-source')
    landmarks = [[30,22],[66,22],[48,34],[35,42],[61,42]]
    face = {'bbox': {'x':12,'y':8,'w':72,'h':52},'landmarks':landmarks}
    values = []
    rng=np.random.default_rng(31)
    for index,level in enumerate(rng.integers(10,115,100)):
        rgb=np.full((64,96,3),160,dtype=np.uint8)
        rgb[33:51,35:61]=int(level)
        Image.fromarray(rgb).save(path/f'f{index:03}.png')
        values.append(xcorr.aperture(lipsync.mouth_patch(rgb,face)))
    # Source audio starts400ms after source video. Its envelope is a controlled
    # +200ms copy of the mouth signal, so raw lag accounting is measurable.
    levels=np.asarray([.15+.6*values[max(0,10+tick-5)] for tick in range(90)])
    time=np.arange(90*640)/16000
    pcm=(np.repeat(levels,640)*np.sin(2*np.pi*400*time)).astype(np.float32)
    (path/'audio.wav').write_bytes(wav_bytes(pcm))
    source=path/'source.mkv'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-framerate','25','-i',str(path/'f%03d.png'),
                    '-itsoffset','0.4','-i',str(path/'audio.wav'),'-map','0:v:0','-map','1:a:0','-c:v','ffv1',
                    '-c:a','pcm_s16le','-output_ts_offset','5',str(source)],check=True,timeout=30)
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    table=probe_frames(source); decoded=decode_audio(source,source_sha256=sha,max_seconds=120)
    assert table.seconds(0)==5 and decoded.mapping['timeline'][0]['source_pts']==86400
    return source,table,decoded,face


@pytest.fixture
def av_fixture(tmp_path,av_source):
    path,table,decoded,face=av_source
    engine=create_engine('sqlite://');Base.metadata.create_all(engine);store=ObjectStore(tmp_path/'objects')
    with Session(engine) as db:
        obj=store.put_file(path)
        analysis=Analysis(id='video-av-test',kind='video',filename='controlled-av.mkv',content_type='video/x-matroska',
                          sha256=obj.sha256,size_bytes=obj.size_bytes,status='done',revision=1,result_schema_version=2)
        db.add(analysis);db.commit()
        ctx=SimpleNamespace(sessions=lambda:Session(engine),store=store,analysis_id=analysis.id)
        frozen=video_audio.freeze_input(ctx,decoded)
        frames=FrameStore(store.path_for(obj.sha256),table)
        audio=SimpleNamespace(pcm=decoded.pcm,status='analysed',samples=len(decoded.pcm))
        by_frame={i:{**face,'frame_index':i,'pts':table.pts[i]} for i in range(len(table))}
        inputs=AVInputs(table,({'track_id':'track:0'},),{'track:0':list(by_frame.values())},audio,frames,decoded.mapping)
        grid=lipsync.tick_grid(table.times(),sampling.duration_s(table),len(decoded.pcm),audio_start_s=source_audio_origin(inputs))
        ticks=range(25,75);pixels=frames.get_many({grid.frames[t] for t in ticks})
        patches={'track:0':{t:lipsync.mouth_patch(pixels[grid.frames[t]],by_frame[grid.frames[t]]) for t in ticks}}
        aperture={'track:0':{t:xcorr.aperture(p) for t,p in patches['track:0'].items()}}
        level=np.sqrt(np.mean(np.asarray(decoded.pcm,np.float64).reshape(-1,640)**2,axis=1))
        speech=lipsync.Speech(20*np.log10(np.maximum(level,1e-5)),level,np.zeros(90),np.ones(90,dtype=bool),-50)
        windows=[lipsync.Window('track:0',j,1.0,52,12.5) for j in (2,3,4)]
        sync=lipsync.track_sync([xcorr.window_sync(w,aperture['track:0'],level) for w in windows],xcorr.SETTINGS)
        proof=capture_correlation(inputs,grid,speech,{'track:0':sync},patches,aperture,{'track:0':by_frame},{'track:0':.9})
        units=[{'unit_ref':f'track:0@window={w.window.index}','raw':w.confidence,'raw_kind':'score',
                'quality':{'offset_frames':w.offset,'start_s':grid.seconds(w.start_tick),'end_s':grid.seconds(w.end_tick)}} for w in sync.windows]
        avraw={'detector_id':'vid.lipsync.landmark_xcorr','status':'ok','unit_scores':units,'details':{'av_input':proof},'exhibit_ids':[]}
        voiceraw={'detector_id':'aud.gustking_xlsr','status':'ok','unit_scores':[{'unit_ref':'window:0','raw':.85,'raw_kind':'probability',
            'quality':{'start_sample':0,'end_sample':len(decoded.pcm),'start_s':0,'end_s':len(decoded.pcm)/16000}}],
            'details':{'audio_input':{'pcm_sha256':decoded.mapping['pcm_sha256'],'samples':len(decoded.pcm),'sample_rate':16000}},'exhibit_ids':[]}
        avref=store.put_bytes(canonical_json(avraw)).sha256;voiceref=store.put_bytes(canonical_json(voiceraw)).sha256
        records={'version':'video-audio-inputs/1','input':frozen['input'],'source_audio_present':True,'decode_status':'analysed',
                 'voice_checks':[{'detector_id':voiceraw['detector_id'],'raw_output_sha256':voiceref}],
                 'av_checks':[{'detector_id':avraw['detector_id'],'raw_output_sha256':avref}]}
        result={'input':{'sha256':obj.sha256},'manifest':{'sampling':{'video_audio_evidence':records}},'findings':[],
                'signals':[{'id':'aud.gustking_xlsr:global','detector_id':'aud.gustking_xlsr','family':'voice','status':'ok','display_id':'S01'},
                           {'id':'vid.lipsync.landmark_xcorr:track:0','detector_id':'vid.lipsync.landmark_xcorr','family':'av-sync','status':'ok','display_id':'S02','scope':'track','scope_ref':'track:0'}]}
        analysis.result_hash=canonical_hash(result)
        anchors={a['frame_index']:a for a in frames.native_anchors()}
        source={'sha256':obj.sha256,'width':table.width,'height':table.height,'frame_count':len(table),'time_base':table.time_base,
                'timeline_origin_s':5.,'frame_table_sha256':canonical_hash(table.to_json()),'frame_anchor_sha256':canonical_hash(list(anchors.values()))}
        yield SimpleNamespace(db=db,store=store,analysis=analysis,result=result,table=table,anchors=anchors,source=source,
                              proof=proof,raw=avraw,decoded=decoded,inputs=inputs,grid=grid,speech=speech,sync=sync,
                              patches=patches,aperture=aperture,by_frame=by_frame)


def _channels(f):
    audio=video_audio.build_channel(f.db,f.analysis,f.store,f.result,set_id='parent-video-set')
    av=video_av.build_channel(f.db,f.analysis,f.store,f.result,set_id='parent-video-set',video_source=f.source,
                             table=f.table,anchors=f.anchors,audio_channel=audio)
    return audio,av


def test_voice_fractional_sample_seconds_survive_canonical_freezing(av_fixture):
    f=av_fixture
    record=f.result['manifest']['sampling']['video_audio_evidence']
    reference=record['voice_checks'][0]
    raw=json.loads(f.store.read_bytes(reference['raw_output_sha256']))
    end=len(f.decoded.pcm)-1  # 3.5999375 seconds: not exactly representable with six decimals
    raw['unit_scores'][0]['quality'].update(end_sample=end,end_s=end/16000)
    reference['raw_output_sha256']=f.store.put_bytes(canonical_json(raw)).sha256
    audio=video_audio.build_channel(f.db,f.analysis,f.store,f.result,set_id='parent-video-set')
    frozen=json.loads(canonical_json(audio))
    assert frozen['packets'][0]['interval']['end_sample']==end
    assert frozen['packets'][0]['end_s']==round(end/16000,6)
    video_audio.validate_channel(audio,source_sha256=f.analysis.sha256)
    video_audio.validate_channel(frozen,source_sha256=f.analysis.sha256)
    video_audio.verify_source_content(frozen,result=f.result,store=f.store,source_sha256=f.analysis.sha256)
    forged=deepcopy(frozen);forged['packets'][0]['end_s']+=1/16000
    with pytest.raises(ValueError,match='clock'):
        video_audio.validate_channel(forged,source_sha256=f.analysis.sha256)


def test_vfr_repeated_inputs_preserve_actual_tick_frame_and_clip_bindings(tmp_path, av_source):
    path,_,_,face=av_source
    vfr=tmp_path/'offset-vfr-av.mkv'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-copyts','-i',str(path),
                    '-vf',r'select=not(eq(mod(n\,5)\,2))','-fps_mode','vfr',
                    '-c:v','ffv1','-c:a','copy','-avoid_negative_ts','disabled',str(vfr)],check=True,timeout=30)
    sha=hashlib.sha256(vfr.read_bytes()).hexdigest();table=probe_frames(vfr)
    decoded=decode_audio(vfr,source_sha256=sha,max_seconds=120)
    assert table.seconds(0)==5 and len(set(np.diff(table.pts)))>1
    fixture=av_fixture.__wrapped__(tmp_path,(vfr,table,decoded,face))
    try:
        f=next(fixture);audio,av=_channels(f)
        assert av['status']=='ready'
        assert any(len(set(p['window']['frame_indices']))<25 for p in av['packets'])
        for packet in av['packets']:
            assert len(packet['window']['frame_inputs'])==25
            assert packet['clip']['sampled_frame_indices']==packet['window']['frame_indices']
        assets={a['sha256']:f.store.read_bytes(a['sha256']) for p in av['packets'] for a in p['artifacts'].values()}
        video_av.verify_source_assets(av,video_source=f.source,result=f.result,store=f.store,
                                      table=table,anchors=f.anchors,assets=assets)
    finally:
        fixture.close()


def test_exact_voice_and_measured_av_inputs_share_parent_authority(av_fixture):
    f=av_fixture;audio,av=_channels(f)
    video_audio.validate_channel(audio,source_sha256=f.analysis.sha256);validate_av_channel(av,f.source)
    assert audio['status']=='ready' and len(audio['packets'])==1
    assert av['status']=='ready' and len(av['packets'])==3
    assert all(p['measurement']['offset_ticks']==5 for p in av['packets'])
    assert av['packets'][0]['window']['frame_indices']==list(range(35,60))
    assert av['packets'][0]['anchor']['source_timestamp_ranges'][0]['source_start_pts']==102400
    for channel in (audio,av):
        assert not {'id','sha256','analysis_id','revision','original_result_hash'}.intersection(channel)
        for packet in channel['packets']:
            for asset in packet['artifacts'].values(): assert '/parent-video-set/assets/' in asset['url']
    for packet in av['packets']:
        assert packet['clip']['status']=='ready' and packet['clip']['artifact_ref']=='context_clip_video'
        assert len(packet['measurement']['lag_profile'])==31 and len(packet['measurement']['audio_curve'])==55
        for row in packet['window']['frame_inputs']:
            image=Image.open(io.BytesIO(f.store.read_bytes(packet['artifacts'][row['artifact_ref']]['sha256']))).convert('L')
            assert hashlib.sha256(np.asarray(image,dtype='<f4').tobytes()).hexdigest()==row['mouth_patch_sha256']
        for role,span in [('selected_clip',packet['interval']),('context_clip',packet['context_interval'])]:
            actual=read_wav(f.store.read_bytes(packet['artifacts'][role]['sha256']))
            np.testing.assert_array_equal(actual,f.decoded.pcm[span['start_sample']:span['end_sample']])
    # The existing projection keeps registered, model-free method IDs so check
    # joins remain compatible; actual model identities are still opaque.
    assert av['packets'][0]['signal_refs'][0]['id']==alias_signal('vid.lipsync.landmark_xcorr:track:0')
    assert 'gustking' not in json.dumps(audio)


@pytest.mark.parametrize('mutation',['lag','missing_lag','tick','source_pts','frame','pcm','sample_interval','patch','clip'])
def test_public_av_binding_rejects_invented_measurements_and_clocks(av_fixture,mutation):
    f=av_fixture;_,av=_channels(f);packet=av['packets'][0]
    if mutation=='lag':packet['measurement']['offset_ms']+=40
    elif mutation=='missing_lag':packet['measurement']['lag_profile'].pop()
    elif mutation=='tick':packet['window']['frame_inputs'][0]['tick']+=1
    elif mutation=='source_pts':packet['window']['frame_inputs'][0]['source_audio_pts']+=1
    elif mutation=='frame':packet['frames'][0]['anchor']['pts']+=1
    elif mutation=='pcm':packet['anchor']['pcm_sha256']='0'*64
    elif mutation=='sample_interval':packet['interval']['start_sample']+=1
    elif mutation=='patch':packet['artifacts'][packet['window']['frame_inputs'][0]['artifact_ref']]['width']+=1
    else:packet['clip']['clip_pts'][1]+=1
    with pytest.raises(ValueError):validate_av_channel(av,f.source)


def test_missing_research_feature_bindings_are_unavailable_not_successful(av_fixture):
    f=av_fixture;raw=deepcopy(f.raw);raw['details']={}
    sha=f.store.put_bytes(canonical_json(raw)).sha256
    f.result['manifest']['sampling']['video_audio_evidence']['av_checks'][0]['raw_output_sha256']=sha
    _,av=_channels(f)
    assert av['status']=='unavailable' and av['packets']==[]
    assert av['coverage']['unsupported_checks'][0]['reason']=='Measured windows lack a frozen supported input binding.'


def test_raw_av_input_tampering_is_detected_before_packet_creation(av_fixture):
    f=av_fixture;raw=deepcopy(f.raw)
    raw['details']['av_input']['tracks'][0]['windows'][0]['lag_profile'][0]['correlation']+=.1
    sha=f.store.put_bytes(canonical_json(raw)).sha256
    f.result['manifest']['sampling']['video_audio_evidence']['av_checks'][0]['raw_output_sha256']=sha
    with pytest.raises(PacketIntegrityError,match='profile cannot be reproduced'):_channels(f)


def test_frozen_av_report_verification_uses_measured_inputs_without_redecoding(av_fixture,monkeypatch):
    f=av_fixture;_,av=_channels(f)
    assets={a['sha256']:f.store.read_bytes(a['sha256']) for p in av['packets'] for a in p['artifacts'].values()}
    monkeypatch.setattr(video_av,'decode_frames',lambda *a,**k:pytest.fail('Frozen verification decoded mutable source'))
    video_av.verify_source_assets(av,video_source=f.source,result=f.result,store=f.store,table=f.table,anchors=f.anchors,assets=assets)


@pytest.mark.parametrize('damage',['mouth_pixels','pcm_samples','curve','raw_reference'])
def test_frozen_av_report_verification_detects_tampered_assets_and_measurements(av_fixture,damage):
    f=av_fixture;_,av=_channels(f);packet=av['packets'][0]
    assets={a['sha256']:f.store.read_bytes(a['sha256']) for p in av['packets'] for a in p['artifacts'].values()}
    if damage=='mouth_pixels':
        role=packet['window']['frame_inputs'][0]['artifact_ref'];old=packet['artifacts'][role]
        im=Image.open(io.BytesIO(assets[old['sha256']])).convert('L');im.putpixel((0,0),(im.getpixel((0,0))+1)%256)
        stream=io.BytesIO();im.save(stream,format='PNG');data=stream.getvalue();sha=hashlib.sha256(data).hexdigest()
        old['sha256']=sha;assets[sha]=data
    elif damage=='pcm_samples':
        old=packet['artifacts']['context_clip'];data=wav_bytes(np.zeros(old['samples'],np.float32));sha=hashlib.sha256(data).hexdigest()
        old['sha256']=sha;assets[sha]=data
    elif damage=='curve':
        packet['measurement']['audio_curve'][0]['rms']+=.001
    else:packet['source_record']['raw_output_sha256']='f'*64
    with pytest.raises((ValueError,PacketIntegrityError)):
        video_av.verify_source_assets(av,video_source=f.source,result=f.result,store=f.store,table=f.table,anchors=f.anchors,assets=assets)
