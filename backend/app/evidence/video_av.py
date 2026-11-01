"""Source-bound measured audio/video correlation windows inside a video set."""
from __future__ import annotations

import copy
import hashlib
import io
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from app.api.public_projection import alias_signal
from app.core.canonical import canonical_hash
from app.detectors.audio.decode import SAMPLE_RATE, source_ranges, wav_bytes
from app.detectors.video import lipsync, sampling, xcorr
from app.detectors.video.av_evidence import VERSION as INPUT_VERSION
from app.detectors.video.decode import decode_frames
from app.evidence import audio_packets
from app.evidence.av_binding import validate_av_channel
from app.evidence.image_packets import PacketIntegrityError, _bounds, _png
from app.evidence.video_audio import unavailable
from app.evidence.video_clips import ClipUnavailable, build_clip
from app.storage.artefacts import register_artefact

MAX_PACKETS = 4
CREATOR = 'video-av-evidence/1'


def _signals(result, entry, track_id):
    rows = [s for s in result.get('signals', []) if s.get('status', 'ok') == 'ok' and s.get('family') == 'av-sync'
            and (s.get('detector_id') == entry['detector_id'] or set(s.get('exhibit_ids', [])).intersection(entry.get('exhibit_ids', [])))]
    scoped = [s for s in rows if s.get('scope_ref') == track_id or s['id'].endswith(':' + track_id)]
    return scoped or [s for s in rows if s.get('scope', 'global') == 'global']


def _sources(record, result, store, source, table, anchors):
    candidates, unsupported = [], []
    grid = lipsync.tick_grid(table.times(), sampling.duration_s(table), source['samples'],
                            audio_start_s=source['timeline'][0]['source_pts']/SAMPLE_RATE)
    for reference in record.get('av_checks', []):
        entry = json.loads(audio_packets._verified(store, reference['raw_output_sha256']))
        if entry.get('detector_id') != reference['detector_id'] or entry.get('status') != 'ok':
            raise PacketIntegrityError('An AV raw run differs from its published check commitment')
        proof = entry.get('details', {}).get('av_input') or {}
        if proof.get('version') != INPUT_VERSION or proof.get('status') != 'ready' or proof.get('method') != 'mouth_rms_correlation':
            has_windows = any('@window=' in str(u.get('unit_ref')) for u in entry.get('unit_scores', []))
            unsupported.append({'raw_output_sha256': reference['raw_output_sha256'],
                                'status': 'unavailable' if has_windows else 'not_assessed',
                                'reason': 'Measured windows lack a frozen supported input binding.' if has_windows else 'No measurable alignment window was recorded.'})
            continue
        expected = {'source_sha256': source['sha256'], 'pcm_sha256': source['pcm_sha256'], 'mapping_sha256': source['mapping_sha256'],
                    'samples': source['samples'], 'sample_rate': SAMPLE_RATE, 'frame_table_sha256': source['frame_table_sha256'],
                    'frame_time_base': table.time_base, 'audio_origin_pts': source['timeline'][0]['source_pts'],
                    'audio_time_base': '1/16000', 'tick_rate': 25, 'samples_per_tick': 640, 'lag_search_ticks': [-15, 15]}
        if any(proof.get(k) != v for k, v in expected.items()):
            raise PacketIntegrityError('AV raw input clocks differ from the frozen audio/video source')
        units = {u['unit_ref']: u for u in entry.get('unit_scores', [])}
        for track in proof['tracks']:
            refs = [{'id': alias_signal(s['id']), 'display_id': s.get('display_id')} for s in _signals(result, entry, track['track_id'])]
            if not refs:
                raise PacketIntegrityError('Measured AV windows have no corresponding successful published signal')
            ticks = {row['tick']: row for row in track['frame_inputs']}
            if len(ticks) != len(track['frame_inputs']):
                raise PacketIntegrityError('AV input ticks have duplicate identities')
            for row in ticks.values():
                anchor = anchors.get(row['frame_index'])
                if (anchor is None or anchor['pts'] != row['frame_pts'] or anchor['decoded_rgb_sha256'] != row['decoded_rgb_sha256']
                        or not 0 <= row['tick'] < len(grid.frames) or grid.frames[row['tick']] != row['frame_index']
                        or row['source_audio_pts'] != proof['audio_origin_pts'] + row['tick'] * 640):
                    raise PacketIntegrityError('AV measured frame differs from its committed native input')
            for window in track['windows']:
                unit = units.get(window['unit_ref'])
                q = (unit or {}).get('quality') or {}
                start, end = window['start_tick'], window['end_tick']
                if (unit is None or unit.get('raw_kind') != 'score' or abs(unit['raw'] - window['peak_correlation']) > .000002
                        or q.get('offset_frames') != window['offset_ticks']
                        or abs(q.get('start_s', -1) - (proof['audio_origin_pts'] + start * 640) / SAMPLE_RATE) > .000002
                        or abs(q.get('end_s', -1) - (proof['audio_origin_pts'] + end * 640) / SAMPLE_RATE) > .000002
                        or any(k not in ticks for k in range(start, end))):
                    raise PacketIntegrityError('AV packet window differs from its actual raw measurement')
                candidates.append({'window': window, 'track': track, 'inputs': [ticks[k] for k in range(start, end)],
                                   'signal_refs': refs, 'raw_output_sha256': reference['raw_output_sha256'],
                                   'patch_transform': proof['patch_transform']})
    return candidates, unsupported


def _patch(rgb, region):
    x, y, w, h = (region[k] for k in ('x', 'y', 'w', 'h'))
    grey = cv2.cvtColor(np.ascontiguousarray(rgb[y:y+h, x:x+w]), cv2.COLOR_RGB2GRAY)
    return cv2.resize(grey, (32, 24), interpolation=cv2.INTER_AREA).astype('<f4')


def _patch_key(row):
    return (row['frame_index'], *(row['mouth_region'][key] for key in ('x','y','w','h')))


def _decoded_inputs(path, table, selected, anchors):
    """Stream native frames, retaining only tiny measured patches and 3 representatives per packet."""
    rows = {}
    representatives = set()
    for item in selected:
        inputs = item['inputs']
        representatives.update(inputs[pos]['frame_index'] for pos in (0,12,-1))
        for row in inputs:
            rows.setdefault(row['frame_index'], []).append(row)
    originals, patches, found = {}, {}, set()
    for frame in decode_frames(path,table,sorted(rows)):
        if hashlib.sha256(frame.rgb.tobytes()).hexdigest() != anchors[frame.index]['decoded_rgb_sha256']:
            raise PacketIntegrityError('AV source pixels differ from the frames measured during analysis')
        found.add(frame.index)
        if frame.index in representatives:
            originals[frame.index] = frame.rgb
        for row in rows[frame.index]:
            key = _patch_key(row)
            patch = patches.setdefault(key, _patch(frame.rgb,row['mouth_region']))
            if (hashlib.sha256(patch.tobytes()).hexdigest() != row['mouth_patch_sha256']
                    or abs(xcorr.aperture(patch)-row['mouth_value']) > .000002):
                raise PacketIntegrityError('AV measured mouth input differs from its exact source crop')
    if found != set(rows):
        raise PacketIntegrityError('An actual measured AV frame could not be decoded')
    return originals,patches


def _plot(window, inputs, *, lag):
    image = Image.new('RGB', (960, 300), (250, 249, 245))
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = 60, 55, 938, 245
    draw.rectangle((x0, y0, x1, y1), outline=(140, 150, 140))
    if lag:
        draw.text((x0, 12), 'Measured lag search: correlation, not an AI probability', fill=(25, 45, 35))
        draw.line((x0, (y0+y1)/2, x1, (y0+y1)/2), fill=(190, 200, 190))
        points = [(x0 + (r['offset_ticks'] + 15) / 30 * (x1-x0), y1 - (r['correlation'] + 1) / 2 * (y1-y0)) for r in window['lag_profile']]
        draw.line(points, fill=(30, 100, 75), width=3)
        px, py = points[window['offset_ticks'] + 15]
        draw.ellipse((px-5, py-5, px+5, py+5), fill=(195, 100, 30))
        for ticks in (-15, 0, 15):
            draw.text((x0 + (ticks+15)/30*(x1-x0)-15, y1+12), f'{ticks*40:+d} ms', fill=(40, 60, 45))
        draw.text((12, y0), '+1', fill=(40, 60, 45))
        draw.text((12, y1-10), '-1', fill=(40, 60, 45))
        draw.text((x0, 280), 'Positive lag: audio later than the picture. Selected lag is the highest measured correlation.', fill=(40, 60, 45))
    else:
        draw.text((x0, 12), 'Mouth and audio curves: each standardised separately for visual comparison', fill=(25, 45, 35))
        values = [r['mouth_value'] for r in inputs]
        levels = {r['tick']: r['rms'] for r in window['audio_curve']}
        ticks = list(range(window['start_tick'], window['end_tick']))
        curves = [(values, (30, 100, 75), 'mouth proxy'), ([levels[k] for k in ticks], (100, 115, 155), 'audio at source time'),
                  ([levels[k+window['offset_ticks']] for k in ticks], (195, 100, 30), 'audio at selected lag')]
        for i, (values, colour, label) in enumerate(curves):
            z = np.clip(xcorr.zscore(np.asarray(values)), -3, 3)
            points = [(x0+j/24*(x1-x0), (y0+y1)/2-float(v)/6*(y1-y0)) for j,v in enumerate(z)]
            draw.line(points, fill=colour, width=2)
            draw.text((x0+i*270, 35), label, fill=colour)
        draw.text((x0, y1+12), '25 measured ticks / 1 second; source frame choices may repeat with variable frame rate.', fill=(40, 60, 45))
        draw.text((x0, 280), 'Z-scores are descriptive. Raw mouth values and RMS measurements remain in JSON.', fill=(40, 60, 45))
    return _png(image)


def build_channel(db, analysis, store, result, *, set_id, video_source, table, anchors, audio_channel):
    from app.evidence.video_packets import _contact, _context

    record = result.get('manifest', {}).get('sampling', {}).get('video_audio_evidence') or {}
    if audio_channel['status'] == 'absent':
        return unavailable(analysis.sha256, 'The original video has no audio stream to align with its frames.', status='absent')
    binding = record.get('input')
    if not binding:
        return unavailable(analysis.sha256, 'No frozen PCM and source timeline are available for audio/video evidence.')
    pcm, mapping = audio_packets._input(store, analysis, binding)
    if len(source_ranges(mapping, 0, len(pcm))) != 1:
        return unavailable(analysis.sha256, 'Audio timestamps contain gaps; audio/video correlation does not support this timeline.')
    source = {**copy.deepcopy(audio_channel['source']), 'frame_table_sha256': video_source['frame_table_sha256'],
              'frame_anchor_sha256': video_source['frame_anchor_sha256'], 'video_time_base': video_source['time_base'],
              'tick_rate': 25, 'samples_per_tick': 640}
    candidates, unsupported = _sources(record, result, store, source, table, anchors)
    candidates.sort(key=lambda item: (not (item['window']['overlaps_offset_step'] or item['window']['overlaps_correlation_collapse']),
                                      item['window']['start_tick'], item['track']['track_id']))
    selected = candidates[:MAX_PACKETS]
    decoded,measured_patches = (_decoded_inputs(store.path_for(analysis.sha256),table,selected,anchors)
                               if selected else ({},{}))
    channel = {'status': 'ready' if candidates else ('unavailable' if any(r['status']=='unavailable' for r in unsupported) else 'not_assessed'),
               'reason': None if candidates else 'No source-bound correlation windows were recorded; absent evidence does not establish synchronization.',
               'source': source, 'coverage': {'recorded_windows': len(candidates), 'selected_windows': len(selected),
                    'omitted_windows': max(0, len(candidates)-MAX_PACKETS), 'unsupported_checks': unsupported,
                    'selection': 'flagged windows first, then chronological informational measured examples'},
               'packets': [], 'limitations': ['Audio/video misalignment is not proof of AI generation or a deepfake.',
                   'Constant offsets can be container timing; changing offsets can also reflect dropped frames or clock jitter.',
                   'Mouth shading, facial hair, occlusion and speech from another person can alter correlation.',
                   'The nearest source frame is selected at each 25 Hz tick; repeated frames and timestamp errors are retained.',
                   'A track labels measurements, not a verified person or speaker identity.']}
    for number, item in enumerate(selected, 1):
        window, track = item['window'], item['track']
        packet_id, artifacts = f'AV{number:02d}', {}
        def add(role, data, media_type='image/png', *, target=artifacts, **metadata):
            art = register_artefact(db, store, data, media_type=media_type,
                role='working_copy' if media_type == 'audio/wav' else 'preview', created_by=CREATOR, analysis_id=analysis.id)
            extension = {'image/png': 'png', 'audio/wav': 'wav', 'video/mp4': 'mp4'}[media_type]
            target[role] = {'sha256': art.sha256, 'media_type': media_type, **metadata,
                'url': f'/api/v2/analyses/{analysis.id}/evidence-packets/{set_id}/assets/{art.sha256}.{extension}'}
        inputs, actual_values = copy.deepcopy(item['inputs']), []
        for row in inputs:
            patch = measured_patches[_patch_key(row)]
            value = xcorr.aperture(patch)
            if hashlib.sha256(patch.tobytes()).hexdigest() != row['mouth_patch_sha256'] or abs(value-row['mouth_value']) > .000002:
                raise PacketIntegrityError('AV measured mouth input differs from its exact source crop')
            actual_values.append(value)
            role = f"mouth_tick_{row['tick']}"
            add(role, _png(Image.fromarray(patch.astype(np.uint8))), width=32, height=24)
            row['artifact_ref'] = role
        start, end = window['start_tick'], window['end_tick']
        first, last = (start-15)*640, (end+15)*640
        levels = np.sqrt(np.mean(np.asarray(pcm[first:last], np.float64).reshape(-1, 640) ** 2, axis=1))
        if any(abs(float(value)-row['rms']) > .000002 for value,row in zip(levels,window['audio_curve'],strict=True)):
            raise PacketIntegrityError('AV audio RMS curve differs from its exact frozen PCM')
        profile = xcorr.correlation_profile(np.asarray(actual_values), levels, 15)
        if any(abs(float(value)-row['correlation']) > .000002 for value,row in zip(profile,window['lag_profile'],strict=True)):
            raise PacketIntegrityError('AV lag profile cannot be reproduced from its frozen measured inputs')
        for role, span in (('selected_clip',window['audio_interval']), ('context_clip',window['search_audio_interval'])):
            a,b=span['start_sample'],span['end_sample']
            add(role,wav_bytes(pcm[a:b]),'audio/wav',samples=b-a,sample_rate=SAMPLE_RATE,channels=1)
        add('waveform',audio_packets._chart(pcm, [], start=first,end=last,selected=(start*640,end*640),scores=False),width=960,height=280)
        add('lag_profile',_plot(window,inputs,lag=True),width=960,height=300)
        add('alignment',_plot(window,inputs,lag=False),width=960,height=300)
        frames, images, labels = [], [], []
        for pos, row in enumerate([inputs[0],inputs[12],inputs[-1]],1):
            index=row['frame_index']
            image=Image.fromarray(decoded[index])
            region=row['mouth_region']
            label=f'{packet_id}.{pos} - frame {index}, tick {row["tick"]}'
            context=_context(image,region,label)
            roles={'original':f'frame_{pos}_original','context':f'frame_{pos}_context','crop':f'frame_{pos}_crop','measured_patch':row['artifact_ref']}
            for role,picture in [('original',image),('context',context),('crop',image.crop(_bounds(region,table.width,table.height)))]:
                add(roles[role],_png(picture),width=picture.width,height=picture.height)
            frames.append({'id':f'{packet_id}.{pos}','av_tick':row['tick'],'score':None,'is_peak':False,
                'anchor':{**anchors[index],'original_sha256':analysis.sha256,'stream_selector':'v:0','stream_time_s':table.seconds(index),
                    'playback_time_s':None if video_source['timeline_origin_s'] is None else table.seconds(index)-video_source['timeline_origin_s'],
                    'coordinate_space':'display-oriented-native-frame'},'context_region':region,'region_meaning':'mouth_measurement_input','artifact_refs':roles})
            images.append(context)
            labels.append(f"Frame {index}, measured tick {row['tick']}")
        sheet=_contact(images,labels)
        add('contact_sheet',_png(sheet),width=sheet.width,height=sheet.height)
        flagged=window['overlaps_offset_step'] or window['overlaps_correlation_collapse']
        refs=item['signal_refs']
        ids={r['id'] for r in refs}
        packet={'id':packet_id,'title':'Audio/video correlation anomaly' if flagged else 'Measured audio/video alignment',
            'evidence_type':'anomaly','claim_scope':'sampled_av_window','semantics':'mouth_audio_correlation',
            'finding_ids':[f['id'] for f in result.get('findings',[]) if ids.intersection(alias_signal(v) for v in f.get('evidence_ids',[]))],
            'signal_refs':refs,'observation':f"The measured 1 s window selected a {window['offset_ticks']*40:+d} ms audio lag with peak correlation {window['peak_correlation']:.3f}.",
            'interpretation':('This window overlaps a recorded offset change or correlation collapse; inspect the source and the full lag profile.' if flagged else 'This is an informational measured example. No offset-change or correlation-collapse segment overlaps this window.'),
            'interval':copy.deepcopy(window['audio_interval']),'context_interval':copy.deepcopy(window['search_audio_interval']),
            'anchor':{'original_sha256':analysis.sha256,'pcm_sha256':binding['pcm_sha256'],'sample_rate':SAMPLE_RATE,
                      'source_timestamp_ranges':source_ranges(mapping,start*640,end*640)},
            'window':{'start_tick':start,'end_tick':end,'frame_indices':[row['frame_index'] for row in inputs],'frame_inputs':inputs},
            'measurement':{**{key:copy.deepcopy(window[key]) for key in ['offset_ticks','peak_correlation','lag_profile','audio_curve','low_correlation','overlaps_offset_step','overlaps_correlation_collapse']},
                'offset_ms':window['offset_ticks']*40,**{key:track[key] for key in ['baseline_offset_ticks','collapse_threshold','minimum_trusted_peak','reading']}},
            'measurements':[{'label':'Selected audio lag','value':window['offset_ticks']*40,'unit':'ms; positive is audio later'},
                            {'label':'Peak mouth/audio correlation','value':window['peak_correlation'],'unit':'Pearson correlation, not probability'}],
            'frames':frames,'artifacts':artifacts,'source_record':{'raw_output_sha256':item['raw_output_sha256'],'unit_ref':window['unit_ref'],'track_id':track['track_id']},
            'derivation':{'version':CREATOR,'patch_transform':copy.deepcopy(item['patch_transform']),'lag_direction':'positive means audio later than video'},
            'limitations':['The 31 measured lag candidates are retained; a high correlation does not establish authenticity.',
                           'Selected audio is the unshifted source-time window. Search context includes the additional ±600 ms actually compared.',
                           'Gray mouth patches are the exact measured downsampled input, not an edit mask.']}
        try:
            data,metadata,description=build_clip(store.path_for(analysis.sha256),table,source_sha256=analysis.sha256,
                focus_index=inputs[12]['frame_index'],sampled_frame_indices=packet['window']['frame_indices'],timeline_origin_s=video_source['timeline_origin_s'])
            add('context_video',data,'video/mp4',**metadata)
            # The shared silent-clip validator uses one canonical artifact key.
            artifacts['context_clip_video']=artifacts.pop('context_video')
            description['artifact_ref']='context_clip_video'
            packet['clip']=description
        except ClipUnavailable as error:
            packet['clip']={'status':'unavailable','reason':str(error)}
        channel['packets'].append(packet)
    validate_av_channel(channel,video_source)
    return channel


def verify_source_assets(channel, *, video_source, result, store, table, anchors, assets):
    """Validate a frozen channel against raw run commitments and existing assets.

    This reads no mutable analysis rows and performs no decoding, inference or
    regeneration. Native frame hashes and every measured mouth-patch hash were
    committed during the actual detector run.
    """
    validate_av_channel(channel, video_source)
    record=result.get('manifest',{}).get('sampling',{}).get('video_audio_evidence') or {}
    if channel['coverage'] is None:
        if record.get('source_audio_present') is False:
            if channel['status'] != 'absent':
                raise PacketIntegrityError('Frozen AV absence differs from the original audio stream')
        elif record.get('input'):
            from types import SimpleNamespace
            pcm,mapping=audio_packets._input(store,SimpleNamespace(sha256=video_source['sha256']),record['input'])
            if len(source_ranges(mapping,0,len(pcm))) == 1:
                raise PacketIntegrityError('Frozen AV channel omits an available source timeline')
        return
    binding=record.get('input')
    if not binding or any(channel['source'].get(k)!=binding.get(k) for k in ('pcm_sha256','wav_sha256','mapping_sha256','samples','sample_rate')):
        raise PacketIntegrityError('Frozen AV audio input differs from the published source commitment')
    from types import SimpleNamespace

    from app.detectors.audio.decode import read_wav
    pcm,_=audio_packets._input(store,SimpleNamespace(sha256=video_source['sha256']),binding)
    candidates,unsupported=_sources(record,result,store,channel['source'],table,anchors)
    candidates.sort(key=lambda item:(not (item['window']['overlaps_offset_step'] or item['window']['overlaps_correlation_collapse']),
                                     item['window']['start_tick'],item['track']['track_id']))
    if (channel['coverage']['recorded_windows']!=len(candidates) or channel['coverage']['omitted_windows']!=max(0,len(candidates)-MAX_PACKETS)
            or len(channel['packets'])!=min(len(candidates),MAX_PACKETS)
            or canonical_hash(channel['coverage']['unsupported_checks'])!=canonical_hash(unsupported)):
        raise PacketIntegrityError('Frozen AV coverage differs from its actual supported runs')
    for packet,expected in zip(channel['packets'],candidates[:MAX_PACKETS],strict=True):
        raw,track=expected['window'],expected['track']
        if packet['source_record']!={'raw_output_sha256':expected['raw_output_sha256'],'unit_ref':raw['unit_ref'],'track_id':track['track_id']}:
            raise PacketIntegrityError('Frozen AV packet selection differs from its raw run')
        if canonical_hash(packet['signal_refs'])!=canonical_hash(expected['signal_refs']):
            raise PacketIntegrityError('Frozen AV signal references differ from their actual run')
        if canonical_hash([{k:v for k,v in row.items() if k!='artifact_ref'} for row in packet['window']['frame_inputs']])!=canonical_hash(expected['inputs']):
            raise PacketIntegrityError('Frozen AV frame inputs differ from the measured input record')
        measurement=packet['measurement']
        for key in ('offset_ticks','peak_correlation','lag_profile','audio_curve','low_correlation','overlaps_offset_step','overlaps_correlation_collapse'):
            if canonical_hash(measurement[key])!=canonical_hash(raw[key]):
                raise PacketIntegrityError('Frozen AV measurement differs from its raw output')
        for key in ('baseline_offset_ticks','collapse_threshold','minimum_trusted_peak','reading'):
            if canonical_hash(measurement[key])!=canonical_hash(track[key]):
                raise PacketIntegrityError('Frozen AV track reading differs from its raw output')
        if packet['derivation']['patch_transform']!=expected['patch_transform']:
            raise PacketIntegrityError('Frozen AV crop transformation differs from its actual input')
        media=packet['artifacts']
        actual_values=[]
        for row in packet['window']['frame_inputs']:
            with Image.open(io.BytesIO(assets[media[row['artifact_ref']]['sha256']])) as picture:
                patch=np.asarray(picture.convert('L'),dtype='<f4')
            if hashlib.sha256(patch.tobytes()).hexdigest()!=row['mouth_patch_sha256']:
                raise PacketIntegrityError('Frozen AV mouth pixels differ from the actual measured patch')
            value=xcorr.aperture(patch)
            if abs(value-row['mouth_value'])>.000002:
                raise PacketIntegrityError('Frozen AV mouth response differs from its measured pixels')
            actual_values.append(value)
        for role,span in (('selected_clip',packet['interval']),('context_clip',packet['context_interval'])):
            clip=read_wav(assets[media[role]['sha256']])
            if not np.array_equal(clip,pcm[span['start_sample']:span['end_sample']]):
                raise PacketIntegrityError('Frozen AV audio clip differs from its measured PCM samples')
        a,b=packet['context_interval']['start_sample'],packet['context_interval']['end_sample']
        rms=np.sqrt(np.mean(np.asarray(pcm[a:b],np.float64).reshape(-1,640)**2,axis=1))
        profile=xcorr.correlation_profile(np.asarray(actual_values),rms,15)
        if any(abs(float(v)-p['correlation'])>.000002 for v,p in zip(profile,measurement['lag_profile'],strict=True)):
            raise PacketIntegrityError('Frozen AV lag profile does not follow its exact measured inputs')
        for frame in packet['frames']:
            anchor=frame['anchor']
            native=anchors.get(anchor['frame_index'])
            if not native or any(anchor[k]!=v for k,v in native.items()):
                raise PacketIntegrityError('Frozen AV representative frame differs from its native source anchor')
            with Image.open(io.BytesIO(assets[media[frame['artifact_refs']['original']]['sha256']])) as picture:
                original=picture.convert('RGB')
            if hashlib.sha256(original.tobytes()).hexdigest()!=anchor['decoded_rgb_sha256']:
                raise PacketIntegrityError('Frozen AV representative pixels differ from the actual analysed frame')
            with Image.open(io.BytesIO(assets[media[frame['artifact_refs']['crop']]['sha256']])) as picture:
                crop=picture.convert('RGB')
            expected_crop=original.crop(_bounds(frame['context_region'],table.width,table.height))
            if crop.size!=expected_crop.size or crop.tobytes()!=expected_crop.tobytes():
                raise PacketIntegrityError('Frozen AV crop is not from its recorded source rectangle')
        clip=packet.get('clip') or {}
        if clip.get('status')=='ready':
            start,end=clip['start_frame_index'],clip['end_frame_index_exclusive']
            if clip['source_pts']!=list(table.pts[start:end]) or set(table.interpolated).intersection(range(start,end)):
                raise PacketIntegrityError('Frozen AV context video differs from its source frame clock')
            if end<len(table) and clip['last_frame_duration']['ticks']!=(table.pts[end]-table.pts[end-1])*table.tb.numerator:
                raise PacketIntegrityError('Frozen AV context duration differs from its next source frame')
