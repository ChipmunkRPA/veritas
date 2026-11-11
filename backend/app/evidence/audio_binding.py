"""Standard-library validation of audio content nested in a public video set."""
from __future__ import annotations

import math
import re

from app.core.canonical import canonical_hash


def validate_audio_channel(channel: dict, source_sha256: str) -> None:
    try:
        _validate(channel, source_sha256)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('Video voice evidence contains missing fields or invalid types') from error


def _validate(channel, source_sha256):
    def require(value, message):
        if not value:
            raise ValueError(message)

    def integer(value):
        return isinstance(value, int) and not isinstance(value, bool)

    def sha(value):
        return bool(re.fullmatch(r'[a-f0-9]{64}', str(value)))

    def seconds(value, sample):
        # Integer sample coordinates remain authoritative. Presentation seconds
        # have the canonical JSON's six decimals after freezing; accept that
        # exact representation as well as the pre-freeze arithmetic value.
        exact = sample / 16000
        return not isinstance(value, bool) and value in (exact, round(exact, 6))

    require(not {'schema', 'version', 'id', 'sha256', 'analysis_id', 'revision', 'original_result_hash'}.intersection(channel),
            'A video voice channel cannot claim independent authority')
    source = channel['source']
    require(source['sha256'] == source_sha256, 'Video voice source differs from its parent')
    require(channel['status'] in {'ready', 'not_assessed', 'absent', 'unavailable'}, 'Invalid video voice availability')
    if channel['coverage'] is None:
        require(channel['status'] in {'absent', 'unavailable'} and not channel['packets'] and not channel.get('overview')
                and not channel.get('acoustic_context')
                and not {'stereo_context', 'stereo_context_sha256', 'repetition_context', 'similarity_context'}.intersection(channel)
                and bool(channel['reason']), 'Unavailable video voice contains unexplained assets')
        return
    samples = source['samples']
    require(integer(samples) and 0 < samples <= 120*16000 and source['sample_rate'] == 16000 and source['channels'] == 1,
            'Invalid video voice PCM dimensions')
    require(all(sha(source[k]) for k in ('pcm_sha256', 'wav_sha256', 'mapping_sha256')), 'Uncommitted video voice input')
    require(source['timeline_time_base'] == '1/16000', 'Video voice source clock differs from its PCM mapping')
    cursor = 0
    for block in source['timeline']:
        require(all(integer(block[k]) for k in ('start_sample', 'end_sample', 'source_pts'))
                and block['start_sample'] == cursor < block['end_sample'] <= samples, 'Invalid video voice source mapping')
        cursor = block['end_sample']
    require(cursor == samples, 'Video voice source mapping omits decoded samples')

    def interval(span):
        a,b = span['start_sample'], span['end_sample']
        require(integer(a) and integer(b) and 0 <= a < b <= samples, 'Voice samples are outside the analysed input')
        return a,b

    def ranges(a,b):
        return [{'start_sample': max(a,row['start_sample']), 'end_sample': min(b,row['end_sample']),
                 'source_start_pts': row['source_pts']+max(a,row['start_sample'])-row['start_sample'],
                 'source_end_pts': row['source_pts']+min(b,row['end_sample'])-row['start_sample'], 'time_base':'1/16000'}
                for row in source['timeline'] if max(a,row['start_sample']) < min(b,row['end_sample'])]

    if {'stereo_context', 'stereo_context_sha256'}.intersection(channel):
        from app.evidence.stereo_binding import validate

        stereo = channel['stereo_context']
        require(sha(channel['stereo_context_sha256']) and canonical_hash(stereo) == channel['stereo_context_sha256'],
                'Video stereo context differs from its committed measurements')
        validate(stereo, source, ranges=ranges)

    if channel.get('repetition_context') is not None:
        from app.evidence.audio_repetition_binding import validate

        validate(channel['repetition_context'], source)
    if 'similarity_context' in channel:
        from app.evidence.audio_similarity_binding import validate

        validate(channel['similarity_context'], source)

    coverage=channel['coverage']
    for window in coverage['windows']:
        interval(window)
        require(isinstance(window['score'], (float,int)) and not isinstance(window['score'],bool)
                and math.isfinite(window['score']) and 0 <= window['score'] <= 1
                and sha(window['raw_output_sha256']) and bool(window['signal_refs']), 'Invalid recorded voice response')
    require((channel['status']=='ready') == bool(coverage['windows']), 'Voice status differs from its actual windows')
    require(coverage['decoded_samples'] == samples and coverage['scored_samples']+coverage['unscored_samples']==samples,
            'Voice sample coverage does not balance')
    ids=set()
    for packet in channel['packets']:
        require(packet['id'].startswith('A') and packet['id'] not in ids and packet['evidence_type']=='statistical',
                'Invalid voice packet identity or semantics')
        ids.add(packet['id'])
        a,b=interval(packet['interval'])
        ca,cb=interval(packet['context_interval'])
        require(ca<=a<b<=cb and seconds(packet['start_s'],a) and seconds(packet['end_s'],b), 'Voice clock or context differs')
        anchor=packet['anchor']
        require(anchor['original_sha256']==source_sha256 and anchor['pcm_sha256']==source['pcm_sha256']
                and anchor['sample_rate']==16000 and anchor['source_timestamp_ranges']==ranges(a,b), 'Voice source anchor differs')
        require(set(packet['artifacts'])=={'selected_clip','context_clip','waveform','scores'}, 'Voice evidence lacks required assets')
        for role,start,end in (('selected_clip',a,b),('context_clip',ca,cb)):
            asset=packet['artifacts'][role]
            require(asset['media_type']=='audio/wav' and asset['sample_rate']==16000 and asset['channels']==1
                    and asset['samples']==end-start and sha(asset['sha256']), 'Voice WAV differs from its source interval')
        for observation in packet['observations']:
            require(any(w['start_sample']==a and w['end_sample']==b and w['score']==observation['value']
                        and w['raw_output_sha256']==observation['raw_output_sha256'] and w['unit_ref']==observation['unit_ref']
                        and w['signal_refs']==observation['signal_refs'] for w in coverage['windows']),
                    'Voice observation is not one of the recorded windows')
    if channel.get('acoustic_context') is not None:
        from app.evidence.audio_background_binding import validate_acoustic_context

        validate_acoustic_context(channel['acoustic_context'], source=source, voice_packets=channel['packets'])
    if channel.get('sound_events') is not None:
        from app.evidence.sound_event_binding import validate
        validate(channel['sound_events'], source)
