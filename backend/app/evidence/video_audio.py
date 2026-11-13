"""Frozen audio inputs and voice presentation inside one parent video evidence set."""
from __future__ import annotations

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.evidence import audio_packets, stereo_listening, stereo_repetition
from app.evidence.audio_binding import VOICE_CLOCK_POLICY
from app.evidence.audio_gain_repetition_binding import GAIN_INPUT, GAIN_VERSION
from app.evidence.stereo_repetition_binding import INPUT as STEREO_REPEAT_INPUT
from app.storage.artefacts import get_artefact, register_artefact
from app.storage.db import DetectorRun


def freeze_input(ctx, decoded) -> dict:
    """Persist the already-decoded PCM once; never decode again to invent evidence."""
    source_mapping = dict(decoded.mapping)
    stereo_context = source_mapping.pop('stereo_context', None)
    with ctx.sessions() as db:
        wav = register_artefact(db, ctx.store, decoded.wav, media_type='audio/wav', role='working_copy',
                               created_by='video-audio-input/1', analysis_id=ctx.analysis_id)
        listening = stereo_listening.freeze(db, ctx.store, ctx.analysis_id, decoded)
        mapping = register_artefact(db, ctx.store, canonical_json(source_mapping), media_type='application/json',
                                   role='stage_output', created_by='video-audio-input/1', analysis_id=ctx.analysis_id)
        binding = {'version': 'audio-input/1', 'pcm_sha256': decoded.mapping['pcm_sha256'],
                   'wav_sha256': wav.sha256, 'mapping_sha256': mapping.sha256,
                   'sample_rate': decoded.mapping['sample_rate'], 'samples': len(decoded.pcm)}
        objects = [wav, mapping]
        if listening is not None:
            # Preserve the mono-only mapping hash used by the AV timing proof.
            binding['stereo_listening_pcm_sha256'] = listening.sha256
            objects.append(listening)
        if repeated := stereo_repetition.freeze(db, ctx.store, ctx.analysis_id, decoded):
            binding[STEREO_REPEAT_INPUT] = repeated.sha256
            if settings.environment == 'dev' and settings.audio_stereo_gain_repeat_review:
                binding[GAIN_INPUT] = GAIN_VERSION
            objects.append(repeated)
        output = {'input': binding}
        if stereo_context is not None:
            stereo = register_artefact(db, ctx.store, canonical_json(stereo_context), media_type='application/json',
                                       role='stage_output', created_by='video-audio-stereo/1', analysis_id=ctx.analysis_id)
            objects.append(stereo)
            output['stereo_context_sha256'] = stereo.sha256
        output['artefacts'] = [{'id': a.id, 'sha256': a.sha256, 'size_bytes': a.bytes, 'media_type': a.media_type,
                                'role': a.role, 'created_by': a.created_by} for a in objects]
        db.commit()
    return output


def published_record(ctx) -> dict:
    """References to exact successful voice/AV raw outputs, committed with the result."""
    from app.engine.stages import common

    decoded = (ctx.outputs.get('decode') or {}).get('audio', {})
    record = {'version': 'video-audio-inputs/1', 'decode_status': decoded.get('status'),
              'source_audio_present': (ctx.outputs.get('probe') or {}).get('info', {}).get('has_audio'),
              'input': decoded.get('evidence_input'), 'voice_checks': [], 'av_checks': [], 'sound_checks': []}
    if (ctx.outputs.get('detect') or {}).get('voice_clock_policy') == VOICE_CLOCK_POLICY:
        record['voice_clock_policy'] = VOICE_CLOCK_POLICY
    if stereo_sha := (ctx.outputs.get('render') or {}).get('audio_stereo_context_sha256'):
        record['stereo_context_sha256'] = stereo_sha
    with ctx.sessions() as db:
        for entry in common.entries(ctx, ('detect',)):
            family = entry.get('family')
            if family not in {'voice', 'av-sync', 'acoustic-context'} or entry.get('status') != 'ok' or not entry.get('run_id'):
                continue
            run = db.get(DetectorRun, entry['run_id'])
            if run is None or run.analysis_id != ctx.analysis_id or not run.raw_output_artefact_id:
                raise ValueError('A video audio check has no stored raw input record')
            raw = get_artefact(db, run.raw_output_artefact_id, analysis_id=ctx.analysis_id)
            record[{'voice': 'voice_checks', 'av-sync': 'av_checks', 'acoustic-context': 'sound_checks'}[family]].append(
                {'detector_id': entry['detector_id'], 'raw_output_sha256': raw.sha256})
    return record


def unavailable(source_sha256: str, reason: str, *, status: str = 'unavailable') -> dict:
    return {'status': status, 'reason': reason, 'source': {'sha256': source_sha256},
            'coverage': None, 'overview': {}, 'packets': [], 'omitted_windows': 0, 'limitations': [reason]}


def build_channel(db, analysis, store, result: dict, *, set_id: str) -> dict:
    record = result.get('manifest', {}).get('sampling', {}).get('video_audio_evidence')
    if not record:
        return unavailable(analysis.sha256, 'This video predates frozen audio inputs. Reanalyse it to create source-bound voice evidence.')
    if record.get('source_audio_present') is False:
        return unavailable(analysis.sha256, 'The original video has no audio stream.', status='absent')
    if not record.get('input'):
        return unavailable(analysis.sha256, 'The source audio could not be decoded and frozen with a verified timeline.')
    content = audio_packets.build_content(db, analysis, store, result,
        record={'input': record['input'], 'checks': record['voice_checks'], 'sound_checks': record.get('sound_checks', []),
                'voice_clock_policy': record.get('voice_clock_policy'),
                'stereo_context_sha256': record.get('stereo_context_sha256')}, set_id=set_id)
    assessed = bool(content['coverage']['windows'])
    return {'status': 'ready' if assessed else 'not_assessed',
            'reason': None if assessed else 'No voice-check windows were scored. Frozen audio context is not a synthesis finding.',
            **content}


def validate_channel(channel: dict, *, source_sha256: str) -> None:
    from app.evidence.audio_binding import validate_audio_channel

    validate_audio_channel(channel, source_sha256)
    if (any(key in channel for key in ('id', 'sha256', 'analysis_id', 'revision', 'original_result_hash'))
            or channel.get('source', {}).get('sha256') != source_sha256
            or channel.get('status') not in {'ready', 'not_assessed', 'absent', 'unavailable'}):
        raise audio_packets.PacketIntegrityError('Video audio channel has independent or inconsistent source authority')
    if channel.get('coverage') is None:
        if channel.get('status') == 'ready' or channel.get('packets') or channel.get('overview') or not channel.get('reason'):
            raise audio_packets.PacketIntegrityError('Unavailable video audio channel contains unexplained evidence')
        return
    audio_packets.validate_content(channel)
    if (channel['status'] == 'ready') != bool(channel['coverage']['windows']):
        raise audio_packets.PacketIntegrityError('Video audio status differs from its measured coverage')


def verify_source_content(channel: dict, *, result: dict, store, source_sha256: str, read_waveform=None) -> None:
    """Verify measured voice content against its parent's frozen raw-run commitments."""
    from types import SimpleNamespace

    from app.detectors.audio.coverage import interval_union

    validate_channel(channel, source_sha256=source_sha256)
    record = result.get('manifest', {}).get('sampling', {}).get('video_audio_evidence') or {}
    binding = record.get('input')
    if channel['coverage'] is None:
        expected = 'absent' if record.get('source_audio_present') is False else 'unavailable'
        if channel['status'] != expected or (binding is not None and expected != 'absent'):
            raise audio_packets.PacketIntegrityError('Video voice availability disagrees with its frozen inputs')
        return
    if not binding or any(channel['source'].get(key) != binding.get(key)
                          for key in ('pcm_sha256', 'wav_sha256', 'mapping_sha256', 'samples', 'sample_rate')):
        raise audio_packets.PacketIntegrityError('Video voice input differs from its published commitment')
    pcm, mapping = audio_packets._input(store, SimpleNamespace(sha256=source_sha256), binding)
    if channel.get('stereo_context_sha256') != record.get('stereo_context_sha256'):
        raise audio_packets.PacketIntegrityError('Video stereo context differs from its published commitment')
    stereo_listening.verify(store, channel, binding.get('stereo_listening_pcm_sha256'), pcm, read_waveform=read_waveform)
    stereo_repetition.verify_frozen(store, channel, mapping, binding.get(STEREO_REPEAT_INPUT), store.read_bytes,
                                    gain_version=binding.get(GAIN_INPUT))
    if channel.get('sound_events') is not None:
        from app.evidence.sound_events import verify
        verify(channel['sound_events'], store=store, record=record, binding=binding,
               source=channel['source'], pcm=pcm, mapping=mapping, waveform=audio_packets._chart)
    windows = audio_packets._windows(store, result, {'checks': record['voice_checks']}, binding)
    coverage = channel['coverage']
    union = interval_union((w['start_sample'], w['end_sample']) for w in windows)
    intervals = [{'start_sample': a, 'end_sample': b} for a, b in union]
    if (canonical_hash(coverage['windows']) != canonical_hash(windows) or coverage['scored_intervals'] != intervals
            or coverage['scored_samples'] != sum(b-a for a,b in union)
            or coverage['unscored_samples'] != len(pcm)-sum(b-a for a,b in union)
            or coverage['decode_limit_reached'] != (len(pcm) >= mapping['limit_samples'])):
        raise audio_packets.PacketIntegrityError('Video voice coverage differs from actual scored windows')
    grouped = {}
    for window in windows:
        grouped.setdefault((window['start_sample'], window['end_sample']), []).append(window)
    candidates = [(span, values) for span, values in grouped.items() if max(v['score'] for v in values) >= audio_packets.THRESHOLD]
    candidates.sort(key=lambda item: (-max(v['score'] for v in item[1]), item[0]))
    if len(channel['packets']) != min(len(candidates), audio_packets.MAX_PACKETS) or channel['omitted_windows'] != max(0, len(candidates)-audio_packets.MAX_PACKETS):
        raise audio_packets.PacketIntegrityError('Video voice packet selection differs from its recorded responses')
    for packet, ((a,b), values) in zip(channel['packets'], candidates[:audio_packets.MAX_PACKETS], strict=True):
        expected = [{'label': 'Voice-check window response', 'value': v['score'], 'unit': 'uncalibrated score',
                     'signal_refs': v['signal_refs'], 'unit_ref': v['unit_ref'], 'raw_output_sha256': v['raw_output_sha256']}
                    for v in values]
        refs = {s['id']: s for value in values for s in value['signal_refs']}
        if (packet['interval'] != {'start_sample': a, 'end_sample': b}
                or canonical_hash(packet['observations']) != canonical_hash(expected)
                or canonical_hash(packet['signal_refs']) != canonical_hash(list(refs.values()))):
            raise audio_packets.PacketIntegrityError('Video voice packet differs from its actual scored input')
    if channel.get('acoustic_context') is not None:
        from app.evidence.audio_background import verify_source_content as verify_acoustics

        source = {key: channel['source'][key] for key in
                  ('sha256', 'pcm_sha256', 'samples', 'sample_rate', 'coordinate_space')}
        try:
            verify_acoustics(channel['acoustic_context'], pcm=pcm, mapping=mapping, source=source,
                             voice_packets=channel['packets'], read_asset=store.read_bytes, waveform=audio_packets._chart)
        except ValueError as error:
            raise audio_packets.PacketIntegrityError(str(error)) from error
