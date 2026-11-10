"""Frozen audio inputs and voice presentation inside one parent video evidence set."""
from __future__ import annotations

from app.core.canonical import canonical_json
from app.evidence import audio_packets
from app.storage.artefacts import get_artefact, register_artefact
from app.storage.db import DetectorRun


def freeze_input(ctx, decoded) -> dict:
    """Persist the already-decoded PCM once; never decode again to invent evidence."""
    with ctx.sessions() as db:
        wav = register_artefact(db, ctx.store, decoded.wav, media_type='audio/wav', role='working_copy',
                               created_by='video-audio-input/1', analysis_id=ctx.analysis_id)
        mapping = register_artefact(db, ctx.store, canonical_json(decoded.mapping), media_type='application/json',
                                   role='stage_output', created_by='video-audio-input/1', analysis_id=ctx.analysis_id)
        binding = {'version': 'audio-input/1', 'pcm_sha256': decoded.mapping['pcm_sha256'],
                   'wav_sha256': wav.sha256, 'mapping_sha256': mapping.sha256,
                   'sample_rate': decoded.mapping['sample_rate'], 'samples': len(decoded.pcm)}
        objects = [{'id': a.id, 'sha256': a.sha256, 'size_bytes': a.bytes, 'media_type': a.media_type,
                    'role': a.role, 'created_by': a.created_by} for a in (wav, mapping)]
        db.commit()
    return {'input': binding, 'artefacts': objects}


def published_record(ctx, signals: list[dict]) -> dict:
    """References to exact successful voice/AV raw outputs, committed with the result."""
    from app.engine.stages import common

    decoded = (ctx.outputs.get('decode') or {}).get('audio', {})
    published_voice_ids = {s['detector_id'] for s in signals if s.get('family') == 'voice' and s.get('status') == 'ok'}
    record = {'version': 'video-audio-inputs/1', 'decode_status': decoded.get('status'),
              'source_audio_present': (ctx.outputs.get('probe') or {}).get('info', {}).get('has_audio'),
              'input': decoded.get('evidence_input'), 'voice_checks': [], 'av_checks': [], 'sound_checks': []}
    with ctx.sessions() as db:
        for entry in common.entries(ctx, ('detect',)):
            family = entry.get('family')
            if family not in {'voice', 'av-sync', 'acoustic-context'} or entry.get('status') != 'ok' or not entry.get('run_id'):
                continue
            if family == 'voice' and entry['detector_id'] not in published_voice_ids:
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
        record={'input': record['input'], 'checks': record['voice_checks'], 'sound_checks': record.get('sound_checks', [])}, set_id=set_id)
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


def verify_source_content(channel: dict, *, result: dict, store, source_sha256: str) -> None:
    """Verify measured voice content against its parent's frozen raw-run commitments."""
    from types import SimpleNamespace

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
    if channel.get('sound_events') is not None:
        from app.evidence.sound_events import verify
        verify(channel['sound_events'], store=store, record=record, binding=binding,
               source=channel['source'], pcm=pcm, mapping=mapping, waveform=audio_packets._chart)
    audio_packets.verify_responses(channel, store=store, result=result,
                                   record={'checks': record['voice_checks']}, binding=binding)
    if channel['coverage']['decode_limit_reached'] != (len(pcm) >= mapping['limit_samples']):
        raise audio_packets.PacketIntegrityError('Video voice decoding limit differs from its frozen input')
    if channel.get('acoustic_context') is not None:
        from app.evidence.audio_background import verify_source_content as verify_acoustics

        source = {key: channel['source'][key] for key in
                  ('sha256', 'pcm_sha256', 'samples', 'sample_rate', 'coordinate_space')}
        try:
            verify_acoustics(channel['acoustic_context'], pcm=pcm, mapping=mapping, source=source,
                             voice_packets=channel['packets'], read_asset=store.read_bytes, waveform=audio_packets._chart)
        except ValueError as error:
            raise audio_packets.PacketIntegrityError(str(error)) from error
