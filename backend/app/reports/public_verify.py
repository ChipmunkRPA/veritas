"""Offline public-derivative checks: standard library and canonical JSON only.

This intentionally does not import application settings, model registries, databases,
or modality runtimes. It verifies signed public commitments, not inference accuracy.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
import wave

from app.core.canonical import canonical_hash
from app.evidence.face_interval_binding import validate_packet as validate_face_interval
from app.evidence.mp4 import inspect_mp4, validate_clip_binding

SCHEMA = 'chipmunk.public-evidence/1'
_SHA = re.compile(r'[a-f0-9]{64}')
TYPES = {'image/png': 'png', 'audio/wav': 'wav', 'video/mp4': 'mp4'}
VERSIONS = {'image': {'image-evidence-1', 'image-evidence-2', 'image-evidence-3'}, 'video': {'video-evidence-1', 'video-evidence-2', 'video-evidence-3', 'video-evidence-4', 'video-evidence-5', 'video-evidence-6', 'video-evidence-7', 'video-evidence-8'},
            'audio': {'audio-evidence-1', 'audio-evidence-2', 'audio-evidence-3', 'audio-evidence-4'}, 'document': {'document-evidence-1', 'document-evidence-2'}}


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise ValueError(detail)


def _document(archive, names, reference, expected_file):
    _require(isinstance(reference, dict) and reference.get('file') == expected_file and expected_file in names,
             f'The {expected_file} reference is missing or invalid')
    data = archive.read(expected_file)
    _require(hashlib.sha256(data).hexdigest() == reference.get('sha256'), f'{expected_file} has a different committed hash')
    document = json.loads(data)
    _require(isinstance(document, dict), f'{expected_file} must be a JSON object')
    return document


def _checks(archive, names: set[str], bundle: dict) -> list[tuple[str, bool, str]]:
    _require(bundle.get('reproduction_supported') is False, 'A public derivative cannot claim inference reproduction')
    scope = bundle.get('verification_scope')
    if scope is not None:  # Historical public packages retain their original format.
        _require(isinstance(scope, dict) and set(scope) == {'type', 'inference_reproduced', 'confidence_recalculated', 'calibration_parameters_included'}
                 and scope['type'] == 'document-integrity-only'
                 and all(scope[key] is False for key in ('inference_reproduced', 'confidence_recalculated', 'calibration_parameters_included')),
                 'A public derivative cannot claim calibration recalculation or included private parameters')
    analysis = bundle['analysis']
    _require(isinstance(analysis, dict) and analysis.get('kind') in {'image', 'video', 'audio', 'document'},
             'The public analysis identity is unsupported')
    source = bundle['source_commitment']
    _require(isinstance(source, dict) and bool(_SHA.fullmatch(str(source.get('sha256', '')))), 'Invalid source commitment')
    _require(bool(_SHA.fullmatch(str(analysis.get('original_result_sha256', '')))), 'Invalid original-result commitment')
    result = _document(archive, names, bundle.get('result'), 'result.json')
    _require(result.get('schema') == 'chipmunk.public-result/1' and result.get('is_projection') is True
             and bundle['result'].get('is_projection') is True, 'The result must identify itself as a public projection')
    _require(result.get('original_result_sha256') == analysis['original_result_sha256'], 'The result commits to a different original')
    _require(set(result) <= {'schema', 'is_projection', 'original_result_sha256', 'decision', 'findings', 'checks', 'note'},
             'The public result contains unsupported internal fields')
    evidence = _document(archive, names, bundle.get('evidence'), 'evidence.json')
    binding = bundle['evidence']
    _require(evidence.get('schema') == f"chipmunk.{analysis['kind']}-evidence/1" and evidence.get('status') == 'ready',
             'The evidence schema or readiness does not match the analysis')
    _require(evidence.get('version') in VERSIONS[analysis['kind']], 'The evidence version is not supported by this verifier')
    actual = canonical_hash({k: v for k, v in evidence.items() if k != 'sha256'})
    _require(actual == evidence.get('sha256') == binding.get('set_sha256'), 'The evidence set hash does not recompute')
    _require(evidence.get('id') == binding.get('id') and evidence.get('version') == binding.get('version')
             and evidence['schema'] == binding.get('schema'), 'The evidence set identity or version differs')
    _require(evidence.get('analysis_id') == analysis.get('id') and evidence.get('revision') == analysis.get('revision')
             and evidence.get('original_result_hash') == analysis['original_result_sha256']
             and evidence.get('source', {}).get('sha256') == source['sha256'] == binding.get('source_sha256'),
             'The evidence source/result/revision binding differs')
    if analysis['kind'] == 'video':
        for packet in evidence.get('packets', []):
            validate_clip_binding(packet, evidence['source'])
            validate_face_interval(packet, evidence['source'])
        if evidence['version'] in {'video-evidence-3', 'video-evidence-4', 'video-evidence-5', 'video-evidence-6', 'video-evidence-7', 'video-evidence-8'}:
            from app.evidence.audio_binding import validate_audio_channel
            from app.evidence.av_binding import validate_av_channel

            validate_audio_channel(evidence['audio'], source['sha256'])
            if evidence['version'] == 'video-evidence-8':
                from app.evidence.audio_response_binding import validate as validate_responses

                validate_responses(evidence['audio'], required=True)
            if evidence['version'] in {'video-evidence-4', 'video-evidence-5', 'video-evidence-6', 'video-evidence-7', 'video-evidence-8'} and evidence['audio']['coverage'] is not None:
                _require(bool(evidence['audio'].get('acoustic_context')), 'Video acoustic context is missing')
            if evidence['version'] in {'video-evidence-5', 'video-evidence-6', 'video-evidence-7', 'video-evidence-8'} and evidence['audio']['coverage'] is not None:
                _require(evidence['audio']['acoustic_context']['schema'] == 'chipmunk.acoustic-context/2', 'Video continuity measurements are missing')
            validate_av_channel(evidence['av'], evidence['source'])
            packet_ids = [p['id'] for part in (evidence, evidence['audio'], evidence['av']) for p in part['packets']]
            _require(len(packet_ids) == len(set(packet_ids)), 'Video evidence packet identifiers collide across channels')
            checks = {check['id'] for check in result.get('checks', [])}
            _require(all(ref['id'] in checks for key in ('audio', 'av') for packet in evidence[key]['packets']
                         for ref in packet['signal_refs']), 'A nested video observation has no corresponding published check')
            if all(evidence[key]['coverage'] is not None for key in ('audio', 'av')):
                _require(all(evidence['audio']['source'][key] == evidence['av']['source'][key]
                             for key in ('pcm_sha256', 'wav_sha256', 'mapping_sha256', 'samples', 'sample_rate', 'timeline')),
                         'Video voice and alignment evidence use different audio inputs')
        else:
            _require(not {'audio', 'av'}.intersection(evidence), 'Historical video evidence has unexpected nested channels')
    if analysis['kind'] == 'image':
        from app.evidence.image_eligibility import validate_inspection_coverage

        validate_inspection_coverage(evidence)

    refs = {}
    parts = [evidence]
    if evidence['version'] in {'video-evidence-3', 'video-evidence-4', 'video-evidence-5', 'video-evidence-6', 'video-evidence-7', 'video-evidence-8'}:
        parts.extend(evidence[key] for key in ('audio', 'av'))
    if analysis['kind'] == 'audio':
        if evidence['version'] == 'audio-evidence-4':
            from app.evidence.audio_response_binding import validate as validate_responses

            validate_responses(evidence, required=True)
        acoustic = evidence.get('acoustic_context')
        if evidence['version'] in {'audio-evidence-2', 'audio-evidence-3', 'audio-evidence-4'}:
            _require(bool(acoustic), 'Audio acoustic context is missing')
        if evidence['version'] in {'audio-evidence-3', 'audio-evidence-4'}:
            _require(acoustic['schema'] == 'chipmunk.acoustic-context/2', 'Audio continuity measurements are missing')
        if acoustic is not None:
            from app.evidence.audio_background_binding import validate_acoustic_context

            validate_acoustic_context(acoustic, source=evidence['source'], voice_packets=evidence['packets'])
    groups = [group for part in parts for group in
              [part.get('overview') or {}, *(p.get('artifacts') or {} for p in part.get('packets', []))]]
    for part in parts:
        acoustic = part.get('acoustic_context')
        if acoustic:
            groups.extend([acoustic.get('overview') or {}, *(event.get('artifacts') or {} for event in acoustic.get('events', []))])
        sound = part.get('sound_events')
        if sound is not None:
            from app.evidence.sound_event_binding import validate as validate_sound_events
            validate_sound_events(sound, part['source'])
            groups.extend(window.get('artifacts') or {} for window in sound['windows'])
    for group in groups:
        _require(isinstance(group, dict), 'Invalid evidence assets')
        for asset in group.values():
            _require(isinstance(asset, dict) and asset.get('media_type') in TYPES
                     and bool(_SHA.fullmatch(str(asset.get('sha256', '')))), 'Invalid evidence asset reference')
            digest = asset['sha256']
            intrinsic = {k: asset.get(k) for k in ('media_type', 'width', 'height', 'samples', 'sample_rate', 'channels',
                                                 'frame_count', 'time_base', 'pts', 'durations', 'duration_ticks',
                                                 'duration_s', 'codec', 'has_audio')}
            _require(digest not in refs or refs[digest][1] == intrinsic, 'One evidence object has conflicting metadata')
            refs[digest] = (asset, intrinsic)
    entries = bundle.get('artifacts')
    _require(isinstance(entries, list), 'The artifact index is missing')
    indexed = {}
    for entry in entries:
        _require(isinstance(entry, dict) and entry.get('sha256') not in indexed, 'Duplicate artifact index entry')
        indexed[entry.get('sha256')] = entry
    _require(set(indexed) == set(refs), 'Artifact index does not exactly cover the frozen evidence set')
    for digest, (asset, _) in refs.items():
        entry = indexed[digest]
        path = f"assets/{digest}.{TYPES[asset['media_type']]}"
        _require(entry.get('file') == path and path in names and entry.get('media_type') == asset['media_type'],
                 'An artifact path or media type differs from its frozen reference')
        _require(entry.get('bytes') == archive.getinfo(path).file_size, 'An artifact size differs from its index')
        recorded = hashlib.sha256()
        with archive.open(path) as stream:
            while chunk := stream.read(1024 * 1024):
                recorded.update(chunk)
        _require(recorded.hexdigest() == digest, 'An artifact differs from its frozen evidence hash')
        with archive.open(path) as stream:
            if asset['media_type'] == 'image/png':
                header = stream.read(24)
                _require(header[:8] == b'\x89PNG\r\n\x1a\n' and header[12:16] == b'IHDR', 'A PNG asset has an invalid header')
                width, height = struct.unpack('>II', header[16:24])
                _require(width == asset.get('width') and height == asset.get('height'), 'A PNG asset has different dimensions')
            elif asset['media_type'] == 'audio/wav':
                with wave.open(stream, 'rb') as audio:
                    _require(audio.getnframes() == asset.get('samples') and audio.getframerate() == asset.get('sample_rate')
                             and audio.getnchannels() == asset.get('channels') and audio.getsampwidth() == 2,
                             'A WAV asset does not match its frozen sample dimensions')
            else:
                dimensions = inspect_mp4(stream)
                _require(all(canonical_hash(asset.get(k)) == canonical_hash(v) for k, v in dimensions.items()),
                         'An MP4 asset does not match its frozen frames, timing or track format')
    expected = {'README.txt', 'reproduce.md', 'bundle.json', 'evidence.json', 'result.json', 'report.pdf',
                'instance_public_key.pem', 'manifest.sha256', 'manifest.sha256.sig', *(a['file'] for a in entries)}
    original = bundle.get('source')
    if original is not None:
        _require(isinstance(original, dict) and bool(re.fullmatch(r'source/original\.[a-z0-9]{1,8}', str(original.get('file')))),
                 'Invalid optional original path')
        path = original['file']
        _require(path in names and original.get('sha256') == source['sha256'] and original.get('bytes') == source.get('bytes')
                 and archive.getinfo(path).file_size == source.get('bytes'), 'The optional original identity or size differs')
        digest = hashlib.sha256()
        with archive.open(path) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        _require(digest.hexdigest() == source['sha256'], 'The optional original does not match its source commitment')
        expected.add(path)
    _require(names == expected, 'Unexpected files or missing declared files in the public derivative')
    reports = bundle.get('reports')
    _require(isinstance(reports, list) and len(reports) == 1, 'Exactly one frozen report must be included')
    report = reports[0]
    _require(bundle.get('signing') == {'algorithm': 'Ed25519', 'key_id': report.get('signing_key_id'),
                                      'public_key_file': 'instance_public_key.pem'}, 'The signing declaration differs from the report signer')
    _require(report.get('file') == 'report.pdf' and report.get('analysis_id') == analysis['id']
             and report.get('evidence_set_sha256') == actual
             and bool(_SHA.fullmatch(str(report.get('snapshot_sha256', '')))), 'The report identity does not bind to this evidence')
    _require(bundle.get('kind') in ('draft', 'issue') and report.get('issued') == (bundle['kind'] == 'issue'),
             'The report and package draft/issue states disagree')

    custody = bundle.get('custody')
    _require(isinstance(custody, dict) and custody.get('schema') == 'chipmunk.custody-commitments/1'
             and custody.get('verification') == 'commitments_only' and custody.get('complete_event_chain_included') is False,
             'Public custody must be labelled as commitments only, not a verified original chain')
    events, anchor = custody['events'], custody['anchor']
    _require(isinstance(events, list) and isinstance(anchor, dict) and isinstance(anchor.get('head_seq'), int)
             and bool(_SHA.fullmatch(str(anchor.get('head_hash', '')))), 'Invalid custody commitment anchor')
    previous = 0
    for event in events:
        _require(isinstance(event, dict) and set(event) == {'seq', 'event_hash'} and isinstance(event['seq'], int)
                 and previous < event['seq'] <= anchor['head_seq'] and bool(_SHA.fullmatch(str(event['event_hash']))),
                 'Invalid or unordered custody commitment')
        if event['seq'] == anchor['head_seq']:
            _require(event['event_hash'] == anchor['head_hash'], 'The final custody commitment differs from its anchor')
        previous = event['seq']
    return [('public evidence', True, f'{len(refs)} declared assets; source, result, version and report commitments agree'),
            ('custody commitments', True, f'{len(events)} ordered commitments; original event bodies and chain completeness are NOT verified'),
            ('verification scope', True, 'Signed public derivative integrity only; confidence estimates and calibration are NOT recalculated; no inference reproduction or AI-authorship proof')]


def checks(archive, names: set[str], bundle: dict) -> list[tuple[str, bool, str]]:
    try:
        return _checks(archive, names, bundle)
    except (KeyError, TypeError, ValueError, AttributeError, struct.error, wave.Error) as error:
        return [('public evidence', False, str(error) or 'Malformed public evidence derivative')]
